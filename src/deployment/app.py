"""FastAPI inference service for the supply chain risk optimizer.

Exposes a versioned ``POST /v1/predict`` endpoint (the ``/v1`` prefix is a
strict SAP AI Core inference-routing requirement). Each request:

    1. Ingests a tabular feature record.
    2. Runs the PyTorch stock-out classifier.
    3. Extracts a SHAP explanation.
    4. Triggers the LangGraph agent, which proposes a mitigation and pauses at
       the human-in-the-loop interrupt.
    5. Returns the risk score, explanation, and proposed action.

For offline runnability the service trains a lightweight model on startup. In
production the model + SHAP explainer would be loaded from the MLflow registry.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph import build_graph
from src.agent.state import SupplyChainState
from src.features.build_features import FEATURE_COLUMNS
from src.models.explainability import StockoutExplainer
from src.models.train import train

# Populated at startup; holds the trained model, explainer, and agent graph.
_STATE: Dict[str, object] = {}


class PredictRequest(BaseModel):
    """Request payload: one feature record keyed by feature name."""

    features: Dict[str, float] = Field(
        description="Mapping of feature name -> value for a single observation."
    )


class PredictResponse(BaseModel):
    """Response payload: risk score, SHAP explanation, and agent proposal."""

    risk_score: float
    is_stockout: bool
    explanation: Dict[str, float]
    root_cause_summary: str
    proposed_action: Dict[str, object]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train a lightweight model and build the explainer/agent at startup."""
    artifacts = train(max_epochs=3)
    explainer = StockoutExplainer(
        artifacts.model,
        background=artifacts.data.x_train,
        feature_names=list(artifacts.data.feature_names),
    )
    _STATE["model"] = artifacts.model
    _STATE["scaler"] = artifacts.data.scaler
    _STATE["explainer"] = explainer
    _STATE["graph"] = build_graph()
    yield
    _STATE.clear()


app = FastAPI(title="Supply Chain Risk Optimizer", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
def health() -> Dict[str, str]:
    """Liveness probe for Kubernetes / KServe."""
    return {"status": "ok"}


def _vectorize(features: Dict[str, float]) -> np.ndarray:
    """Order and scale an incoming feature record into a model input row.

    Args:
        features: Feature-name -> value mapping from the request.

    Returns:
        A scaled 1-row float32 array aligned to ``FEATURE_COLUMNS``.

    Raises:
        HTTPException: If any required feature is missing.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    row = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
    scaler = _STATE["scaler"]
    return scaler.transform(row).astype(np.float32)


@app.post("/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run inference, explanation, and the mitigation agent for one record."""
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    scaled = _vectorize(request.features)

    explainer: StockoutExplainer = _STATE["explainer"]  # type: ignore[assignment]
    explanation = explainer.explain(scaled[0], top_k=5)

    # Trigger the agent; it proposes a mitigation and pauses at the HITL gate.
    graph = _STATE["graph"]
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial = SupplyChainState(
        prediction_data={"prediction": explanation["prediction"]},
        shap_explanation=explanation["top_features"],
        audit_log=[],
    )
    graph.invoke(initial, config=config)  # type: ignore[attr-defined]
    snapshot = graph.get_state(config)  # type: ignore[attr-defined]

    return PredictResponse(
        risk_score=explanation["prediction"],
        is_stockout=explanation["is_stockout"],
        explanation=explanation["top_features"],
        root_cause_summary=snapshot.values.get("root_cause_summary", ""),
        proposed_action=snapshot.values.get("proposed_action", {}),
    )
