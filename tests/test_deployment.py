"""Phase 5 tests: FastAPI inference service and endpoint contract."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from fastapi.testclient import TestClient

from src.config import PROCESSED_DATA_DIR
from src.deployment.app import app
from src.features.build_features import FEATURE_COLUMNS


def _sample_features() -> Dict[str, float]:
    """Pull one real feature record from the processed dataset."""
    df = pd.read_parquet(PROCESSED_DATA_DIR / "features.parquet")
    row = df.iloc[0]
    return {c: float(row[c]) for c in FEATURE_COLUMNS}


def test_healthz() -> None:
    """Liveness probe must return ok."""
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_predict_endpoint_contract() -> None:
    """The versioned predict endpoint must return risk + explanation + action."""
    with TestClient(app) as client:
        resp = client.post("/v1/predict", json={"features": _sample_features()})
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert isinstance(body["explanation"], dict)
        assert body["root_cause_summary"]
        assert body["proposed_action"]["action"] in {
            "expedite_purchase_order",
            "transfer_plant_stock",
        }


def test_predict_rejects_missing_features() -> None:
    """Missing features must yield a 422 with a descriptive message."""
    with TestClient(app) as client:
        resp = client.post("/v1/predict", json={"features": {"days_of_cover": 1.0}})
        assert resp.status_code == 422
