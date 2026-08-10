# AI-Driven Supply Chain Risk & Inventory Optimizer

An end-to-end, enterprise-style ML system that predicts rare inventory
stock-out events from simulated SAP S/4HANA Materials Management data,
explains each prediction with SHAP, and routes the intelligence into a
human-in-the-loop LangGraph agent that proposes and (upon approval) simulates
an ERP write-back. The system is MLflow-tracked and packaged for a KServe /
SAP AI Core BYOM deployment.

Everything runs **fully offline** — no API keys or network access required.
The agent's LLM calls use a deterministic mock (`LLM_PROVIDER=mock`).

## Architecture

| Phase | Layer | Key modules |
|-------|-------|-------------|
| 1 | Data engineering & leakage prevention | `src/data/make_dataset.py`, `src/features/build_features.py`, `src/data/split_data.py` |
| 2 | Modeling & explainability | `src/models/focal_loss.py`, `src/models/pytorch_model.py`, `src/models/baseline.py`, `src/models/explainability.py` |
| 3 | Agentic AI (LangGraph, HITL) | `src/agent/state.py`, `src/agent/nodes.py`, `src/agent/graph.py`, `src/agent/tools.py` |
| 4 | MLOps (MLflow + SAP mock) | `src/mlops/track.py`, `src/mlops/sap_tracker.py` |
| 5 | CI/CD & deployment | `src/deployment/app.py`, `Dockerfile`, `deployment/serving-template.yaml`, `.github/workflows/deploy.yml` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; sensible defaults are built in
```

## Running the pipeline

```bash
# Phase 1: synthesize data -> engineer features -> validated chronological split
python -m src.data.make_dataset
python -m src.features.build_features
python -m src.data.split_data

# Phase 2: train the Focal-Loss MLP and report imbalance metrics
python -m src.models.train

# Phase 4: MLflow-tracked training + model registration
python -m src.mlops.track

# Phase 5: serve the inference + agent API
uvicorn src.deployment.app:app --host 0.0.0.0 --port 8080
```

### Example request

```bash
curl -X POST localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"planned_lead_time": 10, "actual_lead_time": 25,
       "vendor_lead_time_deviation": 15, "unrestricted_stock": 5, "eisbe": 100,
       "safety_stock_penetration_ratio": 20, "daily_consumption": 12,
       "rolling_7d_consumption": 11, "rolling_30d_consumption": 10,
       "days_of_cover": 0.4, "po_open_qty": 0, "unit_price": 42,
       "mean_delay_days": 6}}'
```

## Testing

```bash
pytest            # full suite across all 5 phases
flake8 src tests  # linting
```

## Key design decisions

- **No temporal leakage.** Splits are strictly chronological (out-of-time);
  `validate_no_leakage` raises `RuntimeError` on any overlap.
- **Imbalance-first.** Focal Loss + PR-AUC / F2 / Precision / Recall. Accuracy
  is deliberately never used as a metric.
- **Glass-box.** SHAP decomposes each prediction into a top-feature JSON payload
  consumed by the agent.
- **Human-in-the-loop.** The LangGraph graph interrupts before the ERP
  write-back; execution resumes only after approval is injected into the
  persisted state.
- **Deployment-ready.** Multi-stage Docker image runs as non-root `nobody`,
  binds `0.0.0.0:$PORT`, and ships a KServe `serving-template.yaml` with SAP AI
  Core annotations and a dynamic `{{inputs.parameters.image}}` placeholder.

## Docker

```bash
docker build -t supply-chain-optimizer .
docker run -p 8080:8080 supply-chain-optimizer
```
