"""Phase 4 tests: SAP AI Core tracking mock and MLflow run integration."""

from __future__ import annotations

import mlflow

from src.config import settings
from src.mlops.sap_tracker import MockTracking, log_to_sap_ai_core


def test_sap_tracker_records_metrics_and_tags() -> None:
    """The mock SAP tracker must capture the exact metrics and tags."""
    tracker = log_to_sap_ai_core(
        metrics={"PR-AUC": 0.81, "F2": 0.74},
        tags={"model_type": "focal_mlp"},
    )
    assert isinstance(tracker, MockTracking)
    metric_names = {m["name"] for m in tracker.logged_metrics}
    assert metric_names == {"PR-AUC", "F2"}
    assert tracker.logged_tags[0]["value"] == "focal_mlp"


def test_mlflow_tracked_training_logs_metrics(tmp_path, monkeypatch) -> None:
    """A short tracked run must log imbalance metrics to MLflow and no accuracy."""
    # Isolate the tracking store in a temp SQLite DB.
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(settings, "mlflow_tracking_uri", uri)
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_exp")

    from src.mlops.track import run_tracked_training

    metrics = run_tracked_training(max_epochs=1)

    assert set(metrics) == {"test_precision", "test_recall", "test_f2", "test_prauc"}
    assert "accuracy" not in metrics

    # Verify the run was actually persisted to the tracking store.
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("test_exp")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 1
    assert "metrics.test_prauc" in runs.columns
