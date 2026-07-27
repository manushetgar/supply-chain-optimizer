"""MLflow-tracked training run for the stock-out classifier.

Wraps the Phase 2 training loop in an MLflow run: logs all hyperparameters,
per-epoch and final imbalance-aware metrics (PR-AUC, F2, Precision, Recall),
and persists the model, SHAP explainer, and requirements as artifacts. The
best model is registered to the MLflow Model Registry. A mocked SAP AI Core
tracking call demonstrates the productive SAP integration pattern.

The tracking URI is never hardcoded; it is resolved from environment-driven
settings (``MLFLOW_TRACKING_URI``).
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Dict

import mlflow
import mlflow.pytorch

from src.config import PROJECT_ROOT, settings
from src.mlops.sap_tracker import log_to_sap_ai_core
from src.models.explainability import StockoutExplainer
from src.models.train import TrainingArtifacts, train


def _configure_mlflow() -> None:
    """Point MLflow at the configured tracking URI and experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _log_artifacts(artifacts: TrainingArtifacts, tmp: Path) -> str:
    """Persist model weights, SHAP explainer, and requirements as artifacts.

    The PyTorch model is logged via ``mlflow.pytorch.log_model`` (excluding
    optimizer/gradient state) so it is registrable and servable. The SHAP
    explainer and requirements are logged as supplementary artifacts.

    Returns:
        The model URI usable for Model Registry registration.
    """
    # Log the network (weights + architecture) as a first-class MLflow model.
    # A representative input example lets MLflow infer the serving signature.
    input_example = artifacts.data.x_train[:2]
    model_info = mlflow.pytorch.log_model(
        pytorch_model=artifacts.model.model,
        name="model",
        input_example=input_example,
    )

    # SHAP explainer built over a background sample of the training set.
    explainer = StockoutExplainer(
        artifacts.model,
        background=artifacts.data.x_train,
        feature_names=list(artifacts.data.feature_names),
    )
    explainer_path = tmp / "shap_explainer.pkl"
    with explainer_path.open("wb") as fh:
        pickle.dump(explainer, fh)
    mlflow.log_artifact(str(explainer_path), artifact_path="explainer")

    # Reproducible environment snapshot.
    req = PROJECT_ROOT / "requirements.txt"
    if req.exists():
        mlflow.log_artifact(str(req))

    return model_info.model_uri


def run_tracked_training(
    max_epochs: int = 15,
    gamma: float = 2.0,
    alpha: float = 0.25,
    learning_rate: float = 1e-3,
) -> Dict[str, float]:
    """Execute a fully MLflow-tracked training run.

    Args:
        max_epochs: Maximum training epochs.
        gamma: Focal Loss focusing parameter.
        alpha: Focal Loss positive-class weight.
        learning_rate: AdamW learning rate.

    Returns:
        The out-of-time test metrics for the run.
    """
    _configure_mlflow()

    with mlflow.start_run(run_name="focal_mlp_stockout") as run:
        artifacts = train(
            max_epochs=max_epochs,
            gamma=gamma,
            alpha=alpha,
            learning_rate=learning_rate,
        )

        # Parameters: architecture, focal-loss config, optimizer, data meta.
        mlflow.log_params(artifacts.hyperparams)
        mlflow.log_param("n_train_rows", len(artifacts.data.y_train))
        mlflow.log_param("n_features", artifacts.data.x_train.shape[1])
        mlflow.set_tag("loss_function", "focal_loss")

        # Metrics: imbalance-aware only (accuracy is intentionally excluded).
        mlflow.log_metrics(artifacts.test_metrics)

        with tempfile.TemporaryDirectory() as td:
            model_uri = _log_artifacts(artifacts, Path(td))

        # Register the model to the MLflow Model Registry.
        mlflow.register_model(
            model_uri=model_uri,
            name="supply_chain_stockout_predictor",
        )

        # Demonstrate the productive SAP AI Core tracking pattern (mocked).
        log_to_sap_ai_core(
            metrics=artifacts.test_metrics,
            tags={
                "scenario": settings.sap_ai_core_scenario,
                "executable": settings.sap_ai_core_executable,
                "loss_function": "focal_loss",
            },
        )

    return artifacts.test_metrics


def main() -> None:
    """CLI entry point: run tracked training and print metrics."""
    metrics = run_tracked_training()
    print("MLflow run complete. Out-of-time test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
