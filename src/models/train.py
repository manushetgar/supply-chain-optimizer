"""End-to-end training orchestration for the stock-out classifier.

Loads the leakage-safe chronological split, trains the Focal-Loss MLP with
PyTorch Lightning, evaluates on the out-of-time test set with imbalance-aware
metrics, and returns the trained model plus prepared data for explainability
and MLOps logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pytorch_lightning as pl
import torch
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryFBetaScore,
    BinaryPrecision,
    BinaryRecall,
)

from src.data.split_data import load_split
from src.models.datamodule import PreparedData, make_loaders, prepare_arrays
from src.models.pytorch_model import StockoutClassifier
from src.utils.seed import seed_everything


@dataclass
class TrainingArtifacts:
    """Outputs of a training run.

    Attributes:
        model: The trained classifier.
        data: The prepared, scaled arrays and fitted scaler.
        test_metrics: Imbalance-aware metrics on the out-of-time test set.
        hyperparams: The resolved hyperparameters used for the run.
    """

    model: StockoutClassifier
    data: PreparedData
    test_metrics: Dict[str, float]
    hyperparams: Dict[str, float]


def _evaluate(model: StockoutClassifier, x, y) -> Dict[str, float]:
    """Compute imbalance-aware test metrics with torchmetrics."""
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x)))
    target = torch.from_numpy(y).long()

    return {
        "test_precision": float(BinaryPrecision()(probs, target)),
        "test_recall": float(BinaryRecall()(probs, target)),
        "test_f2": float(BinaryFBetaScore(beta=2.0)(probs, target)),
        "test_prauc": float(BinaryAveragePrecision()(probs, target)),
    }


def train(
    max_epochs: int = 15,
    gamma: float = 2.0,
    alpha: float = 0.25,
    learning_rate: float = 1e-3,
    batch_size: int = 512,
) -> TrainingArtifacts:
    """Train and evaluate the stock-out classifier.

    Args:
        max_epochs: Maximum training epochs.
        gamma: Focal Loss focusing parameter.
        alpha: Focal Loss positive-class weight.
        learning_rate: AdamW learning rate.
        batch_size: Mini-batch size.

    Returns:
        A :class:`TrainingArtifacts` bundle.
    """
    seed_everything()
    split, _ = load_split()
    data = prepare_arrays(split)
    train_loader, val_loader, _ = make_loaders(data, batch_size=batch_size)

    model = StockoutClassifier(
        input_dim=data.x_train.shape[1],
        gamma=gamma,
        alpha=alpha,
        learning_rate=learning_rate,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)

    test_metrics = _evaluate(model, data.x_test, data.y_test)
    hyperparams = {
        "gamma": gamma,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "input_dim": data.x_train.shape[1],
    }
    return TrainingArtifacts(
        model=model, data=data, test_metrics=test_metrics, hyperparams=hyperparams
    )


def main() -> None:
    """CLI entry point: train and print out-of-time test metrics."""
    artifacts = train()
    print("Out-of-time test metrics (accuracy intentionally excluded):")
    for name, value in artifacts.test_metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
