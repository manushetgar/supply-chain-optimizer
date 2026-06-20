"""PyTorch Lightning MLP for imbalanced tabular stock-out prediction.

The architecture is a multi-layer perceptron with Batch Normalization,
Dropout, and ReLU activations, optimized with AdamW under a Focal Loss
objective. Evaluation deliberately excludes accuracy in favor of metrics
appropriate for extreme imbalance: Precision, Recall, F2-Score, and PR-AUC.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryFBetaScore,
    BinaryPrecision,
    BinaryRecall,
)

from src.models.focal_loss import BinaryFocalLoss


class TabularMLP(nn.Module):
    """A configurable MLP encoder for tabular feature vectors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ) -> None:
        """Build the MLP.

        Args:
            input_dim: Number of input features.
            hidden_dims: Widths of successive hidden layers.
            dropout: Dropout probability applied after each hidden block.
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        layers.append(nn.Linear(prev, 1))  # single logit output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(N,)``."""
        return self.net(x).view(-1)


class StockoutClassifier(pl.LightningModule):
    """LightningModule wrapping the MLP with Focal Loss and imbalance metrics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        gamma: float = 2.0,
        alpha: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        """Initialize the classifier.

        Args:
            input_dim: Number of input features.
            hidden_dims: Hidden layer widths.
            dropout: Dropout probability.
            gamma: Focal Loss focusing parameter.
            alpha: Focal Loss positive-class weight.
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = TabularMLP(input_dim, hidden_dims, dropout)
        self.loss_fn = BinaryFocalLoss(gamma=gamma, alpha=alpha)

        # Metrics appropriate for extreme imbalance (accuracy is forbidden).
        # F2 weights recall higher than precision (beta=2) to penalize missed
        # stock-outs, which are operationally catastrophic.
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f2 = BinaryFBetaScore(beta=2.0)
        self.val_prauc = BinaryAveragePrecision()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of feature vectors."""
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        """Compute and log the training focal loss."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: int) -> None:
        """Update imbalance metrics and log validation loss."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        target = y.view(-1).long()

        self.val_precision.update(probs, target)
        self.val_recall.update(probs, target)
        self.val_f2.update(probs, target)
        self.val_prauc.update(probs, target)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self) -> None:
        """Compute, log, and reset the epoch-level imbalance metrics."""
        metrics: Dict[str, torch.Tensor] = {
            "val_precision": self.val_precision.compute(),
            "val_recall": self.val_recall.compute(),
            "val_f2": self.val_f2.compute(),
            "val_prauc": self.val_prauc.compute(),
        }
        self.log_dict(metrics, prog_bar=True)
        for metric in (
            self.val_precision,
            self.val_recall,
            self.val_f2,
            self.val_prauc,
        ):
            metric.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the AdamW optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
