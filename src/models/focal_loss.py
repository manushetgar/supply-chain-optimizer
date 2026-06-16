"""Numerically stable Binary Focal Loss for extreme class imbalance.

Focal Loss (Lin et al., 2017) reshapes standard cross-entropy so that
well-classified examples contribute little to the gradient, forcing the model
to focus on the hard, rare minority class (stock-outs).

Formulation:
    FL(p_t) = -alpha_t * (1 - p_t) ** gamma * log(p_t)

where ``p_t`` is the model's estimated probability of the true class. This
implementation operates on raw logits and uses the log-sum-exp-stable
``binary_cross_entropy_with_logits`` primitive to avoid overflow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss operating on logits.

    Attributes:
        gamma: Focusing parameter; higher values down-weight easy examples
            more aggressively (typically 2.0-5.0).
        alpha: Class-balancing weight for the positive class in [0, 1].
        reduction: One of ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        """Initialize the loss.

        Args:
            gamma: Focusing parameter (>= 0).
            alpha: Positive-class weight in [0, 1].
            reduction: Reduction mode applied to the per-sample losses.

        Raises:
            ValueError: If ``reduction`` is not a supported mode.
        """
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        Args:
            logits: Raw model outputs of shape ``(N,)`` or ``(N, 1)``.
            targets: Binary ground-truth labels broadcastable to ``logits``.

        Returns:
            The reduced focal loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).to(logits.dtype)

        # Stable per-sample BCE: -log(p_t).
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = probability assigned to the true class.
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)

        # Modulating factor (1 - p_t) ** gamma.
        modulating_factor = (1.0 - p_t).pow(self.gamma)

        # Per-class alpha weighting.
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * modulating_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
