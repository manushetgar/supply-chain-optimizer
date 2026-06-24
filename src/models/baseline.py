"""Scikit-Learn RandomForest baseline for comparative benchmarking.

A class-weight-balanced RandomForest provides a strong, interpretable baseline
against which the Focal-Loss MLP is compared using imbalance-aware metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from src.config import settings


class BaselineRandomForest:
    """Wraps a balanced RandomForest with imbalance-aware evaluation."""

    def __init__(self, n_estimators: int = 300) -> None:
        """Initialize the baseline.

        Args:
            n_estimators: Number of trees in the forest.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=settings.random_seed,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaselineRandomForest":
        """Fit the forest on training features/labels."""
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities of shape ``(N,)``."""
        return self.model.predict_proba(x)[:, 1]

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute imbalance-aware metrics (never accuracy).

        Args:
            x: Feature matrix.
            y: Ground-truth binary labels.
            threshold: Decision threshold for hard predictions.

        Returns:
            Dict with precision, recall, F2-score, and PR-AUC.
        """
        proba = self.predict_proba(x)
        preds = (proba >= threshold).astype(int)
        return {
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f2": float(fbeta_score(y, preds, beta=2.0, zero_division=0)),
            "prauc": float(average_precision_score(y, proba)),
        }
