"""SHAP-based explainability wrapper for the PyTorch stock-out classifier.

Converts an opaque neural-network prediction into an auditable, additive
feature-attribution vector using SHAP. The output is a structured JSON payload
consumed programmatically by the downstream LangGraph agent (Phase 3).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import shap
import torch
from torch import nn

from src.models.pytorch_model import StockoutClassifier


class _TwoDimWrapper(nn.Module):
    """Wraps the MLP so its output keeps a trailing dim of ``(N, 1)``.

    SHAP's :class:`~shap.DeepExplainer` indexes ``output.shape[1]``, so a
    flattened ``(N,)`` logit tensor must be reshaped to ``(N, 1)``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1, 1)


class StockoutExplainer:
    """Wraps a trained classifier with a SHAP DeepExplainer.

    Attributes:
        model: The trained :class:`StockoutClassifier` (set to eval mode).
        feature_names: Ordered names aligning with the model's input columns.
    """

    def __init__(
        self,
        model: StockoutClassifier,
        background: np.ndarray,
        feature_names: List[str],
        max_background: int = 100,
    ) -> None:
        """Build the explainer.

        Args:
            model: A trained Lightning classifier.
            background: Representative background samples for the explainer.
            feature_names: Ordered feature column names.
            max_background: Cap on background rows for tractable KernelExplainer
                / DeepExplainer computation.
        """
        self.model = model.eval()
        self.feature_names = feature_names

        bg = background[:max_background]
        bg_tensor = torch.from_numpy(bg.astype(np.float32))

        # DeepExplainer integrates gradients through the PyTorch network for
        # efficient, model-specific Shapley value estimation. The wrapper
        # exposes a (N, 1) output shape that SHAP expects.
        self._wrapped = _TwoDimWrapper(self.model.model).eval()
        self._explainer = shap.DeepExplainer(self._wrapped, bg_tensor)

    def _shap_values(self, x: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a batch, returned as ``(N, n_features)``."""
        x_tensor = torch.from_numpy(x.astype(np.float32))
        values = self._explainer.shap_values(x_tensor, check_additivity=False)
        arr = np.asarray(values)
        # DeepExplainer may return shape (N, F, 1) for a single output.
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def explain(self, x_row: np.ndarray, top_k: int = 5) -> Dict:
        """Explain a single prediction as a structured JSON-serializable dict.

        Args:
            x_row: A single scaled feature vector of shape ``(n_features,)``.
            top_k: Number of top-contributing features to return.

        Returns:
            A dict with the predicted probability and the top-``k`` features
            ranked by absolute SHAP contribution.
        """
        x_row = np.asarray(x_row, dtype=np.float32).reshape(1, -1)

        with torch.no_grad():
            logit = self.model(torch.from_numpy(x_row))
            prob = float(torch.sigmoid(logit).item())

        shap_vals = self._shap_values(x_row)[0]  # (n_features,)

        order = np.argsort(-np.abs(shap_vals))[:top_k]
        top_features = {
            self.feature_names[i]: round(float(shap_vals[i]), 6) for i in order
        }

        return {
            "prediction": round(prob, 6),
            "is_stockout": bool(prob >= 0.5),
            "top_features": top_features,
        }
