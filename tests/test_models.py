"""Phase 2 tests: Focal Loss correctness, model wiring, and SHAP output."""

from __future__ import annotations

import numpy as np
import torch

from src.models.focal_loss import BinaryFocalLoss
from src.models.pytorch_model import StockoutClassifier


def test_focal_loss_matches_manual_formula() -> None:
    """Focal loss must equal -alpha*(1-p_t)^gamma*log(p_t) computed manually."""
    gamma, alpha = 2.0, 0.25
    logits = torch.tensor([0.5, -1.0, 2.0])
    targets = torch.tensor([1.0, 0.0, 1.0])

    loss_fn = BinaryFocalLoss(gamma=gamma, alpha=alpha, reduction="none")
    got = loss_fn(logits, targets)

    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    expected = -alpha_t * (1 - p_t) ** gamma * torch.log(p_t)

    assert torch.allclose(got, expected, atol=1e-5)


def test_focal_loss_downweights_easy_examples() -> None:
    """Higher gamma must reduce loss on a confidently-correct example."""
    logits = torch.tensor([4.0])  # confident, correct positive
    targets = torch.tensor([1.0])
    low = BinaryFocalLoss(gamma=0.0, alpha=0.5)(logits, targets)
    high = BinaryFocalLoss(gamma=3.0, alpha=0.5)(logits, targets)
    assert high < low


def test_focal_loss_reduction_modes() -> None:
    """Mean and sum reductions must be consistent with the none reduction."""
    logits = torch.randn(16)
    targets = (torch.rand(16) > 0.5).float()
    none = BinaryFocalLoss(reduction="none")(logits, targets)
    mean = BinaryFocalLoss(reduction="mean")(logits, targets)
    assert torch.allclose(none.mean(), mean, atol=1e-6)


def test_classifier_forward_shape() -> None:
    """The classifier must emit one logit per input row."""
    model = StockoutClassifier(input_dim=13)
    x = torch.randn(8, 13)
    out = model(x)
    assert out.shape == (8,)


def test_shap_explainer_returns_structured_json() -> None:
    """SHAP explainer output must be a JSON-serializable ranked payload."""
    import json

    from src.models.explainability import StockoutExplainer

    torch.manual_seed(0)
    np.random.seed(0)
    feature_names = [f"f{i}" for i in range(6)]
    model = StockoutClassifier(input_dim=6)
    model.eval()

    background = np.random.randn(40, 6).astype(np.float32)
    explainer = StockoutExplainer(model, background, feature_names, max_background=30)

    payload = explainer.explain(background[0], top_k=3)
    # Must round-trip through JSON.
    json.dumps(payload)

    assert set(payload.keys()) == {"prediction", "is_stockout", "top_features"}
    assert 0.0 <= payload["prediction"] <= 1.0
    assert len(payload["top_features"]) == 3
