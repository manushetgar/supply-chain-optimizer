"""Global reproducibility utilities.

Seeding every source of randomness (Python, numpy, torch) and enabling
deterministic algorithm selection is a hard requirement for enterprise ML,
guaranteeing that training runs and SHAP attributions are reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed all relevant RNGs for full reproducibility.

    Args:
        seed: The integer seed applied to every RNG.
        deterministic: If True, force torch (when available) to use
            deterministic algorithms and disable cuDNN benchmarking.

    Returns:
        The seed that was applied (useful for logging into MLflow).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Best-effort determinism; warn_only avoids hard failures on
            # ops lacking deterministic kernels.
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is optional for pure data-engineering workflows.
        pass

    return seed
