"""Tensor dataset preparation and scaling for the tabular classifier.

Bridges the leakage-safe pandas splits into scaled numpy arrays and PyTorch
tensors. The scaler is fit on the training split only, then applied to
validation and test to avoid information leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.split_data import TimeSplit
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class PreparedData:
    """Scaled numpy arrays for each partition plus the fitted scaler.

    Attributes:
        x_train, y_train: Training features/labels.
        x_val, y_val: Validation features/labels.
        x_test, y_test: Test features/labels.
        scaler: The StandardScaler fitted on the training features.
        feature_names: Ordered feature column names.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: Tuple[str, ...]


def prepare_arrays(split: TimeSplit) -> PreparedData:
    """Scale features (fit on train only) and return numpy arrays.

    Args:
        split: A validated chronological :class:`TimeSplit`.

    Returns:
        A :class:`PreparedData` bundle with float32 arrays.
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(split.train[FEATURE_COLUMNS]).astype(np.float32)
    x_val = scaler.transform(split.val[FEATURE_COLUMNS]).astype(np.float32)
    x_test = scaler.transform(split.test[FEATURE_COLUMNS]).astype(np.float32)

    return PreparedData(
        x_train=x_train,
        y_train=split.train[TARGET_COLUMN].to_numpy(dtype=np.float32),
        x_val=x_val,
        y_val=split.val[TARGET_COLUMN].to_numpy(dtype=np.float32),
        x_test=x_test,
        y_test=split.test[TARGET_COLUMN].to_numpy(dtype=np.float32),
        scaler=scaler,
        feature_names=tuple(FEATURE_COLUMNS),
    )


def make_loaders(
    data: PreparedData, batch_size: int = 512
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from prepared arrays.

    Args:
        data: A :class:`PreparedData` bundle.
        batch_size: Mini-batch size.

    Returns:
        A tuple of ``(train_loader, val_loader, test_loader)``.
    """

    def _ds(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    train = DataLoader(
        _ds(data.x_train, data.y_train), batch_size=batch_size, shuffle=True
    )
    val = DataLoader(_ds(data.x_val, data.y_val), batch_size=batch_size)
    test = DataLoader(_ds(data.x_test, data.y_test), batch_size=batch_size)
    return train, val, test
