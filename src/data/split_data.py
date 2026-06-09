"""Strict out-of-time (chronological) data splitting and leakage validation.

Supply chain forecasting must predict the future from the past. Random
shuffling (``train_test_split``) or standard ``KFold`` are mathematically
invalid here because they leak future information into training. This module
enforces an expanding-window chronological split and provides a validator that
raises ``RuntimeError`` on any temporal overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.config import PROCESSED_DATA_DIR, settings

_DATE_COLUMN = "obs_date"


@dataclass(frozen=True)
class TimeSplit:
    """Container for a chronological train/validation/test partition.

    Attributes:
        train: Earliest slice of the timeline (model fitting).
        val: Middle slice (early stopping / model selection).
        test: Final slice (out-of-time generalization estimate).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = settings.train_frac,
    val_frac: float = settings.val_frac,
    date_column: str = _DATE_COLUMN,
) -> TimeSplit:
    """Partition a frame into out-of-time train/val/test slices.

    The frame is sorted by ``date_column`` and cut at the ``train_frac`` and
    ``train_frac + val_frac`` quantile boundaries of the ordered rows.

    Args:
        df: Feature frame containing a datetime ``date_column``.
        train_frac: Fraction of the ordered rows used for training.
        val_frac: Fraction used for validation (test is the remainder).
        date_column: Name of the chronological ordering column.

    Returns:
        A :class:`TimeSplit` with non-overlapping temporal slices.

    Raises:
        ValueError: If the fractions are invalid or the column is missing.
    """
    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column!r}")
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must lie in (0, 1).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 to leave a test set.")

    ordered = df.sort_values(date_column).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return TimeSplit(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        val=ordered.iloc[train_end:val_end].reset_index(drop=True),
        test=ordered.iloc[val_end:].reset_index(drop=True),
    )


def validate_no_leakage(split: TimeSplit, date_column: str = _DATE_COLUMN) -> None:
    """Assert strict chronological ordering between the split partitions.

    Verifies that every training timestamp precedes or equals the earliest
    validation timestamp, and every validation timestamp precedes or equals the
    earliest test timestamp.

    Args:
        split: The :class:`TimeSplit` to validate.
        date_column: Name of the chronological column.

    Raises:
        RuntimeError: If any future data has leaked into an earlier partition.
    """
    train_max = split.train[date_column].max()
    val_min = split.val[date_column].min()
    val_max = split.val[date_column].max()
    test_min = split.test[date_column].min()

    if train_max > val_min:
        raise RuntimeError(
            f"Temporal leakage: train_max ({train_max}) > val_min ({val_min})."
        )
    if val_max > test_min:
        raise RuntimeError(
            f"Temporal leakage: val_max ({val_max}) > test_min ({test_min})."
        )


def load_split() -> Tuple[TimeSplit, pd.DataFrame]:
    """Load processed features and produce a validated chronological split.

    Returns:
        A tuple of the validated :class:`TimeSplit` and the full feature frame.
    """
    features = pd.read_parquet(PROCESSED_DATA_DIR / "features.parquet")
    split = chronological_split(features)
    validate_no_leakage(split)
    return split, features


def main() -> None:
    """CLI entry point: build and report a validated chronological split."""
    split, _ = load_split()
    print(
        f"train={len(split.train):,}  val={len(split.val):,}  "
        f"test={len(split.test):,}"
    )
    print(
        f"train dates: {split.train[_DATE_COLUMN].min()} -> "
        f"{split.train[_DATE_COLUMN].max()}"
    )
    print(
        f"test dates:  {split.test[_DATE_COLUMN].min()} -> "
        f"{split.test[_DATE_COLUMN].max()}"
    )
    print("Leakage validation passed: no chronological overlap.")


if __name__ == "__main__":
    main()
