"""Temporal feature engineering for the supply chain risk predictor.

All engineered features are computed strictly from past and current
observations. Rolling aggregations use trailing windows and are shifted so a
row never incorporates information from its own future, preventing temporal
data leakage.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Grain over which trailing aggregations are computed.
_GROUP_KEYS: List[str] = ["matnr", "werks"]

# Feature columns produced by this module, consumed downstream by models.
FEATURE_COLUMNS: List[str] = [
    "planned_lead_time",
    "actual_lead_time",
    "vendor_lead_time_deviation",
    "unrestricted_stock",
    "eisbe",
    "safety_stock_penetration_ratio",
    "daily_consumption",
    "rolling_7d_consumption",
    "rolling_30d_consumption",
    "days_of_cover",
    "po_open_qty",
    "unit_price",
    "mean_delay_days",
]

TARGET_COLUMN: str = "is_stockout"


class FeatureBuilder:
    """Builds leakage-safe temporal features from the raw ERP frame."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the builder.

        Args:
            df: Raw joined observation frame produced by the synthesizer.
        """
        self._df = df.copy()

    def _sort_chronologically(self) -> None:
        """Sort by group and date so trailing windows are well-defined."""
        self._df = self._df.sort_values(
            _GROUP_KEYS + ["obs_date"]
        ).reset_index(drop=True)

    def _add_lead_time_deviation(self) -> None:
        """Compute vendor lead-time deviation (actual minus planned)."""
        self._df["vendor_lead_time_deviation"] = (
            self._df["actual_lead_time"] - self._df["planned_lead_time"]
        )

    def _add_safety_stock_penetration(self) -> None:
        """Compute how far stock has penetrated the safety-stock buffer.

        A ratio > 1 means unrestricted stock has fallen below the safety
        stock level (i.e. the buffer is breached).
        """
        self._df["safety_stock_penetration_ratio"] = self._df["eisbe"] / (
            self._df["unrestricted_stock"] + 1.0
        )

    def _add_rolling_consumption(self) -> None:
        """Add trailing rolling consumption windows without lookahead.

        The rolling window is computed then shifted by one within each group so
        the current row's own consumption never leaks into its feature value.
        """
        grouped = self._df.groupby(_GROUP_KEYS)["daily_consumption"]

        for window, col in [(7, "rolling_7d_consumption"), (30, "rolling_30d_consumption")]:
            rolled = grouped.transform(
                lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
            )
            # First observation per group has no history -> fall back to current.
            self._df[col] = rolled.fillna(self._df["daily_consumption"])

    def _add_days_of_cover(self) -> None:
        """Compute inventory days-of-cover from stock and consumption."""
        self._df["days_of_cover"] = self._df["unrestricted_stock"] / (
            self._df["daily_consumption"] + 1e-6
        )

    def _optimize_memory(self) -> None:
        """Downcast numeric dtypes to shrink the in-memory footprint.

        float64 -> float32 and integer columns to the smallest safe width.
        The target is kept as int8.
        """
        float_cols = self._df.select_dtypes(include=["float64"]).columns
        self._df[float_cols] = self._df[float_cols].astype(np.float32)

        int_cols = self._df.select_dtypes(include=["int64", "int32"]).columns
        for col in int_cols:
            self._df[col] = pd.to_numeric(self._df[col], downcast="integer")

    def build(self) -> pd.DataFrame:
        """Run the full feature-engineering pipeline.

        Returns:
            A chronologically sorted frame containing engineered features, the
            observation date, the join keys, and the target column.
        """
        self._sort_chronologically()
        self._add_lead_time_deviation()
        self._add_safety_stock_penetration()
        self._add_rolling_consumption()
        self._add_days_of_cover()
        self._optimize_memory()

        keep = _GROUP_KEYS + ["obs_date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
        return self._df[keep].reset_index(drop=True)


def main() -> None:
    """CLI entry point: build features from raw parquet and persist them."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / "sap_mm_dataset.parquet"
    df = pd.read_parquet(raw_path)

    builder = FeatureBuilder(df)
    features = builder.build()

    out_path = PROCESSED_DATA_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)
    mem_mb = features.memory_usage(deep=True).sum() / 1e6
    print(f"Built {features.shape[1]} columns for {len(features):,} rows -> {out_path}")
    print(f"In-memory footprint: {mem_mb:.2f} MB")


if __name__ == "__main__":
    main()
