"""Synthetic SAP S/4HANA Materials Management (MM) dataset generator.

This module programmatically synthesizes a relational dataset mirroring the
structural relationships of SAP MM tables and joins them into a single,
model-ready feature frame with an extreme (2%) stock-out class imbalance.

Simulated tables:
    * MARA  - Client-level material master data.
    * MARC  - Plant-level material data (MRP settings, safety stock).
    * MARD  - Storage-location stock levels (unrestricted stock).
    * EKKO  - Purchasing document header (vendor, document date).
    * EKPO  - Purchasing document items (material, quantity, plant).
    * MATDOC - Unified S/4HANA goods movement table (consumption).

The synthesis deliberately encodes causal structure: a delayed purchase order
(EKKO) combined with low unrestricted stock (MARD) and high consumption
(MATDOC) drives a higher probability of a stock-out event.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, settings
from src.utils.seed import seed_everything


@dataclass(frozen=True)
class SynthesisConfig:
    """Configuration for the synthetic ERP data generator.

    Attributes:
        n_records: Total number of joined observation rows to produce.
        stockout_rate: Exact fraction of rows flagged as stock-outs.
        n_materials: Distinct material master records (MARA).
        n_plants: Distinct plants (MARC/MARD).
        n_vendors: Distinct vendors referenced by purchasing documents.
        start_date: First calendar date in the simulated timeline.
        horizon_days: Length of the simulated timeline in days.
        seed: Global RNG seed for reproducibility.
    """

    n_records: int = settings.n_records
    stockout_rate: float = settings.stockout_rate
    n_materials: int = 1_200
    n_plants: int = 8
    n_vendors: int = 60
    start_date: datetime = datetime(2024, 1, 1)
    horizon_days: int = 540
    seed: int = settings.random_seed


class SAPDataSynthesizer:
    """Generates a relational, imbalanced, time-series SAP MM dataset."""

    def __init__(self, config: SynthesisConfig | None = None) -> None:
        """Initialize the synthesizer.

        Args:
            config: Optional synthesis configuration; defaults are used when
                omitted.
        """
        self.config = config or SynthesisConfig()
        self._rng = np.random.default_rng(self.config.seed)
        seed_everything(self.config.seed)

    # ------------------------------------------------------------------ #
    # Master data tables
    # ------------------------------------------------------------------ #
    def _make_mara(self) -> pd.DataFrame:
        """Synthesize client-level material master data (MARA)."""
        cfg = self.config
        material_ids = np.arange(1, cfg.n_materials + 1)
        material_groups = self._rng.choice(
            ["RAW", "SEMI", "FIN", "PACK", "SPARE"], size=cfg.n_materials
        )
        base_unit = self._rng.choice(["EA", "KG", "L", "M"], size=cfg.n_materials)
        return pd.DataFrame(
            {
                "matnr": [f"MAT-{m:06d}" for m in material_ids],
                "mtart": material_groups,  # material type
                "meins": base_unit,  # base unit of measure
                "unit_price": self._rng.uniform(5.0, 500.0, size=cfg.n_materials),
            }
        )

    def _make_marc(self, mara: pd.DataFrame) -> pd.DataFrame:
        """Synthesize plant-level material data (MARC) with MRP settings."""
        cfg = self.config
        rows: List[Dict] = []
        for matnr in mara["matnr"]:
            for plant_idx in range(cfg.n_plants):
                safety_stock = float(self._rng.integers(10, 200))
                rows.append(
                    {
                        "matnr": matnr,
                        "werks": f"PL{plant_idx:02d}",  # plant
                        "eisbe": safety_stock,  # safety stock
                        "plifz": int(self._rng.integers(3, 45)),  # planned deliv time
                        "minbe": safety_stock * 0.5,  # reorder point
                    }
                )
        return pd.DataFrame(rows)

    def _make_vendors(self) -> pd.DataFrame:
        """Synthesize vendor master with intrinsic reliability profiles."""
        cfg = self.config
        vendor_ids = [f"VEN-{v:05d}" for v in range(1, cfg.n_vendors + 1)]
        # Each vendor has a latent mean delay and volatility.
        return pd.DataFrame(
            {
                "lifnr": vendor_ids,  # vendor number
                "mean_delay_days": self._rng.gamma(2.0, 2.0, size=cfg.n_vendors),
                "delay_volatility": self._rng.uniform(0.5, 5.0, size=cfg.n_vendors),
            }
        )

    # ------------------------------------------------------------------ #
    # Transactional tables
    # ------------------------------------------------------------------ #
    def _make_observations(
        self,
        mara: pd.DataFrame,
        marc: pd.DataFrame,
        vendors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Synthesize the joined transactional observation frame.

        Each row represents a material/plant/date observation combining
        purchasing (EKKO/EKPO), stock (MARD), and goods-movement (MATDOC)
        signals. A latent risk score derived from these signals is used to
        assign exactly ``stockout_rate`` positive labels.
        """
        cfg = self.config
        n = cfg.n_records

        # Sample the material/plant grain from MARC (guarantees valid joins).
        marc_idx = self._rng.integers(0, len(marc), size=n)
        obs = marc.iloc[marc_idx].reset_index(drop=True).copy()

        # Attach material master attributes (MARA join).
        obs = obs.merge(mara, on="matnr", how="left")

        # Assign vendors (EKKO relation) and derive lead-time behavior.
        ven_idx = self._rng.integers(0, len(vendors), size=n)
        ven = vendors.iloc[ven_idx].reset_index(drop=True)
        obs = pd.concat([obs, ven], axis=1)

        # --- Temporal spine: spread observations across the timeline. --- #
        day_offsets = self._rng.integers(0, cfg.horizon_days, size=n)
        obs["obs_date"] = [
            cfg.start_date + timedelta(days=int(d)) for d in day_offsets
        ]

        # --- EKKO / EKPO: purchasing document signals. --- #
        # Actual lead time = planned + vendor delay noise.
        vendor_delay = self._rng.normal(
            obs["mean_delay_days"].to_numpy(),
            obs["delay_volatility"].to_numpy(),
        )
        obs["po_open_qty"] = self._rng.uniform(0, 500, size=n)
        obs["planned_lead_time"] = obs["plifz"].astype(float)
        obs["actual_lead_time"] = np.clip(
            obs["planned_lead_time"] + vendor_delay, 1.0, None
        )

        # --- MARD: storage-location unrestricted stock. --- #
        # Stock correlates inversely with safety-stock breaches.
        obs["unrestricted_stock"] = np.clip(
            self._rng.normal(obs["eisbe"].to_numpy() * 1.5, 50.0), 0.0, None
        )

        # --- MATDOC: goods-movement consumption. --- #
        obs["daily_consumption"] = np.clip(
            self._rng.gamma(2.0, 8.0, size=n), 0.0, None
        )

        return obs

    def _assign_labels(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Assign exactly ``stockout_rate`` positive labels via latent risk.

        A continuous latent risk score is computed from the causal drivers,
        then the top ``stockout_rate`` fraction of rows (by risk) are labeled
        as stock-outs, guaranteeing the exact class ratio while preserving the
        causal signal that models must learn.
        """
        cfg = self.config

        # Days of cover = stock / consumption; low cover -> high risk.
        days_of_cover = obs["unrestricted_stock"] / (obs["daily_consumption"] + 1e-6)
        lead_gap = obs["actual_lead_time"] - obs["planned_lead_time"]
        safety_breach = obs["eisbe"] - obs["unrestricted_stock"]

        # Standardize drivers and combine into a latent risk score.
        def _z(series: pd.Series) -> np.ndarray:
            arr = series.to_numpy(dtype=np.float64)
            std = arr.std() or 1.0
            return (arr - arr.mean()) / std

        latent_risk = (
            1.5 * _z(lead_gap)
            + 1.2 * _z(safety_breach)
            - 1.8 * _z(days_of_cover)
            + 0.4 * self._rng.normal(0, 1, size=len(obs))  # irreducible noise
        )

        n_positive = int(round(cfg.n_records * cfg.stockout_rate))
        threshold_idx = np.argsort(-latent_risk)[:n_positive]
        labels = np.zeros(len(obs), dtype=np.int8)
        labels[threshold_idx] = 1

        obs = obs.copy()
        obs["is_stockout"] = labels
        return obs

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self) -> pd.DataFrame:
        """Generate the full joined, labeled dataset.

        Returns:
            A chronologically unsorted observation frame containing raw signals
            and the ``is_stockout`` target at exactly the configured ratio.
        """
        mara = self._make_mara()
        marc = self._make_marc(mara)
        vendors = self._make_vendors()
        obs = self._make_observations(mara, marc, vendors)
        obs = self._assign_labels(obs)
        return obs


def main() -> None:
    """CLI entry point: synthesize and persist the raw dataset to parquet."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    synthesizer = SAPDataSynthesizer()
    df = synthesizer.generate()
    out_path = RAW_DATA_DIR / "sap_mm_dataset.parquet"
    df.to_parquet(out_path, index=False)
    ratio = df["is_stockout"].mean()
    print(f"Generated {len(df):,} records -> {out_path}")
    print(f"Stock-out ratio: {ratio:.4f} ({df['is_stockout'].sum():,} positives)")


if __name__ == "__main__":
    main()
