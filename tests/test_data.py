"""Phase 1 tests: dataset synthesis integrity and leakage prevention."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.make_dataset import SAPDataSynthesizer, SynthesisConfig
from src.data.split_data import chronological_split, validate_no_leakage
from src.features.build_features import FeatureBuilder, TARGET_COLUMN


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Generate a smaller synthetic dataset once for the test module."""
    cfg = SynthesisConfig(n_records=10_000, n_materials=300, seed=7)
    return SAPDataSynthesizer(cfg).generate()


@pytest.fixture(scope="module")
def feature_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build features from the synthetic raw frame."""
    return FeatureBuilder(raw_df).build()


def test_stockout_ratio_is_exactly_two_percent(raw_df: pd.DataFrame) -> None:
    """The positive class must be exactly 2% of the dataset."""
    ratio = raw_df[TARGET_COLUMN].mean()
    assert ratio == pytest.approx(0.02, abs=1e-9)


def test_expected_row_count(raw_df: pd.DataFrame) -> None:
    """Synthesizer must emit exactly the configured number of records."""
    assert len(raw_df) == 10_000


def test_referential_integrity(raw_df: pd.DataFrame) -> None:
    """Every observation must reference valid master-data keys."""
    assert raw_df["matnr"].notna().all()
    assert raw_df["werks"].notna().all()
    assert raw_df["lifnr"].notna().all()


def test_features_have_no_nulls(feature_df: pd.DataFrame) -> None:
    """Engineered feature frame must be fully populated."""
    assert not feature_df.isna().any().any()


def test_memory_downcasting(feature_df: pd.DataFrame) -> None:
    """Float features must be downcast to float32 to shrink footprint."""
    float_cols = feature_df.select_dtypes(include=["float"]).columns
    assert all(str(feature_df[c].dtype) == "float32" for c in float_cols)


def test_chronological_split_has_zero_date_overlap(feature_df: pd.DataFrame) -> None:
    """Mathematically verify no date overlap between the partitions."""
    split = chronological_split(feature_df)

    train_max = split.train["obs_date"].max()
    val_min = split.val["obs_date"].min()
    val_max = split.val["obs_date"].max()
    test_min = split.test["obs_date"].min()

    assert train_max <= val_min
    assert val_max <= test_min
    # Validator must not raise on a correct split.
    validate_no_leakage(split)


def test_leakage_validator_raises_on_shuffled_data(feature_df: pd.DataFrame) -> None:
    """A randomly shuffled (leaky) split must trip the validator."""
    split = chronological_split(feature_df)

    # Swap a late test row into the training set to fabricate leakage.
    leaky_train = pd.concat(
        [split.train, split.test.tail(1)], ignore_index=True
    )
    from src.data.split_data import TimeSplit

    leaky = TimeSplit(train=leaky_train, val=split.val, test=split.test)
    with pytest.raises(RuntimeError):
        validate_no_leakage(leaky)


def test_split_partitions_cover_all_rows(feature_df: pd.DataFrame) -> None:
    """Train + val + test must partition the dataset without loss."""
    split = chronological_split(feature_df)
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(feature_df)
