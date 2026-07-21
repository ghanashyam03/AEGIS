"""Schema validation tests for required source fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from aegis.data.schema import (
    validate_raw_metadata,
    validate_test_metadata,
    validate_true_population,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _valid_raw_frame() -> pd.DataFrame:
    """Minimal valid frame with study-class and non-study-class rows."""
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "hostgal_photoz": [0.1, 0.2, 0.5, 0.8],
            "hostgal_photoz_err": [0.01, 0.02, 0.05, 0.08],
            "true_target": [64, 90, 95, 15],  # 15 is NOT a study class
        }
    )


def _valid_true_frame() -> pd.DataFrame:
    """Minimal valid TRUE population frame (study classes only)."""
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "hostgal_photoz": [0.1, 0.2, 0.5],
            "hostgal_photoz_err": [0.01, 0.02, 0.05],
            "true_target": [64, 90, 95],
        }
    )


# ---------------------------------------------------------------------------
# RAW schema tests
# ---------------------------------------------------------------------------


def test_valid_metadata_passes_schema_validation() -> None:
    """Backward-compat alias still works and passes valid data."""
    result = validate_test_metadata(_valid_raw_frame())
    assert len(result) == 4


def test_valid_raw_frame_passes() -> None:
    result = validate_raw_metadata(_valid_raw_frame())
    assert len(result) == 4


def test_missing_required_field_fails_loudly() -> None:
    with pytest.raises(SchemaErrors):
        validate_raw_metadata(_valid_raw_frame().drop(columns=["true_target"]))


def test_negative_photoz_fails_loudly() -> None:
    malformed = _valid_raw_frame().copy()
    malformed.loc[0, "hostgal_photoz"] = -0.1
    with pytest.raises(SchemaErrors):
        validate_raw_metadata(malformed)


def test_nan_photoz_fails_loudly() -> None:
    malformed = _valid_raw_frame().copy()
    malformed.loc[0, "hostgal_photoz"] = float("nan")
    with pytest.raises(SchemaErrors):
        validate_raw_metadata(malformed)


def test_inf_photoz_fails_loudly() -> None:
    malformed = _valid_raw_frame().copy()
    malformed.loc[1, "hostgal_photoz"] = np.inf
    with pytest.raises(SchemaErrors):
        validate_raw_metadata(malformed)


def test_duplicate_object_id_fails_loudly() -> None:
    malformed = _valid_raw_frame().copy()
    malformed.loc[1, "object_id"] = malformed.loc[0, "object_id"]
    with pytest.raises(SchemaErrors):
        validate_raw_metadata(malformed)


# ---------------------------------------------------------------------------
# TRUE population schema tests
# ---------------------------------------------------------------------------


def test_valid_true_population_passes() -> None:
    result = validate_true_population(_valid_true_frame())
    assert len(result) == 3
    assert set(result["true_target"].unique()) == {64, 90, 95}


def test_non_study_class_excluded_from_true_population() -> None:
    """Class 15 (TDE) must not pass the TRUE population schema."""
    frame_with_tde = _valid_true_frame().copy()
    extra = pd.DataFrame(
        {
            "object_id": [99],
            "hostgal_photoz": [0.3],
            "hostgal_photoz_err": [0.03],
            "true_target": [15],
        }
    )
    bad_frame = pd.concat([frame_with_tde, extra], ignore_index=True)
    with pytest.raises(SchemaErrors):
        validate_true_population(bad_frame)


def test_nan_photoz_fails_true_population_schema() -> None:
    malformed = _valid_true_frame().copy()
    malformed.loc[0, "hostgal_photoz"] = float("nan")
    with pytest.raises(SchemaErrors):
        validate_true_population(malformed)
