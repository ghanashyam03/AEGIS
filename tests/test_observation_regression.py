"""Concrete regression tests for observation truncation using real object profiles.

Tests epoch-based truncation (e = 0, 2, 7 days), S/N thresholds,
multi-object batching, and schema rejection on realistic light curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from aegis.data.observation import (
    generate_as_of_epoch_sequences,
    get_first_detection_mjd,
    truncate_light_curve_at_epoch,
    truncate_light_curve_at_mjd,
)


@pytest.fixture
def kilonova_light_curve() -> pd.DataFrame:
    """Fixture producing a realistic Kilonova (class 64) light curve profile.

    Kilonovae evolve rapidly in red bands (i, z, Y) over ~10 days.
    """
    # Object 64001
    mjds = [
        59995.0,  # Pre-alert non-detection
        59998.0,  # Pre-alert non-detection
        60000.0,  # Alert! First detection (S/N = 10.0) -> t_0 = 60000.0
        60000.5,  # Alert day (+0.5d)
        60001.2,  # Day +1.2d
        60002.0,  # Day +2.0d
        60003.5,  # Day +3.5d
        60005.0,  # Day +5.0d
        60006.8,  # Day +6.8d
        60010.0,  # Day +10.0d (faded)
        60020.0,  # Day +20.0d (late epoch)
    ]
    passbands = [0, 1, 2, 3, 2, 3, 4, 4, 5, 5, 5]
    fluxes = [1.2, -0.5, 50.0, 120.0, 300.0, 280.0, 180.0, 90.0, 40.0, 5.0, 0.2]
    flux_errs = [5.0, 5.0, 5.0, 6.0, 8.0, 8.0, 7.0, 6.0, 5.0, 5.0, 5.0]
    detected_bools = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    return pd.DataFrame(
        {
            "object_id": 64001,
            "mjd": mjds,
            "passband": passbands,
            "flux": fluxes,
            "flux_err": flux_errs,
            "detected_bool": detected_bools,
        }
    )


@pytest.fixture
def sn_ia_light_curve() -> pd.DataFrame:
    """Fixture producing a Type Ia Supernova (class 90) light curve profile."""
    # Object 90001
    mjds = [
        60100.0,  # First alert (S/N = 8.0) -> t_0 = 60100.0
        60101.5,
        60103.0,
        60107.0,
        60114.0,
        60120.0,
    ]
    return pd.DataFrame(
        {
            "object_id": 90001,
            "mjd": mjds,
            "passband": [1, 2, 2, 3, 3, 4],
            "flux": [40.0, 150.0, 450.0, 800.0, 600.0, 200.0],
            "flux_err": [5.0, 10.0, 15.0, 20.0, 18.0, 10.0],
            "detected_bool": [1, 1, 1, 1, 1, 1],
        }
    )


def test_kilonova_epoch_truncation(kilonova_light_curve: pd.DataFrame) -> None:
    """Regression test: Truncating a KN light curve at e=0, 2, 7 days."""
    t0 = get_first_detection_mjd(kilonova_light_curve, detection_snr_threshold=5.0)
    assert t0 == 60000.0

    sequences = generate_as_of_epoch_sequences(
        kilonova_light_curve, epochs=(0.0, 2.0, 7.0)
    )

    e0 = sequences[0.0]
    e2 = sequences[2.0]
    e7 = sequences[7.0]

    # At e=0 (t <= 60000.0): 2 pre-alert non-detections + 1 alert observation = 3 rows
    assert len(e0) == 3
    assert (e0["mjd"] <= 60000.0).all()

    # At e=2 (t <= 60002.0): includes 60000.5, 60001.2, 60002.0 = 6 rows
    assert len(e2) == 6
    assert (e2["mjd"] <= 60002.0).all()

    # At e=7 (t <= 60007.0): includes 60003.5, 60005.0, 60006.8 = 9 rows
    assert len(e7) == 9
    assert (e7["mjd"] <= 60007.0).all()

    # Strict prefix integrity checks
    pd.testing.assert_frame_equal(e0, e2.iloc[:3].reset_index(drop=True))
    pd.testing.assert_frame_equal(e2, e7.iloc[:6].reset_index(drop=True))


def test_no_qualifying_detection_handling() -> None:
    """Regression test: Object with low S/N data (< 5) returns empty frame."""
    low_snr_df = pd.DataFrame(
        {
            "object_id": 99901,
            "mjd": [60000.0, 60002.0, 60004.0],
            "passband": [0, 1, 2],
            "flux": [2.0, 3.0, 1.5],
            "flux_err": [5.0, 5.0, 5.0],  # S/N <= 0.6
            "detected_bool": [0, 0, 0],
        }
    )

    t0 = get_first_detection_mjd(low_snr_df, detection_snr_threshold=5.0)
    assert t0 is None

    truncated = truncate_light_curve_at_epoch(
        low_snr_df, days_since_first_detection=2.0
    )
    assert truncated.empty
    assert list(truncated.columns) == list(low_snr_df.columns)


def test_multi_object_batch_truncation(
    kilonova_light_curve: pd.DataFrame, sn_ia_light_curve: pd.DataFrame
) -> None:
    """Regression test: Truncating multi-object light curve with different t_0."""

    combined = pd.concat([kilonova_light_curve, sn_ia_light_curve], ignore_index=True)

    first_mjds = get_first_detection_mjd(combined, detection_snr_threshold=5.0)
    assert isinstance(first_mjds, dict)
    assert first_mjds[64001] == 60000.0
    assert first_mjds[90001] == 60100.0

    # Truncate at epoch e=2 days
    truncated_e2 = truncate_light_curve_at_epoch(
        combined, days_since_first_detection=2.0
    )

    # KN object 64001: mjd <= 60002.0 (6 rows)
    kn_sub = truncated_e2[truncated_e2["object_id"] == 64001]
    assert len(kn_sub) == 6
    assert (kn_sub["mjd"] <= 60002.0).all()

    # SN Ia object 90001: mjd <= 60102.0 (2 rows: 60100.0, 60101.5)
    sn_sub = truncated_e2[truncated_e2["object_id"] == 90001]
    assert len(sn_sub) == 2
    assert (sn_sub["mjd"] <= 60102.0).all()


def test_explicit_dict_cutoff(
    kilonova_light_curve: pd.DataFrame, sn_ia_light_curve: pd.DataFrame
) -> None:
    """Regression test: Passing a dictionary of explicit cutoff MJDs per object."""
    combined = pd.concat([kilonova_light_curve, sn_ia_light_curve], ignore_index=True)

    cutoffs = {64001: 60001.0, 90001: 60105.0}

    truncated = truncate_light_curve_at_mjd(combined, cutoff_mjd=cutoffs)

    kn_rows = truncated[truncated["object_id"] == 64001]
    assert (kn_rows["mjd"] <= 60001.0).all()

    sn_rows = truncated[truncated["object_id"] == 90001]
    assert (sn_rows["mjd"] <= 60105.0).all()


def test_invalid_schema_rejection() -> None:
    """Regression test: Non-numeric flux or negative flux_err fails loud via Pandera."""
    invalid_df = pd.DataFrame(
        {
            "object_id": [101],
            "mjd": [60000.0],  # mjd <= cutoff so row is retained
            "passband": [1],
            "flux": [np.nan],  # NaN flux
            "flux_err": [-2.0],  # Negative flux_err
            "detected_bool": [1],
        }
    )

    with pytest.raises((SchemaError, SchemaErrors, ValueError)):
        truncate_light_curve_at_mjd(invalid_df, cutoff_mjd=60000.0)
