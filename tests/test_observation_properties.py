"""Property-based tests for observation truncation and no-leakage contract.

Uses Hypothesis to verify property invariants across randomly generated light curves:
1. Cutoff bound guarantee: every returned observation's mjd <= cutoff_mjd.
2. Invariance to future mutations (no future leakage): changes to t > T_cutoff do not
   affect output at T_cutoff.
3. Append-only consistency: truncating at T1 and extending to T2 > T1 preserves all
   T1 observations unchanged.
4. Monotonicity: length of truncate(T1) <= length of truncate(T2) for T1 <= T2.
5. Cadence & value preservation: timestamps and flux values are exact subsets.
6. Forbidden field stripping: global summary statistics are stripped.
7. First detection t_0 chronological invariance: t_0 depends strictly on first alert.
8. Epoch sequence generation: sequences at e=(0, 2, 7) preserve append-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from aegis.data.observation import (
    generate_as_of_epoch_sequences,
    get_first_detection_mjd,
    truncate_light_curve_at_epoch,
    truncate_light_curve_at_mjd,
)
from aegis.data.schema import OBSERVATION_FORBIDDEN_FIELDS


@st.composite
def light_curve_strategy(draw, object_id: int = 1001) -> pd.DataFrame:
    """Hypothesis strategy to generate valid astronomical light curve DataFrames."""
    n_obs = draw(st.integers(min_value=5, max_value=30))
    start_mjd = draw(st.floats(min_value=60000.0, max_value=60100.0, allow_nan=False))
    dt_list = draw(
        st.lists(
            st.floats(
                min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
            ),
            min_size=n_obs - 1,
            max_size=n_obs - 1,
        )
    )
    mjds = [start_mjd] + list(start_mjd + np.cumsum(dt_list))
    passbands = draw(
        st.lists(st.integers(min_value=0, max_value=5), min_size=n_obs, max_size=n_obs)
    )
    fluxes = draw(
        st.lists(
            st.floats(
                min_value=-500.0,
                max_value=5000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n_obs,
            max_size=n_obs,
        )
    )
    flux_errs = draw(
        st.lists(
            st.floats(
                min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False
            ),
            min_size=n_obs,
            max_size=n_obs,
        )
    )
    detected_bools = draw(
        st.lists(st.integers(min_value=0, max_value=1), min_size=n_obs, max_size=n_obs)
    )

    df = pd.DataFrame(
        {
            "object_id": object_id,
            "mjd": mjds,
            "passband": passbands,
            "flux": fluxes,
            "flux_err": flux_errs,
            "detected_bool": detected_bools,
        }
    )
    return df


@given(
    df=light_curve_strategy(),
    cutoff_offset=st.floats(min_value=-10.0, max_value=200.0),
)
@settings(max_examples=50, deadline=None)
def test_property_cutoff_bound(df: pd.DataFrame, cutoff_offset: float) -> None:
    """Property 1: Every returned observation's mjd is strictly <= cutoff_mjd."""
    min_mjd = df["mjd"].min()
    cutoff_mjd = float(min_mjd + cutoff_offset)

    truncated = truncate_light_curve_at_mjd(df, cutoff_mjd=cutoff_mjd)

    if not truncated.empty:
        assert (truncated["mjd"] <= cutoff_mjd).all()


@given(df=light_curve_strategy(), fraction=st.floats(min_value=0.2, max_value=0.8))
@settings(max_examples=50, deadline=None)
def test_property_future_invariance(df: pd.DataFrame, fraction: float) -> None:
    """Property 2: Mutating observations at t > cutoff does not change output."""
    mjds = df["mjd"].sort_values().values
    idx = int(len(mjds) * fraction)
    cutoff_mjd = float(mjds[idx])

    base_truncated = truncate_light_curve_at_mjd(df, cutoff_mjd=cutoff_mjd)

    mutated_df = df.copy()
    future_mask = mutated_df["mjd"] > cutoff_mjd
    mutated_df.loc[future_mask, "flux"] = mutated_df.loc[future_mask, "flux"] + 9999.0
    mutated_df.loc[future_mask, "flux_err"] = 0.001

    extra_row = pd.DataFrame(
        {
            "object_id": [mutated_df["object_id"].iloc[0]],
            "mjd": [cutoff_mjd + 100.0],
            "passband": [1],
            "flux": [8888.0],
            "flux_err": [1.0],
            "detected_bool": [1],
        }
    )
    mutated_df = pd.concat([mutated_df, extra_row], ignore_index=True)

    mutated_truncated = truncate_light_curve_at_mjd(mutated_df, cutoff_mjd=cutoff_mjd)

    pd.testing.assert_frame_equal(base_truncated, mutated_truncated)


@given(
    df=light_curve_strategy(),
    frac1=st.floats(min_value=0.1, max_value=0.4),
    frac2=st.floats(min_value=0.5, max_value=0.9),
)
@settings(max_examples=50, deadline=None)
def test_property_append_only_consistency(
    df: pd.DataFrame, frac1: float, frac2: float
) -> None:
    """Property 3: Extending T1 to T2 > T1 preserves all T1 rows unchanged."""
    sorted_mjds = df["mjd"].sort_values().values
    t1 = float(sorted_mjds[int(len(sorted_mjds) * frac1)])
    t2 = float(sorted_mjds[int(len(sorted_mjds) * frac2)])
    assert t1 <= t2

    r1 = truncate_light_curve_at_mjd(df, cutoff_mjd=t1)
    r2 = truncate_light_curve_at_mjd(df, cutoff_mjd=t2)

    r2_prefix = r2[r2["mjd"] <= t1].reset_index(drop=True)
    pd.testing.assert_frame_equal(r1, r2_prefix)


@given(
    df=light_curve_strategy(),
    offset1=st.floats(min_value=0.0, max_value=50.0),
    offset2=st.floats(min_value=50.0, max_value=100.0),
)
@settings(max_examples=50, deadline=None)
def test_property_monotonicity(
    df: pd.DataFrame, offset1: float, offset2: float
) -> None:
    """Property 4: len(truncate(T1)) <= len(truncate(T2)) for T1 <= T2."""
    t_min = float(df["mjd"].min())
    t1 = t_min + offset1
    t2 = t_min + offset2

    r1 = truncate_light_curve_at_mjd(df, cutoff_mjd=t1)
    r2 = truncate_light_curve_at_mjd(df, cutoff_mjd=t2)

    assert len(r1) <= len(r2)


@given(
    df=light_curve_strategy(),
    cutoff_offset=st.floats(min_value=0.0, max_value=150.0),
)
@settings(max_examples=50, deadline=None)
def test_property_cadence_preservation(df: pd.DataFrame, cutoff_offset: float) -> None:
    """Property 5: Output timestamps are a strict subset of original timestamps."""
    cutoff_mjd = float(df["mjd"].min() + cutoff_offset)
    truncated = truncate_light_curve_at_mjd(df, cutoff_mjd=cutoff_mjd)

    if not truncated.empty:
        assert set(truncated["mjd"]).issubset(set(df["mjd"]))

        merged = pd.merge(
            truncated,
            df,
            on=["object_id", "mjd", "passband"],
            suffixes=("_trunc", "_orig"),
        )
        assert len(merged) == len(truncated)
        np.testing.assert_allclose(merged["flux_trunc"], merged["flux_orig"])
        np.testing.assert_allclose(merged["flux_err_trunc"], merged["flux_err_orig"])


@given(df=light_curve_strategy())
@settings(max_examples=30, deadline=None)
def test_property_forbidden_fields_stripped(df: pd.DataFrame) -> None:
    """Property 6: Any forbidden global statistics fields are stripped."""
    poisoned_df = df.copy()
    poisoned_df["peak_mjd"] = 60050.0
    poisoned_df["peak_flux"] = 12345.0
    poisoned_df["total_obs_count"] = 999
    poisoned_df["true_target"] = 64
    poisoned_df["hostgal_specz"] = 0.25

    cutoff_mjd = float(df["mjd"].median())
    truncated = truncate_light_curve_at_mjd(poisoned_df, cutoff_mjd=cutoff_mjd)

    for field in OBSERVATION_FORBIDDEN_FIELDS:
        assert field not in truncated.columns


@given(df=light_curve_strategy())
@settings(max_examples=30, deadline=None)
def test_property_t0_chronological_invariance(df: pd.DataFrame) -> None:
    """Property 7: First detection t_0 is invariant to future data changes."""
    df_with_det = df.copy()
    df_with_det.loc[0, "flux"] = 100.0
    df_with_det.loc[0, "flux_err"] = 1.0  # S/N = 100

    t0_base = get_first_detection_mjd(df_with_det, detection_snr_threshold=5.0)
    assert t0_base is not None
    assert isinstance(t0_base, float)

    mutated_df = df_with_det.copy()
    after_mask = mutated_df["mjd"] > t0_base
    mutated_df.loc[after_mask, "flux"] = 99999.0
    mutated_df.loc[after_mask, "detected_bool"] = 1

    t0_mutated = get_first_detection_mjd(mutated_df, detection_snr_threshold=5.0)

    assert t0_base == t0_mutated


@given(df=light_curve_strategy())
@settings(max_examples=30, deadline=None)
def test_property_as_of_epoch_sequences(df: pd.DataFrame) -> None:
    """Property 8: Epoch sequences at e=(0, 2, 7) preserve append-only consistency."""
    df_alert = df.copy()
    df_alert.loc[0, "flux"] = 100.0
    df_alert.loc[0, "flux_err"] = 2.0  # S/N = 50

    seq = generate_as_of_epoch_sequences(df_alert, epochs=(0.0, 2.0, 7.0))
    assert 0.0 in seq and 2.0 in seq and 7.0 in seq

    e0, e2, e7 = seq[0.0], seq[2.0], seq[7.0]

    assert len(e0) <= len(e2) <= len(e7)

    if not e0.empty:
        pd.testing.assert_frame_equal(e0, e2.iloc[: len(e0)].reset_index(drop=True))

    if not e2.empty:
        pd.testing.assert_frame_equal(e2, e7.iloc[: len(e2)].reset_index(drop=True))


@given(df=light_curve_strategy())
@settings(max_examples=20, deadline=None)
def test_property_truncate_at_epoch_single(df: pd.DataFrame) -> None:
    """Property 9: Truncate at epoch e equals truncate at MJD (t_0 + e)."""
    df_alert = df.copy()
    df_alert.loc[0, "flux"] = 100.0
    df_alert.loc[0, "flux_err"] = 2.0  # S/N = 50

    t0 = get_first_detection_mjd(df_alert, detection_snr_threshold=5.0)
    assert t0 is not None and isinstance(t0, float)

    truncated_epoch = truncate_light_curve_at_epoch(
        df_alert, days_since_first_detection=2.0
    )
    truncated_mjd = truncate_light_curve_at_mjd(df_alert, cutoff_mjd=t0 + 2.0)

    pd.testing.assert_frame_equal(truncated_epoch, truncated_mjd)
