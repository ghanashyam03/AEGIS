"""Tests for the logistic selection function and population contract (ADR 004).

These tests use synthetic in-memory DataFrames; no real PLAsTiCC files are
required to run the test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegis.config.data import SelectionConfig
from aegis.data.population import apply_selection_function, logistic_spec_probability

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_sel(**overrides: object) -> SelectionConfig:
    defaults = {"seed": 42, "p_floor": 0.10, "p_bright": 0.80, "z_50": 0.5, "w_z": 0.15}
    defaults.update(overrides)  # type: ignore[arg-type]
    return SelectionConfig(**defaults)


def _valid_kwargs() -> dict[str, object]:
    return {"seed": 42, "p_floor": 0.10, "p_bright": 0.80, "z_50": 0.5, "w_z": 0.15}


def test_config_is_frozen() -> None:
    from pydantic import ValidationError

    cfg = SelectionConfig(**_valid_kwargs())
    with pytest.raises((ValidationError, TypeError)):
        cfg.seed = 0  # type: ignore[misc]


def _true_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic TRUE population DataFrame with realistic photo-z spread."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "object_id": np.arange(1, n + 1, dtype=int),
            "hostgal_photoz": rng.uniform(0.0, 2.0, size=n),
            "hostgal_photoz_err": rng.uniform(0.01, 0.2, size=n),
            "target": rng.choice([64, 90, 95], size=n),
        }
    )


def _run_selection(
    true_df: pd.DataFrame,
    sel: SelectionConfig,
    tmp_path: pytest.TempPathType,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write true_df to tmp_path, run selection, return (true_df, biased_df)."""

    from aegis.config.data import (
        DatasetConfig,
        PathsConfig,
        PopulationConfig,
    )

    # Write the true population to a temp file so apply_selection_function can read it.
    tmp_path.mkdir(parents=True, exist_ok=True)
    true_path = tmp_path / "true_population.csv.gz"
    true_df.to_csv(true_path, index=False, compression="gzip")

    # Build a minimal PopulationConfig pointing at tmp_path.
    config = PopulationConfig(
        dataset=DatasetConfig(
            name="test",
            url="https://example.com/test.csv.gz",  # type: ignore[arg-type]
            filename="test.csv.gz",
            license="test",
        ),
        paths=PathsConfig(
            raw_dir=tmp_path / "raw",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        ),
        classes={"kilonova": 64, "sn_ia": 90, "slsn_i": 95},
        selection=sel,
    )

    biased_path = apply_selection_function(config, true_path)
    biased_df = pd.read_csv(biased_path, compression="gzip")
    return true_df, biased_df


# ---------------------------------------------------------------------------
# logistic_spec_probability unit tests
# ---------------------------------------------------------------------------


def test_logistic_returns_array_of_correct_shape() -> None:
    z = np.array([0.0, 0.5, 1.0, 2.0])
    result = logistic_spec_probability(z, p_floor=0.1, p_bright=0.8, z_50=0.5, w_z=0.15)
    assert result.shape == z.shape


def test_logistic_monotonically_decreasing_with_z() -> None:
    """Higher redshift → lower p_spec (proxy mirrors spectroscopic reality)."""
    z = np.linspace(0.0, 3.0, 300)
    p = logistic_spec_probability(z, p_floor=0.1, p_bright=0.8, z_50=0.5, w_z=0.15)
    assert np.all(np.diff(p) <= 0), "p_spec must be non-increasing with z"


def test_logistic_at_low_z_approaches_p_bright() -> None:
    # At z=0 with z_50=0.5, w_z=0.15: exponent = (0-0.5)/0.15 ≈ -3.33
    # logistic(−3.33) ≈ 0.965, so p ≈ 0.1 + 0.7*0.965 ≈ 0.776.
    # We verify "substantially above the midpoint (0.45)" and below p_bright.
    p = logistic_spec_probability(
        np.array([0.0]), p_floor=0.1, p_bright=0.8, z_50=0.5, w_z=0.15
    )
    midpoint = (0.1 + 0.8) / 2
    assert p[0] > midpoint, f"p at z=0 must exceed midpoint {midpoint}, got {p[0]}"
    assert p[0] <= 0.8, f"p at z=0 must not exceed p_bright=0.80, got {p[0]}"


def test_logistic_at_high_z_approaches_p_floor() -> None:
    p = logistic_spec_probability(
        np.array([10.0]), p_floor=0.1, p_bright=0.8, z_50=0.5, w_z=0.15
    )
    assert p[0] < 0.11, f"Expected close to p_floor=0.10, got {p[0]}"


def test_logistic_midpoint_at_z50() -> None:
    """At z = z_50 the result must equal the midpoint (p_floor + p_bright) / 2."""
    p_floor, p_bright, z_50, w_z = 0.1, 0.8, 0.5, 0.15
    midpoint = (p_floor + p_bright) / 2
    p = logistic_spec_probability(np.array([z_50]), p_floor, p_bright, z_50, w_z)
    assert abs(p[0] - midpoint) < 1e-10


def test_logistic_all_values_in_range() -> None:
    z = np.linspace(0.0, 5.0, 500)
    p = logistic_spec_probability(z, p_floor=0.1, p_bright=0.8, z_50=0.5, w_z=0.15)
    assert np.all((p >= 0.1) & (p <= 0.8))


# ---------------------------------------------------------------------------
# Population contract tests
# ---------------------------------------------------------------------------


def test_biased_is_strict_subset_of_true(tmp_path: pytest.TempPathType) -> None:
    """Every object_id in BIASED must appear in TRUE (no fabricated objects)."""
    true_df, biased_df = _run_selection(_true_frame(), _default_sel(), tmp_path)
    true_ids = set(true_df["object_id"])
    biased_ids = set(biased_df["object_id"])
    assert biased_ids.issubset(true_ids), "BIASED contains object_ids not in TRUE"


def test_biased_is_proper_subset_not_equal(tmp_path: pytest.TempPathType) -> None:
    """Selection function must reject at least some objects (complement nonempty)."""
    true_df, biased_df = _run_selection(_true_frame(), _default_sel(), tmp_path)
    assert len(biased_df) < len(true_df), "BIASED must be strictly smaller than TRUE"


def test_no_reverse_leakage(tmp_path: pytest.TempPathType) -> None:
    """No object can be in BIASED without being in TRUE.

    This is the 'no-leakage' contract: the biased ⊆ true direction is
    tested in test_biased_is_strict_subset_of_true; here we verify that the
    complement TRUE \\ BIASED is consistent (objects excluded from BIASED are
    indeed present in TRUE, not invented elsewhere).
    """
    true_df, biased_df = _run_selection(_true_frame(), _default_sel(), tmp_path)
    true_ids = set(true_df["object_id"])
    biased_ids = set(biased_df["object_id"])
    not_in_true = biased_ids - true_ids
    assert len(not_in_true) == 0, f"{len(not_in_true)} BIASED IDs are absent from TRUE"
    # At least some are excluded (non-trivial selection)
    excluded = true_ids - biased_ids
    assert len(excluded) > 0, "Selection function did not exclude any object"


def test_determinism_same_seed(tmp_path: pytest.TempPathType) -> None:
    """Same seed + same input must produce byte-identical biased populations."""
    frame = _true_frame(n=300, seed=7)
    sel = _default_sel(seed=99)

    _, biased_a = _run_selection(frame, sel, tmp_path / "run_a")
    _, biased_b = _run_selection(frame, sel, tmp_path / "run_b")

    pd.testing.assert_frame_equal(
        biased_a.reset_index(drop=True), biased_b.reset_index(drop=True)
    )


def test_different_seeds_differ(tmp_path: pytest.TempPathType) -> None:
    """Different seeds must produce different biased populations.

    The probability that two independent seeds select identical subsets is
    astronomically small for any nontrivial population size.
    """
    frame = _true_frame(n=400, seed=3)
    _, biased_a = _run_selection(frame, _default_sel(seed=11), tmp_path / "run_a")
    _, biased_b = _run_selection(frame, _default_sel(seed=99), tmp_path / "run_b")
    assert set(biased_a["object_id"]) != set(biased_b["object_id"])


def test_selection_monotone_direction(tmp_path: pytest.TempPathType) -> None:
    """Low-z objects must have higher mean p_spec than high-z objects.

    We verify the direction of the proxy: objects with z < z_50 are selected
    at a higher average rate than objects with z > z_50.
    """
    from aegis.data.population import logistic_spec_probability

    sel = _default_sel()
    n = 1000
    rng = np.random.default_rng(42)
    z_low = rng.uniform(0.0, sel.z_50, size=n)
    z_high = rng.uniform(sel.z_50, 2.0, size=n)
    p_low = logistic_spec_probability(
        z_low, sel.p_floor, sel.p_bright, sel.z_50, sel.w_z
    )
    p_high = logistic_spec_probability(
        z_high, sel.p_floor, sel.p_bright, sel.z_50, sel.w_z
    )
    assert p_low.mean() > p_high.mean(), "Low-z objects must have higher mean p_spec"


def test_missing_photoz_raises_before_selection(tmp_path: pytest.TempPathType) -> None:
    """Non-finite hostgal_photoz must cause loud failure, not a silent selection.

    Schema validation upstream catches NaN/inf; this test confirms the guard
    inside apply_selection_function also raises if the check is somehow bypassed.
    """
    frame = _true_frame(n=50)
    # Manually corrupt one row AFTER creating a valid-looking frame, bypassing schema
    frame_corrupted = frame.copy()
    frame_corrupted.loc[0, "hostgal_photoz"] = float("nan")

    # Write the corrupted frame directly (bypassing validate_true_population)
    true_path = tmp_path / "true_population.csv.gz"
    frame_corrupted.to_csv(true_path, index=False, compression="gzip")

    from aegis.config.data import DatasetConfig, PathsConfig, PopulationConfig

    config = PopulationConfig(
        dataset=DatasetConfig(
            name="test",
            url="https://example.com/test.csv.gz",  # type: ignore[arg-type]
            filename="test.csv.gz",
            license="test",
        ),
        paths=PathsConfig(
            raw_dir=tmp_path / "raw",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        ),
        classes={"kilonova": 64, "sn_ia": 90, "slsn_i": 95},
        selection=_default_sel(),
    )

    # The schema validation inside apply_selection_function will catch the NaN
    from pandera.errors import SchemaErrors

    with pytest.raises((ValueError, SchemaErrors)):
        apply_selection_function(config, true_path)


def test_manifest_biased_json_written(tmp_path: pytest.TempPathType) -> None:
    """The BIASED manifest must be written alongside the population file."""
    import json

    true_df, _ = _run_selection(_true_frame(), _default_sel(), tmp_path)
    manifest_path = tmp_path / "processed" / "manifest_biased.json"
    assert manifest_path.exists(), "manifest_biased.json must be written"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["population"] == "BIASED"
    assert "selection_config" in manifest
    assert "summary" in manifest
    assert manifest["summary"]["biased_total"] > 0
