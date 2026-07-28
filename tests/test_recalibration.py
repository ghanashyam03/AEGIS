"""Unit tests for selection-aware recalibration and positivity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegis.recalibration.selection_recalibration import (
    PositivityDiagnostic,
    SelectionAwareRecalibrator,
    compute_covariate_balance,
    compute_selection_weights,
    diagnose_positivity_overlap,
)


def test_compute_selection_weights() -> None:
    photoz = np.array([0.0, 0.5, 1.0, 2.0])
    weights, diagnostics = compute_selection_weights(
        photoz, p_floor=0.1, p_bright=0.8, z_50=0.5, w_z=0.15
    )

    assert len(weights) == 4
    # Low z (0.0) -> p_spec ~ 0.8 -> w ~ 1.25
    assert 1.20 <= weights[0] <= 1.30
    # Midpoint z (0.5) -> p_spec ~ 0.45 -> w ~ 2.22
    assert 2.10 <= weights[1] <= 2.30
    # High z (2.0) -> p_spec ~ 0.10 -> w ~ 10.0
    assert 9.90 <= weights[3] <= 10.01

    assert "min" in diagnostics
    assert "median" in diagnostics
    assert "p95" in diagnostics
    assert "max" in diagnostics
    assert "cv" in diagnostics
    assert "ess" in diagnostics
    assert diagnostics["ess"] > 0.0
    assert diagnostics["cv"] > 0.0


def test_compute_covariate_balance() -> None:
    rng = np.random.default_rng(42)
    z_true = rng.normal(loc=0.9, scale=0.3, size=1000)
    z_s1 = rng.normal(loc=0.5, scale=0.3, size=500)
    weights_s1 = np.ones(500) * 2.0  # mock weights

    df_true = pd.DataFrame({"hostgal_photoz": z_true})
    df_s1 = pd.DataFrame({"hostgal_photoz": z_s1})

    balance = compute_covariate_balance(
        df_s1, df_true, weights_s1, features=["hostgal_photoz"]
    )

    assert "hostgal_photoz" in balance
    res = balance["hostgal_photoz"]
    assert "smd_unweighted" in res
    assert "smd_weighted" in res
    assert res["smd_unweighted"] < 0.0  # s1 shifted lower than true


def test_diagnose_positivity_overlap() -> None:
    df_true = pd.DataFrame({"hostgal_photoz": np.linspace(0.1, 2.5, 100)})
    df_s1 = pd.DataFrame({"hostgal_photoz": np.linspace(0.1, 1.2, 50)})

    diag = diagnose_positivity_overlap(df_true, df_s1, z_cutoff=1.5)

    assert isinstance(diag, PositivityDiagnostic)
    assert diag.n_affected_objects > 0
    assert 0.0 < diag.pct_true_population < 100.0
    assert "hostgal_photoz" in diag.affected_feature_range
    assert diag.action_taken == "flagged_and_masked"

    diag_dict = diag.to_dict()
    assert diag_dict["n_affected_objects"] == diag.n_affected_objects


def test_selection_aware_recalibrator() -> None:
    rng = np.random.default_rng(42)
    n_samples = 100
    y_prob_s1 = rng.dirichlet(alpha=[1, 1, 1], size=n_samples)
    y_true_s1 = rng.choice([64, 90, 95], size=n_samples)
    photoz_s1 = rng.uniform(0.1, 1.2, size=n_samples)

    recal = SelectionAwareRecalibrator(z_cutoff=1.5, apply_extrapolation_mask=True)

    with pytest.raises(RuntimeError):
        recal.predict_proba(y_prob_s1)

    recal.fit(y_prob_s1, y_true_s1, photoz_s1)
    assert recal.is_fitted

    p_recal = recal.predict_proba(y_prob_s1, photoz=photoz_s1)
    assert p_recal.shape == (n_samples, 3)
    np.testing.assert_allclose(np.sum(p_recal, axis=1), 1.0, atol=1e-5)

    # Test extrapolation masking
    photoz_test = np.array([0.5, 2.0])  # second object > z_cutoff (1.5)
    y_prob_test = np.array([[0.2, 0.7, 0.1], [0.3, 0.4, 0.3]])

    p_test_recal = recal.predict_proba(y_prob_test, photoz=photoz_test)
    # The high-z object (idx 1) must be masked to equal unextrapolated y_prob_test[1]
    np.testing.assert_allclose(p_test_recal[1], y_prob_test[1], atol=1e-5)
