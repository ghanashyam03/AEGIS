"""Tests for feature configuration, representation, and synthetic recovery."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aegis.config.features import FeatureConfig
from aegis.features.representation import (
    FeatureStatus,
    extract_early_representation,
)
from aegis.features.synthetic import (
    generate_synthetic_light_curve,
    sample_empirical_sparsity,
)


def test_feature_config_validation() -> None:
    """Verify FeatureConfig parameters and validation rules."""
    config = FeatureConfig()
    assert config.detection_snr_threshold == 5.0
    assert config.max_color_dt_days == 0.5
    assert config.min_points_for_slope == 2
    assert config.passbands == (0, 1, 2, 3, 4, 5)

    with pytest.raises(
        ValueError, match="Thresholds and window durations must be positive"
    ):
        FeatureConfig(detection_snr_threshold=0.0)

    with pytest.raises(ValueError, match="Fitting rates requires at least 2 points"):
        FeatureConfig(min_points_for_slope=1)


def test_synthetic_sparsity_distribution_matching() -> None:
    """Verify synthetic dataset reproduces empirical sparsity distributions.

    Benchmark reference (docs/results/early_lightcurve_sparsity.md) for Kilonova:
    - Epoch 0d: 100.0% 1 point, 100.0% 1 passband, 0.00d span.
    - Epoch 2d: 50.0% 1 pt, 43.3% 2 pts, 5.8% 3-4 pts, 1.0% 5+ pts.
                64.4% 1 band, 30.8% 2 bands, 4.8% 3+ bands.
    Tolerance: within 5 percentage points of benchmark percentages.
    """
    rng = np.random.default_rng(12345)
    n_samples = 5000

    # Test epoch 0d
    e0_samples = [
        sample_empirical_sparsity(0.0, target_class=64, rng=rng)
        for _ in range(n_samples)
    ]
    assert all(s["n_det"] == 1 for s in e0_samples)
    assert all(s["n_passbands"] == 1 for s in e0_samples)
    assert all(s["det_span_days"] == 0.0 for s in e0_samples)

    # Test epoch 2d
    e2_samples = [
        sample_empirical_sparsity(2.0, target_class=64, rng=rng)
        for _ in range(n_samples)
    ]

    n_det_1_pct = float(np.mean([s["n_det"] == 1 for s in e2_samples]) * 100)
    n_det_2_pct = float(np.mean([s["n_det"] == 2 for s in e2_samples]) * 100)
    n_det_3_4_pct = float(np.mean([s["n_det"] in (3, 4) for s in e2_samples]) * 100)

    pb_1_pct = float(np.mean([s["n_passbands"] == 1 for s in e2_samples]) * 100)
    pb_2_pct = float(np.mean([s["n_passbands"] == 2 for s in e2_samples]) * 100)

    # Compare against empirical benchmark (tolerance: 5.0%)
    assert abs(n_det_1_pct - 50.0) < 5.0, (
        f"e2 n_det=1 pct {n_det_1_pct}% outside tolerance"
    )
    assert abs(n_det_2_pct - 43.3) < 5.0, (
        f"e2 n_det=2 pct {n_det_2_pct}% outside tolerance"
    )
    assert abs(n_det_3_4_pct - 5.8) < 4.0, (
        f"e2 n_det=3-4 pct {n_det_3_4_pct}% outside tolerance"
    )

    assert abs(pb_1_pct - 64.4) < 5.0, f"e2 pb=1 pct {pb_1_pct}% outside tolerance"
    assert abs(pb_2_pct - 30.8) < 5.0, f"e2 pb=2 pct {pb_2_pct}% outside tolerance"


def test_synthetic_ground_truth_recovery() -> None:
    """Verify feature extractor recovers known injected ground truth."""
    rng = np.random.default_rng(999)
    config = FeatureConfig()

    true_rates = {1: 50.0, 2: 30.0}  # g-band slope = 50.0, r-band slope = 30.0
    true_f0 = {1: 100.0, 2: 150.0}  # g-band F0 = 100.0, r-band F0 = 150.0
    noise_std = 0.1  # very small measurement noise for precise recovery test

    # 2 passbands, 4 detections, 1.5d span
    sparsity = {"n_det": 4, "n_passbands": 2, "det_span_days": 1.5}

    df_obs, meta_row = generate_synthetic_light_curve(
        true_rise_rates=true_rates,
        true_initial_fluxes=true_f0,
        sparsity=sparsity,
        noise_std=noise_std,
        rng=rng,
    )

    result = extract_early_representation(df_obs, meta_row, config=config, epoch=2.0)

    # 1. Check rise rate recovery
    rate_g = result.features["rise_rate_pb1"]
    assert rate_g.is_constrained
    diff_g = abs(rate_g.value - true_rates[1])
    assert diff_g <= 3.0 * rate_g.uncertainty + 1e-4

    # 2. Check color recovery (g - r)
    color_gr = result.features["color_pb1_pb2"]
    assert color_gr.is_constrained

    # Calculate actual ground-truth flux at the matched observation times
    f1_obs = color_gr.diagnostics["flux_b1"]
    f2_obs = color_gr.diagnostics["flux_b2"]

    # Check that ratio match expected log ratio within analytical uncertainty
    expected_color = -2.5 * math.log10(f1_obs / f2_obs)
    assert abs(color_gr.value - expected_color) < 1e-5
    assert color_gr.uncertainty > 0.0

    # 3. Check S/N_0 and host photo-z
    assert result.features["alert_snr_0"].is_constrained
    assert result.features["hostgal_photoz"].is_constrained
    assert result.features["hostgal_photoz"].value == 0.15


def test_undersampled_identifiability_flagging() -> None:
    """Verify under-sampled light curves return NaN and UNCONSTRAINED."""
    rng = np.random.default_rng(42)
    config = FeatureConfig()

    true_rates = {1: 40.0}
    true_f0 = {1: 100.0}
    sparsity = {"n_det": 1, "n_passbands": 1, "det_span_days": 0.0}

    df_obs, meta_row = generate_synthetic_light_curve(
        true_rise_rates=true_rates,
        true_initial_fluxes=true_f0,
        sparsity=sparsity,
        noise_std=1.0,
        rng=rng,
    )

    # Epoch 0d representation (1 point total)
    result = extract_early_representation(df_obs, meta_row, config=config, epoch=0.0)

    # Rise rate must be UNCONSTRAINED and NaN
    rate = result.features["rise_rate_pb1"]
    assert not rate.is_constrained
    assert rate.status == FeatureStatus.UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS
    assert math.isnan(rate.value)
    assert math.isnan(rate.uncertainty)

    # S/N growth rate must be UNCONSTRAINED and NaN
    growth = result.features["snr_growth_rate"]
    assert not growth.is_constrained
    assert growth.status == FeatureStatus.UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS
    assert math.isnan(growth.value)

    # Colors must be UNCONSTRAINED
    color = result.features["color_pb1_pb2"]
    assert not color.is_constrained
    assert color.status == FeatureStatus.UNCONSTRAINED_NO_PASSBAND_PAIR
    assert math.isnan(color.value)

    # Alert S/N_0 should be constrained because N_det >= 1
    snr0 = result.features["alert_snr_0"]
    assert snr0.is_constrained
    assert not math.isnan(snr0.value)
