"""Unit tests for NoveltyDetector and Class 15 Ingestion Isolation (ADR 006)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.config.data import load_population_config
from aegis.models.novelty import (
    EPOCH_IDENTIFIABLE_FEATURES,
    NoveltyDetector,
)


def test_novelty_detector_fit_and_score() -> None:
    """Test standard fitting and distance scoring on complete feature matrices."""
    # S=1 Reference distribution around mean=0, std=1
    rng = np.random.default_rng(42)
    s1_data = rng.normal(loc=0.0, scale=1.0, size=(100, 3))
    df_s1 = pd.DataFrame(s1_data, columns=["f1", "f2", "f3"])

    detector = NoveltyDetector()
    detector.fit(df_s1)

    assert detector.is_fitted_
    assert len(detector.feature_names_) == 3

    # In-distribution sample near mean=0
    eval_in = pd.DataFrame([[0.0, 0.0, 0.0]], columns=["f1", "f2", "f3"])
    score_in = detector.score_samples(eval_in)
    assert score_in[0] < 1.0

    # Out-of-distribution anomaly sample near (5.0, 5.0, 5.0)
    eval_out = pd.DataFrame([[5.0, 5.0, 5.0]], columns=["f1", "f2", "f3"])
    score_out = detector.score_samples(eval_out)
    assert score_out[0] > 4.0
    assert score_out[0] > score_in[0]


def test_novelty_detector_missing_features() -> None:
    """Test that missing/NaN feature values are handled without silent imputation."""
    rng = np.random.default_rng(42)
    s1_data = rng.normal(loc=0.0, scale=1.0, size=(100, 2))
    df_s1 = pd.DataFrame(s1_data, columns=["f1", "f2"])

    detector = NoveltyDetector()
    detector.fit(df_s1)

    # Object with f1=3.0 and f2=NaN (missing)
    df_missing = pd.DataFrame([[3.0, np.nan]], columns=["f1", "f2"])
    score_missing = detector.score_samples(df_missing)

    # Expected distance: sqrt( 1/1 * ((3.0 - mean_f1)/std_f1)^2 )
    mean_f1 = detector.means_["f1"]
    std_f1 = detector.stds_["f1"]
    expected_dist = abs((3.0 - mean_f1) / std_f1)

    np.testing.assert_allclose(score_missing[0], expected_dist, rtol=1e-4)


def test_novelty_detector_unfitted_raises() -> None:
    """Test that scoring prior to fitting raises RuntimeError."""
    detector = NoveltyDetector()
    df_eval = pd.DataFrame([[1.0, 2.0]], columns=["f1", "f2"])
    with pytest.raises(RuntimeError, match="NoveltyDetector must be fitted"):
        detector.score_samples(df_eval)


def test_class15_strict_isolation_contract() -> None:
    """Verify Class 15 is NEVER present in PopulationConfig study classes or schemas."""

    config_path = Path("configs/data.yaml")
    if config_path.exists():
        config = load_population_config(config_path)
        study_classes = list(config.classes.values())
        assert 15 not in study_classes
        assert set(study_classes) == {64, 90, 95}

    # Also verify frozen BaselineClassifierConfig study_classes
    from aegis.config.models import BaselineClassifierConfig

    clf_config = BaselineClassifierConfig()
    assert 15 not in clf_config.study_classes
    assert set(clf_config.study_classes) == {64, 90, 95}


def test_epoch_identifiable_subspace() -> None:
    """Verify identifiable feature subspaces conform strictly to ADR 006 & P6."""
    assert 0.0 in EPOCH_IDENTIFIABLE_FEATURES
    assert 2.0 in EPOCH_IDENTIFIABLE_FEATURES
    assert 7.0 in EPOCH_IDENTIFIABLE_FEATURES

    # At e=0.0d, only hostgal_photoz is identifiable
    assert EPOCH_IDENTIFIABLE_FEATURES[0.0] == ["hostgal_photoz"]
