"""Unit tests for baseline probabilistic multiclass classifier module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegis.config.models import BaselineClassifierConfig
from aegis.features.representation import (
    FeatureStatus,
    RepresentationResult,
    SingleFeatureResult,
)
from aegis.models.baseline import (
    BaselineClassifier,
    representation_results_to_dataframe,
)


@pytest.fixture
def sample_representation_results() -> list[RepresentationResult]:
    """Generate a list of sample RepresentationResult objects for testing."""
    results = []
    for i in range(10):
        feat = {
            "rise_rate_pb0": SingleFeatureResult(
                value=0.5 if i % 2 == 0 else float("nan"),
                uncertainty=0.1 if i % 2 == 0 else float("nan"),
                status=(
                    FeatureStatus.WELL_CONSTRAINED
                    if i % 2 == 0
                    else FeatureStatus.UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS
                ),
                diagnostics={},
            ),
            "hostgal_photoz": SingleFeatureResult(
                value=0.2 + 0.1 * i,
                uncertainty=0.02,
                status=FeatureStatus.WELL_CONSTRAINED,
                diagnostics={},
            ),
        }
        diag = {
            "n_obs_total": 5 + i,
            "n_det_total": 2 if i % 2 == 0 else 1,
            "n_det_passbands": 2 if i % 2 == 0 else 1,
            "det_time_span_days": 1.5 if i % 2 == 0 else 0.0,
            "well_constrained_features": 2 if i % 2 == 0 else 1,
            "unconstrained_features": 0 if i % 2 == 0 else 1,
        }
        results.append(
            RepresentationResult(
                object_id=100 + i,
                epoch=2.0,
                features=feat,
                summary_diagnostics=diag,
            )
        )
    return results


def test_representation_results_to_dataframe(
    sample_representation_results: list[RepresentationResult],
) -> None:
    """Test conversion of RepresentationResult list into feature DataFrame."""
    df = representation_results_to_dataframe(sample_representation_results)
    assert len(df) == 10
    assert "object_id" in df.columns
    assert "epoch" in df.columns
    assert "rise_rate_pb0" in df.columns
    assert "rise_rate_pb0_err" in df.columns
    assert "rise_rate_pb0_status" in df.columns
    assert "diag_n_det_total" in df.columns
    assert "diag_well_constrained_features" in df.columns
    assert pd.isna(df.loc[1, "rise_rate_pb0"])
    assert not pd.isna(df.loc[0, "rise_rate_pb0"])


def test_probabilistic_simplex(
    sample_representation_results: list[RepresentationResult],
) -> None:
    """Test that classifier outputs valid probability simplex per object."""
    X = representation_results_to_dataframe(sample_representation_results)
    y = np.array([64, 90, 95, 64, 90, 95, 64, 90, 95, 64])

    config = BaselineClassifierConfig(random_seed=42)
    clf = BaselineClassifier(config=config)
    clf.fit_epoch(X, y, epoch=2.0, population_type="BIASED")

    probs = clf.predict_proba(X, epoch=2.0)

    assert probs.shape == (10, 3)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)


def test_true_population_exclusion() -> None:
    """Test explicit exclusion of TRUE (S=0) population from fitting."""
    X = np.random.randn(20, 5)
    y = np.random.choice([64, 90, 95], size=20)

    clf = BaselineClassifier()

    with pytest.raises(ValueError, match="EXCLUSIVELY on the BIASED"):
        clf.fit_epoch(X, y, epoch=0.0, population_type="TRUE")

    with pytest.raises(ValueError, match="EXCLUSIVELY on the BIASED"):
        clf.fit_epoch(X, y, epoch=0.0, population_type="UNLABELED")

    bad_meta_1 = pd.DataFrame({"population": ["BIASED"] * 10 + ["TRUE"] * 10})
    with pytest.raises(ValueError, match="non-BIASED population objects"):
        clf.fit_epoch(X, y, epoch=0.0, population_type="BIASED", meta_df=bad_meta_1)

    bad_meta_2 = pd.DataFrame({"S": [1] * 15 + [0] * 5})
    with pytest.raises(ValueError, match="objects with S != 1"):
        clf.fit_epoch(X, y, epoch=0.0, population_type="BIASED", meta_df=bad_meta_2)


def test_seed_reproducibility(
    sample_representation_results: list[RepresentationResult],
) -> None:
    """Test exact prediction reproducibility under a fixed random seed."""
    X = representation_results_to_dataframe(sample_representation_results)
    y = np.array([64, 90, 95, 64, 90, 95, 64, 90, 95, 64])

    config = BaselineClassifierConfig(random_seed=12345)
    clf1 = BaselineClassifier(config=config)
    clf1.fit_epoch(X, y, epoch=2.0, population_type="BIASED")
    probs1 = clf1.predict_proba(X, epoch=2.0)

    clf2 = BaselineClassifier(config=config)
    clf2.fit_epoch(X, y, epoch=2.0, population_type="BIASED")
    probs2 = clf2.predict_proba(X, epoch=2.0)

    np.testing.assert_array_equal(probs1, probs2)


def test_fit_quality_diagnostics_input_included(
    sample_representation_results: list[RepresentationResult],
) -> None:
    """Test that support/fit-quality diagnostics are included as feature inputs."""
    X = representation_results_to_dataframe(sample_representation_results)
    y = np.array([64, 90, 95, 64, 90, 95, 64, 90, 95, 64])

    clf = BaselineClassifier()
    clf.fit_epoch(X, y, epoch=2.0, population_type="BIASED")

    fitted_features = clf.feature_names_[2.0]
    assert "diag_n_det_total" in fitted_features
    assert "diag_well_constrained_features" in fitted_features
    assert "diag_unconstrained_features" in fitted_features
    assert "rise_rate_pb0_status" in fitted_features
