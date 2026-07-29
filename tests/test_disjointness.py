# ruff: noqa: E501
"""Mandatory Disjointness & Zero-Data-Leakage Automated Tests (Step 2).

Verifies zero overlap between every object evaluated in the AEGIS triage policy
(headline kilonova and SLSN-I generalization) and every object used in classifier
training, novelty reference-population (S=1) construction, or threshold/config selection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

STUDY_CLASSES = [64, 90, 95]


@pytest.fixture
def train_s1_object_ids() -> set[int]:
    """Load all object IDs used in S=1 classifier training and novelty reference construction."""
    train_meta_path = Path("data/raw/plasticc_train_metadata.csv.gz")
    assert train_meta_path.exists(), f"Missing training metadata at {train_meta_path}"
    df_train = pd.read_csv(train_meta_path)
    return set(df_train["object_id"].astype(int))


@pytest.fixture
def true_population_object_ids() -> set[int]:
    """Load all object IDs in the TRUE population metadata."""
    true_meta_path = Path("data/processed/true_population.csv.gz")
    assert true_meta_path.exists(), (
        f"Missing TRUE population metadata at {true_meta_path}"
    )
    df_true = pd.read_csv(true_meta_path)
    return set(df_true["object_id"].astype(int))


@pytest.fixture
def evaluation_cohort_01_object_ids() -> set[int]:
    """Load object IDs present in the preliminary evaluation slice (plasticc_test_lightcurves_01)."""
    lc01_path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz")
    assert lc01_path.exists(), f"Missing test lightcurve partition 01 at {lc01_path}"
    df_lc01 = pd.read_csv(lc01_path, usecols=["object_id"])
    return set(df_lc01["object_id"].astype(int))


def test_true_population_disjoint_from_s1_training(
    train_s1_object_ids: set[int], true_population_object_ids: set[int]
) -> None:
    """Test that TRUE population has zero overlap with S=1 training data."""
    overlap = train_s1_object_ids.intersection(true_population_object_ids)
    assert len(overlap) == 0, (
        f"Data leakage detected! {len(overlap)} objects overlap between S=1 training "
        f"and TRUE evaluation population: {list(overlap)[:5]}"
    )


def test_preliminary_cohort_disjoint_from_s1_training(
    train_s1_object_ids: set[int], evaluation_cohort_01_object_ids: set[int]
) -> None:
    """Test that preliminary evaluation slice (lightcurves_01) has zero overlap with S=1 training."""
    overlap = train_s1_object_ids.intersection(evaluation_cohort_01_object_ids)
    assert len(overlap) == 0, (
        f"Data leakage detected! {len(overlap)} objects overlap between S=1 training "
        f"and evaluation slice lightcurves_01: {list(overlap)[:5]}"
    )


def test_target_kilonova_disjointness(train_s1_object_ids: set[int]) -> None:
    """Test that all 133 kilonova objects in TRUE population are disjoint from S=1 training set."""
    true_meta_path = Path("data/processed/true_population.csv.gz")
    df_true = pd.read_csv(true_meta_path)
    kn_true_ids = set(df_true[df_true["true_target"] == 64]["object_id"].astype(int))

    assert len(kn_true_ids) == 133, (
        f"Expected 133 kilonovae in TRUE population, got {len(kn_true_ids)}"
    )

    overlap = train_s1_object_ids.intersection(kn_true_ids)
    assert len(overlap) == 0, (
        f"Data leakage detected! {len(overlap)} kilonova objects overlap between "
        f"S=1 training set and TRUE evaluation kilonovae."
    )


def test_target_slsn_disjointness(train_s1_object_ids: set[int]) -> None:
    """Test that all 35,782 SLSN-I objects in TRUE population are disjoint from S=1 training set."""
    true_meta_path = Path("data/processed/true_population.csv.gz")
    df_true = pd.read_csv(true_meta_path)
    slsn_true_ids = set(df_true[df_true["true_target"] == 95]["object_id"].astype(int))

    assert len(slsn_true_ids) == 35782, (
        f"Expected 35,782 SLSN-I in TRUE population, got {len(slsn_true_ids)}"
    )

    overlap = train_s1_object_ids.intersection(slsn_true_ids)
    assert len(overlap) == 0, (
        f"Data leakage detected! {len(overlap)} SLSN-I objects overlap between "
        f"S=1 training set and TRUE evaluation SLSN-I objects."
    )


def test_expanded_population_strict_disjointness_guarantee(
    train_s1_object_ids: set[int],
) -> None:
    """Test that any subset of TRUE population satisfies zero-overlap contract."""
    true_meta_path = Path("data/processed/true_population.csv.gz")
    df_true = pd.read_csv(true_meta_path)
    study_true_ids = set(
        df_true[df_true["true_target"].isin(STUDY_CLASSES)]["object_id"].astype(int)
    )

    assert len(study_true_ids) == 1_695_746

    overlap = train_s1_object_ids.intersection(study_true_ids)
    assert len(overlap) == 0, (
        "Disjointness contract violated for study class objects in TRUE population."
    )
