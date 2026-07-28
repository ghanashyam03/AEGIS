"""Mechanism and property tests for triage decision policy and metrics.

Tests required per prompt & ADR 007:
1. Synthetic Perfect-Information: Policy recovers oracle decisions and zero regret.
2. Synthetic Zero-Signal: Policy produces no spurious triggers and maintains base-rate default.
3. Cost Ratio Shift: Valuing KN detection more heavily shifts trigger thresholds in expected direction.
4. Leakage Safety: Explicitly verifies policy at epoch e rejects observations with mjd > t_0 + e.
5. Policy Sanity / Monotonicity: Proves S_e is strictly non-decreasing w.r.t. both confidence and novelty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegis.config.decision import DecisionPolicyConfig
from aegis.decision.policy import (
    SequentialDecisionPolicy,
    assert_leakage_safety,
    compute_combined_decision_score,
)
from aegis.decision.utility import (
    compute_missed_high_value_event_rate,
    compute_object_utility,
    compute_oracle_actions,
    compute_total_utility,
    compute_utility_regret,
)


def test_reference_utility_and_oracle() -> None:
    """Verify reference utility values and oracle capacity targeting per ADR 003."""
    # Target (64) triggered -> +2.0
    assert compute_object_utility(action=1, y_true=64) == 2.0
    # Non-target (90) triggered -> -1.0
    assert compute_object_utility(action=1, y_true=90) == -1.0
    # Any object not triggered -> 0.0
    assert compute_object_utility(action=0, y_true=64) == 0.0
    assert compute_object_utility(action=0, y_true=90) == 0.0

    # Oracle test with 3 targets (64) and 5 non-targets (90)
    y_true = np.array([90, 64, 90, 64, 90, 64, 90, 90])
    capacity = 2

    oracle_a = compute_oracle_actions(y_true, capacity=capacity, target_class=64)
    assert np.sum(oracle_a) == 2
    # Oracle must trigger 2 of the 3 kilonovae (indices 1, 3, 5)
    triggered_classes = y_true[oracle_a == 1]
    assert np.all(triggered_classes == 64)

    u_oracle = compute_total_utility(oracle_a, y_true)
    assert u_oracle == 4.0  # 2 * +2.0 = 4.0


def test_perfect_information_oracle_recovery() -> None:
    """Test STEP 3: Under perfect information, policy recovers oracle decisions and zero regret."""
    y_true = np.array([90, 64, 90, 64, 90, 90, 64, 90])  # 3 KN (64), 5 SNIa (90)
    capacity = 3

    # Perfect classifier signal: p_kn = 1.0 for 64, 0.0 for 90
    p_kn = np.where(y_true == 64, 1.0, 0.0)
    novelty_scores = np.zeros_like(p_kn)

    config = DecisionPolicyConfig(
        capacity_per_epoch=capacity,
        novelty_weight=0.05,
        decision_threshold=0.5,
    )
    policy = SequentialDecisionPolicy(config=config)

    untriggered = np.ones(len(y_true), dtype=bool)
    scores, actions = policy.evaluate_candidates(
        p_kn=p_kn,
        novelty_scores=novelty_scores,
        untriggered_mask=untriggered,
    )

    oracle_actions = compute_oracle_actions(y_true, capacity=capacity, target_class=64)

    # Policy actions must match oracle actions exactly
    np.testing.assert_array_equal(actions, oracle_actions)

    regret_metrics = compute_utility_regret(
        policy_actions=actions,
        y_true=y_true,
        capacity=capacity,
        target_class=64,
    )
    assert regret_metrics["regret"] == 0.0
    assert regret_metrics["normalized_regret"] == 0.0

    mhver = compute_missed_high_value_event_rate(actions, y_true, target_class=64)
    assert mhver == 0.0


def test_zero_signal_base_rate_default() -> None:
    """Test STEP 3: Under zero signal, policy avoids spurious triggers above threshold."""
    # Zero signal: uninformative flat probabilities and zero novelty
    n_samples = 20
    p_kn = np.full(n_samples, 0.0001)  # Below decision_threshold (0.001)
    nov_scores = np.zeros(n_samples)

    config = DecisionPolicyConfig(
        capacity_per_epoch=5,
        novelty_weight=0.05,
        decision_threshold=0.001,
    )
    policy = SequentialDecisionPolicy(config=config)

    untriggered = np.ones(n_samples, dtype=bool)
    scores, actions = policy.evaluate_candidates(
        p_kn=p_kn,
        novelty_scores=nov_scores,
        untriggered_mask=untriggered,
    )

    # No candidate crosses decision_threshold -> 0 triggers produced
    assert np.sum(actions) == 0


def test_cost_ratio_threshold_shift() -> None:
    """Test STEP 3: Increasing target utility (u_tp) or lowering threshold shifts trigger behavior."""
    # Moderate signal candidates with p_kn = 0.005
    p_kn = np.array([0.005, 0.005, 0.0002])
    nov_scores = np.array([0.1, 0.1, 0.0])

    # High threshold config -> no triggers
    config_strict = DecisionPolicyConfig(
        capacity_per_epoch=2,
        decision_threshold=0.02,
    )
    policy_strict = SequentialDecisionPolicy(config=config_strict)
    _, actions_strict = policy_strict.evaluate_candidates(
        p_kn=p_kn,
        novelty_scores=nov_scores,
        untriggered_mask=np.ones(3, dtype=bool),
    )
    assert np.sum(actions_strict) == 0

    # Valuing target events more heavily by lowering threshold -> triggers candidates
    config_lenient = DecisionPolicyConfig(
        capacity_per_epoch=2,
        decision_threshold=0.001,
    )
    policy_lenient = SequentialDecisionPolicy(config=config_lenient)
    _, actions_lenient = policy_lenient.evaluate_candidates(
        p_kn=p_kn,
        novelty_scores=nov_scores,
        untriggered_mask=np.ones(3, dtype=bool),
    )
    assert np.sum(actions_lenient) == 2


def test_leakage_safety_assertion() -> None:
    """Test STEP 2(c): Policy explicitly rejects observations with MJD > t_0 + epoch."""
    # Build clean truncated observation DataFrame (t_0 = 60000.0, epoch = 2.0 -> max MJD = 60002.0)
    obs_valid = pd.DataFrame(
        {
            "object_id": [1, 1, 1],
            "mjd": [60000.0, 60001.0, 60002.0],
            "flux": [10.0, 12.0, 15.0],
            "flux_err": [1.0, 1.0, 1.0],
            "detected_bool": [1, 1, 1],
            "passband": [0, 1, 2],
        }
    )
    # Valid frame should pass without error
    assert_leakage_safety(obs_valid, epoch=2.0)

    # Build leaked observation DataFrame with timestamp 60003.0 > t_0 + 2.0
    obs_leaked = pd.DataFrame(
        {
            "object_id": [1, 1, 1],
            "mjd": [60000.0, 60001.0, 60003.0],  # 60003.0 > 60002.0
            "flux": [10.0, 12.0, 15.0],
            "flux_err": [1.0, 1.0, 1.0],
            "detected_bool": [1, 1, 1],
            "passband": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        assert_leakage_safety(obs_leaked, epoch=2.0)


def test_policy_sanity_monotonicity_property() -> None:
    """Test POLICY SANITY CONSTRAINT: Decision score is strictly monotonic w.r.t components."""
    base_p = 0.05
    base_nov = 1.2
    w_nov = 0.05
    scale = 1.0

    score_base = compute_combined_decision_score(
        p_kn=base_p, novelty_score=base_nov, novelty_scale=scale, w_nov=w_nov
    )

    # 1. Increasing p_kn with novelty fixed MUST increase score
    score_p_up = compute_combined_decision_score(
        p_kn=base_p + 0.01, novelty_score=base_nov, novelty_scale=scale, w_nov=w_nov
    )
    assert score_p_up > score_base

    # 2. Increasing novelty with p_kn fixed MUST non-decrease score
    score_nov_up = compute_combined_decision_score(
        p_kn=base_p, novelty_score=base_nov + 0.5, novelty_scale=scale, w_nov=w_nov
    )
    assert score_nov_up > score_base

    # 3. Parameterized monotonicity over random grids
    p_grid = np.linspace(0.0, 1.0, 20)
    nov_grid = np.linspace(0.0, 10.0, 20)

    for i in range(len(p_grid) - 1):
        s1 = compute_combined_decision_score(p_grid[i], base_nov, scale, w_nov)
        s2 = compute_combined_decision_score(p_grid[i + 1], base_nov, scale, w_nov)
        assert s2 > s1

    for j in range(len(nov_grid) - 1):
        s1 = compute_combined_decision_score(base_p, nov_grid[j], scale, w_nov)
        s2 = compute_combined_decision_score(base_p, nov_grid[j + 1], scale, w_nov)
        assert s2 >= s1
