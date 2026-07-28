"""Sensitivity analysis test suite for decision policy across cost ratios, capacity K, and novelty weight."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from aegis.config.decision import DecisionPolicyConfig
from aegis.decision.policy import SequentialDecisionPolicy
from aegis.decision.utility import compute_utility_regret

# Pre-specified grid: 3 cost ratios x 3 capacities x 3 novelty weights = 27 grid cells
COST_RATIOS = [1.0, 2.0, 5.0]  # u_tp values (u_fp fixed at -1.0 per ADR 003)
CAPACITIES = [1, 5, 10]
NOVELTY_WEIGHTS = [0.00, 0.05, 0.10]


@pytest.mark.parametrize(
    ("u_tp", "capacity", "w_nov"),
    list(itertools.product(COST_RATIOS, CAPACITIES, NOVELTY_WEIGHTS)),
)
def test_sensitivity_grid_oracle_recovery(
    u_tp: float,
    capacity: int,
    w_nov: float,
) -> None:
    """Verify that under perfect information, oracle recovery holds across the full 27-cell grid."""
    u_fp = -1.0
    # Population of 20 objects: 5 targets (64), 15 non-targets (90)
    rng = np.random.default_rng(42)
    y_true = np.array([64] * 5 + [90] * 15)
    rng.shuffle(y_true)

    # Perfect classifier signal: p_kn = 1.0 for 64, 0.0 for 90
    p_kn = np.where(y_true == 64, 1.0, 0.0)
    nov_scores = np.zeros_like(p_kn)

    config = DecisionPolicyConfig(
        capacity_per_epoch=capacity,
        novelty_weight=w_nov,
        decision_threshold=0.1,
        u_tp=u_tp,
        u_fp=u_fp,
    )
    policy = SequentialDecisionPolicy(config=config)

    untriggered = np.ones(len(y_true), dtype=bool)
    _, actions = policy.evaluate_candidates(
        p_kn=p_kn,
        novelty_scores=nov_scores,
        untriggered_mask=untriggered,
    )

    regret_metrics = compute_utility_regret(
        policy_actions=actions,
        y_true=y_true,
        capacity=capacity,
        target_class=64,
        u_tp=u_tp,
        u_fp=u_fp,
    )

    # Oracle recovery holds robustly across all 27 grid cells
    assert regret_metrics["regret"] == 0.0
    assert regret_metrics["normalized_regret"] == 0.0


def test_sensitivity_cost_ratio_valuation_effect() -> None:
    """Verify that increasing target valuation (u_tp) increases maximum potential target utility gain."""
    y_true = np.array([64, 64, 90, 90, 90])  # length 5
    actions = np.array([1, 1, 0, 0, 0])  # length 5

    r_1_1 = compute_utility_regret(actions, y_true, capacity=2, u_tp=1.0, u_fp=-1.0)
    r_2_1 = compute_utility_regret(actions, y_true, capacity=2, u_tp=2.0, u_fp=-1.0)
    r_5_1 = compute_utility_regret(actions, y_true, capacity=2, u_tp=5.0, u_fp=-1.0)

    # Utility achieved scales with u_tp
    assert r_1_1["u_oracle"] == 2.0  # 2 * 1.0
    assert r_2_1["u_oracle"] == 4.0  # 2 * 2.0
    assert r_5_1["u_oracle"] == 10.0  # 2 * 5.0


def test_sensitivity_zero_signal_behavior() -> None:
    """Verify zero-signal behavior across w_nov settings."""
    n_samples = 20
    p_kn = np.full(n_samples, 0.0001)  # Below decision_threshold (0.001)
    nov_scores = np.zeros(n_samples)

    for w_nov in [0.00, 0.05, 0.10]:
        config = DecisionPolicyConfig(
            capacity_per_epoch=5,
            novelty_weight=w_nov,
            decision_threshold=0.001,
        )
        policy = SequentialDecisionPolicy(config=config)
        _, actions = policy.evaluate_candidates(
            p_kn=p_kn,
            novelty_scores=nov_scores,
            untriggered_mask=np.ones(n_samples, dtype=bool),
        )
        assert np.sum(actions) == 0
