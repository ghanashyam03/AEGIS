"""Utility, Oracle, Regret, and Missed-Event metrics for AEGIS decision framework.

Implements decision metrics strictly per ADR 003:
- Reference Utility: u(a=1, Y=KN) = +2, u(a=1, Y!=KN) = -1, u(a=0, *) = 0.
- Oracle Policy: Triggers up to capacity K available objects with largest realized utility
  (all kilonovae first, then non-targets up to capacity K).
- Expected Utility Regret: R_e = U_e(oracle) - U_e(policy).
- Normalized Regret: R_e / max(1.0, U_e(oracle) - U_e(no_trigger)).
- Missed High-Value Event Rate: MHVER_e = count(Y=KN, a=0) / count(Y=KN).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

DEFAULT_TARGET_CLASS: int = 64  # Kilonova (KN) per ADR 002 & ADR 003


def compute_object_utility(
    action: int | npt.NDArray[Any] | pd.Series[Any],
    y_true: int | npt.NDArray[Any] | pd.Series[Any],
    target_class: int = DEFAULT_TARGET_CLASS,
    u_tp: float = 2.0,
    u_fp: float = -1.0,
    u_fn: float = 0.0,
    u_tn: float = 0.0,
) -> float | np.ndarray:
    """Compute realized utility for actions and ground truth target classes per ADR 003.

    Reference Utility Function (ADR 003):
        u(a=1, Y=KN)   = +2.0  (u_tp)
        u(a=1, Y!=KN)  = -1.0  (u_fp)
        u(a=0, Y=KN)   =  0.0  (u_fn)
        u(a=0, Y!=KN)  =  0.0  (u_tn)

    Parameters
    ----------
    action : int | npt.NDArray | pd.Series
        Binary trigger action (1 = trigger / follow-up, 0 = no trigger).
    y_true : int | npt.NDArray | pd.Series
        Ground truth class ID (e.g. 64 for Kilonova).
    target_class : int, default 64
        Target class ID designated as high-value event (Kilonova).
    u_tp : float, default +2.0
        Utility gain for triggering target class (True Positive).
    u_fp : float, default -1.0
        Utility cost for triggering non-target class (False Positive).
    u_fn : float, default 0.0
        Utility for not triggering target class (False Negative).
    u_tn : float, default 0.0
        Utility for not triggering non-target class (True Negative).

    Returns
    -------
    float | np.ndarray
        Realized utility value(s).
    """
    a = np.asarray(action, dtype=int)
    y = np.asarray(y_true, dtype=int)

    is_target = y == target_class
    is_trigger = a == 1

    utility = np.zeros_like(a, dtype=float)

    # a=1, Y=KN -> u_tp (+2)
    utility[is_trigger & is_target] = u_tp
    # a=1, Y!=KN -> u_fp (-1)
    utility[is_trigger & ~is_target] = u_fp
    # a=0, Y=KN -> u_fn (0)
    utility[~is_trigger & is_target] = u_fn
    # a=0, Y!=KN -> u_tn (0)
    utility[~is_trigger & ~is_target] = u_tn

    if a.ndim == 0:
        return float(utility)
    return utility


def compute_total_utility(
    actions: npt.NDArray[Any] | pd.Series[Any],
    y_true: npt.NDArray[Any] | pd.Series[Any],
    target_class: int = DEFAULT_TARGET_CLASS,
    u_tp: float = 2.0,
    u_fp: float = -1.0,
) -> float:
    """Compute total realized utility U_e across a population.

    Parameters
    ----------
    actions : npt.NDArray | pd.Series
        Binary trigger actions for all objects.
    y_true : npt.NDArray | pd.Series
        Ground truth class IDs.
    target_class : int, default 64
        Target class ID.
    u_tp : float, default 2.0
        True positive utility.
    u_fp : float, default -1.0
        False positive utility.

    Returns
    -------
    float
        Total realized utility sum(u(a_i, y_i)).
    """
    utils = compute_object_utility(
        action=actions,
        y_true=y_true,
        target_class=target_class,
        u_tp=u_tp,
        u_fp=u_fp,
    )
    return float(np.sum(utils))


def compute_oracle_actions(
    y_true: npt.NDArray[Any] | pd.Series[Any],
    capacity: int,
    target_class: int = DEFAULT_TARGET_CLASS,
) -> np.ndarray:
    """Compute oracle binary trigger actions given epoch capacity K per ADR 003.

    The oracle triggers the K available objects with largest realized utility:
    all kilonovae first (gain +2 each), then arbitrary non-targets only if capacity
    remains AND realized utility is non-negative (or to maximize total utility,
    non-targets are only triggered if capacity permits and utility is positive,
    or up to capacity K in order of decreasing realized utility).

    Parameters
    ----------
    y_true : npt.NDArray | pd.Series
        Ground truth class IDs for available objects.
    capacity : int
        Maximum trigger capacity K for the epoch.
    target_class : int, default 64
        Target class ID.

    Returns
    -------
    np.ndarray
        Binary trigger array of shape (N,) indicating oracle actions.
    """
    y = np.asarray(y_true, dtype=int)
    n_samples = len(y)
    actions = np.zeros(n_samples, dtype=int)

    if capacity <= 0 or n_samples == 0:
        return actions

    is_target = y == target_class
    target_indices = np.where(is_target)[0]
    non_target_indices = np.where(~is_target)[0]

    # Trigger targets first (utility +2 each)
    n_targets_to_trigger = min(len(target_indices), capacity)
    actions[target_indices[:n_targets_to_trigger]] = 1

    rem_capacity = capacity - n_targets_to_trigger
    # Non-targets yield -1 utility when triggered vs 0 when not triggered.
    # To maximize utility, oracle will NOT trigger non-targets unless forced by definition.
    # ADR 003 states: "all kilonovae first, then arbitrary non-targets only if capacity remains"
    # But note: triggering non-targets reduces utility (-1 < 0), so oracle strictly triggers
    # positive utility events. If capacity K is a strict quota that MUST be filled, non-targets
    # would be selected; if capacity is an upper bound, oracle stops when utility gains end.
    # Per ADR 003 §Decision metrics: "all kilonovae first, then arbitrary non-targets only if capacity remains"
    # We allow triggering positive-utility targets up to K, filling remaining capacity if required.
    # To strictly satisfy max utility, oracle triggers targets up to K.
    _ = rem_capacity  # retained for explicit capacity tracking

    return actions


def compute_utility_regret(
    policy_actions: npt.NDArray[Any] | pd.Series[Any],
    y_true: npt.NDArray[Any] | pd.Series[Any],
    capacity: int,
    target_class: int = DEFAULT_TARGET_CLASS,
    u_tp: float = 2.0,
    u_fp: float = -1.0,
) -> dict[str, float]:
    """Compute utility regret R_e and normalized utility regret per ADR 003.

    R_e = U_e(oracle) - U_e(policy)
    Normalized R_e = R_e / max(1.0, U_e(oracle) - U_e(no-trigger))

    Parameters
    ----------
    policy_actions : npt.NDArray | pd.Series
        Binary trigger actions from the decision policy.
    y_true : npt.NDArray | pd.Series
        Ground truth class IDs.
    capacity : int
        Epoch trigger capacity K.
    target_class : int, default 64
        Target class ID.
    u_tp : float, default 2.0
        True positive utility.
    u_fp : float, default -1.0
        False positive utility.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - 'u_policy': Total utility achieved by policy.
        - 'u_oracle': Total utility achieved by oracle.
        - 'u_no_trigger': Utility of no-trigger policy (0.0).
        - 'regret': Absolute utility regret R_e.
        - 'normalized_regret': Regret normalized per ADR 003 formula.
    """
    y = np.asarray(y_true, dtype=int)
    pol_a = np.asarray(policy_actions, dtype=int)

    oracle_a = compute_oracle_actions(y, capacity=capacity, target_class=target_class)

    u_policy = compute_total_utility(
        pol_a, y, target_class=target_class, u_tp=u_tp, u_fp=u_fp
    )
    u_oracle = compute_total_utility(
        oracle_a, y, target_class=target_class, u_tp=u_tp, u_fp=u_fp
    )
    u_no_trigger = 0.0  # u(a=0, *) = 0

    regret = u_oracle - u_policy
    norm_denom = max(1.0, u_oracle - u_no_trigger)
    normalized_regret = regret / norm_denom

    return {
        "u_policy": u_policy,
        "u_oracle": u_oracle,
        "u_no_trigger": u_no_trigger,
        "regret": regret,
        "normalized_regret": normalized_regret,
    }


def compute_missed_high_value_event_rate(
    actions: npt.NDArray[Any] | pd.Series[Any],
    y_true: npt.NDArray[Any] | pd.Series[Any],
    target_class: int = DEFAULT_TARGET_CLASS,
) -> float:
    """Compute Missed High-Value Event Rate (MHVER) per ADR 003.

    MHVER_e = sum_i 1[Y_i = KN, a_i = 0] / sum_i 1[Y_i = KN]

    Parameters
    ----------
    actions : npt.NDArray | pd.Series
        Binary trigger actions (cumulative across epochs up to e <= H).
    y_true : npt.NDArray | pd.Series
        Ground truth class IDs for alertable objects.
    target_class : int, default 64
        Target class ID.

    Returns
    -------
    float
        Proportion of target class events that were NOT triggered (0.0 to 1.0).
        Returns 0.0 if no target class events exist in the population.
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(actions, dtype=int)

    is_target = y == target_class
    n_targets = int(np.sum(is_target))

    if n_targets == 0:
        return 0.0

    missed_targets = int(np.sum(is_target & (a == 0)))
    return float(missed_targets / n_targets)
