# ruff: noqa: E501
"""Sequential early-alert triage decision policy for AEGIS.

Implements ADR 007 sequential stopping rule and combined decision score:
- Operates strictly on leakage-safe truncated observation data (mjd <= t_0 + e).
- Combines calibrated classifier confidence and normalized novelty score:
    S_e(x_i) = p_{i, KN, e} + w_nov * N_{e, norm}(x_i)
- Enforces Policy Sanity Constraint: monotonic w.r.t. confidence and novelty.
- Sequentially evaluates epochs e <= H, triggering top capacity K untriggered alerts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from aegis.config.decision import DecisionPolicyConfig
from aegis.data.observation import get_first_detection_mjd
from aegis.decision.utility import (
    compute_missed_high_value_event_rate,
    compute_utility_regret,
)


def compute_combined_decision_score(
    p_kn: float | npt.NDArray[Any] | pd.Series[Any],
    novelty_score: float | npt.NDArray[Any] | pd.Series[Any],
    novelty_scale: float = 1.0,
    w_nov: float = 0.05,
) -> float | np.ndarray:
    """Compute combined decision score S_e(x_i) = p_KN + w_nov * N_norm per ADR 007.

    POLICY SANITY & MONOTONICITY GUARANTEE:
    - d(S_e)/d(p_KN) = 1.0 > 0  (strictly increasing w.r.t. confidence)
    - d(S_e)/d(N)    = w_nov / novelty_scale >= 0  (non-decreasing w.r.t. novelty)

    Parameters
    ----------
    p_kn : float | npt.NDArray | pd.Series
        Predicted kilonova probability (in [0, 1]).
    novelty_score : float | npt.NDArray | pd.Series
        Raw standardized Euclidean novelty score N_e(x_i).
    novelty_scale : float, default 1.0
        Reference scale (standard deviation of N_e on S=1) for normalization.
    w_nov : float, default 0.05
        Decision weight for novelty score (w_nov >= 0).

    Returns
    -------
    float | np.ndarray
        Combined decision score S_e.
    """
    p = np.asarray(p_kn, dtype=float)
    nov = np.asarray(novelty_score, dtype=float)

    if w_nov < 0.0:
        raise ValueError(f"Novelty weight w_nov must be >= 0, got {w_nov}")
    if novelty_scale <= 0.0:
        scale = 1.0
    else:
        scale = float(novelty_scale)

    nov_norm = nov / scale
    score = p + w_nov * nov_norm

    if p.ndim == 0 and nov.ndim == 0:
        return float(score)
    return score


def assert_leakage_safety(
    df_obs: pd.DataFrame,
    epoch: float,
    detection_snr_threshold: float = 5.0,
) -> None:
    """Assert that observation DataFrame contains no timestamps strictly after t_0 + epoch.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Light curve observation DataFrame containing 'mjd'.
    epoch : float
        Elapsed decision epoch e in days.
    detection_snr_threshold : float, default 5.0
        S/N threshold for t_0.

    Raises
    ------
    ValueError
        If any observation MJD exceeds t_0 + epoch + 1e-6 (future leakage).
    """
    if df_obs.empty:
        return

    first_mjd_info = get_first_detection_mjd(
        df_obs, detection_snr_threshold=detection_snr_threshold
    )

    if isinstance(first_mjd_info, dict):
        for obj_id, t0 in first_mjd_info.items():
            if t0 is None:
                continue
            max_allowed = t0 + float(epoch) + 1e-6
            obj_rows = df_obs[df_obs["object_id"] == obj_id]
            if not obj_rows.empty:
                max_actual = float(obj_rows["mjd"].max())
                if max_actual > max_allowed:
                    raise ValueError(
                        f"LEAKAGE VIOLATION: object_id {obj_id} has observation MJD "
                        f"{max_actual:.4f} exceeding t_0 + epoch ({t0:.4f} + {epoch:.1f} "
                        f"= {max_allowed:.4f})."
                    )
    elif first_mjd_info is not None:
        max_allowed = first_mjd_info + float(epoch) + 1e-6
        max_actual = float(df_obs["mjd"].max())
        if max_actual > max_allowed:
            raise ValueError(
                f"LEAKAGE VIOLATION: observation MJD {max_actual:.4f} exceeds "
                f"t_0 + epoch ({first_mjd_info:.4f} + {epoch:.1f} = {max_allowed:.4f})."
            )


class SequentialDecisionPolicy:
    """Sequential capacity-constrained early-alert triage policy (ADR 007).

    Operates sequentially over epochs e <= H. At each epoch e:
    1. Evaluates candidate alerts that have not been previously triggered.
    2. Computes combined score S_e(x_i) = p_{i, KN, e} + w_nov * N_{e, norm}(x_i).
    3. Triggers top K candidates satisfying S_e(x_i) >= decision_threshold.
    """

    def __init__(self, config: DecisionPolicyConfig | None = None) -> None:
        if config is None:
            config = DecisionPolicyConfig()
        self.config = config

    def evaluate_candidates(
        self,
        p_kn: npt.NDArray[Any] | pd.Series[Any],
        novelty_scores: npt.NDArray[Any] | pd.Series[Any],
        untriggered_mask: npt.NDArray[Any] | pd.Series[Any],
        capacity: int | None = None,
        novelty_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate combined decision scores and select binary trigger actions for an epoch.

        Parameters
        ----------
        p_kn : npt.NDArray | pd.Series
            Classifier kilonova probabilities for all evaluation candidates.
        novelty_scores : npt.NDArray | pd.Series
            Novelty scores for all evaluation candidates.
        untriggered_mask : npt.NDArray | pd.Series
            Boolean mask indicating candidates that remain untriggered (True = available).
        capacity : int | None, default None
            Maximum trigger capacity K for this epoch. If None, uses config value.
        novelty_scale : float, default 1.0
            Scale for novelty score normalization.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - scores: Array of combined decision scores S_e for all candidates.
            - actions: Binary array (1 = triggered at this epoch, 0 = not triggered).
        """
        p_arr = np.asarray(p_kn, dtype=float)
        nov_arr = np.asarray(novelty_scores, dtype=float)
        available = np.asarray(untriggered_mask, dtype=bool)

        cap = capacity if capacity is not None else self.config.capacity_per_epoch

        scores_arr = np.asarray(
            compute_combined_decision_score(
                p_kn=p_arr,
                novelty_score=nov_arr,
                novelty_scale=novelty_scale,
                w_nov=self.config.novelty_weight,
            ),
            dtype=float,
        )

        n_samples = len(p_arr)
        actions = np.zeros(n_samples, dtype=int)

        if cap <= 0 or not np.any(available):
            return scores_arr, actions

        # Eligible candidates: untriggered AND score >= decision_threshold
        eligible_mask = available & (scores_arr >= self.config.decision_threshold)
        eligible_indices = np.where(eligible_mask)[0]

        if len(eligible_indices) == 0:
            return scores_arr, actions

        # Sort eligible indices by score in descending order
        sorted_indices = eligible_indices[np.argsort(-scores_arr[eligible_indices])]
        selected_indices = sorted_indices[:cap]

        actions[selected_indices] = 1
        return scores_arr, actions

    def evaluate_sequential_trace(
        self,
        epoch_predictions: dict[float, tuple[npt.NDArray[Any], npt.NDArray[Any]]],
        y_true: npt.NDArray[Any] | pd.Series[Any],
        object_ids: npt.NDArray[Any] | pd.Series[Any] | None = None,
        novelty_scales: dict[float, float] | None = None,
    ) -> dict[str, Any]:
        """Run sequential policy across ordered epochs e <= primary_deadline.

        Parameters
        ----------
        epoch_predictions : dict[float, tuple[npt.NDArray, npt.NDArray]]
            Mapping epoch -> (p_kn_array, novelty_score_array).
        y_true : npt.NDArray | pd.Series
            Ground truth class IDs for the evaluation population.
        object_ids : npt.NDArray | pd.Series | None
            Optional object IDs corresponding to rows.
        novelty_scales : dict[float, float] | None
            Optional per-epoch novelty normalization scales.

        Returns
        -------
        dict[str, Any]
            Execution trace containing cumulative actions, per-epoch triggers,
            regret metrics, and MHVER.
        """
        y_arr = np.asarray(y_true, dtype=int)
        n_samples = len(y_arr)

        if object_ids is None:
            ids_arr = np.arange(n_samples)
        else:
            ids_arr = np.asarray(object_ids)

        cumulative_actions = np.zeros(n_samples, dtype=int)
        trigger_epochs = np.full(n_samples, fill_value=-1.0, dtype=float)

        epoch_traces: dict[float, dict[str, Any]] = {}
        eval_epochs = [
            e
            for e in sorted(epoch_predictions.keys())
            if e <= self.config.primary_deadline
        ]

        for epoch in eval_epochs:
            p_kn, nov_scores = epoch_predictions[epoch]
            scale = novelty_scales.get(epoch, 1.0) if novelty_scales else 1.0

            untriggered_mask = cumulative_actions == 0

            scores, epoch_actions = self.evaluate_candidates(
                p_kn=p_kn,
                novelty_scores=nov_scores,
                untriggered_mask=untriggered_mask,
                capacity=self.config.capacity_per_epoch,
                novelty_scale=scale,
            )

            # Update cumulative actions and trigger epochs
            newly_triggered = (epoch_actions == 1) & untriggered_mask
            cumulative_actions[newly_triggered] = 1
            trigger_epochs[newly_triggered] = epoch

            epoch_traces[epoch] = {
                "scores": scores,
                "epoch_actions": epoch_actions,
                "n_new_triggers": int(np.sum(newly_triggered)),
            }

        # Calculate final decision metrics per ADR 003
        regret_dict = compute_utility_regret(
            policy_actions=cumulative_actions,
            y_true=y_arr,
            capacity=self.config.capacity_per_epoch,
            target_class=self.config.target_class,
            u_tp=self.config.u_tp,
            u_fp=self.config.u_fp,
        )

        mhver = compute_missed_high_value_event_rate(
            actions=cumulative_actions,
            y_true=y_arr,
            target_class=self.config.target_class,
        )

        return {
            "object_ids": ids_arr,
            "cumulative_actions": cumulative_actions,
            "trigger_epochs": trigger_epochs,
            "epoch_traces": epoch_traces,
            "regret": regret_dict,
            "mhver": mhver,
        }
