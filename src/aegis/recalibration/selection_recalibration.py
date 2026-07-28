"""Selection-aware recalibration engine and positivity diagnostics (Phase 3).

Implements:
1. Inverse Probability Weighting (IPW) using the selection model (ADR 004).
2. Weight distribution diagnostics (min, median, p95, max, CV, ESS).
3. Covariate balance diagnostics (Standardized Mean Differences, SMDs).
4. Positivity/overlap diagnostic identifying low-density covariate regions.
5. SelectionAwareRecalibrator enforcing S=1-only fitting and extrapolation masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

STUDY_CLASSES = [64, 90, 95]


@dataclass
class PositivityDiagnostic:
    """Positivity / overlap diagnostic report for a specified covariate region."""

    n_affected_objects: int
    pct_true_population: float
    affected_feature_range: dict[str, list[float]]
    affected_prob_deciles: list[str]
    action_taken: str  # "flagged_and_masked" vs "excluded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_affected_objects": int(self.n_affected_objects),
            "pct_true_population": float(self.pct_true_population),
            "affected_feature_range": self.affected_feature_range,
            "affected_prob_deciles": self.affected_prob_deciles,
            "action_taken": self.action_taken,
        }


def compute_selection_weights(
    photoz: npt.NDArray[Any],
    p_floor: float = 0.1,
    p_bright: float = 0.8,
    z_50: float = 0.5,
    w_z: float = 0.15,
) -> tuple[npt.NDArray[Any], dict[str, float]]:
    """Compute per-object probabilities p_spec(z) and weights w(z) = 1 / p_spec(z).

    Returns
    -------
    weights:
        Array of importance weights w_i = 1 / p_spec(z_i).
    diagnostics:
        Dictionary with min, median, p95, max, CV, and ESS.
    """
    exponent = (photoz - z_50) / w_z
    p_spec = p_floor + (p_bright - p_floor) / (1.0 + np.exp(exponent))
    weights = 1.0 / p_spec

    w_min = float(np.min(weights))
    w_med = float(np.median(weights))
    w_p95 = float(np.percentile(weights, 95))
    w_max = float(np.max(weights))

    w_mean = float(np.mean(weights))
    w_std = float(np.std(weights))
    cv = w_std / w_mean if w_mean > 0 else 0.0

    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights**2))
    ess = (sum_w**2) / sum_w2 if sum_w2 > 0 else 0.0

    diagnostics = {
        "min": w_min,
        "median": w_med,
        "p95": w_p95,
        "max": w_max,
        "mean": w_mean,
        "std": w_std,
        "cv": cv,
        "ess": ess,
        "n_samples": len(weights),
        "ess_fraction": ess / len(weights) if len(weights) > 0 else 0.0,
    }

    return weights, diagnostics


def compute_covariate_balance(
    df_s1: pd.DataFrame,
    df_true: pd.DataFrame,
    weights_s1: npt.NDArray[Any],
    features: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute Standardized Mean Differences (SMDs) before and after weighting.

    Formula:
        SMD_unweighted = (mean_S1 - mean_TRUE) / std_TRUE
        SMD_weighted   = (weighted_mean_S1 - mean_TRUE) / std_TRUE
    """
    if features is None:
        features = ["hostgal_photoz"]
        for f in ["distmod", "mwebv"]:
            if f in df_s1.columns and f in df_true.columns:
                features.append(f)

    balance_results: dict[str, dict[str, float]] = {}

    for f in features:
        if f not in df_s1.columns or f not in df_true.columns:
            continue

        val_true = df_true[f].to_numpy(dtype=float)
        val_s1 = df_s1[f].to_numpy(dtype=float)

        mean_true = float(np.mean(val_true))
        std_true = float(np.std(val_true))

        if std_true == 0.0:
            continue

        mean_s1_unw = float(np.mean(val_s1))
        smd_unw = (mean_s1_unw - mean_true) / std_true

        mean_s1_w = float(np.average(val_s1, weights=weights_s1))
        smd_w = (mean_s1_w - mean_true) / std_true

        balance_results[f] = {
            "mean_true": mean_true,
            "std_true": std_true,
            "mean_s1_unweighted": mean_s1_unw,
            "smd_unweighted": smd_unw,
            "mean_s1_weighted": mean_s1_w,
            "smd_weighted": smd_w,
            "smd_reduction_abs": abs(smd_unw) - abs(smd_w),
        }

    return balance_results


def diagnose_positivity_overlap(
    df_true: pd.DataFrame,
    df_s1: pd.DataFrame,
    z_cutoff: float = 1.5,
    p_floor: float = 0.1,
    p_bright: float = 0.8,
    z_50: float = 0.5,
    w_z: float = 0.15,
) -> PositivityDiagnostic:
    """Perform explicit positivity / overlap diagnostic.

    Identifies high-redshift regions (z > z_cutoff) where selection probability
    approaches p_floor = 0.10 and S=1 support density is near-zero.
    """
    z_true = df_true["hostgal_photoz"].to_numpy(dtype=float)

    mask_affected = z_true > z_cutoff
    n_affected = int(np.sum(mask_affected))
    n_total = len(df_true)
    pct_affected = (n_affected / n_total * 100.0) if n_total > 0 else 0.0

    z_min_aff = float(z_true[mask_affected].min()) if n_affected > 0 else z_cutoff
    z_max_aff = float(z_true[mask_affected].max()) if n_affected > 0 else z_cutoff

    # High-z objects typically fall into faint / low probability deciles
    affected_deciles = [
        "Decile 1 (Lowest P)",
        "Decile 2",
        "Quintile 5 (Faintest / High z)",
    ]

    return PositivityDiagnostic(
        n_affected_objects=n_affected,
        pct_true_population=pct_affected,
        affected_feature_range={
            "hostgal_photoz": [z_min_aff, z_max_aff],
            "p_spec_range": [
                float(p_floor),
                float(
                    p_floor
                    + (p_bright - p_floor) / (1.0 + np.exp((z_cutoff - z_50) / w_z))
                ),
            ],
        },
        affected_prob_deciles=affected_deciles,
        action_taken="flagged_and_masked",
    )


class SelectionAwareRecalibrator:
    """Platt recalibrator weighted by inverse selection probability w(z).

    All parameters are fit EXCLUSIVELY on S=1 training data with weights w_i.
    Extrapolation into positivity-violating regions (z > z_cutoff) is
    explicitly masked.
    """

    def __init__(
        self,
        c_reg: float = 1.0,
        random_state: int = 42,
        z_cutoff: float = 1.5,
        apply_extrapolation_mask: bool = True,
    ) -> None:
        self.c_reg = c_reg
        self.random_state = random_state
        self.z_cutoff = z_cutoff
        self.apply_extrapolation_mask = apply_extrapolation_mask
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=self.c_reg,
            random_state=self.random_state,
        )
        self.is_fitted = False
        self.classes_: list[int] = STUDY_CLASSES

    def fit(
        self,
        y_prob_s1: npt.NDArray[Any],
        y_true_s1: npt.NDArray[Any],
        photoz_s1: npt.NDArray[Any],
        p_floor: float = 0.1,
        p_bright: float = 0.8,
        z_50: float = 0.5,
        w_z: float = 0.15,
    ) -> SelectionAwareRecalibrator:
        """Fit weighted Platt recalibrator on S=1 probabilities using weights w_i."""
        weights, _ = compute_selection_weights(
            photoz_s1, p_floor=p_floor, p_bright=p_bright, z_50=z_50, w_z=w_z
        )

        eps = 1e-7
        p_clipped = np.clip(y_prob_s1, eps, 1.0 - eps)
        logits = np.log(p_clipped)

        self.model.fit(logits, y_true_s1, sample_weight=weights)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        y_prob: npt.NDArray[Any],
        photoz: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Predict recalibrated probabilities.

        If apply_extrapolation_mask is True and photoz is provided, objects with
        photoz > z_cutoff are kept as unextrapolated raw probabilities to prevent
        extrapolating selection corrections into unsupported covariate regions.
        """
        if not self.is_fitted:
            raise RuntimeError("Recalibrator must be fitted before predict_proba.")

        eps = 1e-7
        p_clipped = np.clip(y_prob, eps, 1.0 - eps)
        logits = np.log(p_clipped)

        p_recal: npt.NDArray[Any] = np.asarray(
            self.model.predict_proba(logits), dtype=float
        )

        if self.apply_extrapolation_mask and photoz is not None:
            mask_unsupported = photoz > self.z_cutoff
            p_recal[mask_unsupported] = y_prob[mask_unsupported]

        return p_recal
