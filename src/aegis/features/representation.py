"""Early light-curve feature representation module (ADR 005 Option b).

This module implements low-parameter, strictly identifiable early light-curve
feature extraction from truncated observation DataFrames. Every feature carries an
explicit analytical uncertainty and support/fit-quality diagnostic, flagging
under-constrained estimates rather than returning ungrounded values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from aegis.config.features import FeatureConfig


class FeatureStatus(StrEnum):
    """Enumeration of feature extraction constraint statuses."""

    WELL_CONSTRAINED = "well_constrained"
    UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS = "unconstrained_insufficient_observations"
    UNCONSTRAINED_ZERO_BASELINE = "unconstrained_zero_baseline"
    UNCONSTRAINED_NON_POSITIVE_FLUX = "unconstrained_non_positive_flux"
    UNCONSTRAINED_NO_PASSBAND_PAIR = "unconstrained_no_passband_pair"
    UNCONSTRAINED_NO_DETECTION = "unconstrained_no_detection"


@dataclass(frozen=True)
class SingleFeatureResult:
    """Dataclass holding an extracted feature value, uncertainty, and diagnostic."""

    value: float
    uncertainty: float
    status: FeatureStatus
    diagnostics: dict[str, Any]

    @property
    def is_constrained(self) -> bool:
        """Return True if the feature was successfully and reliably constrained."""
        return self.status == FeatureStatus.WELL_CONSTRAINED


@dataclass(frozen=True)
class RepresentationResult:
    """Complete feature representation result for a light curve at an epoch."""

    object_id: int | None
    epoch: float | None
    features: dict[str, SingleFeatureResult]
    summary_diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result into a flat dictionary suitable for DataFrame creation."""
        res: dict[str, Any] = {
            "object_id": self.object_id,
            "epoch": self.epoch,
        }
        for name, feat in self.features.items():
            res[name] = feat.value
            res[f"{name}_err"] = feat.uncertainty
            res[f"{name}_status"] = feat.status.value
        for k, v in self.summary_diagnostics.items():
            res[f"diag_{k}"] = v
        return res


def _compute_passband_rise_rate(
    df_pb: pd.DataFrame,
    passband_id: int,
    config: FeatureConfig,
) -> SingleFeatureResult:
    """Compute early flux-rise rate (dF/dt) for a specific passband."""
    if df_pb.empty:
        return SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS,
            diagnostics={"n_obs": 0, "n_det": 0, "passband": passband_id},
        )

    # Filter for detected points
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(df_pb["flux_err"] > 0, df_pb["flux"] / df_pb["flux_err"], 0.0)

    is_det = snr >= config.detection_snr_threshold
    if "detected_bool" in df_pb.columns:
        is_det = is_det | (df_pb["detected_bool"] == 1)

    det_pb = df_pb[is_det].sort_values("mjd")
    n_det = len(det_pb)

    if n_det < config.min_points_for_slope:
        return SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS,
            diagnostics={"n_obs": len(df_pb), "n_det": n_det, "passband": passband_id},
        )

    t = det_pb["mjd"].to_numpy(dtype=float)
    f = det_pb["flux"].to_numpy(dtype=float)
    f_err = det_pb["flux_err"].to_numpy(dtype=float)

    dt = float(t[-1] - t[0])
    if dt <= 0.0:
        return SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_ZERO_BASELINE,
            diagnostics={
                "n_obs": len(df_pb),
                "n_det": n_det,
                "dt_days": dt,
                "passband": passband_id,
            },
        )

    if n_det == 2:
        # Two-point analytical slope
        slope = float((f[1] - f[0]) / dt)
        slope_err = float(np.sqrt(f_err[0] ** 2 + f_err[1] ** 2) / dt)
        return SingleFeatureResult(
            value=slope,
            uncertainty=slope_err,
            status=FeatureStatus.WELL_CONSTRAINED,
            diagnostics={
                "n_obs": len(df_pb),
                "n_det": 2,
                "dt_days": dt,
                "passband": passband_id,
                "chi2_red": None,
                "condition_number": 1.0,
            },
        )

    # Weighted least squares for N > 2: f = a + b * t
    weights = 1.0 / (f_err**2)
    sum_w = np.sum(weights)
    sum_wt = np.sum(weights * t)
    sum_wf = np.sum(weights * f)
    sum_wtt = np.sum(weights * t**2)
    sum_wtf = np.sum(weights * t * f)

    delta = sum_w * sum_wtt - sum_wt**2
    if delta <= 0:
        return SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_ZERO_BASELINE,
            diagnostics={
                "n_obs": len(df_pb),
                "n_det": n_det,
                "dt_days": dt,
                "passband": passband_id,
            },
        )

    slope = float((sum_w * sum_wtf - sum_wt * sum_wf) / delta)
    intercept = float((sum_wtt * sum_wf - sum_wt * sum_wtf) / delta)
    slope_err = float(np.sqrt(sum_w / delta))

    # Calculate reduced chi2
    residuals = f - (intercept + slope * t)
    chi2_red = float(np.sum(weights * residuals**2) / (n_det - 2))

    # Condition number of design matrix X = [ones, t]
    design_matrix = np.column_stack([np.ones_like(t), t])
    cond_num = float(np.linalg.cond(design_matrix))

    return SingleFeatureResult(
        value=slope,
        uncertainty=slope_err,
        status=FeatureStatus.WELL_CONSTRAINED,
        diagnostics={
            "n_obs": len(df_pb),
            "n_det": n_det,
            "dt_days": dt,
            "passband": passband_id,
            "chi2_red": chi2_red,
            "condition_number": cond_num,
        },
    )


def _compute_single_epoch_color(
    df_b1: pd.DataFrame,
    df_b2: pd.DataFrame,
    passband_1: int,
    passband_2: int,
    config: FeatureConfig,
) -> SingleFeatureResult:
    """Compute single-epoch cross-band color (m_b1 - m_b2) for passbands."""
    if df_b1.empty or df_b2.empty:
        return SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_NO_PASSBAND_PAIR,
            diagnostics={
                "pb1": passband_1,
                "pb2": passband_2,
                "reason": "missing_passband_obs",
            },
        )

    # Filter detected points
    with np.errstate(divide="ignore", invalid="ignore"):
        snr1 = np.where(df_b1["flux_err"] > 0, df_b1["flux"] / df_b1["flux_err"], 0.0)
        snr2 = np.where(df_b2["flux_err"] > 0, df_b2["flux"] / df_b2["flux_err"], 0.0)

    det1 = df_b1[
        (snr1 >= config.detection_snr_threshold) | (df_b1.get("detected_bool", 0) == 1)
    ]
    det2 = df_b2[
        (snr2 >= config.detection_snr_threshold) | (df_b2.get("detected_bool", 0) == 1)
    ]

    if det1.empty or det2.empty:
        return SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_NO_PASSBAND_PAIR,
            diagnostics={
                "pb1": passband_1,
                "pb2": passband_2,
                "reason": "no_detections_in_one_or_both_bands",
            },
        )

    # Find pair with minimum time difference |t1 - t2|
    best_dt = float("inf")
    best_row1 = None
    best_row2 = None

    for _, r1 in det1.iterrows():
        for _, r2 in det2.iterrows():
            dt = abs(r1["mjd"] - r2["mjd"])
            if dt < best_dt:
                best_dt = dt
                best_row1 = r1
                best_row2 = r2

    if best_row1 is None or best_row2 is None or best_dt > config.max_color_dt_days:
        return SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_NO_PASSBAND_PAIR,
            diagnostics={
                "pb1": passband_1,
                "pb2": passband_2,
                "min_dt_days": float(best_dt) if best_dt < float("inf") else None,
                "max_dt_allowed": config.max_color_dt_days,
            },
        )

    f1, f1_err = float(best_row1["flux"]), float(best_row1["flux_err"])
    f2, f2_err = float(best_row2["flux"]), float(best_row2["flux_err"])

    if f1 <= 0.0 or f2 <= 0.0:
        return SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_NON_POSITIVE_FLUX,
            diagnostics={
                "pb1": passband_1,
                "pb2": passband_2,
                "flux_b1": f1,
                "flux_b2": f2,
                "min_dt_days": float(best_dt),
            },
        )

    # m_b1 - m_b2 = -2.5 * log10(f1 / f2)
    color = float(-2.5 * np.log10(f1 / f2))
    # Analytical error propagation:
    # sigma_c = (2.5 / ln 10) * sqrt((err1/f1)^2 + (err2/f2)^2)
    factor = 2.5 / np.log(10.0)
    color_err = float(factor * np.sqrt((f1_err / f1) ** 2 + (f2_err / f2) ** 2))

    return SingleFeatureResult(
        value=color,
        uncertainty=color_err,
        status=FeatureStatus.WELL_CONSTRAINED,
        diagnostics={
            "pb1": passband_1,
            "pb2": passband_2,
            "dt_days": float(best_dt),
            "flux_b1": f1,
            "flux_b2": f2,
            "snr_b1": float(f1 / f1_err) if f1_err > 0 else 0.0,
            "snr_b2": float(f2 / f2_err) if f2_err > 0 else 0.0,
        },
    )


def extract_early_representation(
    df_obs: pd.DataFrame,
    meta_row: pd.Series | dict[str, Any] | None = None,
    config: FeatureConfig | None = None,
    epoch: float | None = None,
) -> RepresentationResult:
    """Extract early light-curve feature representation (ADR 005 Option b).

    Parameters
    ----------
    df_obs : pd.DataFrame
        Light-curve observation DataFrame truncated to the cutoff epoch via
        aegis.data.observation.
    meta_row : pd.Series | dict[str, Any] | None, default None
        Optional metadata row containing host galaxy photo-z information.
    config : FeatureConfig | None, default None
        Feature extraction configuration. Defaults to FeatureConfig().
    epoch : float | None, default None
        Optional elapsed observer-frame epoch timestamp for record keeping.

    Returns
    -------
    RepresentationResult
        Dataclass containing all extracted feature results, explicit uncertainties,
        statuses, and diagnostic support metrics.
    """
    if config is None:
        config = FeatureConfig()

    obj_id = None
    if not df_obs.empty and "object_id" in df_obs.columns:
        obj_id = int(df_obs["object_id"].iloc[0])
    elif meta_row is not None and "object_id" in meta_row:
        obj_id = int(meta_row["object_id"])

    features: dict[str, SingleFeatureResult] = {}

    # 1. Detected observation filtering across full truncated frame
    if df_obs.empty:
        det_df = df_obs.copy()
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(
                df_obs["flux_err"] > 0, df_obs["flux"] / df_obs["flux_err"], 0.0
            )
        is_det = snr >= config.detection_snr_threshold
        if "detected_bool" in df_obs.columns:
            is_det = is_det | (df_obs["detected_bool"] == 1)
        det_df = df_obs[is_det].sort_values("mjd")

    n_obs_total = len(df_obs)
    n_det_total = len(det_df)
    n_det_pb = det_df["passband"].nunique() if not det_df.empty else 0
    det_dt = (
        float(det_df["mjd"].max() - det_df["mjd"].min()) if n_det_total > 1 else 0.0
    )

    # 2. Per-passband rise rates (dF/dt)
    for pb in config.passbands:
        pb_obs = (
            df_obs[df_obs["passband"] == pb] if not df_obs.empty else pd.DataFrame()
        )
        features[f"rise_rate_pb{pb}"] = _compute_passband_rise_rate(pb_obs, pb, config)

    # 3. Single-epoch cross-band colors (m_b1 - m_b2) for adjacent passband pairs
    # Standard color pairs for LSST: (u-g, g-r, r-i, i-z, z-Y)
    color_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for b1, b2 in color_pairs:
        if b1 in config.passbands and b2 in config.passbands:
            pb1_obs = (
                df_obs[df_obs["passband"] == b1] if not df_obs.empty else pd.DataFrame()
            )
            pb2_obs = (
                df_obs[df_obs["passband"] == b2] if not df_obs.empty else pd.DataFrame()
            )
            features[f"color_pb{b1}_pb{b2}"] = _compute_single_epoch_color(
                pb1_obs, pb2_obs, b1, b2, config
            )

    # 4. Alert S/N statistics (S/N_0 and S/N growth rate)
    if n_det_total >= 1:
        first_det = det_df.iloc[0]
        f0, f0_err = float(first_det["flux"]), float(first_det["flux_err"])
        snr0 = float(f0 / f0_err) if f0_err > 0 else 0.0
        features["alert_snr_0"] = SingleFeatureResult(
            value=snr0,
            uncertainty=1.0,  # Constant measurement error scale for S/N
            status=FeatureStatus.WELL_CONSTRAINED,
            diagnostics={
                "n_det": n_det_total,
                "t0_mjd": float(first_det["mjd"]),
                "t0_passband": int(first_det["passband"]),
            },
        )
    else:
        features["alert_snr_0"] = SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_NO_DETECTION,
            diagnostics={"n_det": 0},
        )

    if n_det_total >= config.min_points_for_snr_rate:
        first_det = det_df.iloc[0]
        last_det = det_df.iloc[-1]
        dt_snr = float(last_det["mjd"] - first_det["mjd"])

        if dt_snr <= 0.0:
            features["snr_growth_rate"] = SingleFeatureResult(
                value=float("nan"),
                uncertainty=float("nan"),
                status=FeatureStatus.UNCONSTRAINED_ZERO_BASELINE,
                diagnostics={"n_det": n_det_total, "dt_days": dt_snr},
            )
        else:
            f0, f0_err = float(first_det["flux"]), float(first_det["flux_err"])
            flast, flast_err = float(last_det["flux"]), float(last_det["flux_err"])
            snr0 = float(f0 / f0_err) if f0_err > 0 else 0.0
            snr_last = float(flast / flast_err) if flast_err > 0 else 0.0

            rate = float((snr_last - snr0) / dt_snr)
            # Analytical error propagation assuming unit S/N measurement uncertainties
            rate_err = float(np.sqrt(2.0) / dt_snr)

            features["snr_growth_rate"] = SingleFeatureResult(
                value=rate,
                uncertainty=rate_err,
                status=FeatureStatus.WELL_CONSTRAINED,
                diagnostics={
                    "n_det": n_det_total,
                    "dt_days": dt_snr,
                    "snr_0": snr0,
                    "snr_latest": snr_last,
                },
            )
    else:
        features["snr_growth_rate"] = SingleFeatureResult(
            value=float("nan"),
            uncertainty=float("nan"),
            status=FeatureStatus.UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS,
            diagnostics={"n_det": n_det_total},
        )

    # 5. Host galaxy photo-z covariate
    if meta_row is not None and "hostgal_photoz" in meta_row:
        photoz = float(meta_row["hostgal_photoz"])
        photoz_err = float(meta_row.get("hostgal_photoz_err", float("nan")))
        if np.isfinite(photoz):
            features["hostgal_photoz"] = SingleFeatureResult(
                value=photoz,
                uncertainty=photoz_err if np.isfinite(photoz_err) else 0.0,
                status=FeatureStatus.WELL_CONSTRAINED,
                diagnostics={"source": "meta_row"},
            )
        else:
            features["hostgal_photoz"] = SingleFeatureResult(
                value=float("nan"),
                uncertainty=float("nan"),
                status=FeatureStatus.UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS,
                diagnostics={"reason": "non_finite_photoz"},
            )

    # Compute summary counts
    constrained_cnt = sum(1 for f in features.values() if f.is_constrained)
    unconstrained_cnt = len(features) - constrained_cnt

    summary_diag = {
        "n_obs_total": n_obs_total,
        "n_det_total": n_det_total,
        "n_det_passbands": n_det_pb,
        "det_time_span_days": det_dt,
        "well_constrained_features": constrained_cnt,
        "unconstrained_features": unconstrained_cnt,
    }

    return RepresentationResult(
        object_id=obj_id,
        epoch=epoch,
        features=features,
        summary_diagnostics=summary_diag,
    )
