"""Synthetic light-curve generation for recovery & identifiability tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sample_empirical_sparsity(
    epoch: float,
    target_class: int = 64,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Sample observation sparsity matching empirical distributions (ADR 005).

    Parameters
    ----------
    epoch : float
        Elapsed observer-frame epoch (0.0 or 2.0 days).
    target_class : int, default 64
        PLAsTiCC class ID (64=KN, 90=SN Ia, 95=SLSN-I).
    rng : np.random.Generator | None, default None
        NumPy random generator instance.

    Returns
    -------
    dict[str, Any]
        Dictionary with sampled 'n_det', 'n_passbands', and 'det_span_days'.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if epoch == 0.0:
        return {
            "n_det": 1,
            "n_passbands": 1,
            "det_span_days": 0.0,
        }

    # Empirical distribution for e = 2.0 days
    # (docs/results/early_lightcurve_sparsity.md)
    if target_class == 64:  # Kilonova
        # Joint distribution of (n_det, n_pb) to accurately reproduce both marginals:
        # n_det=1, n_pb=1: 50.0%
        # n_det=2, n_pb=1: 14.4%
        # n_det=2, n_pb=2: 28.9%
        # n_det=3-4, n_pb=2: 1.9%
        # n_det=3-4, n_pb=3+: 3.9%
        # n_det=5+, n_pb=3+: 0.9%
        joint_cases = [
            (1, 1, 0.500),
            (2, 1, 0.144),
            (2, 2, 0.289),
            (3, 2, 0.019),
            (4, 3, 0.039),
            (5, 3, 0.009),
        ]
        idx = rng.choice(len(joint_cases), p=[c[2] for c in joint_cases])
        n_det_val, n_pb_val, _ = joint_cases[idx]

        span = float(rng.exponential(scale=0.35))
        span = min(span, 1.8) if n_det_val > 1 else 0.0

        return {
            "n_det": n_det_val,
            "n_passbands": n_pb_val,
            "det_span_days": span,
        }
    else:
        # Default distribution for other classes at e=2d
        p_n_det = [0.27, 0.25, 0.38, 0.10]
        n_det_cat = rng.choice([1, 2, 3, 5], p=p_n_det)
        n_det = int(n_det_cat)
        n_pb = min(n_det, int(rng.choice([1, 2, 3])))
        span = float(rng.uniform(0.01, 1.9)) if n_det > 1 else 0.0
        return {
            "n_det": n_det,
            "n_passbands": n_pb,
            "det_span_days": span,
        }


def generate_synthetic_light_curve(
    true_rise_rates: dict[int, float],
    true_initial_fluxes: dict[int, float],
    sparsity: dict[str, Any],
    t0_mjd: float = 58000.0,
    noise_std: float = 5.0,
    object_id: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic light curve with known ground-truth parameters.

    Parameters
    ----------
    true_rise_rates : dict[int, float]
        Ground-truth flux rise rates (dF/dt) per passband ID.
    true_initial_fluxes : dict[int, float]
        Ground-truth initial flux F_0 at t0 per passband ID.
    sparsity : dict[str, Any]
        Sparsity dictionary with 'n_det', 'n_passbands', 'det_span_days'.
    t0_mjd : float, default 58000.0
        Initial detection MJD.
    noise_std : float, default 5.0
        Measurement uncertainty stddev (flux_err).
    object_id : int, default 1
        Synthetic object ID.
    rng : np.random.Generator | None, default None
        NumPy random generator.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Observation DataFrame and metadata Series containing 'hostgal_photoz'.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_det = sparsity["n_det"]
    n_pb = sparsity["n_passbands"]
    span = sparsity["det_span_days"]

    available_pbs = list(true_rise_rates.keys())
    selected_pbs = rng.choice(available_pbs, size=n_pb, replace=False).tolist()

    # Assign observation timestamps
    if n_det == 1:
        mjds = np.array([t0_mjd])
        obs_pbs = [selected_pbs[0]]
    else:
        # Distribute timestamps across span
        times = np.sort(rng.uniform(0.0, span, size=n_det - 1))
        mjds = np.concatenate([[t0_mjd], t0_mjd + times])
        # Distribute passbands across points
        obs_pbs = rng.choice(selected_pbs, size=n_det, replace=True).tolist()
        # Ensure every selected passband is present at least once
        for idx, pb in enumerate(selected_pbs):
            obs_pbs[idx] = pb

    rows = []
    for mjd, pb in zip(mjds, obs_pbs, strict=True):
        dt = mjd - t0_mjd
        rate = true_rise_rates[pb]
        f0 = true_initial_fluxes[pb]
        true_flux = f0 + rate * dt

        # Add Gaussian noise
        observed_flux = rng.normal(true_flux, noise_std)
        flux_err = noise_std
        snr = observed_flux / flux_err
        detected = 1 if snr >= 5.0 else 0

        rows.append(
            {
                "object_id": object_id,
                "mjd": float(mjd),
                "passband": int(pb),
                "flux": float(observed_flux),
                "flux_err": float(flux_err),
                "detected_bool": int(detected),
            }
        )

    df_obs = pd.DataFrame(rows).sort_values("mjd").reset_index(drop=True)
    meta_row = pd.Series(
        {
            "object_id": object_id,
            "hostgal_photoz": 0.15,
            "hostgal_photoz_err": 0.01,
            "true_target": 64,
        }
    )
    return df_obs, meta_row
