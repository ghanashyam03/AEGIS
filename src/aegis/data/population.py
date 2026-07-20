"""Logistic proxy selection function and BIASED population construction (ADR 004).

MODELING ASSUMPTION
-------------------
The selection function implemented here is a literature-motivated proxy, NOT a
measured selection function from PLAsTiCC, DES, Rubin, or any real survey.  All
numeric parameters (p_floor, p_bright, z_50, w_z) are versioned in the Pydantic
configuration and recorded in every manifest.  The proxy is class-blind by
construction: the same logistic curve applies to every study class identically.

See docs/data/selection_function.md for the full formula, parameter rationale,
literature references, and explicit limitations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from aegis.config.data import PopulationConfig
from aegis.data.manifest import selection_summary, sha256sum, write_manifest
from aegis.data.schema import validate_true_population


def logistic_spec_probability(
    photoz: npt.NDArray[Any],
    p_floor: float,
    p_bright: float,
    z_50: float,
    w_z: float,
) -> npt.NDArray[Any]:
    """Compute per-object spectroscopic follow-up inclusion probability.

    Formula (ADR 004):

        p_spec(z) = p_floor + (p_bright - p_floor) / (1 + exp((z - z_50) / w_z))

    This is a logistic / sigmoid function that decreases monotonically from
    ``p_bright`` at z → 0 to ``p_floor`` at z → ∞.  The midpoint is at z = z_50.

    Parameters
    ----------
    photoz:
        Array of host-galaxy photometric redshifts (hostgal_photoz).
    p_floor:
        Asymptotic inclusion probability for very high-z objects.
    p_bright:
        Asymptotic inclusion probability for very low-z (bright) objects.
    z_50:
        Redshift at the logistic midpoint (p_floor + p_bright) / 2.
    w_z:
        Logistic scale; smaller → sharper transition.

    Returns
    -------
    Array of inclusion probabilities with the same shape as ``photoz``.
    """
    exponent = (photoz - z_50) / w_z
    return p_floor + (p_bright - p_floor) / (1.0 + np.exp(exponent))


def apply_selection_function(config: PopulationConfig, true_path: Path) -> Path:
    """Apply the logistic proxy selection function and produce the BIASED population.

    The BIASED population (ADR 004) is a deterministic, seeded Bernoulli subset
    of the TRUE population.  One uniform variate is drawn per object from a
    NumPy Generator seeded with ``config.selection.seed``; an object is included
    when that variate is strictly less than its ``p_spec(hostgal_photoz)`` value.

    Objects with missing or non-finite hostgal_photoz values will have already
    been rejected by upstream schema validation; none are silently converted
    into a selection decision here.  If any survive (which is a pipeline error),
    this function raises ``ValueError``.

    Parameters
    ----------
    config:
        Validated population configuration (includes selection parameters).
    true_path:
        Path to the TRUE population CSV (output of ``build_true_population``).

    Returns
    -------
    Path to ``data/processed/biased_population.csv.gz``.
    """
    true_df = pd.read_csv(true_path, compression="gzip")
    # Re-validate schema defensively
    true_df = validate_true_population(true_df)

    photoz = true_df["hostgal_photoz"].to_numpy(dtype=float)

    # Guard: upstream validation should have caught these, but be explicit.
    if not np.all(np.isfinite(photoz)):
        bad_count = int(np.sum(~np.isfinite(photoz)))
        raise ValueError(
            f"{bad_count} object(s) with non-finite hostgal_photoz reached "
            "the selection stage. Schema validation must have been bypassed."
        )

    sel = config.selection
    p_spec = logistic_spec_probability(
        photoz,
        p_floor=sel.p_floor,
        p_bright=sel.p_bright,
        z_50=sel.z_50,
        w_z=sel.w_z,
    )

    # Seeded Bernoulli draws — one uniform variate per object.
    rng = np.random.default_rng(sel.seed)
    uniforms = rng.uniform(0.0, 1.0, size=len(true_df))
    mask = uniforms < p_spec

    biased_df = true_df[mask].reset_index(drop=True)

    processed_dir = config.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    biased_path = processed_dir / "biased_population.csv.gz"
    biased_df.to_csv(biased_path, index=False, compression="gzip")

    summary = selection_summary(true_df, biased_df)
    write_manifest(
        processed_dir / "manifest_biased.json",
        {
            "population": "BIASED",
            "selection_assumption": (
                "MODELING ASSUMPTION: logistic proxy, not a measured survey "
                "selection function. See docs/data/selection_function.md."
            ),
            "selection_config": {
                "p_bright": sel.p_bright,
                "p_floor": sel.p_floor,
                "seed": sel.seed,
                "w_z": sel.w_z,
                "z_50": sel.z_50,
            },
            "selection_formula": (
                "p_spec(z) = p_floor + (p_bright - p_floor) / "
                "(1 + exp((z - z_50) / w_z))"
            ),
            "summary": summary,
            "true_sha256": sha256sum(true_path),
            "biased_file": biased_path.name,
            "biased_sha256": sha256sum(biased_path),
        },
    )
    return biased_path
