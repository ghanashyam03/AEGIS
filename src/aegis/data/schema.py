"""Strict schema validation for released PLAsTiCC test metadata.

Column-naming note
------------------
The PLAsTiCC test release uses ``true_target`` for the unblinded class label.
The ``target`` column in the released CSV contains placeholder zeros from the
original competition (labels were withheld during the challenge); it carries no
information and is ignored by the AEGIS pipeline.  All class-membership checks
and filtering operate on ``true_target``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Check

# ---------------------------------------------------------------------------
# Study class IDs (ADR 002)
# ---------------------------------------------------------------------------
STUDY_CLASS_IDS: list[int] = [64, 90, 95]

# ---------------------------------------------------------------------------
# Raw / interim schema — applied to the full downloaded table.
# Validates the unblinded label column (true_target) along with the fields
# needed for selection-function computation.
# ---------------------------------------------------------------------------
_FINITE_CHECK = Check(
    lambda s: np.isfinite(s),
    element_wise=True,
    error="hostgal_photoz must be finite (no NaN or inf)",
)

RAW_METADATA_SCHEMA = pa.DataFrameSchema(
    {
        "object_id": pa.Column(
            int,
            nullable=False,
            unique=True,
            coerce=True,
        ),
        "hostgal_photoz": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[Check.ge(0), _FINITE_CHECK],
        ),
        "hostgal_photoz_err": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=Check.ge(0),
        ),
        "true_target": pa.Column(int, nullable=False, coerce=True),
    },
    strict=False,
    coerce=True,
)

# ---------------------------------------------------------------------------
# TRUE population schema — applied after filtering to study classes.
# Enforces class membership and re-checks all raw constraints.
# ---------------------------------------------------------------------------
TRUE_POPULATION_SCHEMA = pa.DataFrameSchema(
    {
        "object_id": pa.Column(
            int,
            nullable=False,
            unique=True,
            coerce=True,
        ),
        "hostgal_photoz": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[Check.ge(0), _FINITE_CHECK],
        ),
        "hostgal_photoz_err": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=Check.ge(0),
        ),
        "true_target": pa.Column(
            int,
            nullable=False,
            coerce=True,
            checks=Check.isin(STUDY_CLASS_IDS),
        ),
    },
    strict=False,
    coerce=True,
)


def validate_raw_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields on the full ingested table, reject malformed rows."""

    return RAW_METADATA_SCHEMA.validate(frame, lazy=True)


def validate_test_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for validate_raw_metadata.

    Retained so existing callers (scripts/ingest_population.py, tests) need
    no changes while the canonical name is ``validate_raw_metadata``.
    """

    return validate_raw_metadata(frame)


def validate_true_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the TRUE population table (study-class rows only)."""

    return TRUE_POPULATION_SCHEMA.validate(frame, lazy=True)


# ---------------------------------------------------------------------------
# Observation / Light-Curve schema — applied to alert-stream time series.
# Enforces required observation columns and guards against future-derived fields.
# ---------------------------------------------------------------------------
_FINITE_NUMERIC_CHECK = Check(
    lambda s: np.isfinite(s),
    element_wise=True,
    error=(
        "Numeric observation fields (mjd, flux, flux_err) "
        "must be finite (no NaN or inf)"
    ),
)


OBSERVATION_FORBIDDEN_FIELDS: set[str] = {
    "peak_mjd",
    "peak_flux",
    "total_obs_count",
    "max_snr",
    "final_flux",
    "true_target",
    "target",
    "hostgal_specz",
    "true_z",
    "true_distmod",
}

OBSERVATION_SCHEMA = pa.DataFrameSchema(
    {
        "object_id": pa.Column(
            int,
            nullable=False,
            coerce=True,
        ),
        "mjd": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=_FINITE_NUMERIC_CHECK,
        ),
        "passband": pa.Column(
            int,
            nullable=False,
            coerce=True,
            checks=Check.isin([0, 1, 2, 3, 4, 5]),
        ),
        "flux": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=_FINITE_NUMERIC_CHECK,
        ),
        "flux_err": pa.Column(
            float,
            nullable=False,
            coerce=True,
            checks=[Check.ge(0), _FINITE_NUMERIC_CHECK],
        ),
        "detected_bool": pa.Column(
            int,
            nullable=False,
            coerce=True,
            checks=Check.isin([0, 1]),
        ),
    },
    strict=False,
    coerce=True,
)


def validate_observation_frame(
    frame: pd.DataFrame, strip_forbidden: bool = True
) -> pd.DataFrame:
    """Validate an observation / light curve DataFrame.

    Parameters
    ----------
    frame : pd.DataFrame
        Light curve observations DataFrame.
    strip_forbidden : bool, default True
        If True, strips any columns in ``OBSERVATION_FORBIDDEN_FIELDS`` that encode
        global or post-event statistics.

    Returns
    -------
    pd.DataFrame
        Validated observation DataFrame.
    """
    if strip_forbidden:
        forbidden_present = [
            col for col in frame.columns if col in OBSERVATION_FORBIDDEN_FIELDS
        ]
        if forbidden_present:
            frame = frame.drop(columns=forbidden_present)

    return OBSERVATION_SCHEMA.validate(frame, lazy=True)
