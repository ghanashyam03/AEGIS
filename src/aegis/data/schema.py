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
