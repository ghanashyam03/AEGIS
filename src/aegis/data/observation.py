"""Observation truncation harness for alert-stream simulation without future leakage.

This module provides functions to truncate full astronomical light curves to partial
observations available as of a cutoff MJD or elapsed days since first alert.

LEAKAGE PREVENTION CONTRACT
---------------------------
1. Truncation operates strictly on observation timestamps (mjd <= cutoff).
2. First detection timestamp t_0 is computed chronologically from first detection
   (S/N >= 5.0 or detected_bool == 1) and never from peak flux or future points.
3. All global summary statistics (peak_mjd, final_flux, total_obs_count, etc.)
   and truth labels are stripped or rejected.
4. Cadence, noise, and measurement values of returned historical observations are
   preserved exactly as recorded in the source data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from aegis.data.schema import (
    OBSERVATION_FORBIDDEN_FIELDS,
    validate_observation_frame,
)


def get_first_detection_mjd(
    df: pd.DataFrame,
    detection_snr_threshold: float = 5.0,
) -> float | dict[int, float] | None:
    """Compute the first detection MJD (t_0) for an object or group of objects.

    First detection t_0 is defined per ADR 003 as the earliest MJD where
    flux / flux_err >= detection_snr_threshold OR detected_bool == 1.

    Parameters
    ----------
    df : pd.DataFrame
        Light curve observation DataFrame containing at least 'mjd', 'flux',
        'flux_err', and optionally 'detected_bool' and 'object_id'.
    detection_snr_threshold : float, default 5.0
        Signal-to-noise ratio threshold for initial detection.

    Returns
    -------
    float | dict[int, float] | None
        - If 'object_id' contains a single unique value (or absent): float MJD or None.
        - If 'object_id' has multiple values: dictionary mapping object_id to MJD.
    """
    if df.empty:
        if "object_id" in df.columns and len(df["object_id"].unique()) > 1:
            return {}
        return None

    # Calculate S/N safely
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(df["flux_err"] > 0, df["flux"] / df["flux_err"], 0.0)

    is_detected = snr >= detection_snr_threshold
    if "detected_bool" in df.columns:
        is_detected = is_detected | (df["detected_bool"] == 1)

    detected_df = df[is_detected]

    # Check if single or multi-object
    if "object_id" not in df.columns:
        if detected_df.empty:
            return None
        return float(detected_df["mjd"].min())

    unique_ids = df["object_id"].unique()
    if len(unique_ids) <= 1:
        if detected_df.empty:
            return None
        return float(detected_df["mjd"].min())

    # Multi-object case
    result: dict[int, float] = {}
    if detected_df.empty:
        for obj_id in unique_ids:
            result[int(obj_id)] = None  # type: ignore[assignment]
        return result

    first_mjds = detected_df.groupby("object_id")["mjd"].min()
    for obj_id in unique_ids:
        if obj_id in first_mjds.index:
            result[int(obj_id)] = float(first_mjds.loc[obj_id])
        else:
            result[int(obj_id)] = None  # type: ignore[assignment]

    return result


def truncate_light_curve_at_mjd(
    df: pd.DataFrame,
    cutoff_mjd: float | dict[int, float] | pd.Series,
    strip_forbidden: bool = True,
    validate_schema: bool = True,
) -> pd.DataFrame:
    """Truncate light curve observations to rows occurring on or before cutoff_mjd.

    Parameters
    ----------
    df : pd.DataFrame
        Light curve observation DataFrame.
    cutoff_mjd : float | dict[int, float] | pd.Series
        Maximum allowed observation timestamp (MJD). Can be a single float for all
        rows, or a mapping/Series keyed by object_id.
    strip_forbidden : bool, default True
        Whether to strip columns that encode future/global statistics.
    validate_schema : bool, default True
        Whether to validate the resulting DataFrame against OBSERVATION_SCHEMA.

    Returns
    -------
    pd.DataFrame
        Sorted observation DataFrame with mjd <= cutoff_mjd.
    """
    if df.empty:
        if validate_schema and not df.empty:
            return validate_observation_frame(df, strip_forbidden=strip_forbidden)
        return df.copy()

    work_df = df.copy()

    # Strip forbidden fields if requested
    if strip_forbidden:
        forbidden = [c for c in work_df.columns if c in OBSERVATION_FORBIDDEN_FIELDS]
        if forbidden:
            work_df = work_df.drop(columns=forbidden)

    if isinstance(cutoff_mjd, (dict, pd.Series)):
        if "object_id" not in work_df.columns:
            raise ValueError(
                "object_id column required when cutoff_mjd is a dict or Series."
            )
        cutoff_series = work_df["object_id"].map(cutoff_mjd)
        mask = work_df["mjd"] <= cutoff_series
    else:
        mask = work_df["mjd"] <= float(cutoff_mjd)

    truncated = work_df[mask].reset_index(drop=True)

    # Sort chronologically by object_id (if present) and mjd
    sort_cols = ["mjd"]
    if "object_id" in truncated.columns:
        sort_cols = ["object_id", "mjd"]
    truncated = truncated.sort_values(sort_cols).reset_index(drop=True)

    if validate_schema and not truncated.empty:
        truncated = validate_observation_frame(
            truncated, strip_forbidden=strip_forbidden
        )

    return truncated


def truncate_light_curve_at_epoch(
    df: pd.DataFrame,
    days_since_first_detection: float,
    detection_snr_threshold: float = 5.0,
    strip_forbidden: bool = True,
    validate_schema: bool = True,
) -> pd.DataFrame:
    """Truncate light curves at elapsed observer-frame days after alert (t_0 + e).

    Parameters
    ----------
    df : pd.DataFrame
        Full light curve observation DataFrame.
    days_since_first_detection : float
        Elapsed observer-frame days 'e' after first alert (t_0).
    detection_snr_threshold : float, default 5.0
        Signal-to-noise ratio threshold used to determine t_0.
    strip_forbidden : bool, default True
        Whether to strip forbidden global summary columns.
    validate_schema : bool, default True
        Whether to validate output schema.

    Returns
    -------
    pd.DataFrame
        Truncated light curve DataFrame with mjd <= t_0 + e.
        Objects with no qualifying initial detection return empty frames.
    """
    if df.empty:
        return df.copy()

    first_mjd_info = get_first_detection_mjd(
        df, detection_snr_threshold=detection_snr_threshold
    )

    if isinstance(first_mjd_info, dict):
        # Multi-object case
        cutoff_map: dict[int, float] = {}
        for obj_id, t0 in first_mjd_info.items():
            if t0 is not None:
                cutoff_map[obj_id] = t0 + float(days_since_first_detection)
            else:
                # No initial detection -> cutoff -inf so no observations match
                cutoff_map[obj_id] = -np.inf
        return truncate_light_curve_at_mjd(
            df,
            cutoff_mjd=cutoff_map,
            strip_forbidden=strip_forbidden,
            validate_schema=validate_schema,
        )
    else:
        # Single object or no object_id
        if first_mjd_info is None:
            # Return empty frame with same columns
            work_df = df.copy()
            if strip_forbidden:
                forbidden = [
                    c for c in work_df.columns if c in OBSERVATION_FORBIDDEN_FIELDS
                ]
                if forbidden:
                    work_df = work_df.drop(columns=forbidden)
            return work_df.iloc[0:0].reset_index(drop=True)

        cutoff_mjd = first_mjd_info + float(days_since_first_detection)
        return truncate_light_curve_at_mjd(
            df,
            cutoff_mjd=cutoff_mjd,
            strip_forbidden=strip_forbidden,
            validate_schema=validate_schema,
        )


def generate_as_of_epoch_sequences(
    df: pd.DataFrame,
    epochs: Sequence[float] = (0.0, 2.0, 7.0),
    detection_snr_threshold: float = 5.0,
    strip_forbidden: bool = True,
) -> dict[float, pd.DataFrame]:
    """Generate sequences of epoch-truncated partial light curves for evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Full light curve observation DataFrame.
    epochs : Sequence[float], default (0.0, 2.0, 7.0)
        Sequence of elapsed observer-frame days after first detection.
    detection_snr_threshold : float, default 5.0
        S/N threshold for first alert t_0.
    strip_forbidden : bool, default True
        Whether to strip forbidden global statistics.

    Returns
    -------
    dict[float, pd.DataFrame]
        Dictionary mapping each epoch 'e' to its truncated observation DataFrame.
    """
    sequences: dict[float, pd.DataFrame] = {}
    for epoch in epochs:
        sequences[float(epoch)] = truncate_light_curve_at_epoch(
            df,
            days_since_first_detection=float(epoch),
            detection_snr_threshold=detection_snr_threshold,
            strip_forbidden=strip_forbidden,
        )
    return sequences
