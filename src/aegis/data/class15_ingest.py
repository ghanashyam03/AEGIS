"""Isolated ingestion path for PLAsTiCC Class 15 (Tidal Disruption Events).

Strict Isolation Contract (ADR 006 & Prompt Constraints):
- Class 15 objects MUST NEVER be added to STUDY_CLASS_IDS ([64, 90, 95]).
- Class 15 objects MUST NEVER pass through TRUE_POPULATION_SCHEMA.
- Class 15 objects MUST NEVER touch classifier training, recalibration fitting,
  or existing calibration audit reports.
- Class 15 is ingested strictly as a held-out, unmodeled validation population
  for novelty detection evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aegis.config.features import FeatureConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import representation_results_to_dataframe

CLASS_15_ID = 15
CLASS_15_NAME = "Tidal Disruption Event (TDE)"


def load_class15_metadata(
    metadata_path: Path = Path("data/interim/plasticc_test_metadata_validated.csv.gz"),
    max_objects: int | None = None,
) -> pd.DataFrame:
    """Load metadata for PLAsTiCC class 15 objects from validated interim table.

    Parameters
    ----------
    metadata_path : Path
        Path to validated interim metadata CSV.
    max_objects : int | None
        Optional limit on number of class 15 objects to load for fast testing.

    Returns
    -------
    pd.DataFrame
        DataFrame containing class 15 metadata rows.
    """
    if not metadata_path.exists():
        # Fallback to raw metadata if interim doesn't exist
        raw_meta = Path("data/raw/plasticc_test_metadata.csv.gz")
        if raw_meta.exists():
            metadata_path = raw_meta
        else:
            raise FileNotFoundError(f"Missing metadata file at {metadata_path}")

    frame = pd.read_csv(metadata_path, compression="gzip")
    c15_frame = frame[frame["true_target"] == CLASS_15_ID].copy().reset_index(drop=True)

    if max_objects is not None and max_objects > 0:
        c15_frame = c15_frame.head(max_objects).copy()

    return c15_frame


def extract_class15_features_at_epoch(
    meta_c15: pd.DataFrame,
    lc_path: Path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz"),
    epoch: float = 0.0,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Extract early light-curve features for class 15 objects at a decision epoch.

    Parameters
    ----------
    meta_c15 : pd.DataFrame
        Metadata DataFrame of class 15 objects.
    lc_path : Path
        Path to raw test light curve CSV.
    epoch : float
        Elapsed decision epoch e in {0.0, 2.0, 7.0} days.
    config : FeatureConfig | None
        Feature extraction configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame of extracted features for class 15 objects.
    """
    if config is None:
        config = FeatureConfig()

    c15_ids = set(meta_c15["object_id"])

    usecols = ["object_id", "mjd", "passband", "flux", "flux_err"]
    head = pd.read_csv(lc_path, nrows=5)
    if "detected_bool" in head.columns:
        usecols.append("detected_bool")
    elif "detected" in head.columns:
        usecols.append("detected")

    obs_list = []
    for chunk in pd.read_csv(lc_path, usecols=usecols, chunksize=1_000_000):
        if "detected" in chunk.columns and "detected_bool" not in chunk.columns:
            chunk = chunk.rename(columns={"detected": "detected_bool"})
        sub = chunk[chunk["object_id"].isin(c15_ids)]
        if not sub.empty:
            obs_list.append(sub)

    if obs_list:
        df_obs_all = pd.concat(obs_list, ignore_index=True)
        obs_by_obj = {
            obj_id: group for obj_id, group in df_obs_all.groupby("object_id")
        }
    else:
        obs_by_obj = {}

    rep_results = []
    for idx in range(len(meta_c15)):
        row_series = meta_c15.iloc[idx]
        obj_id = int(row_series["object_id"])
        raw_obs = obs_by_obj.get(obj_id, pd.DataFrame())
        trunc_obs = truncate_light_curve_at_epoch(
            raw_obs, days_since_first_detection=epoch, validate_schema=False
        )
        res = extract_early_representation(
            df_obs=trunc_obs, meta_row=row_series, config=config, epoch=epoch
        )
        rep_results.append(res)

    return representation_results_to_dataframe(rep_results)
