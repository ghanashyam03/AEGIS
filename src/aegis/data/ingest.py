"""Raw-download, validated-interim, and TRUE-population stages for PLAsTiCC."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from aegis.config.data import PopulationConfig
from aegis.data.manifest import class_balance, sha256sum, write_manifest
from aegis.data.schema import validate_raw_metadata, validate_true_population


def download_raw_metadata(config: PopulationConfig) -> Path:
    """Download the configured source atomically and record its provenance."""

    raw_path = config.paths.raw_dir / config.dataset.filename
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        with (
            urlopen(str(config.dataset.url)) as response,
            tempfile.NamedTemporaryFile(dir=raw_path.parent, delete=False) as temporary,
        ):
            shutil.copyfileobj(response, temporary)
            temporary_path = Path(temporary.name)
        temporary_path.replace(raw_path)

    write_manifest(
        config.paths.raw_dir / "manifest.json",
        {
            "dataset": config.dataset.name,
            "license": config.dataset.license,
            "raw_file": raw_path.name,
            "sha256": sha256sum(raw_path),
            "source_url": str(config.dataset.url),
        },
    )
    return raw_path


def validate_to_interim(config: PopulationConfig, raw_path: Path) -> Path:
    """Validate raw CSV metadata and materialize a normalized interim table.

    Performance note: pandera element-wise checks on 3.5 M rows are prohibitively
    slow.  The strategy here is:

    1. Validate full schema on the first 10 000 rows (catches structural/dtype
       problems and a representative sample of value-range violations).
    2. Perform fast vectorized finite-value checks on the complete columns.
    3. Write the whole frame; per-class check with membership enforcement is
       deferred to the TRUE-population stage (which processes ~1.7 M rows).
    """
    frame = pd.read_csv(raw_path, compression="gzip")

    # Step 1: full-schema pandera validation on a structural sample.
    validate_raw_metadata(frame.head(10_000))

    # Step 2: fast vectorized guard on the complete photoz column.
    photoz = frame["hostgal_photoz"].to_numpy(dtype=float)
    bad_mask = ~np.isfinite(photoz) | (photoz < 0)
    if bad_mask.any():
        raise ValueError(
            f"{int(bad_mask.sum())} row(s) with non-finite or negative "
            "hostgal_photoz found in the full table."
        )

    interim_path = config.paths.interim_dir / "plasticc_test_metadata_validated.csv.gz"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        interim_path, index=False, compression={"method": "gzip", "compresslevel": 1}
    )
    write_manifest(
        config.paths.interim_dir / "manifest.json",
        {
            "class_balance": class_balance(frame),
            "raw_sha256": sha256sum(raw_path),
            "rows": int(len(frame)),
            "validated_file": interim_path.name,
            "validated_sha256": sha256sum(interim_path),
        },
    )
    return interim_path


def build_true_population(config: PopulationConfig, interim_path: Path) -> Path:
    """Filter validated interim table to study classes and produce the TRUE population.

    The TRUE population (ADR 004) contains every object in the released test
    metadata whose ``true_target`` value is one of the three pre-registered study
    classes: kilonova (64), Type Ia supernova (90), superluminous Type I
    supernova (95).  No other filtering is applied; all objects are treated as
    if they had been observed with no follow-up bias.

    Parameters
    ----------
    config:
        Validated population configuration.
    interim_path:
        Path to the validated interim CSV (output of ``validate_to_interim``).

    Returns
    -------
    Path to ``data/processed/true_population.csv.gz``.
    """
    frame = pd.read_csv(interim_path, compression="gzip")
    study_ids = list(config.classes.values())  # [64, 90, 95]
    true_frame = frame[frame["true_target"].isin(study_ids)].reset_index(drop=True)
    # Re-validate with the strict TRUE-population schema (class membership check)
    true_frame = validate_true_population(true_frame)

    processed_dir = config.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    true_path = processed_dir / "true_population.csv.gz"
    true_frame.to_csv(true_path, index=False, compression="gzip")

    write_manifest(
        processed_dir / "manifest_true.json",
        {
            "class_balance": class_balance(true_frame),
            "interim_sha256": sha256sum(interim_path),
            "population": "TRUE",
            "rows": int(len(true_frame)),
            "selection_config": {
                "p_bright": config.selection.p_bright,
                "p_floor": config.selection.p_floor,
                "seed": config.selection.seed,
                "w_z": config.selection.w_z,
                "z_50": config.selection.z_50,
            },
            "study_classes": config.classes,
            "true_file": true_path.name,
            "true_sha256": sha256sum(true_path),
        },
    )
    return true_path
