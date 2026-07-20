"""Run population-construction stages for the configured PLAsTiCC source.

Stages
------
raw      — Download the source file and record its SHA-256 manifest.
interim  — Validate the raw CSV and produce a schema-checked interim table.
true     — Filter the interim table to study classes; produce the TRUE population.
biased   — Apply the logistic selection proxy; produce the BIASED population.
all      — Run all four stages in order (default when --stage is omitted).

Each stage is independently re-runnable: the raw download is skipped if the
file already exists; interim/true/biased stages overwrite their outputs and
rewrite their manifests on every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aegis.config.data import load_population_config
from aegis.data.ingest import (
    build_true_population,
    download_raw_metadata,
    validate_to_interim,
)
from aegis.data.population import apply_selection_function

STAGES = ("raw", "interim", "true", "biased", "all")


def main() -> None:
    """Run the independently rerunnable pipeline stages."""

    parser = argparse.ArgumentParser(
        description="PLAsTiCC population construction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config."
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Which stage(s) to run (default: all).",
    )
    args = parser.parse_args()
    config = load_population_config(args.config)
    stage = args.stage

    raw_path = config.paths.raw_dir / config.dataset.filename
    interim_path = config.paths.interim_dir / "plasticc_test_metadata_validated.csv.gz"
    true_path = config.paths.processed_dir / "true_population.csv.gz"

    if stage in ("raw", "all"):
        print("[1/4] Downloading raw metadata …")
        raw_path = download_raw_metadata(config)
        print(f"      → {raw_path}")

    if stage in ("interim", "all"):
        print("[2/4] Validating → interim …")
        interim_path = validate_to_interim(config, raw_path)
        print(f"      → {interim_path}")

    if stage in ("true", "all"):
        print("[3/4] Building TRUE population …")
        true_path = build_true_population(config, interim_path)
        print(f"      → {true_path}")

    if stage in ("biased", "all"):
        print("[4/4] Applying selection function → BIASED population …")
        biased_path = apply_selection_function(config, true_path)
        print(f"      → {biased_path}")

    print("Done.")


if __name__ == "__main__":
    main()
