# ruff: noqa: E501
"""Stream and extract expanded light curves for all study-class objects across PLAsTiCC test partitions.

Extracts all 133 Kilonovae (class 64), all available SLSN-I (class 95), and background
Type Ia (class 90) objects into data/processed/expanded_test_lightcurves.csv.gz.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

STUDY_CLASSES = [64, 90, 95]

ZENODO_BASE_URL = "https://zenodo.org/records/2539456/files/"


def main() -> None:
    t0 = time.time()
    print("=== AEGIS EXPANDED LIGHT-CURVE EXTRACTION ===", flush=True)

    true_meta_path = Path("data/processed/true_population.csv.gz")
    out_lc_path = Path("data/processed/expanded_test_lightcurves.csv.gz")

    assert true_meta_path.exists(), (
        f"Missing TRUE population metadata at {true_meta_path}"
    )

    df_true = pd.read_csv(true_meta_path)
    study_meta = df_true[df_true["true_target"].isin(STUDY_CLASSES)].copy()

    kn_ids = set(study_meta[study_meta["true_target"] == 64]["object_id"])
    slsn_ids = set(study_meta[study_meta["true_target"] == 95]["object_id"])
    ia_ids = set(study_meta[study_meta["true_target"] == 90]["object_id"])

    print(
        f"TRUE Population target counts: KN (64) = {len(kn_ids)}, SLSN-I (95) = {len(slsn_ids)}, SN Ia (90) = {len(ia_ids)}",
        flush=True,
    )

    all_study_ids = kn_ids | slsn_ids | ia_ids

    # Local file 01
    lc01_path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz")
    extracted_dfs = []

    if lc01_path.exists():
        print(f"Reading local partition 01: {lc01_path}...", flush=True)
        usecols = ["object_id", "mjd", "passband", "flux", "flux_err"]
        head = pd.read_csv(lc01_path, nrows=5)
        if "detected_bool" in head.columns:
            usecols.append("detected_bool")
        elif "detected" in head.columns:
            usecols.append("detected")

        for chunk in pd.read_csv(lc01_path, usecols=usecols, chunksize=500_000):
            if "detected" in chunk.columns and "detected_bool" not in chunk.columns:
                chunk = chunk.rename(columns={"detected": "detected_bool"})
            sub = chunk[chunk["object_id"].isin(all_study_ids)]
            if not sub.empty:
                extracted_dfs.append(sub)

    # Check partitions 02 to 11
    partition_files = [
        f"plasticc_test_lightcurves_{i:02d}.csv.gz" for i in range(2, 12)
    ]

    for fname in partition_files:
        local_part = Path("data/raw") / fname
        if local_part.exists() and local_part.stat().st_size < 500_000_000:
            print(
                f"Removing incomplete file {local_part} ({local_part.stat().st_size} bytes)...",
                flush=True,
            )
            local_part.unlink()

        if not local_part.exists():
            url = f"{ZENODO_BASE_URL}{fname}?download=1"
            print(f"Downloading {fname} from Zenodo...", flush=True)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            try:
                with (
                    urllib.request.urlopen(req) as resp,
                    open(local_part, "wb") as f_out,
                ):
                    while True:
                        buf = resp.read(16 * 1024 * 1024)
                        if not buf:
                            break
                        f_out.write(buf)
                print(
                    f"Saved {local_part} ({os.path.getsize(local_part) / 1e6:.1f} MB)",
                    flush=True,
                )
            except Exception as exc:
                print(f"Error downloading {fname}: {exc}", flush=True)
                if local_part.exists():
                    local_part.unlink()
                continue

        if local_part.exists():
            print(f"Processing partition {fname}...", flush=True)
            usecols = ["object_id", "mjd", "passband", "flux", "flux_err"]
            head = pd.read_csv(local_part, nrows=5)
            if "detected_bool" in head.columns:
                usecols.append("detected_bool")
            elif "detected" in head.columns:
                usecols.append("detected")

            for chunk in pd.read_csv(local_part, usecols=usecols, chunksize=500_000):
                if "detected" in chunk.columns and "detected_bool" not in chunk.columns:
                    chunk = chunk.rename(columns={"detected": "detected_bool"})
                sub = chunk[chunk["object_id"].isin(all_study_ids)]
                if not sub.empty:
                    extracted_dfs.append(sub)

        # Check how many KN found so far
        if extracted_dfs:
            combined_sub = pd.concat(extracted_dfs, ignore_index=True)
            found_ids = set(combined_sub["object_id"].unique())
            found_kn = kn_ids.intersection(found_ids)
            found_slsn = slsn_ids.intersection(found_ids)
            print(
                f"  Cumulative targets extracted: KN = {len(found_kn)}/{len(kn_ids)}, SLSN-I = {len(found_slsn)}/{len(slsn_ids)}, Total objects = {len(found_ids)}",
                flush=True,
            )

    if extracted_dfs:
        df_all_extracted = pd.concat(extracted_dfs, ignore_index=True)
        print(f"\nWriting combined light curves to {out_lc_path}...", flush=True)
        out_lc_path.parent.mkdir(parents=True, exist_ok=True)
        df_all_extracted.to_csv(out_lc_path, index=False, compression="gzip")
        print(f"Finished extracting in {time.time() - t0:.2f}s", flush=True)
    else:
        print("No light curves extracted!", flush=True)


if __name__ == "__main__":
    main()
