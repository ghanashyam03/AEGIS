# ruff: noqa: E501
"""Parallel downloader for PLAsTiCC test partitions from Zenodo."""

from __future__ import annotations

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ZENODO_BASE_URL = "https://zenodo.org/records/2539456/files/"
DEST_DIR = Path("data/raw")


def download_file(partition_idx: int) -> tuple[int, bool, float]:
    fname = f"plasticc_test_lightcurves_{partition_idx:02d}.csv.gz"
    dest = DEST_DIR / fname
    url = f"{ZENODO_BASE_URL}{fname}?download=1"

    # Expected size is ~678 MB to ~680 MB
    if dest.exists() and dest.stat().st_size > 500_000_000:
        print(
            f"[P{partition_idx:02d}] Already exists and complete ({dest.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
        return partition_idx, True, 0.0

    if dest.exists():
        dest.unlink()

    print(f"[P{partition_idx:02d}] Starting parallel download...", flush=True)
    t0 = time.time()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )

    try:
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f_out:
            while True:
                buf = resp.read(16 * 1024 * 1024)
                if not buf:
                    break
                f_out.write(buf)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(dest) / 1e6
        speed_mbps = size_mb / elapsed if elapsed > 0 else 0
        print(
            f"[P{partition_idx:02d}] Finished download: {size_mb:.1f} MB in {elapsed:.1f}s ({speed_mbps:.2f} MB/s)",
            flush=True,
        )
        return partition_idx, True, elapsed
    except Exception as exc:
        print(f"[P{partition_idx:02d}] Download failed: {exc}", flush=True)
        if dest.exists():
            dest.unlink()
        return partition_idx, False, 0.0


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    partitions = list(range(2, 12))  # Partitions 02 through 11

    print(
        f"Launching parallel downloads for {len(partitions)} partitions (max_workers=4)...",
        flush=True,
    )
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, idx): idx for idx in partitions}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[P{idx:02d}] Exception: {e}", flush=True)

    print(f"\nAll parallel downloads finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
