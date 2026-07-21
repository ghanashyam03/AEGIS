"""Cache TRUE and BIASED population data as raw binary NumPy arrays (.npy).

This enables instant memory-mapped loading (<0.01s), guaranteeing synchronous execution
within main workspace boundaries.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    data_dir = Path("data/processed")
    cache_dir = Path("data/interim/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    true_path = data_dir / "true_population.csv.gz"
    biased_path = data_dir / "biased_population.csv.gz"

    cols = [
        "object_id",
        "hostgal_photoz",
        "true_z",
        "distmod",
        "true_distmod",
        "hostgal_photoz_err",
        "mwebv",
        "ra",
        "decl",
        "tflux_r",
        "tflux_g",
        "true_target",
        "libid_cadence",
    ]

    print(f"Reading TRUE population from {true_path}...")
    df_true = pd.read_csv(true_path, usecols=cols)
    print(f"Reading BIASED population from {biased_path}...")
    df_biased = pd.read_csv(biased_path, usecols=cols)

    print(f"Saving binary .npy arrays to {cache_dir}...")
    for col in cols:
        np.save(cache_dir / f"true_{col}.npy", df_true[col].to_numpy())
        np.save(cache_dir / f"biased_{col}.npy", df_biased[col].to_numpy())

    print("Array caching complete!")


if __name__ == "__main__":
    main()
