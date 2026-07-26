"""Evaluate early representation constrainability across the TRUE population."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aegis.config.features import FeatureConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.features.representation import extract_early_representation

EMPTY_DF = pd.DataFrame()


def load_study_metadata() -> pd.DataFrame:
    """Load metadata for study classes 64 (KN), 90 (SN Ia), 95 (SLSN-I)."""
    true_path = Path("data/processed/true_population.csv.gz")
    if not true_path.exists():
        raise FileNotFoundError(f"TRUE population file not found at {true_path}")

    meta = pd.read_csv(
        true_path,
        usecols=["object_id", "true_target", "hostgal_photoz", "hostgal_photoz_err"],
    )
    return meta[meta["true_target"].isin([64, 90, 95])].copy()


def get_needed_columns(csv_path: Path) -> list[str]:
    """Inspect CSV header for required light-curve columns."""
    head = pd.read_csv(csv_path, nrows=2)
    cols = list(head.columns)
    needed = ["object_id", "mjd", "passband", "flux", "flux_err"]
    if "detected_bool" in cols:
        needed.append("detected_bool")
    elif "detected" in cols:
        needed.append("detected")
    return needed


def load_study_observations(lc_path: Path, study_ids: set[int]) -> pd.DataFrame:
    """Read CSV and collect observations for study_ids in a single DataFrame."""
    if not lc_path.exists():
        return pd.DataFrame()

    usecols = get_needed_columns(lc_path)
    chunks = []
    print(f"Reading observation data from {lc_path.name}...", flush=True)

    for chunk in pd.read_csv(lc_path, usecols=usecols, chunksize=1_000_000):
        if "detected" in chunk.columns and "detected_bool" not in chunk.columns:
            chunk = chunk.rename(columns={"detected": "detected_bool"})

        filtered = chunk[chunk["object_id"].isin(study_ids)]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def process_population(
    study_meta: pd.DataFrame,
    df_obs_all: pd.DataFrame,
    config: FeatureConfig,
) -> list[dict]:
    """Process light curves and extract early representation at e=0d and e=2d."""
    meta_dict = study_meta.set_index("object_id").to_dict(orient="index")
    records = []

    print("Truncating light curves at epochs e=0.0d and e=2.0d...", flush=True)
    truncated_epochs = {}
    for epoch in [0.0, 2.0]:
        truncated_epochs[epoch] = truncate_light_curve_at_epoch(
            df_obs_all,
            days_since_first_detection=epoch,
            detection_snr_threshold=config.detection_snr_threshold,
            strip_forbidden=True,
            validate_schema=False,
        )

    print("Extracting feature representations...", flush=True)
    for epoch in [0.0, 2.0]:
        trunc_df = truncated_epochs[epoch]
        obs_by_obj: dict[int, list[dict]] = {}
        if not trunc_df.empty:
            for r in trunc_df.to_dict(orient="records"):
                obs_by_obj.setdefault(int(r["object_id"]), []).append(r)

        for obj_id, obj_meta in meta_dict.items():
            cls_id = int(obj_meta.get("true_target", 0))
            rows = obs_by_obj.get(obj_id, None)
            group = pd.DataFrame(rows) if rows else EMPTY_DF

            rep = extract_early_representation(
                group,
                meta_row=obj_meta,
                config=config,
                epoch=epoch,
            )

            record = {
                "object_id": int(obj_id),
                "class_id": cls_id,
                "epoch": float(epoch),
                "n_obs": rep.summary_diagnostics["n_obs_total"],
                "n_det": rep.summary_diagnostics["n_det_total"],
            }

            for feat_name, feat_res in rep.features.items():
                record[f"{feat_name}_status"] = feat_res.status.value
                record[f"{feat_name}_val"] = feat_res.value

            records.append(record)

    return records


def compute_constrainability_summary(df_res: pd.DataFrame) -> list[dict]:
    """Compute per-class, per-epoch constrainability fractions."""
    summaries = []
    class_names = {
        64: "Kilonova (KN)",
        90: "Type Ia Supernova (SN Ia)",
        95: "Superluminous SN (SLSN-I)",
    }

    feature_keys = (
        ["alert_snr_0", "snr_growth_rate", "hostgal_photoz"]
        + [f"rise_rate_pb{b}" for b in range(6)]
        + [
            f"color_pb{b1}_pb{b2}"
            for b1, b2 in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        ]
    )

    for cls_id in [64, 90, 95]:
        for epoch in [0.0, 2.0]:
            sub = df_res[(df_res["class_id"] == cls_id) & (df_res["epoch"] == epoch)]
            n_total = len(sub)

            feat_stats = {}
            for k in feature_keys:
                status_col = f"{k}_status"
                if status_col in sub.columns and n_total > 0:
                    constrained_cnt = int((sub[status_col] == "well_constrained").sum())
                    pct_constrained = float(constrained_cnt / n_total * 100)
                else:
                    constrained_cnt = 0
                    pct_constrained = 0.0

                feat_stats[k] = {
                    "well_constrained_count": constrained_cnt,
                    "unconstrained_count": n_total - constrained_cnt,
                    "pct_constrained": pct_constrained,
                    "pct_unconstrained": float(100.0 - pct_constrained),
                }

            summaries.append(
                {
                    "class_id": cls_id,
                    "class_name": class_names[cls_id],
                    "epoch": epoch,
                    "n_evaluated_objects": n_total,
                    "feature_constrainability": feat_stats,
                }
            )

    return summaries


def main() -> None:
    print("=== Real TRUE Population Feature Constrainability Benchmark ===", flush=True)
    config = FeatureConfig()
    study_meta = load_study_metadata()
    study_ids = set(study_meta["object_id"])

    obs_dfs = []

    # Read observations for study objects from test chunk 01 and train files
    test_01_path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz")
    if test_01_path.exists():
        df_test = load_study_observations(test_01_path, study_ids)
        if not df_test.empty:
            obs_dfs.append(df_test)

    train_lc_path = Path("data/raw/plasticc_train_lightcurves.csv.gz")
    if train_lc_path.exists():
        df_train = load_study_observations(train_lc_path, study_ids)
        if not df_train.empty:
            obs_dfs.append(df_train)

    if not obs_dfs:
        raise ValueError("No observation data loaded for study objects.")

    df_obs_all = pd.concat(obs_dfs, ignore_index=True)
    records = process_population(study_meta, df_obs_all, config)

    df_res = pd.DataFrame(records)
    summaries = compute_constrainability_summary(df_res)

    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "representation_constrainability.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"Constrainability summary saved to {json_path}", flush=True)


if __name__ == "__main__":
    main()
