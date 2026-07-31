# ruff: noqa: E501
"""Diagnostic script for Data-Completeness Stratification (Step 4).

Within the e=2 day epoch, identifies subsets of objects with more complete early observations:
- More than 1 detection (diag_n_det_total > 1)
- More than 1 passband (diag_n_det_passbands > 1)
- Both conditions
Reports ROC-AUC and PR-AUC separately for these subsets and the full population.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aegis.config.features import FeatureConfig
from aegis.config.models import BaselineClassifierConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import (
    BaselineClassifier,
    representation_results_to_dataframe,
)

STUDY_CLASSES = [64, 90, 95]


def compute_bootstrap_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_class: int,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute point estimate and 95% bootstrap CIs for ROC-AUC and PR-AUC."""
    n_samples = len(y_true)
    y_binary = (y_true == target_class).astype(int)

    # Point estimates
    roc_auc_pt = float(roc_auc_score(y_binary, y_prob))
    pr_auc_pt = float(average_precision_score(y_binary, y_prob))

    rng = np.random.default_rng(seed)
    boot_roc_aucs = []
    boot_pr_aucs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_binary[idx]
        p_boot = y_prob[idx]

        # Guard against zero positive samples in rare resamples
        if len(np.unique(y_boot)) < 2:
            continue

        boot_roc_aucs.append(roc_auc_score(y_boot, p_boot))
        boot_pr_aucs.append(average_precision_score(y_boot, p_boot))

    roc_low, roc_high = np.percentile(boot_roc_aucs, [2.5, 97.5])
    pr_low, pr_high = np.percentile(boot_pr_aucs, [2.5, 97.5])

    return {
        "roc_auc": roc_auc_pt,
        "roc_auc_ci_low": float(roc_low),
        "roc_auc_ci_high": float(roc_high),
        "pr_auc": pr_auc_pt,
        "pr_auc_ci_low": float(pr_low),
        "pr_auc_ci_high": float(pr_high),
        "n_objects": n_samples,
        "n_targets": int(np.sum(y_binary)),
    }


def extract_features_for_object(args):
    obj_id, group, meta_row, epochs, feat_config = args
    res_dict = {}
    meta_row_copy = dict(meta_row)
    meta_row_copy["object_id"] = obj_id
    for epoch in epochs:
        trunc_obs = truncate_light_curve_at_epoch(
            group,
            days_since_first_detection=epoch,
            validate_schema=False,
        )
        res = extract_early_representation(
            df_obs=trunc_obs,
            meta_row=meta_row_copy,
            config=feat_config,
            epoch=epoch,
        )
        res_dict[epoch] = res
    return obj_id, res_dict


def main() -> None:
    t0 = time.time()
    print("=== AEGIS STEP 4: DATA-COMPLETENESS STRATIFICATION ===", flush=True)

    # 1. Load S=1 training data
    train_meta_path = Path("data/raw/plasticc_train_metadata.csv.gz")
    train_lc_path = Path("data/raw/plasticc_train_lightcurves.csv.gz")

    print("Loading S=1 Training Metadata & Light Curves...", flush=True)
    df_train_meta = pd.read_csv(train_meta_path)
    df_train_meta = df_train_meta[df_train_meta["target"].isin(STUDY_CLASSES)].copy()
    df_train_meta["true_target"] = df_train_meta["target"]
    df_train_meta["population"] = "BIASED"
    df_train_meta["S"] = 1

    train_ids = set(df_train_meta["object_id"])

    train_usecols = [
        "object_id",
        "mjd",
        "passband",
        "flux",
        "flux_err",
        "detected_bool",
    ]
    train_obs_list = []
    for chunk in pd.read_csv(train_lc_path, usecols=train_usecols, chunksize=500_000):
        sub = chunk[chunk["object_id"].isin(train_ids)]
        if not sub.empty:
            train_obs_list.append(sub)

    df_train_obs = pd.concat(train_obs_list, ignore_index=True)
    train_obs_by_obj = {
        obj_id: group for obj_id, group in df_train_obs.groupby("object_id")
    }

    # 2. Load Evaluation Metadata & Light curves
    true_meta_path = Path("data/processed/true_population.csv.gz")
    biased_meta_path = Path("data/processed/biased_population.csv.gz")
    expanded_lc_path = Path("data/processed/expanded_test_lightcurves.csv.gz")

    print("Loading Evaluation Metadata & Light Curves...", flush=True)
    df_true_meta = pd.read_csv(true_meta_path)
    df_biased_meta = pd.read_csv(biased_meta_path)
    biased_id_set = set(df_biased_meta["object_id"])

    df_study_meta_all = df_true_meta[
        df_true_meta["true_target"].isin(STUDY_CLASSES)
    ].copy()
    df_study_meta_all["S"] = (
        df_study_meta_all["object_id"].isin(biased_id_set).astype(int)
    )

    df_kn = df_study_meta_all[df_study_meta_all["true_target"] == 64]
    df_slsn = df_study_meta_all[df_study_meta_all["true_target"] == 95]
    df_snia = df_study_meta_all[df_study_meta_all["true_target"] == 90]

    df_snia_sub = df_snia.sample(n=20000, random_state=42)
    df_study_meta = pd.concat([df_kn, df_slsn, df_snia_sub], ignore_index=True)

    study_meta_dict = df_study_meta.set_index("object_id").to_dict(orient="index")

    usecols = ["object_id", "mjd", "passband", "flux", "flux_err"]
    head = pd.read_csv(expanded_lc_path, nrows=5)
    if "detected_bool" in head.columns:
        usecols.append("detected_bool")
    elif "detected" in head.columns:
        usecols.append("detected")

    print(f"Streaming light curves from {expanded_lc_path}...", flush=True)
    collected_lcs = []
    for chunk in pd.read_csv(expanded_lc_path, usecols=usecols, chunksize=5_000_000):
        if "detected" in chunk.columns and "detected_bool" not in chunk.columns:
            chunk = chunk.rename(columns={"detected": "detected_bool"})
        sub = chunk[chunk["object_id"].isin(study_meta_dict)]
        if not sub.empty:
            collected_lcs.append(sub)

    df_eval_obs = pd.concat(collected_lcs, ignore_index=True)
    eval_loaded_ids = set(df_eval_obs["object_id"].unique())

    df_eval_meta = (
        df_study_meta[df_study_meta["object_id"].isin(eval_loaded_ids)]
        .copy()
        .reset_index(drop=True)
    )

    feat_config = FeatureConfig()
    epoch = 2.0

    # Parallel evaluation feature extraction for epoch 2.0d
    tasks = []
    for obj_id_val, group in df_eval_obs.groupby("object_id"):
        obj_id = int(obj_id_val)
        meta_row = study_meta_dict[obj_id]
        tasks.append((obj_id, group, meta_row, [epoch], feat_config))

    print(
        f"Extracting evaluation features in parallel for {len(tasks):,} objects...",
        flush=True,
    )
    eval_feat_dicts = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(extract_features_for_object, tasks, chunksize=100)
        for obj_id, res_dict in results:
            eval_feat_dicts[obj_id] = res_dict[epoch]

    print(f"Feature extraction completed in {time.time() - t0:.1f}s.", flush=True)

    # Fit Baseline Classifier on training features at e=2.0d
    print("\nFitting Baseline Classifier on S=1 train features...", flush=True)
    train_records = df_train_meta.to_dict(orient="records")
    train_rep_results = []
    for row in train_records:
        obj_id = int(row["object_id"])
        raw_obs = train_obs_by_obj.get(obj_id, pd.DataFrame())
        trunc_obs = truncate_light_curve_at_epoch(
            raw_obs, days_since_first_detection=epoch, validate_schema=False
        )
        res = extract_early_representation(
            df_obs=trunc_obs, meta_row=row, config=feat_config, epoch=epoch
        )
        train_rep_results.append(res)

    df_feat_train = representation_results_to_dataframe(train_rep_results)
    y_train = df_train_meta["true_target"].to_numpy(dtype=int)

    feature_cols_all = [
        c for c in df_feat_train.columns if c not in ("object_id", "epoch")
    ]
    varying_cols = [
        c for c in feature_cols_all if df_feat_train[c].dropna().nunique() > 1
    ]

    X_train = df_feat_train[varying_cols]

    model_config = BaselineClassifierConfig(random_seed=42, min_samples_leaf=20)
    clf = BaselineClassifier(config=model_config)
    clf.fit_epoch(
        X_train,
        y_train,
        epoch=epoch,
        population_type="BIASED",
        meta_df=df_train_meta,
    )

    # Get predictions on evaluation set at e=2.0d
    eval_reps = [eval_feat_dicts[obj_id] for obj_id in df_eval_meta["object_id"]]
    df_feat_eval = representation_results_to_dataframe(eval_reps)
    X_eval = df_feat_eval[varying_cols]
    y_eval = df_eval_meta["true_target"].to_numpy(dtype=int)

    probs_eval = clf.predict_proba(X_eval, epoch=epoch)
    p_kn = probs_eval[:, 0]  # Class 64 index is 0

    # Define subsets
    n_det = df_feat_eval["diag_n_det_total"].to_numpy()
    n_pb = df_feat_eval["diag_n_det_passbands"].to_numpy()

    subsets = {
        "Full Population": np.ones(len(y_eval), dtype=bool),
        "More than 1 detection (N_det > 1)": n_det > 1,
        "More than 1 passband (N_pb > 1)": n_pb > 1,
        "Both (N_det > 1 and N_pb > 1)": (n_det > 1) & (n_pb > 1),
    }

    print("\n--- STRATIFICATION RESULTS AT e=2.0d ---", flush=True)
    for name, mask in subsets.items():
        sub_y = y_eval[mask]
        sub_p = p_kn[mask]
        n_sub = len(sub_y)
        n_kn_sub = int(np.sum(sub_y == 64))

        if n_kn_sub < 2 or (n_sub - n_kn_sub) < 2:
            print(
                f"\nSubset '{name}': insufficient sample size (N={n_sub}, KN={n_kn_sub}). Skipping.",
                flush=True,
            )
            continue

        res = compute_bootstrap_auc(
            sub_y, sub_p, target_class=64, n_bootstrap=1000, seed=42
        )
        print(f"\nSubset: {name}")
        print(f"  Total size (N): {n_sub:,}")
        print(f"  Kilonova count (N_KN): {n_kn_sub}")
        print(
            f"  ROC-AUC: {res['roc_auc']:.4f} [95% CI: {res['roc_auc_ci_low']:.4f}, {res['roc_auc_ci_high']:.4f}]"
        )
        print(
            f"  PR-AUC:  {res['pr_auc']:.4f} [95% CI: {res['pr_auc_ci_low']:.4f}, {res['pr_auc_ci_high']:.4f}]"
        )


if __name__ == "__main__":
    main()
