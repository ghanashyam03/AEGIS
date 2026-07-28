# ruff: noqa: E501
"""Validation script for Novelty / Distributional-Distance Signal (Step 3).

Evaluates:
1. Anomaly detection performance (ROC-AUC, PR-AUC with 1,000 bootstrap CIs)
   separating held-out PLAsTiCC Class 15 (TDEs, y=1) from in-distribution study
   classes (64, 90, 95, y=0) across decision epochs e in {0.0, 2.0, 7.0} days.
2. Response to synthetic extreme-value perturbations of in-distribution features.

Saves quantitative summary to docs/results/novelty_signal_metrics.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from aegis.config.features import FeatureConfig
from aegis.data.class15_ingest import (
    extract_class15_features_at_epoch,
    load_class15_metadata,
)
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import representation_results_to_dataframe
from aegis.models.novelty import (
    EPOCH_IDENTIFIABLE_FEATURES,
    compute_epoch_novelty_scores,
)

STUDY_CLASSES = [64, 90, 95]


def compute_bootstrap_anomaly_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute point estimate and 95% bootstrap CIs for anomaly detection AUCs."""
    n_samples = len(y_true)

    roc_auc_pt = float(roc_auc_score(y_true, scores))
    pr_auc_pt = float(average_precision_score(y_true, scores))

    rng = np.random.default_rng(seed)
    boot_roc_aucs = []
    boot_pr_aucs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[idx]
        s_boot = scores[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        boot_roc_aucs.append(roc_auc_score(y_boot, s_boot))
        boot_pr_aucs.append(average_precision_score(y_boot, s_boot))

    roc_low, roc_high = np.percentile(boot_roc_aucs, [2.5, 97.5])
    pr_low, pr_high = np.percentile(boot_pr_aucs, [2.5, 97.5])

    return {
        "roc_auc": roc_auc_pt,
        "roc_auc_ci_low": float(roc_low),
        "roc_auc_ci_high": float(roc_high),
        "pr_auc": pr_auc_pt,
        "pr_auc_ci_low": float(pr_low),
        "pr_auc_ci_high": float(pr_high),
        "n_in_dist": int(np.sum(y_true == 0)),
        "n_class15": int(np.sum(y_true == 1)),
    }


def main() -> None:
    t0 = time.time()
    print("=== AEGIS STEP 3: NOVELTY SIGNAL VALIDATION ===", flush=True)

    # 1. Load S=1 training metadata
    train_meta_path = Path("data/raw/plasticc_train_metadata.csv.gz")
    train_lc_path = Path("data/raw/plasticc_train_lightcurves.csv.gz")

    if not train_meta_path.exists() or not train_lc_path.exists():
        raise FileNotFoundError(f"Missing {train_meta_path} or {train_lc_path}")

    print("Loading S=1 Training Metadata & Light Curves...", flush=True)
    df_train_meta = pd.read_csv(train_meta_path)
    df_train_meta = df_train_meta[df_train_meta["target"].isin(STUDY_CLASSES)].copy()

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

    # 2. Load Evaluation Cohort (In-Distribution study classes 64, 90, 95)
    true_meta_path = Path("data/processed/true_population.csv.gz")
    biased_meta_path = Path("data/processed/biased_population.csv.gz")
    test_lc_path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz")

    print("Loading In-Distribution Evaluation Metadata...", flush=True)
    df_true_meta = pd.read_csv(true_meta_path)
    df_biased_meta = pd.read_csv(biased_meta_path)
    biased_id_set = set(df_biased_meta["object_id"])

    df_study_meta = df_true_meta[df_true_meta["true_target"].isin(STUDY_CLASSES)].copy()
    df_study_meta["S"] = df_study_meta["object_id"].isin(biased_id_set).astype(int)

    test_ids = set(df_study_meta["object_id"])
    usecols = ["object_id", "mjd", "passband", "flux", "flux_err"]
    head = pd.read_csv(test_lc_path, nrows=5)
    if "detected_bool" in head.columns:
        usecols.append("detected_bool")
    elif "detected" in head.columns:
        usecols.append("detected")

    test_obs_list = []
    for chunk in pd.read_csv(test_lc_path, usecols=usecols, chunksize=1_000_000):
        if "detected" in chunk.columns and "detected_bool" not in chunk.columns:
            chunk = chunk.rename(columns={"detected": "detected_bool"})
        sub = chunk[chunk["object_id"].isin(test_ids)]
        if not sub.empty:
            test_obs_list.append(sub)

    df_eval_obs = pd.concat(test_obs_list, ignore_index=True)
    eval_loaded_ids = set(df_eval_obs["object_id"].unique())

    df_eval_meta = (
        df_study_meta[df_study_meta["object_id"].isin(eval_loaded_ids)]
        .copy()
        .reset_index(drop=True)
    )
    eval_obs_by_obj = {
        obj_id: group for obj_id, group in df_eval_obs.groupby("object_id")
    }

    # 3. Load Held-Out Class 15 Metadata (TDEs)
    print("Loading Held-Out Class 15 (TDE) Metadata...", flush=True)
    df_c15_meta = load_class15_metadata(max_objects=2000)
    print(f"Class 15 metadata loaded: N = {len(df_c15_meta):,} objects.", flush=True)

    feat_config = FeatureConfig()
    epochs = [0.0, 2.0, 7.0]

    validation_results = {}
    synthetic_results = {}

    for epoch in epochs:
        print("\n=======================================================", flush=True)
        print(f"VALIDATING NOVELTY SIGNAL AT EPOCH e = {epoch:.1f} days", flush=True)
        print("=======================================================", flush=True)

        # Feature extraction for S=1 reference
        train_records = df_train_meta.to_dict(orient="records")
        train_rep = []
        for row in train_records:
            obj_id = int(row["object_id"])
            raw_obs = train_obs_by_obj.get(obj_id, pd.DataFrame())
            trunc_obs = truncate_light_curve_at_epoch(
                raw_obs, days_since_first_detection=epoch, validate_schema=False
            )
            res = extract_early_representation(
                df_obs=trunc_obs, meta_row=row, config=feat_config, epoch=epoch
            )
            train_rep.append(res)
        df_feat_s1 = representation_results_to_dataframe(train_rep)

        # Feature extraction for In-Distribution evaluation set
        eval_records = df_eval_meta.to_dict(orient="records")
        eval_rep = []
        for row in eval_records:
            obj_id = int(row["object_id"])
            raw_obs = eval_obs_by_obj.get(obj_id, pd.DataFrame())
            trunc_obs = truncate_light_curve_at_epoch(
                raw_obs, days_since_first_detection=epoch, validate_schema=False
            )
            res = extract_early_representation(
                df_obs=trunc_obs, meta_row=row, config=feat_config, epoch=epoch
            )
            eval_rep.append(res)
        df_feat_indist = representation_results_to_dataframe(eval_rep)

        # Feature extraction for Held-Out Class 15
        print(
            f"Extracting features for Class 15 (N={len(df_c15_meta):,})...", flush=True
        )
        df_feat_c15 = extract_class15_features_at_epoch(
            meta_c15=df_c15_meta,
            lc_path=test_lc_path,
            epoch=epoch,
            config=feat_config,
        )

        # Compute Novelty Scores
        scores_indist = compute_epoch_novelty_scores(
            df_s1_train=df_feat_s1, df_eval=df_feat_indist, epoch=epoch
        )
        scores_c15 = compute_epoch_novelty_scores(
            df_s1_train=df_feat_s1, df_eval=df_feat_c15, epoch=epoch
        )

        # Combine for anomaly evaluation (0 = In-Distribution, 1 = Class 15 Novelty)
        y_anomaly = np.concatenate(
            [
                np.zeros(len(scores_indist), dtype=int),
                np.ones(len(scores_c15), dtype=int),
            ]
        )
        scores_all = np.concatenate([scores_indist, scores_c15])

        res_auc = compute_bootstrap_anomaly_auc(
            y_true=y_anomaly, scores=scores_all, n_bootstrap=1000, seed=42
        )
        validation_results[epoch] = res_auc

        print(
            f"  [Class 15 Novelty Separation] ROC-AUC: {res_auc['roc_auc']:.4f} "
            f"[{res_auc['roc_auc_ci_low']:.4f}, {res_auc['roc_auc_ci_high']:.4f}] | "
            f"PR-AUC: {res_auc['pr_auc']:.4f} "
            f"[{res_auc['pr_auc_ci_low']:.4f}, {res_auc['pr_auc_ci_high']:.4f}]",
            flush=True,
        )

        # Secondary Check: Synthetic Extreme-Value Perturbations
        print("  Evaluating Synthetic Extreme-Value Perturbations...", flush=True)
        df_feat_synth = df_feat_indist.copy()
        # Shift hostgal_photoz by +5 stddev of S=1 photoz
        photoz_s1_std = float(df_feat_s1["hostgal_photoz"].std())
        photoz_s1_mean = float(df_feat_s1["hostgal_photoz"].mean())
        df_feat_synth["hostgal_photoz"] = (
            df_feat_synth["hostgal_photoz"] + 5.0 * photoz_s1_std
        )

        scores_synth = compute_epoch_novelty_scores(
            df_s1_train=df_feat_s1, df_eval=df_feat_synth, epoch=epoch
        )

        s1_p95 = float(np.percentile(scores_indist, 95))
        pct_exceeding_p95 = float(np.mean(scores_synth > s1_p95) * 100.0)

        synth_summary = {
            "indist_mean_score": float(np.mean(scores_indist)),
            "synth_mean_score": float(np.mean(scores_synth)),
            "s1_p95_threshold": s1_p95,
            "pct_synth_exceeding_p95": pct_exceeding_p95,
            "photoz_s1_mean": photoz_s1_mean,
            "photoz_s1_std": photoz_s1_std,
        }
        synthetic_results[epoch] = synth_summary
        print(
            f"  [Synthetic Perturbation] Mean Score (Normal: {synth_summary['indist_mean_score']:.4f} -> "
            f"Perturbed: {synth_summary['synth_mean_score']:.4f}) | "
            f"{pct_exceeding_p95:.1f}% exceed P95 threshold",
            flush=True,
        )

    # 4. Save JSON Metrics
    metrics_path = Path("docs/results/novelty_signal_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "metadata": {
            "n_s1_train": len(df_train_meta),
            "n_indist_eval": len(df_eval_meta),
            "n_class15_eval": len(df_c15_meta),
            "epochs": epochs,
            "identifiable_subspaces": EPOCH_IDENTIFIABLE_FEATURES,
        },
        "class15_validation": {str(ep): validation_results[ep] for ep in epochs},
        "synthetic_perturbations": {str(ep): synthetic_results[ep] for ep in epochs},
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"\nSaved novelty validation metrics to {metrics_path}", flush=True)

    print(f"Completed in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
