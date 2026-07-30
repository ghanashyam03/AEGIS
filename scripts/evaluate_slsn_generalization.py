# ruff: noqa: E501
"""SLSN-I (Class 95) Generalization Diagnostic Script.

Evaluates whether the decision policy behavior found for kilonovae replicates
for Superluminous Supernovae Type I (SLSN-I, class 95) under prespecified
operational capacity K=5 and primary deadline H=2.0 days.

Uses existing baseline classifier (class 95 output prob), novelty detector, and
utility functional form (target=+2, non-target=-1, no-trigger=0) unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy.stats import beta

from aegis.config.decision import DecisionPolicyConfig
from aegis.config.features import FeatureConfig
from aegis.config.models import BaselineClassifierConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.decision.policy import SequentialDecisionPolicy
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import (
    BaselineClassifier,
    representation_results_to_dataframe,
)
from aegis.models.novelty import (
    compute_epoch_novelty_scores,
)

STUDY_CLASSES = [64, 90, 95]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return super().default(obj)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute exact Clopper-Pearson 95% binomial confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (low, high)


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


def run_slsn_evaluation() -> dict[str, Any]:
    t0 = time.time()
    print("=== AEGIS SLSN-I (CLASS 95) GENERALIZATION DIAGNOSTIC ===", flush=True)

    # 1. Load S=1 training data
    train_meta_path = Path("data/raw/plasticc_train_metadata.csv.gz")
    train_lc_path = Path("data/raw/plasticc_train_lightcurves.csv.gz")

    print("Loading S=1 Training Metadata & Light Curves...", flush=True)
    df_train_meta = pd.read_csv(train_meta_path)
    df_train_lc = pd.read_csv(train_lc_path)

    # Restrict training set to study classes: 64, 90, 95
    df_train_meta = df_train_meta[df_train_meta["target"].isin(STUDY_CLASSES)].copy()
    if "true_target" not in df_train_meta.columns:
        df_train_meta["true_target"] = df_train_meta["target"]
    train_ids = set(df_train_meta["object_id"])
    df_train_lc = df_train_lc[df_train_lc["object_id"].isin(train_ids)].copy()
    train_obs_by_obj = {
        obj_id: group for obj_id, group in df_train_lc.groupby("object_id")
    }

    # 2. Load Evaluation Metadata
    eval_meta_path = Path("data/processed/true_population.csv.gz")
    expanded_lc_path = Path("data/processed/expanded_test_lightcurves.csv.gz")
    eval_lc_path = (
        expanded_lc_path
        if expanded_lc_path.exists()
        else Path("data/raw/plasticc_test_lightcurves_01.csv.gz")
    )

    print(f"Loading Evaluation Metadata from {eval_meta_path}...", flush=True)
    df_true_meta = pd.read_csv(eval_meta_path)
    df_study_meta_all = df_true_meta[
        df_true_meta["true_target"].isin(STUDY_CLASSES)
    ].copy()

    # Large reproducible subsample of background objects (SN Ia, class 90) to ensure tractable feature extraction
    df_kn = df_study_meta_all[df_study_meta_all["true_target"] == 64]
    df_slsn = df_study_meta_all[df_study_meta_all["true_target"] == 95]
    df_snia = df_study_meta_all[df_study_meta_all["true_target"] == 90]

    df_snia_sub = df_snia.sample(n=20000, random_state=42)
    df_study_meta = pd.concat([df_kn, df_slsn, df_snia_sub], ignore_index=True)

    study_meta_dict = df_study_meta.set_index("object_id").to_dict(orient="index")

    feat_config = FeatureConfig()
    epochs = [0.0, 2.0, 5.0, 10.0]

    # 3. Process S=1 Training Features & Fit Classifiers
    print("\n--- Fitting Baseline Classifiers on S=1 Training Set ---", flush=True)
    train_preds: dict[float, np.ndarray] = {}
    train_feats_by_epoch: dict[float, pd.DataFrame] = {}
    classifiers: dict[float, BaselineClassifier] = {}
    varying_cols_by_epoch: dict[float, list[str]] = {}

    for epoch in epochs:
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
        train_feats_by_epoch[epoch] = df_feat_train

        y_train = df_train_meta["true_target"].to_numpy(dtype=int)
        feature_cols_all = [
            c for c in df_feat_train.columns if c not in ("object_id", "epoch")
        ]
        varying_cols = [
            c
            for c in df_feat_train.columns
            if c in feature_cols_all and df_feat_train[c].dropna().nunique() > 1
        ]
        varying_cols_by_epoch[epoch] = varying_cols

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
        classifiers[epoch] = clf

        probs_train = clf.predict_proba(X_train, epoch=epoch)
        train_preds[epoch] = probs_train[:, 2]  # P(SLSN-I)

    # 4. Stream Evaluation Light Curves & Extract Features for Epochs [0.0, 2.0, 5.0, 10.0]
    print(
        f"\n--- Streaming & Extracting Features from {eval_lc_path.name} ---",
        flush=True,
    )
    import concurrent.futures

    usecols = ["object_id", "mjd", "passband", "flux", "flux_err"]
    head = pd.read_csv(eval_lc_path, nrows=5)
    if "detected_bool" in head.columns:
        usecols.append("detected_bool")
    elif "detected" in head.columns:
        usecols.append("detected")

    t_stream_start = time.time()
    collected_lcs = []
    print(
        "Streaming and collecting light curve rows for selected study objects...",
        flush=True,
    )
    for chunk in pd.read_csv(eval_lc_path, usecols=usecols, chunksize=5_000_000):
        if "detected" in chunk.columns and "detected_bool" not in chunk.columns:
            chunk = chunk.rename(columns={"detected": "detected_bool"})
        sub_chunk = chunk[chunk["object_id"].isin(study_meta_dict)]
        if not sub_chunk.empty:
            collected_lcs.append(sub_chunk)
    df_all_lc = pd.concat(collected_lcs, ignore_index=True)
    print(
        f"Loaded {len(df_all_lc):,} rows for {df_all_lc['object_id'].nunique():,} objects ({time.time() - t_stream_start:.1f}s).",
        flush=True,
    )

    tasks = []
    for obj_id_val, group in df_all_lc.groupby("object_id"):
        obj_id = int(obj_id_val)
        meta_row = study_meta_dict[obj_id]
        tasks.append((obj_id, group, meta_row, epochs, feat_config))

    print(
        f"Extracting features in parallel using ProcessPoolExecutor over {len(tasks):,} objects...",
        flush=True,
    )
    eval_feat_dicts = {e: {} for e in epochs}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(extract_features_for_object, tasks, chunksize=100)
        for obj_id, res_dict in results:
            for epoch in epochs:
                eval_feat_dicts[epoch][obj_id] = res_dict[epoch]

    print(
        f"Completed feature extraction in {time.time() - t_stream_start:.2f}s!",
        flush=True,
    )

    eval_loaded_ids = set(eval_feat_dicts[0.0].keys())
    df_eval_meta = (
        df_study_meta[df_study_meta["object_id"].isin(eval_loaded_ids)]
        .copy()
        .reset_index(drop=True)
    )

    n_total_eval = len(df_eval_meta)
    y_true_eval = df_eval_meta["true_target"].to_numpy(dtype=int)
    slsn_mask = y_true_eval == 95
    n_slsn_eval = int(np.sum(slsn_mask))
    n_non_slsn_eval = n_total_eval - n_slsn_eval

    print("\n=== SLSN-I EVALUATION COHORT SUMMARY ===", flush=True)
    print(f"Total Evaluation Objects (N): {n_total_eval:,}", flush=True)
    print(f"SLSN-I (Class 95) Count: {n_slsn_eval:,}", flush=True)

    # 5. Predict probabilities & compute novelty scores per epoch
    eval_preds: dict[float, np.ndarray] = {}
    eval_novelties: dict[float, np.ndarray] = {}
    novelty_scales: dict[float, float] = {}

    for epoch in epochs:
        eval_reps = [
            eval_feat_dicts[epoch][obj_id] for obj_id in df_eval_meta["object_id"]
        ]
        df_feat_eval = representation_results_to_dataframe(eval_reps)
        varying_cols = varying_cols_by_epoch[epoch]
        X_eval = df_feat_eval[varying_cols]

        clf = classifiers[epoch]
        probs_eval = clf.predict_proba(X_eval, epoch=epoch)
        eval_preds[epoch] = probs_eval[:, 2]  # P(SLSN-I)

        df_feat_train = train_feats_by_epoch[epoch]
        nov_train = compute_epoch_novelty_scores(
            df_s1_train=df_feat_train,
            df_eval=df_feat_train,
            epoch=epoch,
        )
        scale = float(np.std(nov_train))
        novelty_scales[epoch] = scale if scale > 0 else 1.0

        nov_eval = compute_epoch_novelty_scores(
            df_s1_train=df_feat_train,
            df_eval=df_feat_eval,
            epoch=epoch,
        )
        eval_novelties[epoch] = nov_eval

    # 6. Policy Configurations
    capacity = 5
    policy_configs = {
        "naive_baseline": DecisionPolicyConfig(
            novelty_weight=0.0, decision_threshold=0.001, capacity_per_epoch=capacity
        ),
        "fused_policy": DecisionPolicyConfig(
            novelty_weight=0.05, decision_threshold=0.001, capacity_per_epoch=capacity
        ),
        "novelty_ablation": DecisionPolicyConfig(
            novelty_weight=0.0, decision_threshold=0.001, capacity_per_epoch=capacity
        ),
    }

    results: dict[str, Any] = {
        "diagnostic_target": "SLSN-I (class 95)",
        "evaluation_cohort": {
            "total_objects": n_total_eval,
            "slsn_count": n_slsn_eval,
            "kn_count": int(np.sum(y_true_eval == 64)),
            "sn_ia_count": int(np.sum(y_true_eval == 90)),
        },
        "epochs_evaluated": epochs,
        "policies": {},
    }

    # 7. Evaluate Policies for SLSN-I Target
    for pol_key, pcfg in policy_configs.items():
        pol = SequentialDecisionPolicy(config=pcfg)
        epoch_preds_dict = {}
        for epoch in epochs:
            epoch_preds_dict[epoch] = (eval_preds[epoch], eval_novelties[epoch])

        trace = pol.evaluate_sequential_trace(
            epoch_predictions=epoch_preds_dict,
            y_true=y_true_eval,
            object_ids=df_eval_meta["object_id"].to_numpy(dtype=int),
            novelty_scales=novelty_scales,
        )

        actions = trace["cumulative_actions"]
        trigger_epochs = trace["trigger_epochs"]

        slsn_trig_count = int(np.sum((actions == 1) & slsn_mask))
        slsn_trig_rate = (
            float(slsn_trig_count / n_slsn_eval) if n_slsn_eval > 0 else 0.0
        )
        slsn_trig_ci = clopper_pearson_ci(slsn_trig_count, n_slsn_eval)

        # Triggers at primary deadline H = 2.0d
        trig_h2 = int(
            np.sum((actions == 1) & slsn_mask & (np.array(trigger_epochs) <= 2.0))
        )
        trig_h2_rate = float(trig_h2 / n_slsn_eval) if n_slsn_eval > 0 else 0.0
        trig_h2_ci = clopper_pearson_ci(trig_h2, n_slsn_eval)

        fp_count = int(np.sum((actions == 1) & (~slsn_mask)))
        fp_rate = float(fp_count / n_non_slsn_eval) if n_non_slsn_eval > 0 else 0.0
        fp_ci = clopper_pearson_ci(fp_count, n_non_slsn_eval)

        results["policies"][pol_key] = {
            "name": pol_key,
            "w_nov": pcfg.novelty_weight,
            "tau": pcfg.decision_threshold,
            "capacity": pcfg.capacity_per_epoch,
            "slsn_triggered_total": slsn_trig_count,
            "slsn_trigger_rate_total": slsn_trig_rate,
            "slsn_trigger_rate_total_cp_ci_95": list(slsn_trig_ci),
            "slsn_triggered_by_h2": trig_h2,
            "slsn_trigger_rate_by_h2": trig_h2_rate,
            "slsn_trigger_rate_by_h2_cp_ci_95": list(trig_h2_ci),
            "non_slsn_triggers": fp_count,
            "false_trigger_rate": fp_rate,
            "false_trigger_rate_cp_ci_95": list(fp_ci),
            "utility_regret": trace["regret"]["regret"],
            "normalized_regret": trace["regret"]["normalized_regret"],
            "policy_utility": trace["regret"]["u_policy"],
        }

        print(f"\n--- Arm: {pol_key} ---", flush=True)
        print(
            f"  SLSN-I Triggers (Total): {slsn_trig_count}/{n_slsn_eval} ({slsn_trig_rate * 100:.2f}%, 95% CP CI: [{slsn_trig_ci[0] * 100:.2f}%, {slsn_trig_ci[1] * 100:.2f}%])",
            flush=True,
        )
        print(
            f"  SLSN-I Triggers (by H=2.0d): {trig_h2}/{n_slsn_eval} ({trig_h2_rate * 100:.2f}%, 95% CP CI: [{trig_h2_ci[0] * 100:.2f}%, {trig_h2_ci[1] * 100:.2f}%])",
            flush=True,
        )
        print(
            f"  False Triggers (Non-SLSN): {fp_count}/{n_non_slsn_eval} (FTR: {fp_rate * 100:.4f}%)",
            flush=True,
        )

    out_metrics_path = Path("docs/results/slsn_generalization_metrics.json")
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(
        f"\nSaved SLSN generalization metrics to {out_metrics_path} in {time.time() - t0:.2f}s!",
        flush=True,
    )
    return results


if __name__ == "__main__":
    run_slsn_evaluation()
