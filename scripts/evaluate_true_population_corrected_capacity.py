# ruff: noqa: E501
"""AEGIS Headline Kilonova Performance Re-Measurement under Rate-Corrected Capacity (K=22).

Evaluates the 3 policy arms (Naive Baseline, Frozen Policy, Novelty Ablation) with K=22 on the expanded population.
Saves results to docs/results/headline_evaluation_metrics_v2.json.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from scipy.stats import beta

from aegis.config.decision import DecisionPolicyConfig
from aegis.config.features import FeatureConfig
from aegis.config.models import BaselineClassifierConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.decision.policy import (
    SequentialDecisionPolicy,
    compute_utility_regret,
)
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import (
    BaselineClassifier,
    representation_results_to_dataframe,
)
from aegis.models.novelty import (
    compute_epoch_novelty_scores,
)

STUDY_CLASSES = [64, 90, 95]


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute exact two-sided Clopper-Pearson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    lower = max(0.0, lower)
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    upper = min(1.0, upper)
    return (lower, upper)


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
    t_start = time.time()
    print(
        "=== AEGIS STEP 1: EVALUATION UNDER CORRECTED CAPACITY (K=22) ===", flush=True
    )

    # 1. Load S=1 Training Data
    train_meta_path = Path("data/raw/plasticc_train_metadata.csv.gz")
    train_lc_path = Path("data/raw/plasticc_train_lightcurves.csv.gz")

    print("Loading S=1 Training Metadata & Light Curves...", flush=True)
    df_train_meta = pd.read_csv(train_meta_path)
    if "target" in df_train_meta.columns and "true_target" not in df_train_meta.columns:
        df_train_meta["true_target"] = df_train_meta["target"]
    df_train_meta = df_train_meta[
        df_train_meta["true_target"].isin(STUDY_CLASSES)
    ].copy()
    df_train_lc = pd.read_csv(train_lc_path)
    train_ids = set(df_train_meta["object_id"])
    df_train_lc = df_train_lc[df_train_lc["object_id"].isin(train_ids)].copy()

    train_obs_by_obj = {
        obj_id: group for obj_id, group in df_train_lc.groupby("object_id")
    }

    # 2. Load Evaluation Metadata
    true_meta_path = Path("data/processed/true_population.csv.gz")
    biased_meta_path = Path("data/processed/biased_population.csv.gz")
    expanded_lc_path = Path("data/processed/expanded_test_lightcurves.csv.gz")

    print(f"Loading Evaluation Metadata from {true_meta_path}...", flush=True)
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

    feat_config = FeatureConfig()
    epochs = [0.0, 2.0]

    # 3. Process S=1 Training Features & Fit Classifiers per Epoch
    print("\n--- Fitting Baseline Classifiers on S=1 Training Set ---", flush=True)
    train_preds = {}
    train_feats_by_epoch = {}
    classifiers = {}
    varying_cols_by_epoch = {}

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
        train_preds[epoch] = probs_train[:, 0]

    # 4. Stream Evaluation Light Curves & Extract Features for Epochs [0.0, 2.0]
    print(
        f"\n--- Streaming & Extracting Features from {expanded_lc_path.name} ---",
        flush=True,
    )
    usecols = ["object_id", "mjd", "passband", "flux", "flux_err"]
    head = pd.read_csv(expanded_lc_path, nrows=5)
    if "detected_bool" in head.columns:
        usecols.append("detected_bool")
    elif "detected" in head.columns:
        usecols.append("detected")

    collected_lcs = []
    for chunk in pd.read_csv(expanded_lc_path, usecols=usecols, chunksize=5_000_000):
        if "detected" in chunk.columns and "detected_bool" not in chunk.columns:
            chunk = chunk.rename(columns={"detected": "detected_bool"})
        sub_chunk = chunk[chunk["object_id"].isin(study_meta_dict)]
        if not sub_chunk.empty:
            collected_lcs.append(sub_chunk)
    df_all_lc = pd.concat(collected_lcs, ignore_index=True)

    tasks = []
    for obj_id_val, group in df_all_lc.groupby("object_id"):
        obj_id = int(obj_id_val)
        meta_row = study_meta_dict[obj_id]
        tasks.append((obj_id, group, meta_row, epochs, feat_config))

    eval_feat_dicts = {e: {} for e in epochs}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(extract_features_for_object, tasks, chunksize=100)
        for obj_id, res_dict in results:
            for epoch in epochs:
                eval_feat_dicts[epoch][obj_id] = res_dict[epoch]

    eval_loaded_ids = set(eval_feat_dicts[0.0].keys())
    df_eval_meta = (
        df_study_meta[df_study_meta["object_id"].isin(eval_loaded_ids)]
        .copy()
        .reset_index(drop=True)
    )

    n_total_eval = len(df_eval_meta)
    y_true_eval = df_eval_meta["true_target"].to_numpy(dtype=int)
    n_kn_eval = int(np.sum(y_true_eval == 64))
    n_non_kn_eval = n_total_eval - n_kn_eval

    # 5. Predict probabilities & compute novelty scores per epoch
    eval_preds = {}
    eval_novelties = {}
    novelty_scales = {}

    for epoch in epochs:
        eval_reps = [
            eval_feat_dicts[epoch][obj_id] for obj_id in df_eval_meta["object_id"]
        ]
        df_feat_eval = representation_results_to_dataframe(eval_reps)
        varying_cols = varying_cols_by_epoch[epoch]
        X_eval = df_feat_eval[varying_cols]

        clf = classifiers[epoch]
        probs_eval = clf.predict_proba(X_eval, epoch=epoch)
        eval_preds[epoch] = probs_eval[:, 0]

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

    # 6. Calibrate Naive Threshold on S=1 for K=22
    capacity = 22
    s1_eval_ratio = float(capacity) / len(df_eval_meta)
    s1_quota = max(1, int(round(len(df_train_meta) * s1_eval_ratio)))

    tau_naive_by_epoch = {}
    for epoch in epochs:
        p_tr = np.sort(train_preds[epoch])[::-1]
        tau_naive_by_epoch[epoch] = float(p_tr[s1_quota - 1])

    tau_naive_global = max(tau_naive_by_epoch.values())
    print(
        f"\nCalibrated Naive Threshold tau_naive = {tau_naive_global:.6f} (S=1 quota: {s1_quota})",
        flush=True,
    )

    # 7. Run Decision Policies
    np.where(y_true_eval == 64)[0]
    obj_ids_eval = df_eval_meta["object_id"].to_numpy(dtype=int)

    policy_runs = {
        "naive_baseline": {
            "name": "Naive Fixed-Confidence Baseline (K=22)",
            "w_nov": 0.0,
            "tau": tau_naive_global,
            "capacity": capacity,
        },
        "fused_policy": {
            "name": "Frozen Bias-and-Novelty Policy (v2, K=22)",
            "w_nov": 0.05,
            "tau": 0.001,
            "capacity": capacity,
        },
        "novelty_ablation": {
            "name": "Novelty Ablation (w_nov = 0.00, K=22)",
            "w_nov": 0.0,
            "tau": 0.001,
            "capacity": capacity,
        },
    }

    metrics_results = {
        "evaluation_cohort": {
            "total_objects": n_total_eval,
            "kilonovae_count": n_kn_eval,
            "slsn_i_count": int(np.sum(y_true_eval == 95)),
            "sn_ia_count": int(np.sum(y_true_eval == 90)),
            "population_type": "EXPANDED_TRUE_POPULATION",
            "is_expanded_population": True,
        },
        "naive_threshold_calibrated": tau_naive_global,
        "policies": {},
    }

    rng = np.random.default_rng(42)
    B_BOOTSTRAP = 1000

    for pol_key, pol_cfg in policy_runs.items():
        w_nov = pol_cfg["w_nov"]
        tau = pol_cfg["tau"]

        pcfg = DecisionPolicyConfig(
            novelty_weight=w_nov,
            decision_threshold=tau,
            capacity_per_epoch=capacity,
        )
        pol = SequentialDecisionPolicy(config=pcfg)
        epoch_preds_dict = {}
        for epoch in epochs:
            epoch_preds_dict[epoch] = (eval_preds[epoch], eval_novelties[epoch])

        trace = pol.evaluate_sequential_trace(
            epoch_predictions=epoch_preds_dict,
            y_true=y_true_eval,
            object_ids=obj_ids_eval,
            novelty_scales=novelty_scales,
        )

        actions = trace["cumulative_actions"]
        trace["trigger_epochs"]
        regret = trace["regret"]
        mhver = trace["mhver"]

        fp_mask = (actions == 1) & (y_true_eval != 64)
        fp_count = int(np.sum(fp_mask))
        ftr = float(fp_count / n_non_kn_eval)
        ftr_ci = clopper_pearson_ci(fp_count, n_non_kn_eval)

        missed_count = int(round(mhver * n_kn_eval))
        triggered_kn_count = n_kn_eval - missed_count
        mhver_ci = clopper_pearson_ci(missed_count, n_kn_eval)
        target_trigger_rate = triggered_kn_count / n_kn_eval
        target_trigger_ci = clopper_pearson_ci(triggered_kn_count, n_kn_eval)

        # Bootstrap CIs for regret and utility
        boot_regrets = []
        boot_utilities = []
        for _ in range(B_BOOTSTRAP):
            boot_idx = rng.choice(n_total_eval, size=n_total_eval, replace=True)
            boot_y = y_true_eval[boot_idx]
            boot_act = actions[boot_idx]

            boot_reg_dict = compute_utility_regret(
                policy_actions=boot_act,
                y_true=boot_y,
                capacity=capacity,
                target_class=64,
                u_tp=2.0,
                u_fp=-1.0,
            )
            boot_regrets.append(boot_reg_dict["regret"])
            boot_utilities.append(boot_reg_dict["u_policy"])

        regret_ci_95 = (
            float(np.percentile(boot_regrets, 2.5)),
            float(np.percentile(boot_regrets, 97.5)),
        )
        utility_ci_95 = (
            float(np.percentile(boot_utilities, 2.5)),
            float(np.percentile(boot_utilities, 97.5)),
        )

        metrics_results["policies"][pol_key] = {
            "name": pol_cfg["name"],
            "parameters": {"w_nov": w_nov, "tau": tau, "capacity": capacity},
            "performance": {
                "triggered_kilonovae": triggered_kn_count,
                "missed_kilonovae": missed_count,
                "target_trigger_rate": target_trigger_rate,
                "target_trigger_rate_cp_ci_95": list(target_trigger_ci),
                "mhver": mhver,
                "mhver_cp_ci_95": list(mhver_ci),
                "false_positive_triggers": fp_count,
                "false_trigger_rate": ftr,
                "false_trigger_rate_cp_ci_95": list(ftr_ci),
                "utility_regret": regret["regret"],
                "utility_regret_bootstrap_ci_95": list(regret_ci_95),
                "normalized_regret": regret["normalized_regret"],
                "policy_utility": regret["u_policy"],
                "policy_utility_bootstrap_ci_95": list(utility_ci_95),
                "oracle_utility": regret["u_oracle"],
            },
        }

        print(f"\n--- Arm: {pol_cfg['name']} ---", flush=True)
        print(
            f"  Target Triggers (KN): {triggered_kn_count}/{n_kn_eval} ({target_trigger_rate * 100:.1f}%)",
            flush=True,
        )
        print(
            f"  MHVER (Miss Rate): {mhver * 100:.1f}% (95% CP CI: [{mhver_ci[0] * 100:.1f}%, {mhver_ci[1] * 100:.1f}%])",
            flush=True,
        )
        print(
            f"  False Positives (Non-KN): {fp_count}/{n_non_kn_eval} (FTR: {ftr * 100:.4f}%)",
            flush=True,
        )
        print(
            f"  Utility Regret: {regret['regret']:.2f} (95% Boot CI: [{regret_ci_95[0]:.2f}, {regret_ci_95[1]:.2f}])",
            flush=True,
        )
        print(
            f"  Policy Utility: {regret['u_policy']:.2f} / Oracle: {regret['u_oracle']:.2f}",
            flush=True,
        )

    out_metrics_path = Path("docs/results/headline_evaluation_metrics_v2.json")
    with open(out_metrics_path, "w") as f:
        json.dump(metrics_results, f, indent=2)

    print(
        f"\nSaved updated metrics to {out_metrics_path} in {time.time() - t_start:.2f}s!",
        flush=True,
    )


if __name__ == "__main__":
    main()
