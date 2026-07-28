# ruff: noqa: E501
"""Headline Comparative Evaluation on TRUE Population (Step 2 & Step 3).

Evaluates three decision policy configurations under identical prespecified capacity K=5
and primary decision deadline H=2.0 days, adhering to ADR 008 small-sample methodology:

1. Naive fixed-confidence-threshold baseline (calibrated on S=1, no novelty term).
2. Frozen policy (configs/decision_policy_v1.yaml locked: w_nov=0.05, tau=0.001, K=5).
3. Novelty ablation (same frozen policy with w_nov forced to 0.00).

Outputs quantitative metrics, LOKO jackknife ranges, and per-object itemization.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

from aegis.config.decision import DecisionPolicyConfig
from aegis.config.features import FeatureConfig
from aegis.config.models import BaselineClassifierConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.decision.policy import SequentialDecisionPolicy
from aegis.decision.utility import (
    compute_missed_high_value_event_rate,
    compute_utility_regret,
)
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import (
    BaselineClassifier,
    representation_results_to_dataframe,
)
from aegis.models.novelty import (
    EPOCH_IDENTIFIABLE_FEATURES,
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


def run_evaluation() -> dict[
    str, float | dict[str, float | str | list[float]] | list[dict[str, float | str]]
]:
    t0 = time.time()
    print("=== AEGIS STEP 2: HEADLINE COMPARATIVE EVALUATION ===", flush=True)

    # 1. Load S=1 training data
    train_meta_path = Path("data/raw/plasticc_train_metadata.csv.gz")
    train_lc_path = Path("data/raw/plasticc_train_lightcurves.csv.gz")

    print("Loading S=1 Training Metadata & Light Curves...", flush=True)
    df_train_meta = pd.read_csv(train_meta_path)
    df_train_meta = (
        df_train_meta[df_train_meta["target"].isin(STUDY_CLASSES)]
        .copy()
        .reset_index(drop=True)
    )
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
    train_obs_by_obj: dict[int, pd.DataFrame] = {
        obj_id: group for obj_id, group in df_train_obs.groupby("object_id")
    }

    # 2. Load Evaluation Cohort
    true_meta_path = Path("data/processed/true_population.csv.gz")
    biased_meta_path = Path("data/processed/biased_population.csv.gz")
    test_lc_path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz")

    print("Loading Evaluation Metadata & Light Curves...", flush=True)
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

    eval_obs_by_obj: dict[int, pd.DataFrame] = {
        obj_id: group for obj_id, group in df_eval_obs.groupby("object_id")
    }

    feat_config = FeatureConfig()
    epochs = [0.0, 2.0]

    train_preds: dict[float, np.ndarray] = {}
    eval_preds: dict[float, np.ndarray] = {}
    eval_novelties: dict[float, np.ndarray] = {}
    novelty_scales: dict[float, float] = {}

    for epoch in epochs:
        print(f"\nProcessing Epoch e = {epoch:.1f} days...", flush=True)

        # Train representations
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
            c
            for c in df_feat_train.columns
            if c in feature_cols_all and df_feat_train[c].dropna().nunique() > 1
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

        probs_train = clf.predict_proba(X_train, epoch=epoch)
        train_preds[epoch] = probs_train[:, 0]

        # Eval representations
        eval_records = df_eval_meta.to_dict(orient="records")
        eval_rep_results = []
        for row in eval_records:
            obj_id = int(row["object_id"])
            raw_obs = eval_obs_by_obj.get(obj_id, pd.DataFrame())
            trunc_obs = truncate_light_curve_at_epoch(
                raw_obs, days_since_first_detection=epoch, validate_schema=False
            )
            res = extract_early_representation(
                df_obs=trunc_obs, meta_row=row, config=feat_config, epoch=epoch
            )
            eval_rep_results.append(res)

        df_feat_eval = representation_results_to_dataframe(eval_rep_results)
        X_eval = df_feat_eval[varying_cols]

        probs_eval = clf.predict_proba(X_eval, epoch=epoch)
        eval_preds[epoch] = probs_eval[:, 0]

        # Novelty scores
        ident_cols = EPOCH_IDENTIFIABLE_FEATURES[epoch]
        nov_train = compute_epoch_novelty_scores(
            df_feat=df_feat_train,
            df_ref_s1=df_feat_train,
            identifiable_cols=ident_cols,
        )
        scale = float(np.std(nov_train))
        novelty_scales[epoch] = scale if scale > 0 else 1.0

        nov_eval = compute_epoch_novelty_scores(
            df_feat=df_feat_eval,
            df_ref_s1=df_feat_train,
            identifiable_cols=ident_cols,
        )
        eval_novelties[epoch] = nov_eval

    # Calibrate Naive Threshold on S=1 without peeking at TRUE labels
    # Matching target trigger rate: K=5 per 12,740 evaluation objects => 5/12740 = 0.00039246
    # On S=1 (N=2,663), 2663 * (5/12740) = 1.045 => top 1 candidate per epoch on S=1
    s1_eval_ratio = 5.0 / len(df_eval_meta)
    s1_quota = max(1, int(round(len(df_train_meta) * s1_eval_ratio)))

    # Per-epoch S=1 naive thresholds:
    tau_naive_by_epoch = {}
    for epoch in epochs:
        p_tr = np.sort(train_preds[epoch])[::-1]
        tau_naive_by_epoch[epoch] = float(p_tr[s1_quota - 1])

    tau_naive_global = max(tau_naive_by_epoch.values())
    print(
        f"\nS=1 Calibrated Naive Thresholds: epoch 0.0d={tau_naive_by_epoch[0.0]:.6f}, epoch 2.0d={tau_naive_by_epoch[2.0]:.6f} (global max={tau_naive_global:.6f})",
        flush=True,
    )

    # Prepare policy evaluation inputs
    y_true_eval = df_eval_meta["true_target"].to_numpy(dtype=int)
    obj_ids_eval = df_eval_meta["object_id"].to_numpy(dtype=int)
    n_total_eval = len(y_true_eval)
    n_kn_eval = int(np.sum(y_true_eval == 64))
    n_non_kn_eval = n_total_eval - n_kn_eval

    kn_indices = np.where(y_true_eval == 64)[0]

    epoch_preds_dict = {e: (eval_preds[e], eval_novelties[e]) for e in epochs}

    # Configuration definitions:
    # 1. Naive Baseline: Confidence threshold tuned on S=1, w_nov=0, capacity K=5
    # 2. Frozen Policy: w_nov=0.05, tau=0.001, capacity K=5
    # 3. Novelty Ablation: w_nov=0.00, tau=0.001, capacity K=5

    def evaluate_config(
        w_nov: float,
        tau: float,
        capacity: int = 5,
        config_name: str = "Policy",
    ) -> dict[str, Any]:
        pol_cfg = DecisionPolicyConfig()
        # Override w_nov and tau dynamically
        pol_cfg = pol_cfg.model_copy(
            update={
                "novelty_weight": w_nov,
                "decision_threshold": tau,
                "capacity_per_epoch": capacity,
            }
        )
        policy = SequentialDecisionPolicy(config=pol_cfg)

        trace = policy.evaluate_sequential_trace(
            epoch_predictions=epoch_preds_dict,
            y_true=y_true_eval,
            object_ids=obj_ids_eval,
            novelty_scales=novelty_scales,
        )

        actions = trace["cumulative_actions"]
        trigger_epochs = trace["trigger_epochs"]

        # Utility Regret & MHVER
        regret = trace["regret"]
        mhver = trace["mhver"]

        # False positive triggers (non-kilonova triggers)
        fp_mask = (actions == 1) & (y_true_eval != 64)
        fp_count = int(np.sum(fp_mask))
        ftr = float(fp_count / n_non_kn_eval)

        # Clopper-Pearson CIs
        missed_count = int(round(mhver * n_kn_eval))
        triggered_kn_count = n_kn_eval - missed_count
        mhver_ci = clopper_pearson_ci(missed_count, n_kn_eval)
        target_trigger_rate = triggered_kn_count / n_kn_eval
        target_trigger_ci = clopper_pearson_ci(triggered_kn_count, n_kn_eval)

        # LOKO Jackknife analysis
        loko_results = []
        for kn_idx in kn_indices:
            # Mask out kn_idx
            keep_mask = np.ones(n_total_eval, dtype=bool)
            keep_mask[kn_idx] = False

            sub_y = y_true_eval[keep_mask]
            sub_actions = actions[keep_mask]

            sub_regret = compute_utility_regret(
                policy_actions=sub_actions,
                y_true=sub_y,
                capacity=capacity,
                target_class=64,
                u_tp=2.0,
                u_fp=-1.0,
            )
            sub_mhver = compute_missed_high_value_event_rate(
                actions=sub_actions,
                y_true=sub_y,
                target_class=64,
            )
            loko_results.append(
                {
                    "excluded_object_id": int(obj_ids_eval[kn_idx]),
                    "regret": sub_regret["regret"],
                    "norm_regret": sub_regret["normalized_regret"],
                    "mhver": sub_mhver,
                    "policy_utility": sub_regret["policy_utility"],
                }
            )

        loko_regrets = [r["regret"] for r in loko_results]
        loko_norm_regrets = [r["norm_regret"] for r in loko_results]
        loko_mhvers = [r["mhver"] for r in loko_results]

        # Per-object audit for confirmed kilonovae
        kn_audit = []
        for kn_idx in kn_indices:
            obj_id = int(obj_ids_eval[kn_idx])
            meta_row = df_eval_meta[df_eval_meta["object_id"] == obj_id].iloc[0]
            trig_epoch = float(trigger_epochs[kn_idx])
            trig_bool = bool(actions[kn_idx] == 1)

            # Extract scores at epoch 0.0 and 2.0
            p0 = float(eval_preds[0.0][kn_idx])
            nov0 = float(eval_novelties[0.0][kn_idx])
            s0 = float(p0 + w_nov * (nov0 / novelty_scales[0.0]))

            p2 = float(eval_preds[2.0][kn_idx])
            nov2 = float(eval_novelties[2.0][kn_idx])
            s2 = float(p2 + w_nov * (nov2 / novelty_scales[2.0]))

            # Utility contribution
            util_contrib = 2.0 if trig_bool else 0.0

            kn_audit.append(
                {
                    "object_id": obj_id,
                    "z_phot": float(meta_row["hostgal_photoz"]),
                    "true_z": float(meta_row["true_z"]),
                    "triggered": trig_bool,
                    "trigger_epoch": trig_epoch,
                    "p_kn_0d": p0,
                    "p_kn_2d": p2,
                    "score_0d": s0,
                    "score_2d": s2,
                    "utility_contribution": util_contrib,
                }
            )

        return {
            "config_name": config_name,
            "w_nov": w_nov,
            "threshold": tau,
            "capacity": capacity,
            "policy_utility": regret["policy_utility"],
            "oracle_utility": regret["oracle_utility"],
            "no_trigger_utility": regret["no_trigger_utility"],
            "regret": regret["regret"],
            "normalized_regret": regret["normalized_regret"],
            "mhver": mhver,
            "mhver_ci_95": mhver_ci,
            "target_trigger_rate": target_trigger_rate,
            "target_trigger_ci_95": target_trigger_ci,
            "false_positives": fp_count,
            "false_trigger_rate": ftr,
            "loko_jackknife": {
                "regret_range": [float(min(loko_regrets)), float(max(loko_regrets))],
                "norm_regret_range": [
                    float(min(loko_norm_regrets)),
                    float(max(loko_norm_regrets)),
                ],
                "mhver_range": [float(min(loko_mhvers)), float(max(loko_mhvers))],
                "iterations": loko_results,
            },
            "kilonova_per_object_audit": kn_audit,
        }

    eval_naive = evaluate_config(
        w_nov=0.00,
        tau=tau_naive_global,
        capacity=5,
        config_name="Naive Fixed-Confidence Baseline",
    )

    eval_frozen = evaluate_config(
        w_nov=0.05,
        tau=0.001,
        capacity=5,
        config_name="Frozen Bias-and-Novelty Policy",
    )

    eval_ablation = evaluate_config(
        w_nov=0.00,
        tau=0.001,
        capacity=5,
        config_name="Novelty Ablation (w_nov = 0.00)",
    )

    results_summary = {
        "metadata": {
            "evaluation_date": "2026-07-29",
            "evaluation_population": "FULL TRUE cohort (plasticc_test_lightcurves_01.csv.gz slice)",
            "n_total": n_total_eval,
            "n_kilonova": n_kn_eval,
            "n_non_kilonova": n_non_kn_eval,
            "primary_deadline": 2.0,
            "capacity": 5,
            "s1_calibrated_naive_threshold": tau_naive_global,
        },
        "configurations": {
            "naive_baseline": eval_naive,
            "frozen_policy": eval_frozen,
            "novelty_ablation": eval_ablation,
        },
    }

    # Save JSON summary
    json_path = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "results"
        / "headline_evaluation_metrics.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)

    print(f"\nSaved evaluation metrics JSON to {json_path}", flush=True)

    # Print summary tables to stdout
    print(
        "\n=================== HEADLINE COMPARATIVE RESULTS ===================",
        flush=True,
    )
    print(
        f"{'Configuration':<32} | {'Regret':<8} | {'Norm Regret':<12} | {'MHVER [95% CP CI]':<24} | {'False Triggers (Rate)':<20}"
    )
    print("-" * 105)

    for cfg_key in ["naive_baseline", "frozen_policy", "novelty_ablation"]:
        c = results_summary["configurations"][cfg_key]
        mhver_str = (
            f"{c['mhver']:.4f} [{c['mhver_ci_95'][0]:.4f}, {c['mhver_ci_95'][1]:.4f}]"
        )
        ftr_str = f"{c['false_positives']} ({c['false_trigger_rate']:.4%})"
        print(
            f"{c['config_name']:<32} | {c['regret']:<8.1f} | {c['normalized_regret']:<12.4f} | {mhver_str:<24} | {ftr_str:<20}"
        )

    print(
        "\n================ PER-OBJECT KILONOVA DECISION AUDIT ================",
        flush=True,
    )
    for cfg_key in ["naive_baseline", "frozen_policy", "novelty_ablation"]:
        c = results_summary["configurations"][cfg_key]
        print(f"\n--- {c['config_name']} ---")
        for kn in c["kilonova_per_object_audit"]:
            trig_str = (
                f"TRIGGERED at e={kn['trigger_epoch']:.1f}d"
                if kn["triggered"]
                else "NOT TRIGGERED"
            )
            print(
                f"  Obj {kn['object_id']} (z_phot={kn['z_phot']:.4f}): {trig_str} | Score e=0d: {kn['score_0d']:.6f}, e=2d: {kn['score_2d']:.6f}"
            )

    print(f"\nCompleted in {time.time() - t0:.2f}s", flush=True)
    return results_summary


if __name__ == "__main__":
    run_evaluation()
