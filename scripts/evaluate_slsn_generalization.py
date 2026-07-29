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
    EPOCH_IDENTIFIABLE_FEATURES,
    compute_epoch_novelty_scores,
)


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
    df_train_meta = df_train_meta[df_train_meta["target"].isin([64, 90, 95])].copy()
    if "true_target" not in df_train_meta.columns:
        df_train_meta["true_target"] = df_train_meta["target"]
    train_ids = set(df_train_meta["object_id"])
    df_train_lc = df_train_lc[df_train_lc["object_id"].isin(train_ids)].copy()

    # 2. Load Evaluation population (TRUE population slice)
    eval_meta_path = Path("data/processed/true_population.csv.gz")
    eval_lc_path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz")

    print("Loading Evaluation Metadata & Light Curves...", flush=True)
    df_true_meta = pd.read_csv(eval_meta_path)
    df_eval_lc = pd.read_csv(eval_lc_path)

    eval_lc_ids = set(df_eval_lc["object_id"].unique())
    df_eval_meta = df_true_meta[df_true_meta["object_id"].isin(eval_lc_ids)].copy()
    eval_ids = set(df_eval_meta["object_id"])
    df_eval_lc = df_eval_lc[df_eval_lc["object_id"].isin(eval_ids)].copy()

    print(
        f"Evaluation population size: N = {len(df_eval_meta):,}, SLSN-I (class 95) count = {int((df_eval_meta['true_target'] == 95).sum())}",
        flush=True,
    )

    epochs = [0.0, 2.0]
    feat_config = FeatureConfig()

    # Cache observations
    train_obs_by_obj = {
        obj_id: group for obj_id, group in df_train_lc.groupby("object_id")
    }
    eval_obs_by_obj = {
        obj_id: group for obj_id, group in df_eval_lc.groupby("object_id")
    }

    train_preds: dict[float, np.ndarray] = {}
    eval_preds: dict[float, np.ndarray] = {}

    train_novelties: dict[float, np.ndarray] = {}
    eval_novelties: dict[float, np.ndarray] = {}
    novelty_scales: dict[float, float] = {}

    # Process features per epoch
    for epoch in epochs:
        print(f"\nProcessing Epoch e = {epoch} days...", flush=True)

        # S=1 Training representations
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

        # Predict probabilities: study_classes = (64, 90, 95)
        # SLSN-I (class 95) is at index 2
        probs_train = clf.predict_proba(X_train, epoch=epoch)
        train_preds[epoch] = probs_train[:, 2]  # P(SLSN-I)

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
        eval_preds[epoch] = probs_eval[:, 2]  # P(SLSN-I)

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

        train_novelties[epoch] = nov_train
        eval_novelties[epoch] = nov_eval

    # Calibrate Naive Threshold on S=1 training data
    # Target quota = 5 / 12,740 = 0.0003925
    # S=1 N=2,663, top 1 cutoff:
    tau_naive_by_epoch = {}
    for epoch in epochs:
        sorted_train_probs = np.sort(train_preds[epoch])[::-1]
        tau_naive_by_epoch[epoch] = float(sorted_train_probs[0])

    tau_naive_global = max(tau_naive_by_epoch.values())
    print(
        f"\nS=1 Calibrated Naive Thresholds for SLSN-I: epoch 0.0d={tau_naive_by_epoch[0.0]:.6f}, epoch 2.0d={tau_naive_by_epoch[2.0]:.6f} (global max={tau_naive_global:.6f})",
        flush=True,
    )

    # Targets: SLSN-I (class 95)
    y_true_eval = df_eval_meta["true_target"].to_numpy(dtype=int)
    obj_ids_eval = df_eval_meta["object_id"].to_numpy(dtype=int)
    z_phot_eval = df_eval_meta["hostgal_photoz"].to_numpy(dtype=float)

    n_total_eval = len(y_true_eval)
    n_slsn_eval = int(np.sum(y_true_eval == 95))
    n_non_slsn_eval = n_total_eval - n_slsn_eval

    slsn_indices = np.where(y_true_eval == 95)[0]

    epoch_preds_dict = {e: (eval_preds[e], eval_novelties[e]) for e in epochs}

    def evaluate_config(
        w_nov: float,
        tau: float,
        capacity: int = 5,
        config_name: str = "Policy",
    ) -> dict[str, Any]:
        pol_cfg = DecisionPolicyConfig()
        pol_cfg = pol_cfg.model_copy(
            update={
                "novelty_weight": w_nov,
                "decision_threshold": tau,
                "capacity_per_epoch": capacity,
                "target_class": 95,
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

        regret_info = trace["regret"]
        mhver = float(trace["mhver"])

        fp_mask = (actions == 1) & (y_true_eval != 95)
        fp_count = int(np.sum(fp_mask))
        ftr = float(fp_count / n_non_slsn_eval)

        # Clopper-Pearson CIs
        missed_count = int(round(mhver * n_slsn_eval))
        triggered_slsn_count = n_slsn_eval - missed_count
        mhver_ci = clopper_pearson_ci(missed_count, n_slsn_eval)

        # Audit per object
        slsn_audit = []
        for idx in slsn_indices:
            obj_id = int(obj_ids_eval[idx])
            z_p = float(z_phot_eval[idx])
            trig = bool(actions[idx] == 1)
            trig_e = (
                float(trigger_epochs[idx])
                if trig and trigger_epochs[idx] >= 0
                else None
            )

            score_0d = float(
                eval_preds[0.0][idx]
                + w_nov * (eval_novelties[0.0][idx] / novelty_scales[0.0])
            )
            score_2d = float(
                eval_preds[2.0][idx]
                + w_nov * (eval_novelties[2.0][idx] / novelty_scales[2.0])
            )

            slsn_audit.append(
                {
                    "object_id": obj_id,
                    "z_phot": z_p,
                    "triggered": trig,
                    "trigger_epoch": trig_e,
                    "score_0d": score_0d,
                    "score_2d": score_2d,
                    "p_slsn_0d": float(eval_preds[0.0][idx]),
                    "p_slsn_2d": float(eval_preds[2.0][idx]),
                    "nov_0d": float(eval_novelties[0.0][idx]),
                    "nov_2d": float(eval_novelties[2.0][idx]),
                }
            )

        slsn_audit.sort(key=lambda x: max(x["score_0d"], x["score_2d"]), reverse=True)

        # Leave-One-Out Jackknife analysis across N=98 SLSN objects
        jackknife_mhvers = []
        jackknife_regrets = []
        for i in range(len(slsn_indices)):
            mask = np.ones(len(y_true_eval), dtype=bool)
            mask[slsn_indices[i]] = False

            sub_y = y_true_eval[mask]
            sub_actions = actions[mask]

            sub_n_slsn = int(np.sum(sub_y == 95))
            sub_oracle = float(sub_n_slsn * 2.0)
            sub_trig_slsn = int(np.sum((sub_actions == 1) & (sub_y == 95)))
            sub_trig_non = int(np.sum((sub_actions == 1) & (sub_y != 95)))
            sub_u_actual = 2.0 * sub_trig_slsn - 1.0 * sub_trig_non

            sub_regret = sub_oracle - sub_u_actual
            sub_mhver = (
                (sub_n_slsn - sub_trig_slsn) / sub_n_slsn
                if sub_n_slsn > 0
                else 0.0
            )

            jackknife_mhvers.append(float(sub_mhver))
            jackknife_regrets.append(float(sub_regret))

        return {
            "config_name": config_name,
            "w_nov": w_nov,
            "tau": tau,
            "capacity": capacity,
            "total_utility": float(regret_info["actual_utility"]),
            "oracle_utility": float(regret_info["oracle_utility"]),
            "regret": float(regret_info["regret"]),
            "normalized_regret": float(regret_info["normalized_regret"]),
            "mhver": mhver,
            "mhver_95_ci": list(mhver_ci),
            "targets_triggered": triggered_slsn_count,
            "targets_total": n_slsn_eval,
            "false_positives": fp_count,
            "false_trigger_rate": ftr,
            "jackknife_mhver_range": [
                float(np.min(jackknife_mhvers)),
                float(np.max(jackknife_mhvers)),
            ],
            "jackknife_regret_range": [
                float(np.min(jackknife_regrets)),
                float(np.max(jackknife_regrets)),
            ],
            "slsn_audit_top10": slsn_audit[:10],
            "slsn_audit_summary": {
                "total_slsn": len(slsn_audit),
                "num_triggered": triggered_slsn_count,
                "mean_p_slsn_0d": float(
                    np.mean([x["p_slsn_0d"] for x in slsn_audit])
                ),
                "mean_p_slsn_2d": float(
                    np.mean([x["p_slsn_2d"] for x in slsn_audit])
                ),
                "mean_nov_0d": float(np.mean([x["nov_0d"] for x in slsn_audit])),
                "mean_nov_2d": float(np.mean([x["nov_2d"] for x in slsn_audit])),
                "max_p_slsn_0d": float(np.max([x["p_slsn_0d"] for x in slsn_audit])),
                "max_p_slsn_2d": float(np.max([x["p_slsn_2d"] for x in slsn_audit])),
            },
        }

    # Run 3 comparative configurations
    results_summary = {
        "diagnostic_target": "SLSN-I (class 95)",
        "population_size": n_total_eval,
        "slsn_count": n_slsn_eval,
        "slsn_base_rate": float(n_slsn_eval / n_total_eval),
        "capacity_k": 5,
        "primary_deadline_h": 2.0,
        "s1_calibrated_naive_threshold": tau_naive_global,
        "configurations": {
            "naive_baseline": evaluate_config(
                w_nov=0.00,
                tau=tau_naive_global,
                capacity=5,
                config_name="Naive Fixed-Confidence Baseline",
            ),
            "frozen_policy": evaluate_config(
                w_nov=0.05,
                tau=0.001,
                capacity=5,
                config_name="Frozen Bias-and-Novelty Policy",
            ),
            "novelty_ablation": evaluate_config(
                w_nov=0.00,
                tau=0.001,
                capacity=5,
                config_name="Novelty Ablation (w_nov = 0.00)",
            ),
        },
    }

    # Write output JSON
    out_json_path = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "results"
        / "slsn_generalization_metrics.json"
    )
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)

    print(
        f"\nSaved SLSN generalization metrics JSON to {out_json_path.resolve()}",
        flush=True,
    )

    # Print summary tables
    print(
        "\n=================== SLSN-I GENERALIZATION RESULTS ===================",
        flush=True,
    )
    print(
        f"{'Configuration':<32} | {'Regret':<8} | {'Norm Regret':<12} | {'MHVER [95% CP CI]':<24} | {'False Triggers (Rate)':<20}"
    )
    print("-" * 105)

    for cfg_key in ["naive_baseline", "frozen_policy", "novelty_ablation"]:
        c = results_summary["configurations"][cfg_key]
        mhver_str = f"{c['mhver']:.4f} [{c['mhver_95_ci'][0]:.4f}, {c['mhver_95_ci'][1]:.4f}] ({c['targets_triggered']}/{c['targets_total']} trig)"
        ftr_str = f"{c['false_positives']} ({c['false_trigger_rate']:.4%})"
        print(
            f"{c['config_name']:<32} | {c['regret']:<8.1f} | {c['normalized_regret']:<12.4f} | {mhver_str:<24} | {ftr_str:<20}"
        )

    print(
        "\n================ SLSN-I TOP 10 CANDIDATE DECISION AUDIT ================",
        flush=True,
    )
    for cfg_key in ["naive_baseline", "frozen_policy", "novelty_ablation"]:
        c = results_summary["configurations"][cfg_key]
        print(f"\n--- {c['config_name']} ---")
        for slsn in c["slsn_audit_top10"]:
            trig_str = (
                f"TRIGGERED at e={slsn['trigger_epoch']:.1f}d"
                if slsn["triggered"]
                else "NOT TRIGGERED"
            )
            print(
                f"  Obj {slsn['object_id']} (z={slsn['z_phot']:.4f}): {trig_str} | P(SLSN) e=0d: {slsn['p_slsn_0d']:.6f}, e=2d: {slsn['p_slsn_2d']:.6f} | Score e=0d: {slsn['score_0d']:.6f}, e=2d: {slsn['score_2d']:.6f}"
            )

    print(f"\nCompleted in {time.time() - t0:.2f}s", flush=True)
    return results_summary


if __name__ == "__main__":
    run_slsn_evaluation()
