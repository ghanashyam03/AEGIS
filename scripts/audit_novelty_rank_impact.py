# ruff: noqa: E501
"""Diagnostic script for Novelty Rank-Impact (Step 2).

Directly compares the set of objects selected under the frozen fused policy
against the set selected under the novelty-ablation policy, and reports the
size of the symmetric difference.
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
    print("=== AEGIS STEP 2: NOVELTY RANK-IMPACT DIAGNOSTIC ===", flush=True)

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

    # 2. Load Evaluation Metadata
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

    # Sub-sample SN Ia background to 20,000 to match expanded evaluation population
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
    epochs = [0.0, 2.0]

    # Parallel evaluation feature extraction
    tasks = []
    for obj_id_val, group in df_eval_obs.groupby("object_id"):
        obj_id = int(obj_id_val)
        meta_row = study_meta_dict[obj_id]
        tasks.append((obj_id, group, meta_row, epochs, feat_config))

    print(
        f"Extracting evaluation features in parallel for {len(tasks):,} objects...",
        flush=True,
    )
    eval_feat_dicts = {e: {} for e in epochs}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(extract_features_for_object, tasks, chunksize=100)
        for obj_id, res_dict in results:
            for epoch in epochs:
                eval_feat_dicts[epoch][obj_id] = res_dict[epoch]

    print(f"Feature extraction completed in {time.time() - t0:.1f}s.", flush=True)

    # Process training features and fit classifiers per epoch
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

    # Predict probabilities & compute novelty scores per epoch
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
        eval_preds[epoch] = probs_eval[:, 0]  # P(Kilonova)

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

    # Run Fused Policy
    p_fused = DecisionPolicyConfig(
        novelty_weight=0.05,
        decision_threshold=0.001,
        capacity_per_epoch=5,
    )
    pol_fused = SequentialDecisionPolicy(config=p_fused)

    epoch_preds_dict = {}
    for epoch in epochs:
        epoch_preds_dict[epoch] = (eval_preds[epoch], eval_novelties[epoch])

    trace_fused = pol_fused.evaluate_sequential_trace(
        epoch_predictions=epoch_preds_dict,
        y_true=df_eval_meta["true_target"].to_numpy(dtype=int),
        object_ids=df_eval_meta["object_id"].to_numpy(dtype=int),
        novelty_scales=novelty_scales,
    )
    triggered_fused = set(
        trace_fused["object_ids"][trace_fused["cumulative_actions"] == 1]
    )

    # Run Novelty Ablation Policy
    p_ablation = DecisionPolicyConfig(
        novelty_weight=0.00,
        decision_threshold=0.001,
        capacity_per_epoch=5,
    )
    pol_ablation = SequentialDecisionPolicy(config=p_ablation)

    trace_ablation = pol_ablation.evaluate_sequential_trace(
        epoch_predictions=epoch_preds_dict,
        y_true=df_eval_meta["true_target"].to_numpy(dtype=int),
        object_ids=df_eval_meta["object_id"].to_numpy(dtype=int),
        novelty_scales=novelty_scales,
    )
    triggered_ablation = set(
        trace_ablation["object_ids"][trace_ablation["cumulative_actions"] == 1]
    )

    # Compare triggered sets
    sym_diff = triggered_fused.symmetric_difference(triggered_ablation)
    print("\n--- RESULTS ---", flush=True)
    print(f"Fused Policy Triggered IDs: {triggered_fused}", flush=True)
    print(f"Ablation Policy Triggered IDs: {triggered_ablation}", flush=True)
    print(f"Symmetric Difference: {sym_diff}", flush=True)
    print(f"Size of Symmetric Difference: {len(sym_diff)}", flush=True)


if __name__ == "__main__":
    main()
