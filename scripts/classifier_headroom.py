# ruff: noqa: E501
"""Diagnostic script for Classifier Headroom Investigation (Step 3).

Performs leakage-safe hyperparameter tuning via nested cross-validation on the S=1 training set.
Fits retuned classifier on full S=1, and evaluates ROC-AUC against the expanded evaluation population.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from aegis.config.features import FeatureConfig
from aegis.config.models import BaselineClassifierConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import (
    BaselineClassifier,
    representation_results_to_dataframe,
)

STUDY_CLASSES = [64, 90, 95]


def kilonova_auc_scorer(estimator, X, y):
    """Custom scorer targeting Kilonova vs Rest ROC-AUC."""
    probs = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 64 in classes:
        idx = classes.index(64)
        p_kn = probs[:, idx]
    else:
        p_kn = np.zeros(len(X))
    y_binary = (y == 64).astype(int)
    if len(np.unique(y_binary)) < 2:
        return 0.5
    return roc_auc_score(y_binary, p_kn)


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
    print("=== AEGIS STEP 3: CLASSIFIER HEADROOM INVESTIGATION ===", flush=True)

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
    epochs = [0.0, 2.0, 7.0]

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

    # Hyperparameter Grid
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_iter": [50, 100],
        "max_depth": [3, 5],
        "min_samples_leaf": [20, 50],
        "l2_regularization": [1.0, 10.0],
    }

    for epoch in epochs:
        print(f"\n--- Epoch e = {epoch:.1f} days ---", flush=True)

        # Extract features for training
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

        X_train = df_feat_train[varying_cols].to_numpy(dtype=float)

        # 1. Leakage-safe nested cross-validation on S=1
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        outer_scores = []
        for train_idx, val_idx in cv_outer.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            base_estimator = HistGradientBoostingClassifier(
                loss="log_loss", random_state=42
            )
            search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=kilonova_auc_scorer,
                cv=cv_inner,
                n_jobs=-1,
            )
            search.fit(X_tr, y_tr)
            best_model = search.best_estimator_

            # Evaluate best model on outer validation fold
            outer_score = kilonova_auc_scorer(best_model, X_val, y_val)
            outer_scores.append(outer_score)

        mean_outer_auc = np.mean(outer_scores)
        print(
            f"S=1 5-fold Nested CV Kilonova ROC-AUC: {mean_outer_auc:.4f} +/- {np.std(outer_scores):.4f}",
            flush=True,
        )

        # 2. Hyperparameter search on full S=1 train
        base_estimator = HistGradientBoostingClassifier(
            loss="log_loss", random_state=42
        )
        full_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            scoring=kilonova_auc_scorer,
            cv=cv_outer,
            n_jobs=-1,
        )
        full_search.fit(X_train, y_train)
        best_params = full_search.best_params_
        print(f"Best hyperparameters selected on S=1: {best_params}", flush=True)

        # Fit final tuned classifier
        clf_tuned = HistGradientBoostingClassifier(
            loss="log_loss",
            random_state=42,
            **best_params,
        )
        clf_tuned.fit(X_train, y_train)

        # Fit default classifier
        model_config = BaselineClassifierConfig(random_seed=42, min_samples_leaf=20)
        clf_default = BaselineClassifier(config=model_config)
        clf_default.fit_epoch(
            df_feat_train[varying_cols],
            y_train,
            epoch=epoch,
            population_type="BIASED",
            meta_df=df_train_meta,
        )

        # 3. Evaluate both default and tuned on expanded test population
        eval_reps = [
            eval_feat_dicts[epoch][obj_id] for obj_id in df_eval_meta["object_id"]
        ]
        df_feat_eval = representation_results_to_dataframe(eval_reps)
        X_eval = df_feat_eval[varying_cols].to_numpy(dtype=float)
        y_eval = df_eval_meta["true_target"].to_numpy(dtype=int)
        y_eval_binary = (y_eval == 64).astype(int)

        # Default prediction
        probs_default = clf_default.predict_proba(
            df_feat_eval[varying_cols], epoch=epoch
        )
        p_kn_default = probs_default[:, 0]
        auc_default = roc_auc_score(y_eval_binary, p_kn_default)

        # Tuned prediction
        probs_tuned = clf_tuned.predict_proba(X_eval)
        classes_tuned = list(clf_tuned.classes_)
        idx_kn_tuned = classes_tuned.index(64) if 64 in classes_tuned else 0
        p_kn_tuned = probs_tuned[:, idx_kn_tuned]
        auc_tuned = roc_auc_score(y_eval_binary, p_kn_tuned)

        print("Expanded population Kilonova ROC-AUC:", flush=True)
        print(f"  Default model: {auc_default:.4f}", flush=True)
        print(f"  Tuned model:   {auc_tuned:.4f}", flush=True)
        print(f"  Difference:    {auc_tuned - auc_default:+.4f}", flush=True)

    print(f"\nCompleted headroom check in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
