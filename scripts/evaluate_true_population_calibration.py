"""Evaluation harness for True-population calibration audit (ADR 003).

Quantifies early-epoch baseline classifier probabilistic calibration on:
- S=1 (spectroscopically selected test set)
- S=0 (unselected deployment set)
- FULL TRUE population (S=1 + S=0)

Calculates:
1. Multiclass Brier score and Murphy decomposition (Reliability, Resolution,
   Uncertainty).
2. Classwise Expected Calibration Error (ECE) using 10 equal-width bins with bin
   counts.
3. Adaptive-bin reliability statistics (minimum 50 objects per bin where feasible).
4. Stratified evaluation across epoch, predicted-probability deciles, brightness
   quintiles, and redshift quintiles. Strata with < 30 events are tagged as exploratory.
5. Nonparametric object-level bootstrap (1,000 replicates, seed=42) for 95% CIs.

Includes explicit real-time progress logging.
Saves raw metrics to docs/results/calibration_audit_true_population_metrics.json.
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

from aegis.config.features import FeatureConfig
from aegis.config.models import BaselineClassifierConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import (
    BaselineClassifier,
    representation_results_to_dataframe,
)

STUDY_CLASSES = [64, 90, 95]
CLASS_NAMES = {
    64: "Kilonova (KN)",
    90: "Type Ia Supernova (SN Ia)",
    95: "Superluminous SN (SLSN-I)",
}


def calculate_brier_and_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[int] = STUDY_CLASSES,
    n_bins: int = 10,
) -> dict[str, float | list[dict[str, Any]]]:
    """Compute multiclass Brier score and Murphy decomposition (REL, RES, UNC)."""
    n_samples, n_classes = y_prob.shape
    if n_samples == 0:
        return {
            "brier_score": 0.0,
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.0,
            "binned_brier": 0.0,
            "bin_details": [],
        }

    y_onehot = np.zeros((n_samples, n_classes), dtype=float)
    for i, cls in enumerate(classes):
        y_onehot[:, i] = (y_true == cls).astype(float)

    bs_exact = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

    bar_y_c = np.mean(y_onehot, axis=0)
    unc_total = float(np.sum(bar_y_c * (1.0 - bar_y_c)))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    rel_total = 0.0
    res_total = 0.0
    bin_details = []

    for c_idx, cls in enumerate(classes):
        p_c = y_prob[:, c_idx]
        y_c = y_onehot[:, c_idx]
        bar_y_cls = bar_y_c[c_idx]

        cls_rel = 0.0
        cls_res = 0.0

        for b in range(n_bins):
            low, high = bin_edges[b], bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (p_c >= low) & (p_c <= high)
            else:
                mask = (p_c >= low) & (p_c < high)

            n_bin = int(np.sum(mask))
            if n_bin > 0:
                bar_p_mc = float(np.mean(p_c[mask]))
                bar_y_mc = float(np.mean(y_c[mask]))

                weight = n_bin / n_samples
                rel_term = weight * (bar_p_mc - bar_y_mc) ** 2
                res_term = weight * (bar_y_mc - bar_y_cls) ** 2

                cls_rel += rel_term
                cls_res += res_term

                bin_details.append(
                    {
                        "class_id": int(cls),
                        "bin_idx": b,
                        "bin_range": [float(low), float(high)],
                        "count": n_bin,
                        "mean_pred": bar_p_mc,
                        "mean_obs": bar_y_mc,
                        "weight": float(weight),
                    }
                )

        rel_total += cls_rel
        res_total += cls_res

    bs_binned = rel_total - res_total + unc_total

    return {
        "brier_score": bs_exact,
        "reliability": float(rel_total),
        "resolution": float(res_total),
        "uncertainty": float(unc_total),
        "binned_brier": float(bs_binned),
        "bin_details": bin_details,
    }


def calculate_classwise_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[int] = STUDY_CLASSES,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute classwise ECE and overall mean ECE per ADR 003."""
    n_samples, n_classes = y_prob.shape
    if n_samples == 0:
        return {
            "mean_ece": 0.0,
            "class_ece": {str(c): 0.0 for c in classes},
            "bin_counts": {},
        }

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    class_eces: dict[str, float] = {}
    bin_counts_dict: dict[str, list[int]] = {}

    for c_idx, cls in enumerate(classes):
        y_c = (y_true == cls).astype(float)
        p_c = y_prob[:, c_idx]

        ece_c = 0.0
        counts = []

        for b in range(n_bins):
            low, high = bin_edges[b], bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (p_c >= low) & (p_c <= high)
            else:
                mask = (p_c >= low) & (p_c < high)

            bin_size = int(np.sum(mask))
            counts.append(bin_size)

            if bin_size > 0:
                acc = float(np.mean(y_c[mask]))
                conf = float(np.mean(p_c[mask]))
                ece_c += (bin_size / n_samples) * abs(acc - conf)

        class_eces[str(cls)] = float(ece_c)
        bin_counts_dict[str(cls)] = counts

    mean_ece = float(np.mean(list(class_eces.values())))
    return {
        "mean_ece": mean_ece,
        "class_ece": class_eces,
        "bin_counts": bin_counts_dict,
    }


def calculate_adaptive_bin_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[int] = STUDY_CLASSES,
    min_objects_per_bin: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    """Compute adaptive quantile binning reliability data (min 50 objects/bin)."""
    n_samples, n_classes = y_prob.shape
    results: dict[str, list[dict[str, Any]]] = {}

    for c_idx, cls in enumerate(classes):
        y_c = (y_true == cls).astype(float)
        p_c = y_prob[:, c_idx]

        if n_samples < min_objects_per_bin:
            n_bins_target = 1
        else:
            n_bins_target = max(1, min(10, n_samples // min_objects_per_bin))

        quantiles = np.linspace(0, 100, n_bins_target + 1)
        bin_edges = np.unique(np.percentile(p_c, quantiles))

        cls_bins = []
        for b in range(len(bin_edges) - 1):
            low, high = bin_edges[b], bin_edges[b + 1]
            if b == len(bin_edges) - 2:
                mask = (p_c >= low) & (p_c <= high)
            else:
                mask = (p_c >= low) & (p_c < high)

            n_bin = int(np.sum(mask))
            if n_bin > 0:
                mean_p = float(np.mean(p_c[mask]))
                mean_y = float(np.mean(y_c[mask]))
                cls_bins.append(
                    {
                        "bin_idx": b,
                        "bin_range": [float(low), float(high)],
                        "count": n_bin,
                        "mean_pred": mean_p,
                        "mean_obs": mean_y,
                    }
                )

        results[str(cls)] = cls_bins

    return results


def run_evaluation_for_cohort(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    meta_df: pd.DataFrame,
    classes: list[int] = STUDY_CLASSES,
    n_bootstrap: int = 1000,
    seed: int = 42,
    cohort_name: str = "Cohort",
) -> dict[str, Any]:
    """Compute point estimates and 95% object-level bootstrap CIs."""
    n_samples = len(y_true)

    brier_res = calculate_brier_and_decomposition(y_true, y_prob, classes)
    ece_res = calculate_classwise_ece(y_true, y_prob, classes)
    adaptive_res = calculate_adaptive_bin_diagram(y_true, y_prob, classes)

    rng = np.random.default_rng(seed)

    boot_bs = np.zeros(n_bootstrap)
    boot_rel = np.zeros(n_bootstrap)
    boot_res = np.zeros(n_bootstrap)
    boot_unc = np.zeros(n_bootstrap)
    boot_ece_mean = np.zeros(n_bootstrap)
    boot_ece_64 = np.zeros(n_bootstrap)
    boot_ece_90 = np.zeros(n_bootstrap)
    boot_ece_95 = np.zeros(n_bootstrap)

    if n_samples > 0:
        print(
            f"  [Bootstrap] Computing {n_bootstrap:,} resamples for"
            f" {cohort_name} (N={n_samples:,})...",
            flush=True,
        )
        t_boot_start = time.time()
        for b in range(n_bootstrap):
            if (b + 1) % 250 == 0 or b + 1 == n_bootstrap:
                pct = (b + 1) / n_bootstrap * 100
                print(
                    f"    - Bootstrap progress ({cohort_name}):"
                    f" {b + 1}/{n_bootstrap} ({pct:.0f}%) complete",
                    flush=True,
                )

            idx = rng.choice(n_samples, size=n_samples, replace=True)
            y_b = y_true[idx]
            p_b = y_prob[idx]

            brier_b = calculate_brier_and_decomposition(y_b, p_b, classes)
            ece_b = calculate_classwise_ece(y_b, p_b, classes)

            boot_bs[b] = brier_b["brier_score"]
            boot_rel[b] = brier_b["reliability"]
            boot_res[b] = brier_b["resolution"]
            boot_unc[b] = brier_b["uncertainty"]

            boot_ece_mean[b] = ece_b["mean_ece"]
            boot_ece_64[b] = ece_b["class_ece"]["64"]
            boot_ece_90[b] = ece_b["class_ece"]["90"]
            boot_ece_95[b] = ece_b["class_ece"]["95"]
        print(
            f"  [Bootstrap] Completed in {time.time() - t_boot_start:.2f}s",
            flush=True,
        )

    def get_ci(arr: np.ndarray) -> list[float]:
        low, high = np.percentile(arr, [2.5, 97.5])
        return [float(low), float(high)]

    return {
        "n_objects": n_samples,
        "brier_score": float(brier_res["brier_score"]),
        "brier_score_ci95": get_ci(boot_bs),
        "reliability": float(brier_res["reliability"]),
        "reliability_ci95": get_ci(boot_rel),
        "resolution": float(brier_res["resolution"]),
        "resolution_ci95": get_ci(boot_res),
        "uncertainty": float(brier_res["uncertainty"]),
        "uncertainty_ci95": get_ci(boot_unc),
        "binned_brier": float(brier_res["binned_brier"]),
        "mean_ece": float(ece_res["mean_ece"]),
        "mean_ece_ci95": get_ci(boot_ece_mean),
        "class_ece": ece_res["class_ece"],
        "class_ece_ci95": {
            "64": get_ci(boot_ece_64),
            "90": get_ci(boot_ece_90),
            "95": get_ci(boot_ece_95),
        },
        "equal_width_bin_counts": ece_res["bin_counts"],
        "adaptive_bin_diagram": adaptive_res,
    }


def compute_stratified_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    meta_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    group_name: str = "Group",
) -> dict[str, Any]:
    """Compute metrics stratified by prob deciles, brightness & photo-z quintiles."""
    results: dict[str, Any] = {}

    print(
        f"\n  === Computing Stratified Metrics for {group_name} ===",
        flush=True,
    )

    # 1. Predicted-probability deciles for Kilonova (class 64, idx 0)
    p_kn = y_prob[:, 0]
    p_deciles = np.percentile(p_kn, np.linspace(0, 100, 11))
    p_deciles_unique = np.unique(p_deciles)

    p_strata = {}
    print(
        f"  [Strata] Evaluating {len(p_deciles_unique) - 1} Deciles (KN)...",
        flush=True,
    )
    for d in range(len(p_deciles_unique) - 1):
        low, high = p_deciles_unique[d], p_deciles_unique[d + 1]
        mask = (
            (p_kn >= low) & (p_kn <= high)
            if d == len(p_deciles_unique) - 2
            else (p_kn >= low) & (p_kn < high)
        )
        n_sub = int(np.sum(mask))

        tag = "standard" if n_sub >= 30 else "exploratory"
        if n_sub > 0:
            res_sub = run_evaluation_for_cohort(
                y_true[mask],
                y_prob[mask],
                meta_df[mask],
                n_bootstrap=n_bootstrap,
                seed=seed,
                cohort_name=f"{group_name} Prob Decile {d + 1}",
            )
        else:
            res_sub = {"n_objects": 0, "mean_ece": 0.0, "brier_score": 0.0}

        res_sub["stratum_tag"] = tag
        res_sub["prob_range"] = [float(low), float(high)]
        p_strata[f"decile_{d + 1}"] = res_sub

    results["prob_deciles_kn"] = p_strata

    # 2. Apparent-brightness quintiles (derived peak apparent r-band mag m_r)
    m_r = -19.3 + meta_df["distmod"].to_numpy() + 3.1 * meta_df["mwebv"].to_numpy()
    m_quintiles = np.percentile(m_r, np.linspace(0, 100, 6))
    m_quintiles_unique = np.unique(m_quintiles)

    m_strata = {}
    print(
        f"  [Strata] Evaluating {len(m_quintiles_unique) - 1}"
        " Brightness Quintiles (derived m_r)...",
        flush=True,
    )
    for q in range(len(m_quintiles_unique) - 1):
        low, high = m_quintiles_unique[q], m_quintiles_unique[q + 1]
        mask = (
            (m_r >= low) & (m_r <= high)
            if q == len(m_quintiles_unique) - 2
            else (m_r >= low) & (m_r < high)
        )
        n_sub = int(np.sum(mask))

        tag = "standard" if n_sub >= 30 else "exploratory"
        if n_sub > 0:
            res_sub = run_evaluation_for_cohort(
                y_true[mask],
                y_prob[mask],
                meta_df[mask],
                n_bootstrap=n_bootstrap,
                seed=seed,
                cohort_name=f"{group_name} Brightness Quintile {q + 1}",
            )
        else:
            res_sub = {"n_objects": 0, "mean_ece": 0.0, "brier_score": 0.0}

        res_sub["stratum_tag"] = tag
        res_sub["brightness_m_r_range"] = [float(low), float(high)]
        res_sub["quantity_label"] = "derived_physical_approximation"
        m_strata[f"quintile_{q + 1}"] = res_sub

    results["brightness_quintiles_m_r"] = m_strata

    # 3. Redshift quintiles (hostgal_photoz - alert time verified per ADR 005)
    z_photo = meta_df["hostgal_photoz"].to_numpy()
    z_quintiles = np.percentile(z_photo, np.linspace(0, 100, 6))
    z_quintiles_unique = np.unique(z_quintiles)

    z_strata = {}
    print(
        f"  [Strata] Evaluating {len(z_quintiles_unique) - 1}"
        " Redshift Quintiles (hostgal_photoz)...",
        flush=True,
    )
    for q in range(len(z_quintiles_unique) - 1):
        low, high = z_quintiles_unique[q], z_quintiles_unique[q + 1]
        mask = (
            (z_photo >= low) & (z_photo <= high)
            if q == len(z_quintiles_unique) - 2
            else (z_photo >= low) & (z_photo < high)
        )
        n_sub = int(np.sum(mask))

        tag = "standard" if n_sub >= 30 else "exploratory"
        if n_sub > 0:
            res_sub = run_evaluation_for_cohort(
                y_true[mask],
                y_prob[mask],
                meta_df[mask],
                n_bootstrap=n_bootstrap,
                seed=seed,
                cohort_name=f"{group_name} Redshift Quintile {q + 1}",
            )
        else:
            res_sub = {"n_objects": 0, "mean_ece": 0.0, "brier_score": 0.0}

        res_sub["stratum_tag"] = tag
        res_sub["redshift_photoz_range"] = [float(low), float(high)]
        res_sub["quantity_label"] = "directly_measured_catalog_feature"
        z_strata[f"quintile_{q + 1}"] = res_sub

    results["redshift_quintiles_photoz"] = z_strata

    return results


def main() -> None:
    t0 = time.time()
    print(
        "=======================================================================",
        flush=True,
    )
    print("=== STARTING TRUE-POPULATION CALIBRATION AUDIT (ADR 003) ===", flush=True)
    print(
        "=======================================================================",
        flush=True,
    )

    # 1. Load Training Metadata and Light Curves (S=1 Training Population)
    train_meta_path = Path("data/raw/plasticc_train_metadata.csv.gz")
    train_lc_path = Path("data/raw/plasticc_train_lightcurves.csv.gz")

    if not train_meta_path.exists() or not train_lc_path.exists():
        raise FileNotFoundError(f"Missing {train_meta_path} or {train_lc_path}")

    print("[Step 1/4] Loading S=1 Training Metadata & Light Curves...", flush=True)
    df_train_meta = pd.read_csv(train_meta_path)
    df_train_meta = df_train_meta[df_train_meta["target"].isin(STUDY_CLASSES)].copy()
    df_train_meta["true_target"] = df_train_meta["target"]
    df_train_meta["population"] = "BIASED"
    df_train_meta["S"] = 1

    train_ids = set(df_train_meta["object_id"])
    print(
        f"  -> S=1 Training metadata loaded: N = {len(df_train_meta):,} objects.",
        flush=True,
    )
    cls_counts = df_train_meta["true_target"].value_counts().to_dict()
    print(f"  -> Class counts in S=1 training set: {cls_counts}", flush=True)

    print(
        f"  -> Reading training observations from {train_lc_path.name}...",
        flush=True,
    )
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
    print(
        f"  -> Training observations indexed for {len(train_obs_by_obj):,} objects.",
        flush=True,
    )

    # 2. Load Evaluation Cohort Metadata (Test population)
    true_meta_path = Path("data/processed/true_population.csv.gz")
    biased_meta_path = Path("data/processed/biased_population.csv.gz")
    test_lc_path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz")

    print(
        "\n[Step 2/4] Loading Evaluation Metadata & Light Curves...",
        flush=True,
    )
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

    print(
        f"  -> Reading evaluation observations from {test_lc_path.name}...",
        flush=True,
    )
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
    n_cohort = len(df_eval_meta)
    print(
        f"  -> Evaluation cohort loaded: N = {n_cohort:,} objects.",
        flush=True,
    )

    s1_eval_mask = df_eval_meta["S"] == 1
    s0_eval_mask = df_eval_meta["S"] == 0

    n_s1_cnt = int(s1_eval_mask.sum())
    n_s0_cnt = int(s0_eval_mask.sum())
    print(
        f"     - S=1 (Spectroscopically selected test set): N = {n_s1_cnt:,}",
        flush=True,
    )
    print(
        f"     - S=0 (Unselected deployment population):    N = {n_s0_cnt:,}",
        flush=True,
    )

    eval_obs_by_obj: dict[int, pd.DataFrame] = {
        obj_id: group for obj_id, group in df_eval_obs.groupby("object_id")
    }

    feat_config = FeatureConfig()
    epochs = [0.0, 2.0, 7.0]

    metrics_out: dict[str, Any] = {
        "metadata": {
            "n_s1_train": len(df_train_meta),
            "n_s1_test": int(s1_eval_mask.sum()),
            "n_s0_deployment": int(s0_eval_mask.sum()),
            "n_full_true_eval": len(df_eval_meta),
            "epochs": epochs,
            "classes": STUDY_CLASSES,
            "class_names": CLASS_NAMES,
            "n_bootstrap": 1000,
            "seed": 42,
        },
        "epochs": {},
    }

    print(
        "\n[Step 3/4] Fitting & Auditing Classifier...",
        flush=True,
    )

    for epoch in epochs:
        print(
            "=======================================================",
            flush=True,
        )
        print(f"--> AUDITING DECISION EPOCH e = {epoch:.1f} days", flush=True)
        print(
            "=======================================================",
            flush=True,
        )

        # 1. Feature extraction for S=1 training set
        n_tr = len(df_train_meta)
        print(
            f"  -> Extracting features for S=1 training set (N={n_tr:,})...",
            flush=True,
        )
        train_records = df_train_meta.to_dict(orient="records")
        train_rep_results = []
        n_train = len(train_records)
        t_tr_start = time.time()

        for idx_r, row in enumerate(train_records):
            if (idx_r + 1) % 1000 == 0 or idx_r + 1 == n_train:
                pct = (idx_r + 1) / n_train * 100
                print(
                    f"     [Train Progress] {idx_r + 1:5d} / {n_train:5d}"
                    f" ({pct:5.1f}%) objects extracted",
                    flush=True,
                )

            obj_id = int(row["object_id"])
            raw_obs = train_obs_by_obj.get(obj_id, pd.DataFrame())
            trunc_obs = truncate_light_curve_at_epoch(
                raw_obs,
                days_since_first_detection=epoch,
                validate_schema=False,
            )
            res = extract_early_representation(
                df_obs=trunc_obs,
                meta_row=row,
                config=feat_config,
                epoch=epoch,
            )
            train_rep_results.append(res)

        t_tr_elapsed = time.time() - t_tr_start
        print(
            f"  -> Training feature extraction completed in {t_tr_elapsed:.2f}s",
            flush=True,
        )

        df_feat_train = representation_results_to_dataframe(train_rep_results)
        y_train = df_train_meta["true_target"].to_numpy(dtype=int)

        feature_cols_all = [
            c for c in df_feat_train.columns if c not in ("object_id", "epoch")
        ]
        varying_cols = [
            c for c in feature_cols_all if df_feat_train[c].dropna().nunique() > 1
        ]

        X_train = df_feat_train[varying_cols]

        # Fit BaselineClassifier on S=1 training set
        print(
            "  -> Fitting HistGradientBoosting Classifier (S=1)...",
            flush=True,
        )
        model_config = BaselineClassifierConfig(random_seed=42, min_samples_leaf=20)
        clf = BaselineClassifier(config=model_config)
        clf.fit_epoch(
            X_train,
            y_train,
            epoch=epoch,
            population_type="BIASED",
            meta_df=df_train_meta,
        )
        print("  -> Model fitting complete.", flush=True)

        # 2. Feature extraction for evaluation cohort
        n_ev = len(df_eval_meta)
        print(
            f"  -> Extracting features for evaluation cohort (N={n_ev:,})...",
            flush=True,
        )
        eval_records = df_eval_meta.to_dict(orient="records")
        eval_rep_results = []
        n_eval = len(eval_records)
        t_ev_start = time.time()

        for idx_r, row in enumerate(eval_records):
            if (idx_r + 1) % 2500 == 0 or idx_r + 1 == n_eval:
                pct = (idx_r + 1) / n_eval * 100
                print(
                    f"     [Eval Progress] {idx_r + 1:5d} / {n_eval:5d}"
                    f" ({pct:5.1f}%) objects extracted",
                    flush=True,
                )

            obj_id = int(row["object_id"])
            raw_obs = eval_obs_by_obj.get(obj_id, pd.DataFrame())
            trunc_obs = truncate_light_curve_at_epoch(
                raw_obs,
                days_since_first_detection=epoch,
                validate_schema=False,
            )
            res = extract_early_representation(
                df_obs=trunc_obs,
                meta_row=row,
                config=feat_config,
                epoch=epoch,
            )
            eval_rep_results.append(res)

        t_ev_elapsed = time.time() - t_ev_start
        print(
            f"  -> Evaluation feature extraction completed in {t_ev_elapsed:.2f}s",
            flush=True,
        )

        df_feat_eval = representation_results_to_dataframe(eval_rep_results)
        y_eval = df_eval_meta["true_target"].to_numpy(dtype=int)
        X_eval = df_feat_eval[varying_cols]

        # Predict probabilities
        probs_eval = clf.predict_proba(X_eval, epoch=epoch)

        # Evaluation subsets
        y_s1 = y_eval[s1_eval_mask]
        p_s1 = probs_eval[s1_eval_mask]
        meta_s1 = df_eval_meta[s1_eval_mask].copy().reset_index(drop=True)

        y_s0 = y_eval[s0_eval_mask]
        p_s0 = probs_eval[s0_eval_mask]
        meta_s0 = df_eval_meta[s0_eval_mask].copy().reset_index(drop=True)

        y_full = y_eval
        p_full = probs_eval
        meta_full = df_eval_meta.copy().reset_index(drop=True)

        print(
            f"\n  --- Computing Calibration & Bootstrap (e={epoch:.1f}d) ---",
            flush=True,
        )
        res_s1 = run_evaluation_for_cohort(
            y_s1, p_s1, meta_s1, n_bootstrap=1000, seed=42, cohort_name="S=1 Test Set"
        )
        res_s0 = run_evaluation_for_cohort(
            y_s0,
            p_s0,
            meta_s0,
            n_bootstrap=1000,
            seed=42,
            cohort_name="S=0 Deployment Set",
        )
        res_full = run_evaluation_for_cohort(
            y_full,
            p_full,
            meta_full,
            n_bootstrap=1000,
            seed=42,
            cohort_name="FULL TRUE Population",
        )

        strat_s1 = compute_stratified_metrics(
            y_s1, p_s1, meta_s1, n_bootstrap=1000, seed=42, group_name="S=1 Test Set"
        )
        strat_s0 = compute_stratified_metrics(
            y_s0,
            p_s0,
            meta_s0,
            n_bootstrap=1000,
            seed=42,
            group_name="S=0 Deployment Set",
        )
        strat_full = compute_stratified_metrics(
            y_full,
            p_full,
            meta_full,
            n_bootstrap=1000,
            seed=42,
            group_name="FULL TRUE Population",
        )

        print(f"\n  === SUMMARY RESULTS AT EPOCH e = {epoch:.1f}d ===", flush=True)
        print(
            f"  [S=1 Test] BS={res_s1['brier_score']:.4f} "
            f"ECE={res_s1['mean_ece']:.4f} REL={res_s1['reliability']:.4f} "
            f"RES={res_s1['resolution']:.4f} UNC={res_s1['uncertainty']:.4f}",
            flush=True,
        )
        print(
            f"  [S=0 Dev]  BS={res_s0['brier_score']:.4f} "
            f"ECE={res_s0['mean_ece']:.4f} REL={res_s0['reliability']:.4f} "
            f"RES={res_s0['resolution']:.4f} UNC={res_s0['uncertainty']:.4f}",
            flush=True,
        )
        print(
            f"  [FULL TRUE]BS={res_full['brier_score']:.4f} "
            f"ECE={res_full['mean_ece']:.4f} REL={res_full['reliability']:.4f} "
            f"RES={res_full['resolution']:.4f} UNC={res_full['uncertainty']:.4f}",
            flush=True,
        )

        metrics_out["epochs"][str(epoch)] = {
            "S1_test": {**res_s1, "strata": strat_s1},
            "S0_deployment": {**res_s0, "strata": strat_s0},
            "FULL_TRUE": {**res_full, "strata": strat_full},
        }

    print("\n[Step 4/4] Writing raw metrics JSON output...", flush=True)
    out_file = Path("docs/results/calibration_audit_true_population_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics_out, f, indent=2)

    t1 = time.time()
    print(f"\nSaved raw metrics to {out_file.resolve()}", flush=True)
    print(
        f"TRUE-POPULATION CALIBRATION AUDIT COMPLETED IN {t1 - t0:.2f} SECONDS!",
        flush=True,
    )


if __name__ == "__main__":
    main()
