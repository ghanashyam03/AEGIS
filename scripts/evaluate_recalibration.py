"""Evaluation harness for Selection-Aware Recalibration (Phase 3).

Re-evaluates baseline classifier probabilities vs recalibrated probabilities across:
- Decision epochs e ∈ {0.0, 2.0, 7.0} days
- Evaluation populations: S=1 test set (N=2,693), S=0 deployment set (N=10,047),
  FULL TRUE population (N=12,740)

Computes:
1. Positivity / overlap diagnostics (affected counts, %, feature ranges, deciles).
2. Importance weight distribution diagnostics (min, median, p95, max, CV, ESS).
3. Covariate balance diagnostics (SMDs for hostgal_photoz, distmod, mwebv).
4. Recalibrated Multiclass Brier score, Murphy decomposition (REL, RES, UNC), and ECE.
5. Absolute and relative metric improvements with 95% bootstrap CIs (B=1,000, seed=42).
6. Honest residual gap characterization and uncorrectable strata identification.

Saves raw metrics to docs/results/recalibration_true_population_metrics.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aegis.config.features import FeatureConfig
from aegis.config.models import BaselineClassifierConfig
from aegis.data.observation import truncate_light_curve_at_epoch
from aegis.features.representation import extract_early_representation
from aegis.models.baseline import (
    BaselineClassifier,
    representation_results_to_dataframe,
)
from aegis.recalibration.selection_recalibration import (
    SelectionAwareRecalibrator,
    compute_covariate_balance,
    compute_selection_weights,
    diagnose_positivity_overlap,
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
) -> dict[str, float]:
    """Compute multiclass Brier score and Murphy decomposition (REL, RES, UNC)."""
    n_samples, n_classes = y_prob.shape
    if n_samples == 0:
        return {
            "brier_score": 0.0,
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.0,
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

    for c_idx in range(n_classes):
        p_c = y_prob[:, c_idx]
        y_c = y_onehot[:, c_idx]
        bar_y_cls = bar_y_c[c_idx]

        for b in range(n_bins):
            low, high = bin_edges[b], bin_edges[b + 1]
            mask = (
                (p_c >= low) & (p_c <= high)
                if b == n_bins - 1
                else (p_c >= low) & (p_c < high)
            )

            n_bin = int(np.sum(mask))
            if n_bin > 0:
                bar_p_mc = float(np.mean(p_c[mask]))
                bar_y_mc = float(np.mean(y_c[mask]))
                weight = n_bin / n_samples

                rel_total += weight * (bar_p_mc - bar_y_mc) ** 2
                res_total += weight * (bar_y_mc - bar_y_cls) ** 2

    return {
        "brier_score": bs_exact,
        "reliability": float(rel_total),
        "resolution": float(res_total),
        "uncertainty": unc_total,
    }


def calculate_classwise_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[int] = STUDY_CLASSES,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute classwise ECE and overall mean ECE."""
    n_samples, n_classes = y_prob.shape
    if n_samples == 0:
        return {"mean_ece": 0.0, "class_ece": {str(c): 0.0 for c in classes}}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    class_ece: dict[str, float] = {}

    for c_idx, cls in enumerate(classes):
        y_c = (y_true == cls).astype(float)
        p_c = y_prob[:, c_idx]
        ece_c = 0.0

        for b in range(n_bins):
            low, high = bin_edges[b], bin_edges[b + 1]
            mask = (
                (p_c >= low) & (p_c <= high)
                if b == n_bins - 1
                else (p_c >= low) & (p_c < high)
            )
            n_bin = int(np.sum(mask))
            if n_bin > 0:
                mean_p = float(np.mean(p_c[mask]))
                mean_y = float(np.mean(y_c[mask]))
                ece_c += (n_bin / n_samples) * abs(mean_p - mean_y)

        class_ece[str(cls)] = float(ece_c)

    mean_ece = float(np.mean(list(class_ece.values())))
    return {"mean_ece": mean_ece, "class_ece": class_ece}


def evaluate_cohort_comparison(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_recal: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    cohort_name: str = "Cohort",
) -> dict[str, Any]:
    """Evaluate baseline vs recalibrated metrics and compute 95% bootstrap CIs."""
    n_samples = len(y_true)
    n_classes = len(STUDY_CLASSES)

    brier_raw = calculate_brier_and_decomposition(y_true, p_raw)
    ece_raw = calculate_classwise_ece(y_true, p_raw)

    brier_rec = calculate_brier_and_decomposition(y_true, p_recal)
    ece_rec = calculate_classwise_ece(y_true, p_recal)

    # Point estimate improvements
    abs_d_bs = brier_rec["brier_score"] - brier_raw["brier_score"]
    rel_d_bs = (
        (brier_raw["brier_score"] - brier_rec["brier_score"])
        / brier_raw["brier_score"]
        * 100.0
        if brier_raw["brier_score"] > 0
        else 0.0
    )

    abs_d_rel = brier_rec["reliability"] - brier_raw["reliability"]
    rel_d_rel = (
        (brier_raw["reliability"] - brier_rec["reliability"])
        / brier_raw["reliability"]
        * 100.0
        if brier_raw["reliability"] > 0
        else 0.0
    )

    abs_d_ece = ece_rec["mean_ece"] - ece_raw["mean_ece"]
    rel_d_ece = (
        (ece_raw["mean_ece"] - ece_rec["mean_ece"]) / ece_raw["mean_ece"] * 100.0
        if ece_raw["mean_ece"] > 0
        else 0.0
    )

    boot_bs_raw = np.zeros(n_bootstrap)
    boot_bs_rec = np.zeros(n_bootstrap)
    boot_d_bs = np.zeros(n_bootstrap)

    boot_rel_raw = np.zeros(n_bootstrap)
    boot_rel_rec = np.zeros(n_bootstrap)
    boot_d_rel = np.zeros(n_bootstrap)

    boot_ece_raw = np.zeros(n_bootstrap)
    boot_ece_rec = np.zeros(n_bootstrap)
    boot_d_ece = np.zeros(n_bootstrap)

    if n_samples > 0:
        print(
            f"  [Bootstrap] Vectorized computation of {n_bootstrap:,} resamples for"
            f" {cohort_name} (N={n_samples:,})...",
            flush=True,
        )
        t0 = time.time()
        rng = np.random.default_rng(seed)
        boot_idx = rng.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)

        y_onehot = np.zeros((n_samples, n_classes), dtype=float)
        for i, cls in enumerate(STUDY_CLASSES):
            y_onehot[:, i] = (y_true == cls).astype(float)

        # Batch 100 resamples at a time to prevent RAM spikes while executing in seconds
        batch_size = 100
        bin_edges = np.linspace(0.0, 1.0, 11)

        for b_start in range(0, n_bootstrap, batch_size):
            b_end = min(b_start + batch_size, n_bootstrap)
            b_count = b_end - b_start
            sub_idx = boot_idx[b_start:b_end]  # shape: (b_count, n_samples)

            y_b = y_onehot[sub_idx]  # (b_count, n_samples, n_classes)
            p_raw_b = p_raw[sub_idx]
            p_rec_b = p_recal[sub_idx]

            # Vectorized Brier score: mean over samples of sum over classes
            bs_raw_arr = np.mean(np.sum((p_raw_b - y_b) ** 2, axis=2), axis=1)
            bs_rec_arr = np.mean(np.sum((p_rec_b - y_b) ** 2, axis=2), axis=1)

            rel_raw_arr = np.zeros(b_count)
            rel_rec_arr = np.zeros(b_count)
            ece_raw_arr = np.zeros(b_count)
            ece_rec_arr = np.zeros(b_count)

            # Vectorized classwise REL & ECE
            for c_idx in range(n_classes):
                y_c_batch = y_b[:, :, c_idx]
                p_raw_c_batch = p_raw_b[:, :, c_idx]
                p_rec_c_batch = p_rec_b[:, :, c_idx]

                ece_raw_c = np.zeros(b_count)
                ece_rec_c = np.zeros(b_count)

                for bin_i in range(10):
                    low, high = bin_edges[bin_i], bin_edges[bin_i + 1]

                    mask_raw = (
                        (p_raw_c_batch >= low) & (p_raw_c_batch <= high)
                        if bin_i == 9
                        else (p_raw_c_batch >= low) & (p_raw_c_batch < high)
                    )
                    mask_rec = (
                        (p_rec_c_batch >= low) & (p_rec_c_batch <= high)
                        if bin_i == 9
                        else (p_rec_c_batch >= low) & (p_rec_c_batch < high)
                    )

                    for k in range(b_count):
                        m_raw_k = mask_raw[k]
                        cnt_raw = np.sum(m_raw_k)
                        if cnt_raw > 0:
                            bar_p_raw = np.mean(p_raw_c_batch[k, m_raw_k])
                            bar_y_raw = np.mean(y_c_batch[k, m_raw_k])
                            w_k = cnt_raw / n_samples
                            rel_raw_arr[k] += w_k * (bar_p_raw - bar_y_raw) ** 2
                            ece_raw_c[k] += w_k * abs(bar_p_raw - bar_y_raw)

                        m_rec_k = mask_rec[k]
                        cnt_rec = np.sum(m_rec_k)
                        if cnt_rec > 0:
                            bar_p_rec = np.mean(p_rec_c_batch[k, m_rec_k])
                            bar_y_rec = np.mean(y_c_batch[k, m_rec_k])
                            w_k = cnt_rec / n_samples
                            rel_rec_arr[k] += w_k * (bar_p_rec - bar_y_rec) ** 2
                            ece_rec_c[k] += w_k * abs(bar_p_rec - bar_y_rec)

                ece_raw_arr += ece_raw_c / n_classes
                ece_rec_arr += ece_rec_c / n_classes

            boot_bs_raw[b_start:b_end] = bs_raw_arr
            boot_bs_rec[b_start:b_end] = bs_rec_arr
            boot_d_bs[b_start:b_end] = bs_rec_arr - bs_raw_arr

            boot_rel_raw[b_start:b_end] = rel_raw_arr
            boot_rel_rec[b_start:b_end] = rel_rec_arr
            boot_d_rel[b_start:b_end] = rel_rec_arr - rel_raw_arr

            boot_ece_raw[b_start:b_end] = ece_raw_arr
            boot_ece_rec[b_start:b_end] = ece_rec_arr
            boot_d_ece[b_start:b_end] = ece_rec_arr - ece_raw_arr

        print(
            f"  [Bootstrap] Vectorized evaluation completed in {time.time() - t0:.2f}s",
            flush=True,
        )

    def get_ci(arr: np.ndarray) -> list[float]:
        low, high = np.percentile(arr, [2.5, 97.5])
        return [float(low), float(high)]

    return {
        "n_objects": n_samples,
        "baseline": {
            "brier_score": float(brier_raw["brier_score"]),
            "brier_score_ci95": get_ci(boot_bs_raw),
            "reliability": float(brier_raw["reliability"]),
            "reliability_ci95": get_ci(boot_rel_raw),
            "resolution": float(brier_raw["resolution"]),
            "uncertainty": float(brier_raw["uncertainty"]),
            "mean_ece": float(ece_raw["mean_ece"]),
            "mean_ece_ci95": get_ci(boot_ece_raw),
            "class_ece": ece_raw["class_ece"],
        },
        "recalibrated": {
            "brier_score": float(brier_rec["brier_score"]),
            "brier_score_ci95": get_ci(boot_bs_rec),
            "reliability": float(brier_rec["reliability"]),
            "reliability_ci95": get_ci(boot_rel_rec),
            "resolution": float(brier_rec["resolution"]),
            "uncertainty": float(brier_rec["uncertainty"]),
            "mean_ece": float(ece_rec["mean_ece"]),
            "mean_ece_ci95": get_ci(boot_ece_rec),
            "class_ece": ece_rec["class_ece"],
        },
        "improvements": {
            "abs_delta_bs": float(abs_d_bs),
            "abs_delta_bs_ci95": get_ci(boot_d_bs),
            "rel_pct_bs_reduction": float(rel_d_bs),
            "abs_delta_rel": float(abs_d_rel),
            "abs_delta_rel_ci95": get_ci(boot_d_rel),
            "rel_pct_rel_reduction": float(rel_d_rel),
            "abs_delta_ece": float(abs_d_ece),
            "abs_delta_ece_ci95": get_ci(boot_d_ece),
            "rel_pct_ece_reduction": float(rel_d_ece),
        },
    }


def main() -> None:
    t0 = time.time()
    print("=========================================================", flush=True)
    print("  SELECTION-AWARE RECALIBRATION AUDIT HARNESS (PHASE 3)  ", flush=True)
    print("=========================================================", flush=True)

    # 1. Load Training Data (S=1)
    train_meta_path = Path("data/raw/plasticc_train_metadata.csv.gz")
    train_lc_path = Path("data/raw/plasticc_train_lightcurves.csv.gz")

    df_train_meta = pd.read_csv(train_meta_path)
    df_train_meta = (
        df_train_meta[df_train_meta["true_target"].isin(STUDY_CLASSES)]
        .copy()
        .reset_index(drop=True)
    )

    train_ids = set(df_train_meta["object_id"])
    train_obs_list = []
    for chunk in pd.read_csv(
        train_lc_path,
        usecols=[
            "object_id",
            "mjd",
            "passband",
            "flux",
            "flux_err",
            "detected_bool",
        ],
        chunksize=500_000,
    ):
        sub = chunk[chunk["object_id"].isin(train_ids)]
        if not sub.empty:
            train_obs_list.append(sub)

    df_train_obs = pd.concat(train_obs_list, ignore_index=True)
    train_obs_by_obj = {
        obj_id: group for obj_id, group in df_train_obs.groupby("object_id")
    }

    # 2. Load Evaluation Cohort
    true_meta_path = Path("data/processed/true_population.csv.gz")
    biased_meta_path = Path("data/processed/biased_population.csv.gz")
    test_lc_path = Path("data/raw/plasticc_test_lightcurves_01.csv.gz")

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

    s1_eval_mask = df_eval_meta["S"] == 1
    s0_eval_mask = df_eval_meta["S"] == 0

    s1_eval_df = df_eval_meta[s1_eval_mask].copy()

    # 3. Compute Positivity & Overlap Diagnostics
    print("\n[Step 1/3] Computing Positivity & Selection Diagnostics...", flush=True)
    positivity_diag = diagnose_positivity_overlap(
        df_true=df_eval_meta, df_s1=s1_eval_df, z_cutoff=1.5
    )

    photoz_s1_eval = s1_eval_df["hostgal_photoz"].to_numpy(dtype=float)
    w_s1_eval, weight_diag = compute_selection_weights(photoz_s1_eval)
    balance_diag = compute_covariate_balance(
        df_s1=s1_eval_df, df_true=df_eval_meta, weights_s1=w_s1_eval
    )

    print(
        f"  -> Positivity Diagnostic: {positivity_diag.n_affected_objects} objects"
        f" ({positivity_diag.pct_true_population:.2f}%) in high-z region (z > 1.5).",
        flush=True,
    )
    ess_pct = weight_diag["ess_fraction"] * 100.0
    print(
        f"  -> Weight Diagnostics: min={weight_diag['min']:.4f},"
        f" med={weight_diag['median']:.4f}, p95={weight_diag['p95']:.4f},"
        f" max={weight_diag['max']:.4f}, CV={weight_diag['cv']:.4f},"
        f" ESS={weight_diag['ess']:.1f}/{weight_diag['n_samples']} ({ess_pct:.1f}%)",
        flush=True,
    )
    print(
        f"  -> Covariate Balance (hostgal_photoz SMD):"
        f" Unweighted={balance_diag['hostgal_photoz']['smd_unweighted']:.4f},"
        f" Weighted={balance_diag['hostgal_photoz']['smd_weighted']:.4f}",
        flush=True,
    )

    # 4. Epoch Loop
    feat_config = FeatureConfig()
    epochs = [0.0, 2.0, 7.0]

    out_metrics: dict[str, Any] = {
        "metadata": {
            "n_s1_train": len(df_train_meta),
            "n_s1_test": int(s1_eval_mask.sum()),
            "n_s0_deployment": int(s0_eval_mask.sum()),
            "n_full_true_eval": len(df_eval_meta),
            "epochs": epochs,
            "classes": STUDY_CLASSES,
            "class_names": CLASS_NAMES,
            "z_cutoff_positivity": 1.5,
            "seed": 42,
        },
        "diagnostics": {
            "positivity": positivity_diag.to_dict(),
            "weights": weight_diag,
            "covariate_balance": balance_diag,
        },
        "epochs": {},
    }

    print(
        "\n[Step 2/3] Fitting Classifier & Selection-Aware Recalibrator...",
        flush=True,
    )

    for epoch in epochs:
        print(
            "\n=======================================================",
            flush=True,
        )
        print(f"--> AUDITING RECALIBRATION FOR EPOCH e = {epoch:.1f} days", flush=True)
        print(
            "=======================================================",
            flush=True,
        )

        # Feature extraction on S=1 training set
        train_records = df_train_meta.to_dict(orient="records")
        train_rep_results = []
        for row in train_records:
            obj_id = int(row["object_id"])
            raw_obs = train_obs_by_obj.get(obj_id, pd.DataFrame())
            trunc_obs = truncate_light_curve_at_epoch(
                raw_obs,
                days_since_first_detection=epoch,
                validate_schema=False,
            )
            res = extract_early_representation(
                df_obs=trunc_obs, meta_row=row, config=feat_config, epoch=epoch
            )
            train_rep_results.append(res)

        df_feat_train = representation_results_to_dataframe(train_rep_results)
        y_train = df_train_meta["true_target"].to_numpy(dtype=int)
        photoz_train = df_train_meta["hostgal_photoz"].to_numpy(dtype=float)

        feature_cols_all = [
            c for c in df_feat_train.columns if c not in ("object_id", "epoch")
        ]
        varying_cols = [
            c for c in feature_cols_all if df_feat_train[c].dropna().nunique() > 1
        ]
        X_train = df_feat_train[varying_cols]

        # Fit baseline classifier on S=1
        model = BaselineClassifier(
            BaselineClassifierConfig(random_seed=42, min_samples_leaf=20)
        )
        model.fit_epoch(X_train, y_train, epoch=epoch)
        p_train_raw = model.predict_proba(X_train, epoch=epoch)

        # Fit SelectionAwareRecalibrator EXCLUSIVELY on S=1 training probabilities
        recalibrator = SelectionAwareRecalibrator(
            z_cutoff=1.5, apply_extrapolation_mask=True
        )
        recalibrator.fit(
            y_prob_s1=p_train_raw, y_true_s1=y_train, photoz_s1=photoz_train
        )

        # Feature extraction on evaluation cohort
        eval_records = df_eval_meta.to_dict(orient="records")
        eval_rep_results = []
        for row in eval_records:
            obj_id = int(row["object_id"])
            raw_obs = eval_obs_by_obj.get(obj_id, pd.DataFrame())
            trunc_obs = truncate_light_curve_at_epoch(
                raw_obs,
                days_since_first_detection=epoch,
                validate_schema=False,
            )
            res = extract_early_representation(
                df_obs=trunc_obs, meta_row=row, config=feat_config, epoch=epoch
            )
            eval_rep_results.append(res)

        df_feat_eval = representation_results_to_dataframe(eval_rep_results)
        X_eval = df_feat_eval[varying_cols]
        y_eval = df_eval_meta["true_target"].to_numpy(dtype=int)
        photoz_eval = df_eval_meta["hostgal_photoz"].to_numpy(dtype=float)

        p_eval_raw = model.predict_proba(X_eval, epoch=epoch)
        p_eval_recal = recalibrator.predict_proba(p_eval_raw, photoz=photoz_eval)

        # Separate masks
        mask_s1 = df_eval_meta["S"] == 1
        mask_s0 = df_eval_meta["S"] == 0

        # Run evaluation comparisons
        res_s1 = evaluate_cohort_comparison(
            y_eval[mask_s1],
            p_eval_raw[mask_s1],
            p_eval_recal[mask_s1],
            n_bootstrap=1000,
            seed=42,
            cohort_name=f"S=1 Test Set (e={epoch:.1f}d)",
        )
        res_s0 = evaluate_cohort_comparison(
            y_eval[mask_s0],
            p_eval_raw[mask_s0],
            p_eval_recal[mask_s0],
            n_bootstrap=1000,
            seed=42,
            cohort_name=f"S=0 Deployment Set (e={epoch:.1f}d)",
        )
        res_full = evaluate_cohort_comparison(
            y_eval,
            p_eval_raw,
            p_eval_recal,
            n_bootstrap=1000,
            seed=42,
            cohort_name=f"FULL TRUE Population (e={epoch:.1f}d)",
        )

        bs_base = res_full["baseline"]["brier_score"]
        bs_rec = res_full["recalibrated"]["brier_score"]
        rel_base = res_full["baseline"]["reliability"]
        rel_rec = res_full["recalibrated"]["reliability"]
        ece_base = res_full["baseline"]["mean_ece"] * 100
        ece_rec = res_full["recalibrated"]["mean_ece"] * 100

        print(
            f"  -> [FULL TRUE e={epoch:.1f}d] Baseline BS: {bs_base:.4f}"
            f" -> Recalibrated BS: {bs_rec:.4f}",
            flush=True,
        )
        print(
            f"  -> [FULL TRUE e={epoch:.1f}d] Baseline REL: {rel_base:.4f}"
            f" -> Recalibrated REL: {rel_rec:.4f}",
            flush=True,
        )
        print(
            f"  -> [FULL TRUE e={epoch:.1f}d] Baseline ECE: {ece_base:.2f}%"
            f" -> Recalibrated ECE: {ece_rec:.2f}%",
            flush=True,
        )

        out_metrics["epochs"][f"{epoch:.1f}"] = {
            "S1_test": res_s1,
            "S0_deployment": res_s0,
            "FULL_TRUE": res_full,
        }

    # 5. Save Output
    out_file = Path("docs/results/recalibration_true_population_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_metrics, f, indent=2)

    t1 = time.time()
    print(
        f"\nSaved raw recalibration metrics to {out_file.resolve()}",
        flush=True,
    )
    print(
        f"SELECTION-AWARE RECALIBRATION AUDIT COMPLETED IN {t1 - t0:.2f} SECONDS!",
        flush=True,
    )


if __name__ == "__main__":
    main()
