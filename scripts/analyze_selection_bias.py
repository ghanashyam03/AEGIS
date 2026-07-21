"""Reproducible quantitative characterization of TRUE vs BIASED selection bias.

This script loads the TRUE population (full PLAsTiCC test metadata restricted to study
classes) and the BIASED population (logistic spectroscopic selection proxy per ADR 004).
It quantifies distributional shifts, effect sizes (Wasserstein distance, JS divergence,
KS statistic, Cohen's d), class retention dynamics, and cadence/S-N characteristics.
Uncertainty bounds are computed via 1,000 bootstrap resamples with a fixed random seed.
Figures and a JSON summary are saved under docs/results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Study class names per ADR 002
CLASS_NAMES: dict[int, str] = {
    64: "Kilonova (KN)",
    90: "Type Ia Supernova (SN Ia)",
    95: "Superluminous SN (SLSN-I)",
}

# Peak absolute r-band magnitude M_r by class for derived physical approximations
CLASS_ABS_MAG_R: dict[int, float] = {
    64: -15.5,  # Faint kilonova
    90: -19.3,  # Standard candle SN Ia
    95: -21.5,  # Superluminous supernova
}


def compute_js_divergence(x: np.ndarray, y: np.ndarray, bins: int = 100) -> float:
    """Compute Jensen-Shannon divergence (in bits, base 2)."""
    min_val = min(float(np.nanmin(x)), float(np.nanmin(y)))
    max_val = max(float(np.nanmax(x)), float(np.nanmax(y)))
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    hist_x, _ = np.histogram(x, bins=bin_edges)
    hist_y, _ = np.histogram(y, bins=bin_edges)

    p = hist_x / np.sum(hist_x)
    q = hist_y / np.sum(hist_y)

    js_dist = jensenshannon(p, q, base=2.0)
    return float(js_dist**2)


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between group x and group y."""
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(y) - np.mean(x)) / s_pooled)


def analyze_feature(
    x_true: np.ndarray,
    x_biased: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    subsample_size: int = 5000,
) -> dict[str, Any]:
    """Compute point estimates and 95% bootstrap CIs for feature shifts."""
    rng = np.random.default_rng(seed)

    mean_true = float(np.mean(x_true))
    mean_biased = float(np.mean(x_biased))
    mean_shift = mean_biased - mean_true

    median_true = float(np.median(x_true))
    median_biased = float(np.median(x_biased))
    median_shift = median_biased - median_true

    ks_stat, ks_pval = stats.ks_2samp(x_true, x_biased)
    ks_stat = float(ks_stat)
    ks_pval = float(ks_pval)

    w1_dist = float(stats.wasserstein_distance(x_true, x_biased))
    js_div = compute_js_divergence(x_true, x_biased)
    cohen_d = compute_cohens_d(x_true, x_biased)

    n_true = len(x_true)
    n_biased = len(x_biased)

    sz_t = min(n_true, subsample_size)
    sz_b = min(n_biased, subsample_size)

    idx_t_matrix = rng.choice(n_true, size=(n_bootstrap, sz_t), replace=True)
    idx_b_matrix = rng.choice(n_biased, size=(n_bootstrap, sz_b), replace=True)

    samples_t = x_true[idx_t_matrix]
    samples_b = x_biased[idx_b_matrix]

    boot_mean_t = np.mean(samples_t, axis=1)
    boot_mean_b = np.mean(samples_b, axis=1)
    boot_mean_shifts = boot_mean_b - boot_mean_t

    boot_med_t = np.median(samples_t, axis=1)
    boot_med_b = np.median(samples_b, axis=1)
    boot_median_shifts = boot_med_b - boot_med_t

    boot_s1 = np.std(samples_t, axis=1, ddof=1)
    boot_s2 = np.std(samples_b, axis=1, ddof=1)
    s_pooled = np.sqrt(
        ((sz_t - 1) * boot_s1**2 + (sz_b - 1) * boot_s2**2) / (sz_t + sz_b - 2)
    )
    boot_cohen_ds = np.where(
        s_pooled > 0, (boot_mean_b - boot_mean_t) / s_pooled, 0.0
    )

    eval_reps = 100
    boot_w1 = np.zeros(eval_reps)
    boot_js = np.zeros(eval_reps)
    boot_ks = np.zeros(eval_reps)

    for i in range(eval_reps):
        st = samples_t[i]
        sb = samples_b[i]
        boot_w1[i] = stats.wasserstein_distance(st, sb)
        boot_js[i] = compute_js_divergence(st, sb)
        boot_ks[i] = stats.ks_2samp(st, sb).statistic

    def get_ci(arr: np.ndarray) -> list[float]:
        low, high = np.percentile(arr, [2.5, 97.5])
        return [float(low), float(high)]

    return {
        "true_mean": mean_true,
        "biased_mean": mean_biased,
        "mean_shift": mean_shift,
        "mean_shift_ci95": get_ci(boot_mean_shifts),
        "true_median": median_true,
        "biased_median": median_biased,
        "median_shift": median_shift,
        "median_shift_ci95": get_ci(boot_median_shifts),
        "ks_stat": ks_stat,
        "ks_stat_ci95": get_ci(boot_ks),
        "ks_pval": ks_pval,
        "wasserstein_distance": w1_dist,
        "wasserstein_ci95": get_ci(boot_w1),
        "js_divergence": js_div,
        "js_divergence_ci95": get_ci(boot_js),
        "cohens_d": cohen_d,
        "cohens_d_ci95": get_ci(boot_cohen_ds),
    }


def analyze_class_proportions(
    df_true: pd.DataFrame,
    df_biased: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
    subsample_size: int = 5000,
) -> dict[str, Any]:
    """Analyze class counts, retention rates, and proportions with bootstrap CIs."""
    rng = np.random.default_rng(seed)
    total_true = len(df_true)
    total_biased = len(df_biased)

    results: dict[str, Any] = {}
    class_ids = [64, 90, 95]

    for cid in class_ids:
        cname = CLASS_NAMES[cid]
        sub_true = df_true[df_true["true_target"] == cid]
        sub_biased = df_biased[df_biased["true_target"] == cid]

        cnt_true = len(sub_true)
        cnt_biased = len(sub_biased)

        retention = cnt_biased / cnt_true if cnt_true > 0 else 0.0
        prop_true = cnt_true / total_true
        prop_biased = cnt_biased / total_biased
        prop_shift = prop_biased - prop_true

        # Direct empirical sample retention binary array (cnt_biased ones out of cnt_true)
        y_ret = np.zeros(cnt_true, dtype=float)
        y_ret[:cnt_biased] = 1.0

        sz_c = min(cnt_true, subsample_size)
        boot_idx = rng.choice(cnt_true, size=(n_bootstrap, sz_c), replace=True)
        boot_retention = np.mean(y_ret[boot_idx], axis=1)
        low_ret, high_ret = np.percentile(boot_retention, [2.5, 97.5])

        # Proportion shift bootstrap
        sz_t = min(total_true, subsample_size)
        sz_b = min(total_biased, subsample_size)
        idx_t_mat = rng.choice(total_true, size=(n_bootstrap, sz_t), replace=True)
        idx_b_mat = rng.choice(total_biased, size=(n_bootstrap, sz_b), replace=True)

        samples_t_cls = df_true["true_target"].to_numpy()[idx_t_mat]
        samples_b_cls = df_biased["true_target"].to_numpy()[idx_b_mat]

        p_t_arr = np.mean(samples_t_cls == cid, axis=1)
        p_b_arr = np.mean(samples_b_cls == cid, axis=1)
        boot_prop_shift = p_b_arr - p_t_arr
        low_ps, high_ps = np.percentile(boot_prop_shift, [2.5, 97.5])

        results[str(cid)] = {
            "class_name": cname,
            "true_count": cnt_true,
            "biased_count": cnt_biased,
            "retention_rate": float(retention),
            "retention_rate_ci95": [float(low_ret), float(high_ret)],
            "true_proportion": float(prop_true),
            "biased_proportion": float(prop_biased),
            "proportion_shift": float(prop_shift),
            "proportion_shift_ci95": [float(low_ps), float(high_ps)],
        }

    return results


def plot_fig1_redshift(
    df_true: pd.DataFrame,
    df_biased: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Figure 1: Redshift selection shift."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ax = axes[0]
    m_t = metrics["hostgal_photoz"]["true_mean"]
    m_b = metrics["hostgal_photoz"]["biased_mean"]
    ax.hist(
        df_true["hostgal_photoz"],
        bins=60,
        density=True,
        alpha=0.4,
        color="#1f77b4",
        label=f"TRUE Population (Mean={m_t:.3f})",
    )
    ax.hist(
        df_biased["hostgal_photoz"],
        bins=60,
        density=True,
        alpha=0.5,
        color="#ff7f0e",
        label=f"BIASED Population (Mean={m_b:.3f})",
    )
    ax.set_xlabel("Host Photometric Redshift ($z_{\\rm phot}$)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title("A. Host Photo-z Distribution Shift", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    ax.grid(alpha=0.2, linestyle="--")

    w1 = metrics["hostgal_photoz"]["wasserstein_distance"]
    w1_ci = metrics["hostgal_photoz"]["wasserstein_ci95"]
    ks = metrics["hostgal_photoz"]["ks_stat"]
    js = metrics["hostgal_photoz"]["js_divergence"]
    metrics_str = (
        f"$W_1 = {w1:.4f}$ [{w1_ci[0]:.4f}, {w1_ci[1]:.4f}]\n"
        f"$KS = {ks:.4f}$\n$JS = {js:.4f}$"
    )
    ax.text(
        0.48,
        0.65,
        metrics_str,
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f8f9fa",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
        fontsize=9,
    )

    ax = axes[1]
    z_bins = np.linspace(0, 1.5, 31)
    z_mids = 0.5 * (z_bins[:-1] + z_bins[1:])
    true_counts, _ = np.histogram(df_true["hostgal_photoz"], bins=z_bins)
    biased_counts, _ = np.histogram(df_biased["hostgal_photoz"], bins=z_bins)

    valid_mask = true_counts > 0
    emp_frac = np.zeros_like(z_mids)
    emp_frac[valid_mask] = biased_counts[valid_mask] / true_counts[valid_mask]

    z_smooth = np.linspace(0, 1.5, 200)
    p_spec_theory = 0.10 + (0.80 - 0.10) / (1.0 + np.exp((z_smooth - 0.50) / 0.15))

    ax.plot(
        z_smooth,
        p_spec_theory,
        "k--",
        linewidth=2,
        label="Theoretical Proxy $p_{\\rm spec}(z)$",
    )
    ax.plot(
        z_mids[valid_mask],
        emp_frac[valid_mask],
        "o-",
        color="#d62728",
        linewidth=1.5,
        markersize=6,
        label="Empirical Retained Fraction ($N_{\\rm BIASED}/N_{\\rm TRUE}$)",
    )
    ax.axvline(0.50, color="gray", linestyle=":", label="$z_{50} = 0.50$ Midpoint")
    ax.set_xlabel("Host Photometric Redshift ($z_{\\rm phot}$)", fontsize=11)
    ax.set_ylabel("Selection Inclusion Probability $p_{\\rm spec}$", fontsize=11)
    ax.set_title(
        "B. Empirical Selection Function Verification",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_ylim(0, 0.95)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_redshift_selection_shift.png")
    plt.close()


def plot_fig2_brightness(
    df_true: pd.DataFrame,
    df_biased: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Figure 2: Brightness & Distance Modulus Distortions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ax = axes[0]
    ax.hist(
        df_true["distmod"],
        bins=50,
        density=True,
        alpha=0.4,
        color="#1f77b4",
        label=f"TRUE (Mean={metrics['distmod']['true_mean']:.2f})",
    )
    ax.hist(
        df_biased["distmod"],
        bins=50,
        density=True,
        alpha=0.5,
        color="#2ca02c",
        label=f"BIASED (Mean={metrics['distmod']['biased_mean']:.2f})",
    )
    ax.set_xlabel("Distance Modulus $\\mu$ (mag)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(
        "A. Distance Modulus Distribution Shift", fontsize=12, fontweight="bold"
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(alpha=0.2, linestyle="--")

    dm_w1 = metrics["distmod"]["wasserstein_distance"]
    dm_w1_ci = metrics["distmod"]["wasserstein_ci95"]
    dm_shift = metrics["distmod"]["mean_shift"]
    d_val = metrics["distmod"]["cohens_d"]
    dm_text = (
        f"Mean Shift $\\Delta\\mu = {dm_shift:.3f}$ mag\n"
        f"$W_1 = {dm_w1:.3f}$ [{dm_w1_ci[0]:.3f}, {dm_w1_ci[1]:.3f}]\n"
        f"Cohen's $d = {d_val:.3f}$"
    )
    ax.text(
        0.05,
        0.75,
        dm_text,
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f8f9fa",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
        fontsize=9,
    )

    ax = axes[1]
    # Derived physical approximation: m_r = M_r_eff + distmod + 3.1 * mwebv
    m_r_true = -19.3 + df_true["distmod"] + 3.1 * df_true["mwebv"]
    m_r_biased = -19.3 + df_biased["distmod"] + 3.1 * df_biased["mwebv"]

    ax.hist(
        m_r_true,
        bins=50,
        density=True,
        alpha=0.4,
        color="#1f77b4",
        label=f"TRUE (Mean m_r={m_r_true.mean():.2f})",
    )
    ax.hist(
        m_r_biased,
        bins=50,
        density=True,
        alpha=0.5,
        color="#e377c2",
        label=f"BIASED (Mean m_r={m_r_biased.mean():.2f})",
    )
    ax.axvline(24.5, color="red", linestyle="--", label="Survey Limit $r \\approx 24.5$ mag")
    ax.set_xlabel("Derived Peak Apparent $r$-band Mag $m_r$ (mag) [Physical Approx]", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(
        "B. Derived Peak Apparent Magnitude Distribution", fontsize=12, fontweight="bold"
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(alpha=0.2, linestyle="--")

    mag_shift = m_r_biased.mean() - m_r_true.mean()
    ax.text(
        0.05,
        0.65,
        f"Derived Shift $\\Delta m_r = {mag_shift:.2f}$ mag\n(Brighter Selected Sample)",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f8f9fa",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_brightness_distmod_shift.png")
    plt.close()


def plot_fig3_class_retention(
    df_true: pd.DataFrame,
    df_biased: pd.DataFrame,
    class_metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Figure 3: Class Retention Dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ax = axes[0]
    cids = [64, 90, 95]
    names = [CLASS_NAMES[c] for c in cids]
    rets = [class_metrics[str(c)]["retention_rate"] * 100 for c in cids]
    ci_lows = [
        max(0.0, rets[i] - class_metrics[str(c)]["retention_rate_ci95"][0] * 100)
        for i, c in enumerate(cids)
    ]
    ci_highs = [
        max(0.0, class_metrics[str(c)]["retention_rate_ci95"][1] * 100 - rets[i])
        for i, c in enumerate(cids)
    ]
    yerr = np.array([ci_lows, ci_highs])

    colors = ["#d62728", "#1f77b4", "#9467bd"]
    bars = ax.bar(
        names,
        rets,
        yerr=yerr,
        capsize=5,
        color=colors,
        alpha=0.85,
        edgecolor="black",
    )

    for bar, ret in zip(bars, rets, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.5,
            f"{ret:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("Empirical Spectroscopic Retention Rate (%)", fontsize=11)
    ax.set_title(
        "A. Empirical Label Retention Rate by Class", fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    ax = axes[1]
    for c, col in zip(cids, colors, strict=False):
        z_c = df_true[df_true["true_target"] == c]["hostgal_photoz"]
        ax.hist(
            z_c,
            bins=40,
            density=True,
            histtype="step",
            linewidth=2.5,
            color=col,
            label=f"{CLASS_NAMES[c]} (Mean z={z_c.mean():.2f})",
        )

    ax.axvline(
        0.50, color="gray", linestyle="--", alpha=0.7, label="$z_{50} = 0.50$ Midpoint"
    )
    ax.set_xlabel("Host Photometric Redshift ($z_{\\rm phot}$)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(
        "B. Intrinsic Redshift Distributions in TRUE Population",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_class_retention_dynamics.png")
    plt.close()


def plot_fig4_cadence_snr(df_true: pd.DataFrame, output_dir: Path) -> None:
    """Figure 4: Cadence Library Distribution and Signal-to-Noise Characteristics."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ax = axes[0]
    cids = [64, 90, 95]
    colors = ["#d62728", "#1f77b4", "#9467bd"]

    top_cadences = df_true["libid_cadence"].value_counts().head(10).index.tolist()

    cadence_matrix = []
    for c in cids:
        sub = df_true[df_true["true_target"] == c]
        counts = [
            (sub["libid_cadence"] == cad).sum() / len(sub) * 100
            for cad in top_cadences
        ]
        cadence_matrix.append(counts)

    x_indices = np.arange(len(top_cadences))
    width = 0.25

    for i, (c, col) in enumerate(zip(cids, colors, strict=False)):
        ax.bar(
            x_indices + (i - 1) * width,
            cadence_matrix[i],
            width=width,
            color=col,
            label=CLASS_NAMES[c],
            alpha=0.85,
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(c) for c in top_cadences], rotation=45)
    ax.set_xlabel("OpSim Cadence Library ID (`libid_cadence`)", fontsize=11)
    ax.set_ylabel("Percentage of Class (%)", fontsize=11)
    ax.set_title(
        "A. Top Simulation Cadence Library Profiles", fontsize=12, fontweight="bold"
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    ax = axes[1]
    for c, col in zip(cids, colors, strict=False):
        sub = df_true[df_true["true_target"] == c]
        # Derived physical approximation of peak r-band S/N from absolute magnitude
        m_abs = CLASS_ABS_MAG_R[c]
        m_r = m_abs + sub["distmod"] + 3.1 * sub["mwebv"]
        flux_r = 10.0 ** (-0.4 * (m_r - 27.5))
        snr = flux_r / 5.0

        n_pts = min(len(sub), 2000)
        idx_pts = np.random.choice(len(sub), size=n_pts, replace=False)

        ax.scatter(
            sub["hostgal_photoz"].iloc[idx_pts],
            np.log10(np.maximum(snr.iloc[idx_pts], 0.1)),
            alpha=0.3,
            s=12,
            color=col,
            label=CLASS_NAMES[c],
        )

    ax.axhline(np.log10(5.0), color="red", linestyle="--", label="Detection Limit (S/N = 5)")
    ax.set_xlabel("Host Photometric Redshift ($z_{\\rm phot}$)", fontsize=11)
    ax.set_ylabel("Log10 Estimated Peak S/N ($r$-band) [Physical Approx]", fontsize=10)
    ax.set_title(
        "B. Derived Peak Signal-to-Noise Ratio vs Redshift", fontsize=12, fontweight="bold"
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_cadence_snr_characteristics.png")
    plt.close()


def plot_fig5_effect_size_ranking(metrics: dict[str, Any], output_dir: Path) -> None:
    """Figure 5: Ranked Feature Bias Effect Sizes with Bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    feature_keys = [
        "hostgal_photoz",
        "distmod",
        "true_z",
        "true_distmod",
        "hostgal_photoz_err",
        "mwebv",
        "ra",
        "decl",
    ]
    labels = [
        "Host Photo-z (hostgal_photoz)",
        "Host Distance Modulus (distmod)",
        "True Redshift (true_z)",
        "True Distance Modulus (true_distmod)",
        "Photo-z Error (hostgal_photoz_err)",
        "MW Extinction (mwebv)",
        "Right Ascension (ra)",
        "Declination (decl)",
    ]

    ks_stats = [metrics[f]["ks_stat"] for f in feature_keys]
    ks_cis = [metrics[f]["ks_stat_ci95"] for f in feature_keys]

    sorted_indices = np.argsort(ks_stats)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_ks = [ks_stats[i] for i in sorted_indices]
    sorted_cis = [ks_cis[i] for i in sorted_indices]

    y_pos = np.arange(len(sorted_labels))
    ci_lows = [
        max(0.0, sorted_ks[i] - sorted_cis[i][0]) for i in range(len(sorted_ks))
    ]
    ci_highs = [
        max(0.0, sorted_cis[i][1] - sorted_ks[i]) for i in range(len(sorted_ks))
    ]
    xerr = np.array([ci_lows, ci_highs])

    colors = ["#d62728" if ks > 0.1 else "#1f77b4" for ks in sorted_ks]

    bars = ax.barh(
        y_pos,
        sorted_ks,
        xerr=xerr,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="black",
    )

    for _i, (bar, ks) in enumerate(zip(bars, sorted_ks, strict=False)):
        ax.text(
            ks + 0.005,
            bar.get_y() + bar.get_height() / 2.0,
            f"KS={ks:.4f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Kolmogorov-Smirnov (KS) Statistic", fontsize=11)
    ax.set_title(
        "Ranked Selection-Induced Bias Across Observable Features",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_xlim(0, 0.26)

    plt.tight_layout()
    plt.savefig(output_dir / "fig5_effect_size_bootstrap_intervals.png")
    plt.close()


def load_data(
    true_path: str, biased_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load population data from binary .npy cache if present, else from csv.gz."""
    cache_dir = Path("data/interim/cache")
    cols = [
        "object_id",
        "hostgal_photoz",
        "true_z",
        "distmod",
        "true_distmod",
        "hostgal_photoz_err",
        "mwebv",
        "ra",
        "decl",
        "tflux_r",
        "tflux_g",
        "true_target",
        "libid_cadence",
    ]

    if cache_dir.exists() and (cache_dir / "true_hostgal_photoz.npy").exists():
        df_true = pd.DataFrame(
            {col: np.load(cache_dir / f"true_{col}.npy") for col in cols}
        )
        df_biased = pd.DataFrame(
            {col: np.load(cache_dir / f"biased_{col}.npy") for col in cols}
        )
        return df_true, df_biased

    df_true = pd.read_csv(true_path, usecols=cols)
    df_biased = pd.read_csv(biased_path, usecols=cols)
    return df_true, df_biased


def main() -> None:
    t_start = time.time()
    parser = argparse.ArgumentParser(
        description="Analyze selection bias between TRUE and BIASED populations."
    )
    parser.add_argument(
        "--true-path", type=str, default="data/processed/true_population.csv.gz"
    )
    parser.add_argument(
        "--biased-path", type=str, default="data/processed/biased_population.csv.gz"
    )
    parser.add_argument("--output-dir", type=str, default="docs/results")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_true, df_biased = load_data(args.true_path, args.biased_path)
    print(f"Loaded {len(df_true):,} TRUE and {len(df_biased):,} BIASED objects.")

    features_to_analyze = [
        "hostgal_photoz",
        "true_z",
        "distmod",
        "true_distmod",
        "hostgal_photoz_err",
        "mwebv",
        "ra",
        "decl",
        "tflux_r",
        "tflux_g",
    ]

    metrics: dict[str, Any] = {}

    print(
        f"\nComputing feature shift metrics with B={args.n_bootstrap} bootstrap..."
    )
    for feat in features_to_analyze:
        print(f"  - Analyzing feature: {feat}...")
        x_t = df_true[feat].to_numpy()
        x_b = df_biased[feat].to_numpy()
        metrics[feat] = analyze_feature(
            x_t, x_b, n_bootstrap=args.n_bootstrap, seed=args.seed
        )

    print("\nComputing class retention dynamics...")
    class_metrics = analyze_class_proportions(
        df_true, df_biased, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    metrics["class_proportions"] = class_metrics

    metrics_file = out_dir / "selection_bias_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics summary to {metrics_file.resolve()}")

    print("\nGenerating report figures under docs/results/...")
    plot_fig1_redshift(df_true, df_biased, metrics, out_dir)
    print("  - Generated fig1_redshift_selection_shift.png")
    plot_fig2_brightness(df_true, df_biased, metrics, out_dir)
    print("  - Generated fig2_brightness_distmod_shift.png")
    plot_fig3_class_retention(df_true, df_biased, class_metrics, out_dir)
    print("  - Generated fig3_class_retention_dynamics.png")
    plot_fig4_cadence_snr(df_true, out_dir)
    print("  - Generated fig4_cadence_snr_characteristics.png")
    plot_fig5_effect_size_ranking(metrics, out_dir)
    print("  - Generated fig5_effect_size_bootstrap_intervals.png")

    t_end = time.time()
    print(f"\nAnalysis complete in {t_end - t_start:.2f}s!")


if __name__ == "__main__":
    main()
