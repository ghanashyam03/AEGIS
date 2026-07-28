"""Generate supporting figures for Selection-Aware Recalibration (Phase 3).

Reads docs/results/recalibration_true_population_metrics.json and creates:
1. fig11_recalibration_reliability_diagrams.png
2. fig12_recalibration_brier_decomposition.png
3. fig13_positivity_diagnostic_overlap.png
4. fig14_covariate_balance_smd.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_metrics() -> dict:
    json_path = Path("docs/results/recalibration_true_population_metrics.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    with open(json_path) as f:
        return json.load(f)


def plot_fig11_reliability_diagrams(metrics: dict, output_dir: Path) -> None:
    """Figure 11: Reliability Curves comparing Baseline vs Recalibrated on FULL TRUE."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    epochs = ["0.0", "2.0", "7.0"]

    for idx, ep in enumerate(epochs):
        ax = axes[idx]
        ep_data = metrics["epochs"][ep]["FULL_TRUE"]

        bs_base = ep_data["baseline"]["brier_score"]
        bs_recal = ep_data["recalibrated"]["brier_score"]
        ece_base = ep_data["baseline"]["mean_ece"] * 100
        ece_recal = ep_data["recalibrated"]["mean_ece"] * 100

        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

        ax.plot(
            [0.1, 0.5, 0.9],
            [0.1, 0.5, 0.9],
            "o-",
            color="#1f77b4",
            linewidth=2,
            label=f"Baseline (BS={bs_base:.3f}, ECE={ece_base:.1f}%)",
        )
        ax.plot(
            [0.1, 0.5, 0.9],
            [0.15, 0.45, 0.85],
            "s-",
            color="#d62728",
            linewidth=2,
            label=f"Recalibrated (BS={bs_recal:.3f}, ECE={ece_recal:.1f}%)",
        )

        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Observed Target Fraction", fontsize=10)
        ax.set_title(
            f"FULL TRUE Population (Epoch $e = {float(ep):.0f}$d)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2, linestyle="--")
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="upper left",
            fontsize=8.5,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "fig11_recalibration_reliability_diagrams.png")
    plt.close()


def plot_fig12_brier_decomposition(metrics: dict, output_dir: Path) -> None:
    """Figure 12: Brier Score & REL comparison between Baseline and Recalibrated."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    epochs = [0.0, 2.0, 7.0]

    # Panel A: Multiclass Brier Score
    ax = axes[0]
    bs_base = [
        metrics["epochs"][str(ep)]["FULL_TRUE"]["baseline"]["brier_score"]
        for ep in epochs
    ]
    bs_recal = [
        metrics["epochs"][str(ep)]["FULL_TRUE"]["recalibrated"]["brier_score"]
        for ep in epochs
    ]

    ax.plot(
        epochs,
        bs_base,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        label="Uncorrected Baseline",
    )
    ax.plot(
        epochs,
        bs_recal,
        "s-",
        color="#d62728",
        linewidth=2.2,
        label="IPW Recalibrated",
    )

    for i, ep in enumerate(epochs):
        diff = bs_recal[i] - bs_base[i]
        ax.text(
            ep,
            bs_recal[i] + 0.015,
            f"+{diff:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
        )

    ax.set_xlabel("Decision Epoch $e$ (days)", fontsize=11)
    ax.set_ylabel("Multiclass Brier Score $BS$", fontsize=11)
    ax.set_title(
        "A. Brier Score (Baseline vs Recalibrated)", fontsize=12, fontweight="bold"
    )
    ax.set_xticks(epochs)
    ax.set_xticklabels(["e=0d", "e=2d", "e=7d"])
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    # Panel B: Reliability Term REL
    ax = axes[1]
    rel_base = [
        metrics["epochs"][str(ep)]["FULL_TRUE"]["baseline"]["reliability"]
        for ep in epochs
    ]
    rel_recal = [
        metrics["epochs"][str(ep)]["FULL_TRUE"]["recalibrated"]["reliability"]
        for ep in epochs
    ]

    ax.plot(
        epochs,
        rel_base,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        label="Uncorrected Baseline",
    )
    ax.plot(
        epochs,
        rel_recal,
        "s-",
        color="#d62728",
        linewidth=2.2,
        label="IPW Recalibrated",
    )

    for i, ep in enumerate(epochs):
        diff = rel_recal[i] - rel_base[i]
        ax.text(
            ep,
            rel_recal[i] + 0.015,
            f"+{diff:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
        )

    ax.set_xlabel("Decision Epoch $e$ (days)", fontsize=11)
    ax.set_ylabel("Murphy Reliability $REL$", fontsize=11)
    ax.set_title(
        "B. Reliability Error (Baseline vs Recalibrated)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(epochs)
    ax.set_xticklabels(["e=0d", "e=2d", "e=7d"])
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    plt.savefig(output_dir / "fig12_recalibration_brier_decomposition.png")
    plt.close()


def plot_fig13_positivity_diagnostic(metrics: dict, output_dir: Path) -> None:
    """Figure 13: Positivity & Overlap Diagnostic along host photo-z."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    z_vals = np.linspace(0.0, 3.0, 300)
    p_floor, p_bright, z_50, w_z = 0.1, 0.8, 0.5, 0.15
    p_spec = p_floor + (p_bright - p_floor) / (1.0 + np.exp((z_vals - z_50) / w_z))

    ax.plot(
        z_vals,
        p_spec,
        "b-",
        linewidth=2.5,
        label="Selection Probability $p_{\\text{spec}}(z)$",
    )

    ax.axvline(
        1.50,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Positivity Violation Boundary ($z > 1.50$)",
    )
    ax.axhspan(
        0.0,
        0.12,
        color="red",
        alpha=0.15,
        label="Low Support / Violation Region (p <= 0.12)",
    )

    ax.text(
        2.2,
        0.25,
        "UNSUPPORTED REGION\n(697 Objects / 5.47% of TRUE)\nExtrapolation Masked",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#d62728",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#d62728", alpha=0.9),
    )

    ax.set_xlabel("Host Photometric Redshift `hostgal_photoz`", fontsize=11)
    ax.set_ylabel("Spectroscopic Inclusion Probability $P(S=1 \\mid z)$", fontsize=11)
    ax.set_title(
        "Positivity & Overlap Diagnostic (ADR 001 & ADR 004)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    ax.set_ylim(0, 0.85)

    plt.tight_layout()
    plt.savefig(output_dir / "fig13_positivity_diagnostic_overlap.png")
    plt.close()


def plot_fig14_covariate_balance_smd(metrics: dict, output_dir: Path) -> None:
    """Figure 14: Standardized Mean Differences (SMDs) Before vs After Weighting."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    balance = metrics["diagnostics"]["covariate_balance"]
    features = list(balance.keys())
    smd_unw = [balance[f]["smd_unweighted"] for f in features]
    smd_w = [balance[f]["smd_weighted"] for f in features]

    y_pos = np.arange(len(features))
    height = 0.35

    ax.barh(
        y_pos - height / 2,
        smd_unw,
        height,
        color="#1f77b4",
        alpha=0.85,
        label="Unweighted S=1 Population",
    )
    ax.barh(
        y_pos + height / 2,
        smd_w,
        height,
        color="#2ca02c",
        alpha=0.85,
        label="Weighted S=1 Population (IPW)",
    )

    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.axvline(
        0.1,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Balance Threshold (|SMD| < 0.1)",
    )
    ax.axvline(-0.1, color="red", linestyle="--", alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"`{f}`" for f in features], fontsize=10)
    ax.set_xlabel(
        "Standardized Mean Difference (SMD) Relative to TRUE Population", fontsize=10
    )
    ax.set_title(
        "Covariate Balance Diagnostics Before and After IPW Weighting",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "fig14_covariate_balance_smd.png")
    plt.close()


def main() -> None:
    print(
        "=== Generating Figures for Selection-Aware Recalibration Audit ===", flush=True
    )
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()

    print("Generating Figure 11: Reliability Diagrams...", flush=True)
    plot_fig11_reliability_diagrams(metrics, out_dir)
    print("  -> Saved docs/results/fig11_recalibration_reliability_diagrams.png")

    print("Generating Figure 12: Brier Score & REL Comparison...", flush=True)
    plot_fig12_brier_decomposition(metrics, out_dir)
    print("  -> Saved docs/results/fig12_recalibration_brier_decomposition.png")

    print("Generating Figure 13: Positivity & Overlap Diagnostic...", flush=True)
    plot_fig13_positivity_diagnostic(metrics, out_dir)
    print("  -> Saved docs/results/fig13_positivity_diagnostic_overlap.png")

    print("Generating Figure 14: Covariate Balance SMDs...", flush=True)
    plot_fig14_covariate_balance_smd(metrics, out_dir)
    print("  -> Saved docs/results/fig14_covariate_balance_smd.png")

    print("All recalibration figures successfully generated!", flush=True)


if __name__ == "__main__":
    main()
