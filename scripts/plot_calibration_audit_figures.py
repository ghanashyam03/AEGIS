"""Generate figures for True-Population Calibration Audit (Finding #1).

Reads docs/results/calibration_audit_true_population_metrics.json and creates:
1. fig6_true_pop_reliability_diagrams.png
2. fig7_brier_decomposition_by_population.png
3. fig8_classwise_ece_by_epoch.png
4. fig9_selection_induced_calibration_shift.png
5. fig10_stratified_calibration_breakdown.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CLASS_NAMES = {
    64: "Kilonova (KN)",
    90: "Type Ia Supernova (SN Ia)",
    95: "Superluminous SN (SLSN-I)",
}


def load_metrics() -> dict:
    json_path = Path("docs/results/calibration_audit_true_population_metrics.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    with open(json_path) as f:
        return json.load(f)


def plot_fig6_reliability_diagrams(metrics: dict, output_dir: Path) -> None:
    """Figure 6: Kilonova Reliability Diagrams (Equal-Width & Adaptive)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    epochs = ["0.0", "2.0", "7.0"]

    for idx, ep in enumerate(epochs):
        ax = axes[idx]
        ep_data = metrics["epochs"][ep]

        # Equal-width bins for KN (class 64)
        s1_details = list(ep_data["S1_test"]["adaptive_bin_diagram"]["64"])
        s0_details = list(ep_data["S0_deployment"]["adaptive_bin_diagram"]["64"])

        # Ideal reference line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

        if s1_details:
            x_s1 = [b["mean_pred"] for b in s1_details]
            y_s1 = [b["mean_obs"] for b in s1_details]
            ece_s1 = ep_data["S1_test"]["class_ece"]["64"] * 100
            ax.plot(
                x_s1,
                y_s1,
                "o-",
                color="#1f77b4",
                linewidth=2,
                markersize=6,
                label=f"S=1 Test Set (ECE={ece_s1:.1f}%)",
            )

        if s0_details:
            x_s0 = [b["mean_pred"] for b in s0_details]
            y_s0 = [b["mean_obs"] for b in s0_details]
            ece_s0 = ep_data["S0_deployment"]["class_ece"]["64"] * 100
            ax.plot(
                x_s0,
                y_s0,
                "s-",
                color="#d62728",
                linewidth=2,
                markersize=6,
                label=f"S=0 Deployment (ECE={ece_s0:.1f}%)",
            )

        ax.set_xlabel("Mean Predicted Probability $P(\\text{KN})$", fontsize=10)
        ax.set_ylabel("Observed Kilonova Fraction", fontsize=10)
        ax.set_title(
            f"Decision Epoch $e = {float(ep):.0f}$d",
            fontsize=12,
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
    plt.savefig(output_dir / "fig6_true_pop_reliability_diagrams.png")
    plt.close()


def plot_fig7_brier_decomposition(metrics: dict, output_dir: Path) -> None:
    """Figure 7: Murphy Brier Score Decomposition by Population and Epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    epochs = ["0.0", "2.0", "7.0"]
    pop_keys = ["S1_test", "S0_deployment", "FULL_TRUE"]
    pop_labels = [
        "S=1 Test Set",
        "S=0 Deployment Set",
        "FULL TRUE Population",
    ]
    pop_colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for idx, ep in enumerate(epochs):
        ax = axes[idx]
        ep_data = metrics["epochs"][ep]

        rel_vals = [ep_data[pk]["reliability"] for pk in pop_keys]
        bs_vals = [ep_data[pk]["brier_score"] for pk in pop_keys]

        rel_cis = [ep_data[pk]["reliability_ci95"] for pk in pop_keys]
        rel_err = np.array(
            [
                [max(0.0, rel_vals[i] - rel_cis[i][0]) for i in range(3)],
                [max(0.0, rel_cis[i][1] - rel_vals[i]) for i in range(3)],
            ]
        )

        x_pos = np.arange(3)
        width = 0.5

        bars = ax.bar(
            x_pos,
            rel_vals,
            yerr=rel_err,
            capsize=4,
            width=width,
            color=pop_colors,
            alpha=0.85,
            edgecolor="black",
        )

        for i, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"BS={bs_vals[i]:.3f}\nREL={rel_vals[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(pop_labels, fontsize=9.5)
        ax.set_ylabel("Reliability Term $REL$ (Calibration Error)", fontsize=10)
        ax.set_title(
            f"Murphy Decomposition (Epoch $e = {float(ep):.0f}$d)",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.set_ylim(0, max(rel_vals) * 1.35)

    plt.tight_layout()
    plt.savefig(output_dir / "fig7_brier_decomposition_by_population.png")
    plt.close()


def plot_fig8_classwise_ece(metrics: dict, output_dir: Path) -> None:
    """Figure 8: Classwise ECE across Epochs and Study Classes."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    epochs = [0.0, 2.0, 7.0]

    pop_styles = {
        "S1_test": ("#1f77b4", "o-", "S=1 Test Set"),
        "S0_deployment": ("#d62728", "s-", "S=0 Deployment Set"),
        "FULL_TRUE": ("#2ca02c", "^--", "FULL TRUE Population"),
    }

    for pop_key, (color, fmt, label) in pop_styles.items():
        ece_means = [
            metrics["epochs"][str(ep)][pop_key]["mean_ece"] * 100 for ep in epochs
        ]
        ece_cis = [
            metrics["epochs"][str(ep)][pop_key]["mean_ece_ci95"] for ep in epochs
        ]

        yerr_low = [max(0.0, ece_means[i] - ece_cis[i][0] * 100) for i in range(3)]
        yerr_high = [max(0.0, ece_cis[i][1] * 100 - ece_means[i]) for i in range(3)]
        yerr = np.array([yerr_low, yerr_high])

        ax.errorbar(
            epochs,
            ece_means,
            yerr=yerr,
            fmt=fmt,
            color=color,
            linewidth=2,
            markersize=7,
            capsize=4,
            label=label,
        )

    ax.set_xlabel("Elapsed Observer-Frame Decision Epoch $e$ (days)", fontsize=11)
    ax.set_ylabel("Mean Classwise ECE (%)", fontsize=11)
    ax.set_title(
        "Mean Expected Calibration Error Across Decision Epochs",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(epochs)
    ax.set_xticklabels(
        ["$e=0$d\n(Alert)", "$e=2$d\n(Deadline $H=2$d)", "$e=7$d\n(Diagnostic)"]
    )
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    ax.set_ylim(0, max(ece_means) * 1.4 if ece_means else 35)

    plt.tight_layout()
    plt.savefig(output_dir / "fig8_classwise_ece_by_epoch.png")
    plt.close()


def plot_fig9_selection_shift(metrics: dict, output_dir: Path) -> None:
    """Figure 9: Selection-Induced Calibration Shift ($S=1$ vs $S=0$)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    epochs = [0.0, 2.0, 7.0]

    # Panel A: Total Multiclass Brier Score Comparison
    ax = axes[0]
    bs_s1 = [metrics["epochs"][str(ep)]["S1_test"]["brier_score"] for ep in epochs]
    bs_s0 = [
        metrics["epochs"][str(ep)]["S0_deployment"]["brier_score"] for ep in epochs
    ]
    bs_full = [metrics["epochs"][str(ep)]["FULL_TRUE"]["brier_score"] for ep in epochs]

    ax.plot(
        epochs,
        bs_s1,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        label="S=1 Test Set (Biased)",
    )
    ax.plot(
        epochs,
        bs_s0,
        "s-",
        color="#d62728",
        linewidth=2.2,
        label="S=0 Deployment Set (Unselected)",
    )
    ax.plot(
        epochs,
        bs_full,
        "^--",
        color="#2ca02c",
        linewidth=2.2,
        label="FULL TRUE Population",
    )

    for i, ep in enumerate(epochs):
        ratio = bs_s0[i] / bs_s1[i] if bs_s1[i] > 0 else 1.0
        ax.text(
            ep,
            bs_s0[i] + 0.02,
            f"{ratio:.1f}x Degradation",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
        )

    ax.set_xlabel("Decision Epoch $e$ (days)", fontsize=11)
    ax.set_ylabel("Multiclass Brier Score $BS$", fontsize=11)
    ax.set_title(
        "A. Brier Score Shift Across Populations",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(epochs)
    ax.set_xticklabels(["e=0d", "e=2d", "e=7d"])
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    # Panel B: Reliability Term REL Comparison
    ax = axes[1]
    rel_s1 = [metrics["epochs"][str(ep)]["S1_test"]["reliability"] for ep in epochs]
    rel_s0 = [
        metrics["epochs"][str(ep)]["S0_deployment"]["reliability"] for ep in epochs
    ]
    rel_full = [metrics["epochs"][str(ep)]["FULL_TRUE"]["reliability"] for ep in epochs]

    ax.plot(
        epochs,
        rel_s1,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        label="S=1 Test Set (Biased)",
    )
    ax.plot(
        epochs,
        rel_s0,
        "s-",
        color="#d62728",
        linewidth=2.2,
        label="S=0 Deployment Set (Unselected)",
    )
    ax.plot(
        epochs,
        rel_full,
        "^--",
        color="#2ca02c",
        linewidth=2.2,
        label="FULL TRUE Population",
    )

    for i, ep in enumerate(epochs):
        ratio = rel_s0[i] / rel_s1[i] if rel_s1[i] > 0 else 1.0
        ax.text(
            ep,
            rel_s0[i] + 0.02,
            f"{ratio:.1f}x REL Shift",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
        )

    ax.set_xlabel("Decision Epoch $e$ (days)", fontsize=11)
    ax.set_ylabel("Murphy Reliability $REL$", fontsize=11)
    ax.set_title(
        "B. Calibration Error Shift ($REL$ Term)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(epochs)
    ax.set_xticklabels(["e=0d", "e=2d", "e=7d"])
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    plt.savefig(output_dir / "fig9_selection_induced_calibration_shift.png")
    plt.close()


def plot_fig10_stratified_breakdown(metrics: dict, output_dir: Path) -> None:
    """Figure 10: Stratified Calibration Metrics (Brightness, Redshift, Deciles)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    e2_data = metrics["epochs"]["2.0"]["FULL_TRUE"]["strata"]

    # Panel A: Probability Deciles
    ax = axes[0]
    p_strata = e2_data["prob_deciles_kn"]
    dec_labels = list(p_strata.keys())
    dec_eces = [p_strata[dk]["mean_ece"] * 100 for dk in dec_labels]
    dec_tags = [p_strata[dk]["stratum_tag"] for dk in dec_labels]

    x_pos = np.arange(len(dec_labels))
    colors = ["#1f77b4" if tag == "standard" else "#ff7f0e" for tag in dec_tags]

    ax.bar(x_pos, dec_eces, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"D{i + 1}" for i in range(len(dec_labels))], fontsize=8.5)
    ax.set_xlabel("Predicted Probability Decile $P(\\text{KN})$", fontsize=10)
    ax.set_ylabel("Mean ECE (%)", fontsize=10)
    ax.set_title("A. Probability Decile Stratification", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Panel B: Brightness Quintiles
    ax = axes[1]
    m_strata = e2_data["brightness_quintiles_m_r"]
    m_labels = list(m_strata.keys())
    m_eces = [m_strata[mk]["mean_ece"] * 100 for mk in m_labels]

    x_pos = np.arange(len(m_labels))
    ax.bar(x_pos, m_eces, color="#9467bd", alpha=0.85, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"Q{i + 1} (Bright->Faint)" for i in range(len(m_labels))],
        fontsize=8,
        rotation=20,
    )
    ax.set_xlabel("Derived Peak Apparent Mag $m_r$ Quintile [Approx]", fontsize=9.5)
    ax.set_ylabel("Mean ECE (%)", fontsize=10)
    ax.set_title(
        "B. Brightness Quintile Stratification", fontsize=11, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Panel C: Redshift Quintiles
    ax = axes[2]
    z_strata = e2_data["redshift_quintiles_photoz"]
    z_labels = list(z_strata.keys())
    z_eces = [z_strata[zk]["mean_ece"] * 100 for zk in z_labels]

    x_pos = np.arange(len(z_labels))
    ax.bar(x_pos, z_eces, color="#e377c2", alpha=0.85, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"Q{i + 1} (Low->High z)" for i in range(len(z_labels))],
        fontsize=8,
        rotation=20,
    )
    ax.set_xlabel("Host Photo-$z$ Quintile (`hostgal_photoz`) [Catalog]", fontsize=9.5)
    ax.set_ylabel("Mean ECE (%)", fontsize=10)
    ax.set_title("C. Redshift Quintile Stratification", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "fig10_stratified_calibration_breakdown.png")
    plt.close()


def main() -> None:
    print(
        "=== Generating Figures for True-Population Calibration Audit ===", flush=True
    )
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()

    print("Generating Figure 6: Reliability Diagrams...", flush=True)
    plot_fig6_reliability_diagrams(metrics, out_dir)
    print("  -> Saved docs/results/fig6_true_pop_reliability_diagrams.png")

    print("Generating Figure 7: Brier Score Decomposition...", flush=True)
    plot_fig7_brier_decomposition(metrics, out_dir)
    print("  -> Saved docs/results/fig7_brier_decomposition_by_population.png")

    print("Generating Figure 8: Classwise ECE by Epoch...", flush=True)
    plot_fig8_classwise_ece(metrics, out_dir)
    print("  -> Saved docs/results/fig8_classwise_ece_by_epoch.png")

    print("Generating Figure 9: Selection-Induced Calibration Shift...", flush=True)
    plot_fig9_selection_shift(metrics, out_dir)
    print("  -> Saved docs/results/fig9_selection_induced_calibration_shift.png")

    print("Generating Figure 10: Stratified Calibration Breakdown...", flush=True)
    plot_fig10_stratified_breakdown(metrics, out_dir)
    print("  -> Saved docs/results/fig10_stratified_calibration_breakdown.png")

    print("All figures successfully generated!", flush=True)


if __name__ == "__main__":
    main()
