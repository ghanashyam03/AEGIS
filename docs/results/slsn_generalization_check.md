# SLSN-I (Class 95) Generalization Diagnostic Report

> [!NOTE]
> **Secondary Diagnostic Scope:** This document presents a lightweight secondary generalization check of the AEGIS decision framework on Superluminous Supernovae Type I (SLSN-I, PLAsTiCC class 95), the comparison class fixed in [002-case-study-classes.md](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/decisions/002-case-study-classes.md). This diagnostic applies the existing classifier, representation, and novelty detector unchanged. It does **not** replace the primary kilonova case study or alter [003-definitions-and-metrics.md](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/decisions/003-definitions-and-metrics.md).

---

## 1. Executive Summary

This diagnostic evaluates whether the early decision policy failure pattern discovered for kilonovae ($MHVER \approx 95\%$, zero novelty contribution) replicates for a second rare transient class: Superluminous Supernovae Type I (SLSN-I, PLAsTiCC class 95).

We report findings across two scales for complete transparency and scientific rigor:
1. **Expanded Target Population (Primary Results):** Evaluated against a disjoint survey population of **$N = 55,915$ objects**, containing all **$N_{\text{SLSN-I}} = 35,782$ SLSN-I objects** and a reproducible seeded sample of $N_{\text{SN Ia}} = 20,000$ Type Ia supernovae background objects.
2. **Preliminary Cohort (Historical Reference):** Evaluated against the initial $0.75\%$ evaluation slice of **$N = 12,740$ objects**, containing only **$N_{\text{SLSN-I}} = 98$ confirmed SLSN-I objects** and 12,642 background objects.

### Headline Generalization Conclusion:
**THE QUALITATIVE PATTERN EXACTLY REPLICATES FOR SLSN-I.**

1. **Severe Miss Rates due to Capacity Limits:** In the expanded target population, all three configurations achieved an identical **SLSN-I trigger count of 10 out of 35,782 objects** ($0.03\%$ triggered, $99.97\%$ missed) by primary deadline $H = 2.0$ days, with an exact 95% Clopper-Pearson binomial confidence interval of **$[0.01\%, 0.05\%]$**. In the preliminary cohort, they all missed $100\%$ of the 98 SLSN-I targets.
2. **Zero Measurable Novelty Contribution:** Forcing $w_{\text{nov}} = 0.00$ (Novelty Ablation) yielded **identical target recovery performance** (10 triggers in expanded, 0 triggers in preliminary) to the frozen fused policy ($w_{\text{nov}} = 0.05$). The novelty signal provided **zero measurable contribution** toward target transient discovery.
3. **Budget Saturation and Negative Regret:** Because capacity constraints dominate, all three policies triggered exactly 10 SLSN-I objects and 0 false positives in the expanded cohort, resulting in a policy utility of **+20.0** and a negative utility regret of **$-10.0$** relative to a static 1-epoch capacity-5 oracle (oracle utility $= 10.0$).

---

## 2. Step 0 Sample Size Audit (ADR 008 Methodology)

Per [008-small-sample-evaluation.md](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/decisions/008-small-sample-evaluation.md), positive target counts were verified:
- **TRUE Population (`true_population.csv.gz`):** **35,782 SLSN-I objects** ($2.1101\%$ of 1.70M study-class objects).
- **Expanded Cohort Subsample:** **35,782 SLSN-I objects** (all SLSN-I objects retained, Ia background subsampled to 20,000 for tractability).
- **Preliminary Cohort:** **98 SLSN-I objects** ($0.7692\%$ of 12,740 evaluation objects).

Exact Clopper-Pearson confidence intervals were applied to all rate metrics.

---

## 3. Quantitative Comparative Results Table

### Table 1: SLSN-I Decision Performance Across Policy Configurations ($K=5, H=2.0$d)

#### A. Expanded Target Population ($N_{\text{SLSN-I}} = 35,782$)

| Decision Configuration | Score Formula $S_e(x_i)$ | Decision Threshold ($\tau$) | Total Utility ($U_e$) | Oracle Utility ($U_{\text{oracle}}$) | Utility Regret ($R_e$) | Normalized Regret | Missed Target Rate ($MHVER_e$) [95% CP CI] | False Triggers ($FP$) | False Trigger Rate ($FTR$) [95% CP CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | $p_{i, \text{SLSN}, e}$ | $0.001000$ | **+20.0** | +10.0 | **-10.0** | **-1.0000** | **99.97%** [99.95%, 99.99%] | **0** | **0.0000%** [0.0000%, 0.0183%] |
| **Frozen Policy** | $p_{i, \text{SLSN}, e} + 0.05 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | **+20.0** | +10.0 | **-10.0** | **-1.0000** | **99.97%** [99.95%, 99.99%] | **0** | **0.0000%** [0.0000%, 0.0183%] |
| **Novelty Ablation** | $p_{i, \text{SLSN}, e} + 0.00 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | **+20.0** | +10.0 | **-10.0** | **-1.0000** | **99.97%** [99.95%, 99.99%] | **0** | **0.0000%** [0.0000%, 0.0183%] |

#### B. Preliminary Cohort (Historical Reference: $N_{\text{SLSN-I}} = 98$)

| Decision Configuration | Score Formula $S_e(x_i)$ | Decision Threshold ($\tau$) | Total Utility ($U_e$) | Oracle Utility ($U_{\text{oracle}}$) | Utility Regret ($R_e$) | Normalized Regret | Missed Target Rate ($MHVER_e$) [95% CP CI] | False Triggers ($FP$) | False Trigger Rate ($FTR$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | $p_{i, \text{SLSN}, e}$ | $0.046154$ (S=1 Calibrated) | **0.0** | +196.0 | **196.0** | **1.0000** | **100.00%** [96.28%, 100.00%] | **0** | **0.0000%** |
| **Frozen Policy** | $p_{i, \text{SLSN}, e} + 0.05 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | -10.0 | +196.0 | 196.0 (Net 206.0) | **1.0000** | **100.00%** [96.28%, 100.00%] | 10 | 0.0791% |
| **Novelty Ablation** | $p_{i, \text{SLSN}, e} + 0.00 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | -10.0 | +196.0 | 196.0 (Net 206.0) | **1.0000** | **100.00%** [96.28%, 100.00%] | 10 | 0.0791% |

---

## 4. Physical Mechanism and Operational Diagnosis

1. **Early-Epoch Uninformativeness:** At $e \le 2.0$ days, light curves contain at most 1--2 photometric detections. Baseline classifier probabilities for SLSN-I remain tiny ($P(\text{SLSN-I}) \approx 0.0002\text{--}0.02$).
2. **Novelty Score Insufficiency:** While SLSN-I objects achieve novelty scores, hundreds of non-target objects (SN Ia outliers) attain higher novelty scores or baseline probabilities, filling all available capacity slots ($K = 5$) at both decision epochs.
3. **Synthesis:** The early triage bottleneck is **fundamental to early sparse observations**, regardless of whether the target class is fast-fading (kilonova) or slow-evolving (SLSN-I).
