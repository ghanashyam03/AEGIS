# SLSN-I (Class 95) Generalization Diagnostic Report

> [!NOTE]
> **Secondary Diagnostic Scope:** This document presents a lightweight secondary generalization check of the AEGIS decision framework on Superluminous Supernovae Type I (SLSN-I, PLAsTiCC class 95), the comparison class fixed in [002-case-study-classes.md](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/decisions/002-case-study-classes.md). This diagnostic applies the existing classifier, representation, and novelty detector unchanged. It does **not** replace the primary kilonova case study or alter [003-definitions-and-metrics.md](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/decisions/003-definitions-and-metrics.md).

---

## 1. Executive Summary

This diagnostic evaluates whether the early decision policy failure pattern discovered for kilonovae ($MHVER = 1.0000$, zero novelty contribution) in [headline_evaluation_true_population.md](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/headline_evaluation_true_population.md) **replicates** for a second rare transient class: Superluminous Supernovae Type I (SLSN-I, PLAsTiCC class 95).

### Headline Generalization Conclusion:
**THE QUALITATIVE PATTERN EXACTLY REPLICATES FOR SLSN-I.**

1. **Failure to Outperform Naive Baseline:** On the TRUE evaluation population ($N = 12,740$), all three decision policy configurations achieved an identical **Missed High-Value Event Rate of $MHVER = 1.0000$ ($100\%$ of 98 SLSN-I targets missed)** by primary deadline $H = 2.0$ days, with an exact 95% Clopper-Pearson binomial confidence interval of **$[0.9628, 1.0000]$**.
2. **Zero Measurable Novelty Contribution:** Forcing $w_{\text{nov}} = 0.00$ (Novelty Ablation) yielded **identical target recovery performance** ($MHVER = 1.0000$, 0/98 triggered) to the frozen fused policy ($w_{\text{nov}} = 0.05$). The novelty signal provided **zero measurable contribution** toward target transient discovery.
3. **False Alarm Superiority of Naive Baseline:** The Naive Fixed-Confidence Baseline ($\tau_{\text{naive}} = 0.046154$ calibrated on $S=1$) produced **0 false triggers** ($FP = 0$, $FTR = 0.0000\%$, Utility Regret $R_e = 196.0$), whereas both the Frozen Policy and Novelty Ablation triggered **10 false alerts** ($FP = 10$, $FTR = 0.0791\%$, Utility = -10.0, Regret $R_e = 196.0$ / Net Regret = 206.0).

---

## 2. Step 0 Sample Size Audit (ADR 008 Methodology)

Per [008-small-sample-evaluation.md](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/decisions/008-small-sample-evaluation.md), positive target counts were verified prior to running evaluation:

- **Raw PLAsTiCC Test Metadata:** **35,782 SLSN-I objects** ($1.0244\%$ of 3.49M raw objects).
- **TRUE Population (`true_population.csv.gz`):** **35,782 SLSN-I objects** ($2.1101\%$ of 1.70M study-class objects).
- **Evaluation Cohort (`plasticc_test_lightcurves_01.csv.gz` slice):** **98 SLSN-I objects** ($0.7692\%$ of 12,740 evaluation objects).

While $N_{\text{SLSN}} = 98$ provides $49\times$ more positive objects than the $N_{\text{KN}} = 2$ kilonova sample, SLSN-I remains a minor class ($0.77\%$) in the evaluation stream. Exact Clopper-Pearson confidence intervals and Leave-One-Out Jackknife bounds were applied per ADR 008.

---

## 3. Quantitative Comparative Results Table

### Table 1: SLSN-I Decision Performance Across Policy Configurations ($K=5, H=2.0$d)

| Decision Configuration | Score Formula $S_e(x_i)$ | Decision Threshold ($\tau$) | Total Utility ($U_e$) | Oracle Utility ($U_{\text{oracle}}$) | Utility Regret ($R_e$) | Normalized Regret | Missed Target Rate ($MHVER_e$) [95% CP CI] | False Triggers ($FP$) | False Trigger Rate ($FTR$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Fixed-Confidence Baseline** | $p_{i, \text{SLSN}, e}$ | $0.046154$ (S=1 Calibrated) | **0.0** | +196.0 | **196.0** | **1.0000** | **1.0000** [0.9628, 1.0000] (0/98) | **0** | **0.0000%** |
| **Frozen Bias-and-Novelty Policy (`v1.0.0-frozen`)** | $p_{i, \text{SLSN}, e} + 0.05 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | -10.0 | +196.0 | 196.0 (Net 206.0) | **1.0000** | **1.0000** [0.9628, 1.0000] (0/98) | 10 | 0.0791% |
| **Novelty Ablation ($w_{\text{nov}} = 0.00$)** | $p_{i, \text{SLSN}, e} + 0.00 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | -10.0 | +196.0 | 196.0 (Net 206.0) | **1.0000** | **1.0000** [0.9628, 1.0000] (0/98) | 10 | 0.0791% |

---

## 4. Leave-One-Out Jackknife Analysis Across $N = 98$ Objects

Recomputing headline metrics $N_{\text{SLSN}} = 98$ times, omitting one SLSN-I object per iteration:

- **Jackknife $MHVER$ Range:** $[1.0000, 1.0000]$ (97/97 missed across all iterations).
- **Jackknife Regret Range:** $[194.0, 194.0]$ across all iterations.
- **Conclusion:** The findings are 100% invariant across all 98 target objects.

---

## 5. Physical Mechanism and Operational Diagnosis

1. **Early-Epoch Uninformativeness:** At $e \le 2.0$ days, light curves contain at most 1--2 photometric detections ($N_{\text{det}} \le 2$). Baseline classifier probabilities for SLSN-I remain tiny ($P(\text{SLSN-I}) \approx 0.0077\text{--}0.0152$).
2. **Novelty Score Insufficiency:** While SLSN-I objects achieve normalized novelty scores $\mathcal{N}_e \approx 0.77\text{--}0.88$, hundreds of non-target objects (SN Ia outliers) attain higher novelty scores ($\mathcal{N}_e > 1.5$) or baseline probabilities, filling all available capacity slots ($K = 5$) at both decision epochs.
3. **Synthesis:** The early triage bottleneck is **fundamental to early sparse observations**, regardless of whether the target class is fast-fading (kilonova) or slow-evolving (SLSN-I).
