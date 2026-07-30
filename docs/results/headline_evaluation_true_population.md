# Headline Comparative Evaluation Report on the TRUE Population (Step 3)

## 1. Executive Summary

This report presents the formal, comparative headline evaluation of the AEGIS early-alert triage decision policy against the **TRUE evaluation population**. We report findings across two scales for complete transparency and scientific rigor:

1. **Expanded Target Population (Primary Results):** Evaluated against a statistically powerful disjoint survey population of **$N = 55,915$ objects**, containing all **$N_{\text{KN}} = 133$ confirmed kilonovae** and $N_{\text{SLSN-I}} = 35,782$ background SLSN-I, plus a reproducible seeded sample of $N_{\text{SN Ia}} = 20,000$ Type Ia supernovae background objects.
2. **Preliminary Cohort (Historical Reference):** Evaluated against the initial $0.75\%$ evaluation slice of **$N = 12,740$ objects**, containing only **$N_{\text{KN}} = 2$ confirmed kilonovae** and $N_{\text{SLSN-I}} = 98$ background SLSN-I.

Under the identical operational capacity constraint of **$K = 5$ triggers per epoch** and primary decision deadline **$H = 2.0$ days**, three decision configurations were evaluated:
1. **Naive Fixed-Confidence Threshold Baseline:** Calibrated on the $S=1$ reference population to target the quota. (Naive threshold $\tau_{\text{naive}} = 0.979504$ for the expanded population, and $\tau_{\text{naive}} = 0.046154$ for the preliminary population).
2. **Frozen Bias-and-Novelty Policy (`configs/decision_policy_v1.yaml`):** Locked configuration ($w_{\text{nov}} = 0.05$, decision threshold $\tau = 0.001$, capacity $K = 5$).
3. **Novelty Ablation Variant:** The identical frozen policy with novelty weight forced to $w_{\text{nov}} = 0.00$ ($\tau = 0.001$, capacity $K = 5$).

---

### Key Empirical Findings:

1. **Robust Re-Measurement and High Miss Rates:** On the expanded population, all three configurations achieved an identical **Missed High-Value Event Rate of $MHVER = 94.73\%$** (only 7 out of 133 kilonovae triggered), with an exact 95% Clopper-Pearson binomial confidence interval of **$[89.46\%, 97.86\%]$**. This confirms the preliminary finding that the early triage policy misses the vast majority of target events under realistic capacities.
2. **Failure to Outperform Naive Baseline:** The fused bias-and-novelty policy **does not outperform** the naive fixed-confidence baseline on target discovery. On both the preliminary ($100\%$ missed) and expanded ($94.73\%$ missed) populations, the naive baseline, the frozen policy, and the novelty ablation variant perform identically on target discovery.
3. **Zero Contribution from Novelty Signal:** The novelty ablation ($w_{\text{nov}} = 0.00$) produced **identical target recovery performance** to the frozen policy ($w_{\text{nov}} = 0.05$) on both cohorts. The additive novelty term makes **no measurable contribution** to target event discovery.
4. **Capacity Saturation and False Triggers:** Due to sequential capacity limits ($K=5$ per epoch for 2 epochs = 10 triggers maximum), all configurations trigger a total of 10 objects. In the expanded cohort, this results in 7 kilonova triggers and 3 false positives (non-kilonova triggers) for all three arms, yielding a false trigger rate (FTR) of **$0.0054\%$** (95% CP CI: $[0.0011\%, 0.0157\%]$). Because of the sequential budget format, the policies achieve negative utility regret **$-1.0$** relative to a static 1-epoch capacity-5 oracle (oracle utility $= 10.0$, policy utility $= 11.0$).
5. **Jackknife Robustness:** Leave-One-Kilonova-Out (LOKO) jackknife re-evaluation confirms that the results are highly stable. The jackknife range of $MHVER$ is **$[94.70\%, 95.45\%]$** and the jackknife range of Normalized Regret is **$[-0.10, 0.10]$** across the 133 target configurations in the expanded population.

---

## 2. Quantitative Headline Comparison Table

### Table 1: Primary Decision Performance Across Policy Configurations ($K=5, H=2.0$d)

#### A. Expanded Target Population ($N_{\text{KN}} = 133$)

| Decision Configuration | Score Formula $S_e(x_i)$ | Decision Threshold ($\tau$) | Total Utility ($U_e$) | Oracle Utility ($U_{\text{oracle}}$) | Utility Regret ($R_e$) [95% Boot CI] | Normalized Regret | Missed Target Rate ($MHVER_e$) [95% CP CI] | False Triggers ($FP$) | False Trigger Rate ($FTR$) [95% CP CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | $p_{i, \text{KN}, e}$ | $0.979504$ (S=1 Calibrated) | **11.0** | +10.0 | **-1.0** [-13.0, 9.0] | **-0.1000** | **94.74%** [89.46%, 97.86%] | **3** | **0.0054%** [0.0011%, 0.0157%] |
| **Frozen Policy** | $p_{i, \text{KN}, e} + 0.05 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | **11.0** | +10.0 | **-1.0** [-12.02, 9.0] | **-0.1000** | **94.74%** [89.46%, 97.86%] | **3** | **0.0054%** [0.0011%, 0.0157%] |
| **Novelty Ablation** | $p_{i, \text{KN}, e} + 0.00 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | **11.0** | +10.0 | **-1.0** [-13.0, 10.0] | **-0.1000** | **94.74%** [89.46%, 97.86%] | **3** | **0.0054%** [0.0011%, 0.0157%] |

#### B. Preliminary Cohort (Historical Reference: $N_{\text{KN}} = 2$)

| Decision Configuration | Score Formula $S_e(x_i)$ | Decision Threshold ($\tau$) | Total Utility ($U_e$) | Oracle Utility ($U_{\text{oracle}}$) | Utility Regret ($R_e$) | Normalized Regret | Missed Target Rate ($MHVER_e$) [95% CP CI] | False Triggers ($FP$) | False Trigger Rate ($FTR$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | $p_{i, \text{KN}, e}$ | $0.046154$ (S=1 Calibrated) | **0.0** | +4.0 | **4.0** | **1.0000** | **100.00%** [2.50%, 100.00%] | **0** | **0.0000%** |
| **Frozen Policy** | $p_{i, \text{KN}, e} + 0.05 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | -10.0 | +4.0 | 4.0 (Net 14.0) | **1.0000** | **100.00%** [2.50%, 100.00%] | 10 | 0.0785% |
| **Novelty Ablation** | $p_{i, \text{KN}, e} + 0.00 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | -10.0 | +4.0 | 4.0 (Net 14.0) | **1.0000** | **100.00%** [2.50%, 100.00%] | 10 | 0.0785% |

> [!NOTE]
> - Under ADR 003 reference utility ($u_{\text{tp}} = +2.0, u_{\text{fp}} = -1.0$), a static oracle with capacity 5 achieves $U_{\text{oracle}} = 5 \times (+2.0) = +10.0$ in the expanded target population, and $U_{\text{oracle}} = 2 \times (+2.0) = +4.0$ in the preliminary cohort.
> - Exact 95% binomial confidence intervals are computed via Clopper-Pearson quantiles. Percentile bootstrap CIs ($B=1,000$) are reported for the expanded utility metrics.

---

## 3. Leave-One-Kilonova-Out (LOKO) Jackknife Sensitivity Analysis

### Table 2: LOKO Jackknife Metric Ranges

| Cohort | Metric | Naive Baseline | Frozen Policy | Novelty Ablation | Conclusion |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Expanded ($N_{\text{KN}}=133$)** | Regret Range | $[-1.0, 1.0]$ | $[-1.0, 1.0]$ | $[-1.0, 1.0]$ | Stable / Invariant |
| | $MHVER$ Range | $[0.9470, 0.9545]$ | $[0.9470, 0.9545]$ | $[0.9470, 0.9545]$ | Stable / Invariant |
| **Preliminary ($N_{\text{KN}}=2$)** | Regret Range | $[2.0, 2.0]$ | $[2.0, 2.0]$ | $[2.0, 2.0]$ | Stable / Invariant |
| | $MHVER$ Range | $[1.0000, 1.0000]$ | $[1.0000, 1.0000]$ | $[1.0000, 1.0000]$ | Stable / Invariant |

The LOKO jackknife results confirm that the headline findings are robust and do not depend on any single target object.

---

## 4. Physical and Methodological Diagnosis

1. **Why Supervised Confidence Fails Early:** At $e \le 2.0$ days, light curves contain at most 1--2 photometric detections. Baseline classifier probabilities for kilonovae remain tiny ($P(\text{KN}) \approx 0.0002\text{--}0.02$). Supervised confidence alone cannot separate kilonovae from background transients.
2. **Why Novelty Signal Fails to Elevate Target Objects:** While normalized novelty scores boost kilonova scores slightly, hundreds of non-target objects (e.g., high-redshift SN Ia outliers or noisy alerts) achieve much higher novelty scores. Under strict capacity limits, non-target candidates consistently outrank true kilonovae.
3. **Threshold vs. Capacity Interaction:** Sequential policies utilize capacity slots per epoch. In the expanded cohort, the naive threshold $\tau_{\text{naive}} = 0.979504$ and the low threshold $\tau = 0.001$ behave identically because capacity constraints dominate the triage behavior, capping both at 10 total triggers.

---

## 5. Conclusion and Protocol Compliance Verification

- **Hard Constraints Adhered To:** `configs/decision_policy_v1.yaml` was evaluated strictly as locked without modification. Ablations were run as isolated evaluations. No post-hoc tuning was performed after observing deployment labels.
- **Empirical Rigor:** All reported numbers derive from reproducible execution of `scripts/evaluate_true_population_headline.py` against the real repository dataset. Full per-object audits are exported to `docs/results/headline_evaluation_metrics.json`.
