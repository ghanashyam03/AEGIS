# Headline Comparative Evaluation Report on the TRUE Population (Step 3)

## 1. Executive Summary

This report presents the formal, comparative headline evaluation of the AEGIS early-alert triage decision policy against the **FULL TRUE evaluation population** ($N = 12,740$, containing $N_{\text{KN}} = 2$ confirmed kilonovae; base rate $P(Y=64) = 0.000157$).

Following the pre-registered protocol (ADR 003, ADR 007) and the small-sample evaluation methodology established in **ADR 008**, three decision configurations were evaluated under the identical operational capacity constraint of **$K = 5$ triggers per epoch** and primary decision deadline **$H = 2.0$ days**:

1. **Naive Fixed-Confidence Threshold Baseline:** Uses only baseline classifier confidence $p_{i, \text{KN}, e}$ (no novelty term). Threshold $\tau_{\text{naive}} = 0.046154$ was calibrated on the $S=1$ reference population to match the operational target trigger quota ($K/N_{\text{eval}} = 5/12,740 \approx 0.039\%$) without peeking at TRUE deployment labels.
2. **Frozen Bias-and-Novelty Policy (`configs/decision_policy_v1.yaml`):** Pre-registered and locked configuration ($w_{\text{nov}} = 0.05$, decision threshold $\tau = 0.001$, capacity $K = 5$).
3. **Novelty Ablation Variant:** The identical frozen policy with novelty weight forced to $w_{\text{nov}} = 0.00$ ($\tau = 0.001$, capacity $K = 5$).

---

### Key Empirical Findings:

1. **Failure to Outperform Naive Baseline:** The fused bias-and-novelty policy **does not outperform** the naive fixed-confidence baseline on target discovery. All three configurations achieved an identical **Missed High-Value Event Rate of $MHVER = 1.0000$ ($100\%$ missed)**, with an exact 95% Clopper-Pearson binomial confidence interval of **$[0.0250, 1.0000]$**.
2. **Zero Measurable Contribution from Novelty Signal:** The novelty ablation ($w_{\text{nov}} = 0.00$) produced **identical target recovery performance** ($MHVER = 1.0000$, 0 out of 2 kilonovae triggered) to the frozen policy ($w_{\text{nov}} = 0.05$). The additive novelty term made **no measurable contribution** to target event discovery on the TRUE deployment population.
3. **False Alarm Superiority of Naive Baseline:** The Naive Fixed-Confidence Baseline produced **0 false triggers** ($FP = 0$, $FTR = 0.0000\%$), preserving follow-up resources. In contrast, both the Frozen Policy and Novelty Ablation triggered **10 false positive alerts** ($FP = 10$, $FTR = 0.0785\%$) across the two decision epochs ($e \in \{0.0, 2.0\}$ days) due to low score thresholding ($\tau = 0.001$) admitting non-target photometric noise.
4. **Jackknife Robustness:** Leave-One-Kilonova-Out (LOKO) jackknife re-evaluation confirms that $MHVER = 1.0000$ and Normalized Regret $= 1.0000$ hold universally regardless of which positive kilonova object is excluded.

---

## 2. Quantitative Headline Comparison Table

### Table 1: Primary Decision Performance Across Policy Configurations ($K=5, H=2.0$d)

| Decision Configuration | Score Formula $S_e(x_i)$ | Decision Threshold ($\tau$) | Total Utility ($U_e$) | Oracle Utility ($U_{\text{oracle}}$) | Utility Regret ($R_e$) | Normalized Regret | Missed Target Rate ($MHVER_e$) [95% CP CI] | False Triggers ($FP$) | False Trigger Rate ($FTR$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Fixed-Confidence Baseline** | $p_{i, \text{KN}, e}$ | $0.046154$ (S=1 Calibrated) | **0.0** | +4.0 | **4.0** | **1.0000** | **1.0000** [0.0250, 1.0000] | **0** | **0.0000%** |
| **Frozen Policy (`v1.0.0-frozen`)** | $p_{i, \text{KN}, e} + 0.05 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | -10.0 | +4.0 | 4.0 (Net 14.0) | **1.0000** | **1.0000** [0.0250, 1.0000] | 10 | 0.0785% |
| **Novelty Ablation ($w_{\text{nov}} = 0.00$)** | $p_{i, \text{KN}, e} + 0.00 \cdot \mathcal{N}_{\text{norm}}$ | $0.001000$ (Frozen) | -10.0 | +4.0 | 4.0 (Net 14.0) | **1.0000** | **1.0000** [0.0250, 1.0000] | 10 | 0.0785% |

> [!NOTE]
> - Under ADR 003 reference utility ($u_{\text{tp}} = +2.0, u_{\text{fp}} = -1.0$), the oracle achieves $U_{\text{oracle}} = 2 \times (+2.0) = +4.0$. No-trigger utility is $U_{\text{no-trigger}} = 0.0$.
> - Exact 95% binomial confidence intervals are computed via Clopper-Pearson quantiles per ADR 008.

---

## 3. Leave-One-Kilonova-Out (LOKO) Jackknife Sensitivity Analysis

Per ADR 008, headline results were recomputed $N_{\text{KN}} = 2$ times, omitting one positive kilonova object in each iteration to measure dependency on individual targets.

### Table 2: LOKO Jackknife Metric Ranges

| Evaluation Iteration | Excluded Kilonova Object | Naive Baseline Regret | Frozen Policy Regret | Novelty Ablation Regret | Jackknife $MHVER$ | Conclusion Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Iteration 1** | Object `489518` ($z_{\text{phot}}=0.3807$) | 2.0 | 2.0 | 2.0 | 1.0000 (1/1) | Invariant |
| **Iteration 2** | Object `490807` ($z_{\text{phot}}=0.1388$) | 2.0 | 2.0 | 2.0 | 1.0000 (1/1) | Invariant |
| **Full Sample ($N=2$)** | None (Full Cohort) | **4.0** | **4.0** | **4.0** | **1.0000** (2/2) | **Robust** |

The LOKO jackknife range for Normalized Regret is **$[1.0000, 1.0000]$** and for $MHVER$ is **$[1.0000, 1.0000]$** across all configurations. The headline findings do not depend on any single target object.

---

## 4. Granular Per-Object Kilonova Decision Audit

The table below itemizes the exact decision trace and candidate score evolution for each confirmed kilonova object in the TRUE evaluation population.

### Table 3: Individual Decision Trace for Confirmed Kilonovae ($N_{\text{KN}} = 2$)

| Object ID | Host $z_{\text{phot}}$ | True $z_{\text{true}}$ | Decision Epoch ($e$) | Baseline $P(\text{KN})$ | Novelty Score $\mathcal{N}_e$ | Naive Baseline Score | Frozen Policy Score ($w=0.05$) | Ablation Score ($w=0.00$) | Naive Triggered? | Frozen Triggered? | Ablation Triggered? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`489518`** | 0.3807 | 0.4287 | $e = 0.0$d | 0.015152 | 0.77124 | 0.015152 | 0.060505 | 0.015152 | **No** | **No** | **No** |
| **`489518`** | 0.3807 | 0.4287 | $e = 2.0$d | 0.007692 | 0.73507 | 0.007692 | 0.046426 | 0.007692 | **No** | **No** | **No** |
| **`490807`** | 0.1388 | 0.1345 | $e = 0.0$d | 0.015152 | 0.67664 | 0.015152 | 0.054959 | 0.015152 | **No** | **No** | **No** |
| **`490807`** | 0.1388 | 0.1345 | $e = 2.0$d | 0.023077 | 0.87948 | 0.023077 | 0.069411 | 0.023077 | **No** | **No** | **No** |

---

## 5. Physical and Methodological Diagnosis

1. **Why Supervised Confidence Fails Early:** At $e \le 2.0$ days, light curves contain at most 1--2 photometric detections ($N_{\text{det}} \le 2$). Baseline classifier probabilities for kilonovae remain tiny ($P(\text{KN}) \approx 0.0077\text{--}0.0231$). Because classifier resolution is uninformative ($RES \approx 0.0001$, $\text{ROC-AUC} \approx 0.5360$), supervised confidence alone cannot distinguish kilonovae from Type Ia supernovae or photometric noise.
2. **Why Novelty Signal Fails to Elevate Target Objects:** While normalized novelty scores boost kilonova decision scores from $S_e \approx 0.015$ up to $S_e \approx 0.069$, hundreds of non-target objects (e.g. high-redshift SN Ia outliers, noisy alerts) achieve even higher novelty scores ($\mathcal{N}_e > 1.5$) or higher baseline confidence. Under strict capacity $K = 5$, non-target candidates consistently outrank true kilonovae.
3. **Threshold vs. Capacity Interaction:** The Naive Fixed-Confidence Baseline calibrated threshold ($\tau_{\text{naive}} = 0.046154$) correctly suppressed false alarms ($FP = 0$), whereas the low threshold ($\tau = 0.001$) in the frozen policy allowed non-target candidates to fill all available capacity slots ($K = 5$ at $e=0.0$d and $K = 5$ at $e=2.0$d), producing 10 false triggers without capturing any target event.

---

## 6. Conclusion and Protocol Compliance Verification

- **Hard Constraints Adhered To:** `configs/decision_policy_v1.yaml` was evaluated strictly as locked without modification. Ablations were run as isolated evaluations. No post-hoc tuning was performed after observing deployment labels.
- **Empirical Rigor:** All reported numbers derived from reproducible execution of `scripts/evaluate_true_population_headline.py` against the real repository dataset.
