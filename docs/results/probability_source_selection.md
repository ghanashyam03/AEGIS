# Calibrated-Probability Source Selection for Triage Policy (Step 0)

## 1. Executive Summary & Mandate

Per the AEGIS decision policy design protocol, the confidence input feeding into the early-alert triage policy must be selected based on rigorous, empirical probability calibration quality on the **FULL TRUE evaluation population** ($N = 12,740$) at the primary decision deadline ($e = 2.0$ days).

Two calibrated-probability sources currently exist in the AEGIS pipeline:
1. **Uncorrected Baseline Classifier:** Trained on the spectroscopically confirmed ($S=1$) population without post-hoc bias correction or importance weighting.
2. **IPW-Recalibrated Classifier:** Platt scaling fitted on $S=1$ logits with Inverse Probability Weights ($w_i = 1 / p_{\text{spec}}(z_i)$) derived from the selection proxy model ($p_{\text{spec}}(z)$ per ADR 004).

This document presents the mandatory quantitative justification for selecting the **Uncorrected Baseline Classifier** as the confidence input for the decision policy.

---

## 2. Quantitative Empirical Comparison Across Evaluation Populations

Metrics are cited directly from the quantitative audit reports:
- **Baseline Calibration Audit:** `docs/results/calibration_audit_true_population.md`
- **Selection-Aware Recalibration Audit:** `docs/results/recalibration_true_population.md`

All metrics are evaluated across $B = 1,000$ nonparametric object-level bootstrap resamples with 95% confidence intervals.

### Table 1: Calibration Metrics Comparison on FULL TRUE Population ($N=12,740$)

| Decision Epoch | Metric | Uncorrected Baseline Classifier [95% CI] | IPW-Recalibrated Classifier [95% CI] | Absolute Change $\Delta$ [95% CI] | Relative Degradation | Superior Source |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$e = 0.0$d** | **Brier Score $BS$** | 0.6408 [0.6272, 0.6549] | 0.7411 [0.7262, 0.7560] | +0.1003 [+0.0862, +0.1145] | +15.65% (Worse) | **Uncorrected Baseline** |
| **$e = 0.0$d** | **Reliability $REL$** | 0.6240 [0.6098, 0.6383] | 0.7245 [0.7093, 0.7397] | +0.1005 [+0.0863, +0.1147] | +16.12% (Worse) | **Uncorrected Baseline** |
| **$e = 0.0$d** | **Mean ECE (%)** | 26.12% [25.66%, 26.61%] | 26.99% [26.47%, 27.50%] | +0.87% [+0.35%, +1.38%] | +3.34% (Worse) | **Uncorrected Baseline** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$e = 2.0$d (Deadline)** | **Brier Score $BS$** | **0.6323** [0.6177, 0.6471] | **0.7407** [0.7258, 0.7555] | **+0.1084** [+0.0939, +0.1228] | **+17.14%** (Worse) | **Uncorrected Baseline** |
| **$e = 2.0$d (Deadline)** | **Reliability $REL$** | **0.6156** [0.6007, 0.6305] | **0.7243** [0.7092, 0.7394] | **+0.1087** [+0.0942, +0.1232] | **+17.67%** (Worse) | **Uncorrected Baseline** |
| **$e = 2.0$d (Deadline)** | **Mean ECE (%)** | **24.88%** [24.36%, 25.39%] | **26.24%** [25.71%, 26.77%] | **+1.36%** [+0.84%, +1.89%] | **+5.48%** (Worse) | **Uncorrected Baseline** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$e = 7.0$d** | **Brier Score $BS$** | 0.5700 [0.5568, 0.5838] | 0.7065 [0.6917, 0.7212] | +0.1365 [+0.1220, +0.1510] | +23.96% (Worse) | **Uncorrected Baseline** |
| **$e = 7.0$d** | **Reliability $REL$** | 0.5534 [0.5401, 0.5672] | 0.6906 [0.6757, 0.7054] | +0.1372 [+0.1226, +0.1517] | +24.81% (Worse) | **Uncorrected Baseline** |
| **$e = 7.0$d** | **Mean ECE (%)** | 22.87% [22.39%, 23.36%] | 24.52% [24.01%, 25.04%] | +1.65% [+1.14%, +2.17%] | +7.22% (Worse) | **Uncorrected Baseline** |

---

## 3. Methodological and Physical Justification

1. **Empirical Calibration Superiority:** Across all three decision epochs ($e \in \{0.0, 2.0, 7.0\}$ days) and across all three audit metrics (Brier Score, Murphy Reliability $REL$, and Mean ECE), the uncorrected baseline classifier consistently outperforms the IPW-recalibrated model on the FULL TRUE target population. At the primary decision deadline ($e = 2.0$d), the baseline classifier achieves a Brier Score of $0.6323$ vs $0.7407$ for IPW recalibration, representing a statistically significant $17.14\%$ degradation ($p < 0.001$).
2. **Failure Mechanism of IPW Platt Recalibration:** Post-hoc Platt scaling fitted on low-information $S=1$ logits overfits the $99.16\%$ Type Ia Supernova base rate present in the training set. Because early light curves contain minimal discriminative resolution ($RES \approx 0.0001$), probability re-scaling accentuates overconfidence when deployed on unselected $S=0$ targets, inflating the Reliability error ($REL$) from $0.6156$ to $0.7243$.
3. **No Selection Bias Fallacy:** We do not default to IPW recalibration simply because it was implemented more recently in Phase 3. The decision protocol demands selection based on verified calibration quality on the true deployment target.

---

## 4. Operational Conclusion

The **Uncorrected Baseline Classifier** ($P_{\text{base}}(Y = \text{KN} \mid x_{i, \le e})$) is selected as the exclusive calibrated confidence input for the AEGIS triage decision policy.
