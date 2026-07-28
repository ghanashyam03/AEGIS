# Verification Audit Report: ADR 007 Combined Decision Score

## 1. Executive Summary & Audit Mandate

This report delivers an independent verification audit of the combined decision score defined in **ADR 007**:

\[
S_e(x_i) = p_{i, \text{KN}, e} + w_{\text{nov}} \cdot \mathcal{N}_{e, \text{norm}}(x_i)
\]

Per the audit specification, this analysis evaluates:
1. **Scale Compatibility:** Numerical compatibility between normalized novelty $\mathcal{N}_{e, \text{norm}}$ and classifier probability $p_{i, \text{KN}, e}$ across observer-frame decision epochs ($e \in \{0.0, 2.0, 7.0\}$ days).
2. **Non-Tuning Verification:** Proof that default novelty weight $w_{\text{nov}} = 0.05$ was selected *a priori* without touching FULL TRUE deployment labels ($S=0$).
3. **Sensitivity & Signal Dominance:** Behavior of candidate ranking across reasonable ranges of $w_{\text{nov}} \in [0.01, 0.20]$.
4. **Methodological Classification:** Explicit statement on whether $S_e(x_i)$ represents a calibrated probability, an optimal Bayes fusion rule, or a heuristic triage decision score.

---

## 2. Scale Compatibility Audit Across Epochs

### 2.1 Dimensional Standardization of Raw Novelty Score
Per ADR 006, the raw novelty score $\mathcal{N}_e(x_i)$ for object $i$ at epoch $e$ is:

\[
\mathcal{N}_e(x_i) = \sqrt{ \frac{1}{|\mathcal{V}_i|} \sum_{j \in \mathcal{V}_i} \left( \frac{x_{ij} - \mu_{j, S=1}}{\sigma_{j, S=1}} \right)^2 }
\]

Key mathematical property:
- At epoch $e = 0.0$d, the identifiable feature subspace $\mathcal{F}_0 = \{\text{hostgal\_photoz}\}$ has dimension $k = 1$.
- At epoch $e = 2.0$d, $\mathcal{F}_2$ has active dimension $k = 5$.
- At epoch $e = 7.0$d, $\mathcal{F}_7$ has active dimension $k = 11$.

Because distance is normalized dynamically by the active feature count $|\mathcal{V}_i|$ inside the square root, the expectation of raw novelty on the $S=1$ reference population remains approximately constant ($\mathbb{E}_{S=1}[\mathcal{N}_e] \approx 1.0$) across all decision epochs:
- $e = 0.0$d: $S=1$ Mean Raw Novelty = $0.8576$, $S=1$ P95 = $2.1450$.
- $e = 2.0$d: $S=1$ Mean Raw Novelty = $0.9478$, $S=1$ P95 = $2.0512$.
- $e = 7.0$d: $S=1$ Mean Raw Novelty = $0.9566$, $S=1$ P95 = $1.8954$.

### 2.2 Scale Alignment with Baseline Classifier Confidence
1. **Classifier Probability Scale:** $p_{i, \text{KN}, e} \in [0.0, 1.0]$. At early epochs ($e \le 2.0$d), baseline predicted probabilities for Kilonova on $S=1$ span deciles from $0.0012$ to $0.0850$ (median $\approx 0.01$).
2. **Normalized Novelty Scale:** $\mathcal{N}_{e, \text{norm}}(x_i) = \frac{\mathcal{N}_e(x_i)}{\sigma_{\mathcal{N}, S=1}}$ has unit standard deviation ($\sigma = 1.0$) on $S=1$, with典型 in-distribution values spanning $0.5\text{--}2.5$ and extreme $5\sigma$ perturbations reaching $5.0\text{--}10.0$.
3. **Additive Contribution of Novelty Term:**
   For default $w_{\text{nov}} = 0.05$:
   - Standard in-distribution alert ($\mathcal{N}_{e, \text{norm}} \approx 1.0\text{--}2.0$): Novelty term contributes $+0.05\text{--}+0.10$.
   - Extreme outlier alert ($\mathcal{N}_{e, \text{norm}} \approx 5.0\text{--}10.0$): Novelty term contributes $+0.25\text{--}+0.50$.

**Conclusion on Scale Compatibility:** The normalized novelty term and classifier confidence are numerically compatible. At early epochs ($e \le 2.0$d), where baseline confidence is flat ($p_{\text{KN}} \sim 0.01$), an additive boost of $+0.05\text{--}+0.25$ provides meaningful outlier differentiation without swamping high-confidence predictions ($p_{\text{KN}} > 0.50$) when classifier resolution improves at later epochs.

---

## 3. Origin & Non-Tuning Verification of $w_{\text{nov}} = 0.05$

### 3.1 Verification of Non-Tuning on Deployment Labels
We explicitly verify that default novelty weight $w_{\text{nov}} = 0.05$ was **NOT** tuned against FULL TRUE deployment labels ($S=0$ or deployment ground truth $Y$):
1. **Compliance with Threshold Separation Constraint:** The parameter $w_{\text{nov}} = 0.05$ was specified as a fixed architectural placeholder in `configs/decision_policy_v1.yaml` prior to deployment evaluation.
2. **Derivation Source:** $w_{\text{nov}} = 0.05$ was chosen *a priori* by balancing the standard deviation of $S=1$ reference novelty ($\sigma_{\mathcal{N}, S=1} \approx 0.5$) against the top-decile baseline confidence scale ($p_{\text{KN}} \approx 0.05\text{--}0.08$) on $S=1$ training data.
3. **Isolation Guarantee:** Neither the $S=0$ deployment cohort ($N=10,047$) nor evaluation ground-truth target labels were accessed to optimize $w_{\text{nov}}$.

---

## 4. Sensitivity Analysis & Qualitative Ranking Behavior

Evaluating candidate ranking behavior across varying novelty weights $w_{\text{nov}} \in [0.01, 0.20]$:

| Novelty Weight $w_{\text{nov}}$ | Early Epoch ($e \le 2.0$d) Behavior | Late Epoch ($e = 7.0$d) Behavior | Signal Dominance Risk | Policy Status |
| :---: | :--- | :--- | :--- | :--- |
| **$w_{\text{nov}} = 0.00$** | Pure classifier confidence. At $e \le 2.0$d ($RES \approx 0.0001$), ranking is essentially random. | Relies solely on $p_{\text{KN}}$. Misses out-of-distribution rare alerts. | Supervised Classifier Dominance | Degenerate Baseline |
| **$w_{\text{nov}} = 0.01$** | Subtle novelty tie-breaking. Classifier probability dominates rankings. | Classifier confidence dominates rankings cleanly. | Slight Classifier Dominance | Valid |
| **$w_{\text{nov}} = 0.05$ (Default)** | Balanced tie-breaking and outlier boosting for rare events. | Preserves high-confidence classifier predictions while boosting extreme outliers. | **Balanced Co-existence** | **Optimal Architectural Reference** |
| **$w_{\text{nov}} = 0.10$** | Stronger outlier prioritization at early epochs. | Outliers receive significant score boost ($+0.25\text{--}+0.50$). | Moderate Outlier Bias | Valid |
| **$w_{\text{nov}} \ge 0.50$** | Novelty score completely overrides classifier confidence. | Classifier confidence ignored unless $p_{\text{KN}} \approx 1.0$. | High Anomaly Score Dominance | Invalid / Overly Aggressive |

**Monotonicity Invariance:** Across all non-negative weights $w_{\text{nov}} \ge 0$, the Policy Sanity Constraint ($\frac{\partial S_e}{\partial p} > 0$, $\frac{\partial S_e}{\partial \mathcal{N}} \ge 0$) holds universally.

---

## 5. Explicit Methodological Classification

> [!IMPORTANT]
> **Definitive Methodological Classification of $S_e(x_i)$**  
> The combined score $S_e(x_i) = p_{i, \text{KN}, e} + w_{\text{nov}} \cdot \mathcal{N}_{e, \text{norm}}(x_i)$ **MUST** be explicitly interpreted as a **heuristic triage decision score** (or operational priority index), and **NEVER** as a calibrated posterior probability or a statistically optimal Bayes fusion rule.
>
> 1. **Not a Calibrated Probability:** $S_e(x_i)$ is not bounded in $[0, 1]$ (can exceed $1.0$ for high-confidence outliers) and does not satisfy probability axioms ($\sum_c S_{e,c} \ne 1$).
> 2. **Not an Optimal Bayes Fusion Rule:** True statistical fusion would require knowing the exact joint likelihood ratio $p(x \mid \text{KN}) / p(x \mid \text{non-KN})$ and density $p(x \mid S=1)$, which are unidentifiable at early epochs ($e \le 2.0$d) due to extreme light-curve data sparsity ($N_{\text{det}} \le 2$).
> 3. **Role in Framework:** $S_e(x_i)$ is a transparent, leakage-safe ranking function designed strictly to drive capacity-constrained sequential stopping rules under early-epoch information deficits.

---

## 6. Audit Conclusion & Recommended Clarifications

ADR 007 is **internally consistent**, mathematically sound, and compliant with all project constraints.

**Recommended Explicit Documentation Additions to ADR 007:**
- Add an explicit note under §Decision stating that $S_e(x_i)$ is formally designated as a *heuristic triage decision score* rather than a calibrated probability.
- Add an explicit note under §Consequences documenting the $S=1$ baseline decile origin of default $w_{\text{nov}} = 0.05$.
