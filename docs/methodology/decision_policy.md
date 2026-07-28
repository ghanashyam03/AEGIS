# Methodology & Pre-Registration Freeze Report: Triage Decision Framework (Phase 5)

## 1. Executive Summary & Pre-Registration Freeze

This document presents the complete technical methodology, mathematical design, grid sensitivity analysis findings, held-out Class 15 behavioral check results, and pre-registered experiment configuration freeze for the **AEGIS Early-Alert Triage Decision Framework (Phase 5)**.

### Pre-Registration Lock Certificate
- **Configuration Version:** `v1.0.0-frozen`
- **Lock Status:** `LOCKED_PRE_EVALUATION`
- **Random Seed:** `42`
- **Primary Cost Ratio:** 2:1 ($u_{\text{tp}} = +2.0, u_{\text{fp}} = -1.0$)
- **Primary Operational Capacity:** $K = 5$ triggers per epoch
- **Default Novelty Weight:** $w_{\text{nov}} = 0.05$
- **Decision Threshold:** $\tau_e = 0.001$
- **Calibrated Confidence Input:** Uncorrected Baseline Classifier ($P_{\text{base}}$)

> [!IMPORTANT]
> **Pre-Registration Lock Mandate**  
> All configuration parameters and policy definitions in this document and in `configs/decision_policy_v1.yaml` are **locked ahead of Phase 6 deployment evaluation**. They MUST NOT be altered based on outcomes or results from subsequent evaluations against the FULL TRUE deployment population labels.

---

## 2. Decision Metrics & Reference Utility (ADR 003)

Per **ADR 003**, decision performance is evaluated strictly under pre-registered reference definitions:

### 2.1 Reference Utility Function
For binary trigger action $a_i \in \{0, 1\}$ (0 = no trigger, 1 = trigger/follow-up) and ground-truth class $Y_i$:

\[
u(a, Y) = 
\begin{cases} 
+2.0 & \text{if } a = 1 \text{ and } Y = \text{Kilonova (class 64)} \quad (u_{\text{tp}}) \\
-1.0 & \text{if } a = 1 \text{ and } Y \ne \text{Kilonova} \quad (u_{\text{fp}}) \\
0.0 & \text{if } a = 0 \quad (u_{\text{fn}} = u_{\text{tn}} = 0)
\end{cases}
\]

### 2.2 Oracle Policy & Realized Utility
For capacity $K$ per epoch, the **Oracle Policy** triggers up to $K$ available objects with the largest realized utility (all kilonovae first, gaining $+2.0$ each, then arbitrary non-targets up to capacity $K$).

### 2.3 Utility Regret Metrics
- **Absolute Utility Regret:** $R_e = U_e(\text{oracle}) - U_e(\text{policy})$, where $U_e = \sum_i u(a_i, Y_i)$.
- **Normalized Regret:** $R_{e, \text{norm}} = \frac{R_e}{\max(1.0, U_e(\text{oracle}) - U_e(\text{no-trigger}))}$.

### 2.4 Missed High-Value Event Rate (MHVER)
\[
MHVER_e = \frac{\sum_{i \in \text{Alertable}_e} 1[Y_i = \text{KN}, a_{i, \le e} = 0]}{\sum_{i \in \text{Alertable}_e} 1[Y_i = \text{KN}]}
\]
The headline deadline metric is evaluated at $H = 2.0$ days; a target event is missed if it receives no trigger across any epoch $e \le H$.

---

## 3. Sequential Decision Policy Design (ADR 007)

### 3.1 Combined Decision Score
For object $i$ at epoch $e$, the combined decision score $S_e(x_i)$ is:

\[
S_e(x_i) = p_{i, \text{KN}, e} + w_{\text{nov}} \cdot \mathcal{N}_{e, \text{norm}}(x_i)
\]

where $p_{i, \text{KN}, e}$ is the baseline classifier's kilonova probability, $\mathcal{N}_{e, \text{norm}}(x_i) = \frac{\mathcal{N}_e(x_i)}{\sigma_{\mathcal{N}, S=1}}$ is the normalized novelty score, and $w_{\text{nov}} = 0.05$ is the pre-registered novelty weight.

### 3.2 Policy Sanity & Monotonicity
The policy strictly satisfies monotonicity: $\frac{\partial S_e}{\partial p_{i, \text{KN}}} = 1 > 0$ and $\frac{\partial S_e}{\partial \mathcal{N}_{e, \text{norm}}} = w_{\text{nov}} \ge 0$.

### 3.3 Sequential Stopping Rule
1. At epoch $e \in \{0.0, 2.0\}$ days, evaluate untriggered alerts ($a_{i, <e} = 0$).
2. Rank candidates in descending order of $S_e(x_i)$.
3. Trigger $a_{i, e} = 1$ for the top candidates satisfying $S_e(x_i) \ge \tau_e$ up to capacity $K = 5$.
4. Candidates untriggered by epoch $H = 2.0$ days receive $a_i = 0$.

---

## 4. Grid Sensitivity Analysis Summary (Step 1)

Evaluated across a pre-specified $3 \times 3 \times 3 = 27$ cell parameter grid:
- Cost Ratios: 1:1 ($u_{\text{tp}}=+1$), 2:1 ($u_{\text{tp}}=+2$), 5:1 ($u_{\text{tp}}=+5$).
- Capacities: $K \in \{1, 5, 10\}$.
- Novelty Weights: $w_{\text{nov}} \in \{0.00, 0.05, 0.10\}$.

### Key Findings:
- **Robust Conclusions:** Oracle recovery under perfect information ($R_e = 0.0$), decision score monotonicity, and cost-ratio target valuation scaling hold universally across all 27 grid cells.
- **Sensitive Dependencies:** At small capacity ($K=1$), regret is bottlenecked by capacity when target count $> 1$. At $w_{\text{nov}} \ge 0.10$, noise in novelty can trigger false positives if threshold $\tau$ is too small.

---

## 5. Held-Out Class 15 Behavioral Sanity Check (Step 2)

Evaluated on PLAsTiCC Class 15 (TDEs, $N=2,000$) ingested via `src/aegis/data/class15_ingest.py`:
- Mean $P(\text{KN}) \approx 0.0018\text{--}0.0021$ (low supervised confidence).
- Mean Novelty $\mathcal{N}_e \approx 0.51\text{--}0.59$ (low distance due to low-$z$ host galaxy distribution).
- Combined Score $S_e \approx 0.053\text{--}0.060$ (lower than in-distribution target scores $S_e \approx 0.101\text{--}0.113$).
- Trigger Rate: **0.00%** (does not displace target candidates or cause false-alarm storms).

---

## 6. Pre-Registered Configuration Freeze Record

| Parameter | Final Value | Origin | Rationale | Permitted Evidence |
| :--- | :---: | :--- | :--- | :--- |
| `capacity_per_epoch` | `5` | ADR 003 operational time bounds | Realistic nightly spectroscopic alert follow-up quotas | Operational constraints |
| `primary_cost_ratio` | `2:1` | ADR 003 reference utility | Standard reference valuation balancing gain vs cost | Pre-registered specification |
| `novelty_weight` | `0.05` | ADR 007 & $S=1$ scale analysis | Matches novelty unit variance to top-decile confidence | $S=1$ reference training distribution |
| `decision_threshold` | `0.001` | $S=1$ baseline decile distribution | Filters uninformative background noise | $S=1$ probability distribution |
| `probability_source` | `"uncorrected_baseline"` | Step 0 Probability Audit | Superior Brier score ($0.6323$ vs $0.7407$) at $e=2.0$d | Population Brier/Reliability/ECE |
