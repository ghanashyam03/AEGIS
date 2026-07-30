# ADR 007: Early-Alert Sequential Triage Decision Policy

- **Status:** Accepted for the AEGIS decision framework
- **Date:** 2026-07-28

## Context

In time-critical astronomical alert processing (such as kilonova follow-up), observational resources (e.g. spectroscopic telescope time) are strictly capacity-constrained. Under **ADR 003**, decisions must be made at elapsed observer-frame epochs $e \in \{0.0, 2.0\}$ days before the primary decision deadline $H = 2.0$ days.

Prior empirical audit findings establish two key constraints:
1. **Low Classifier Resolution:** The frozen early-epoch baseline classifier's kilonova probability $P(\text{KN})$ at $e \le 2.0$d has near-zero discriminative resolution ($\text{ROC-AUC} \approx 0.5360$, $\text{PR-AUC} \approx 0.0016$, Murphy Resolution $RES \approx 0.0001$; `docs/results/kilonova_discrimination_diagnostic.md`).
2. **Novelty Signal Limitations:** The standardized novelty score $\mathcal{N}_e$ (ADR 006) measures distance from the spectroscopically confirmed ($S=1$) reference distribution in identifiable feature space. While highly sensitive to extreme physical outliers ($5\sigma$ shift detected at $100\%$), it exhibits weak/inverted separation for low-redshift anomalies like TDEs (Class 15 $\text{ROC-AUC} \approx 0.2089$; `docs/methodology/novelty_signal.md`).

Following the architectural precedent set by **ADR 005** (representation complexity), we must select the simplest decision rule that can be rigorously validated and characterized given the actual available signal, rather than an overly complex learned policy.

---

## Options Considered

### Option A: Supervised Class Probability Thresholding Only
Trigger alerts solely when predicted kilonova probability exceeds a cutoff $P(\text{KN}) \ge \tau$.
- *Rejected:* At $e \le 2.0$d, $RES \approx 0.0001$ and $\text{ROC-AUC} \approx 0.5360$. Supervised confidence alone produces essentially random candidate rankings early on.

### Option B: Pure Novelty / Anomaly Score Ranking
Trigger alerts solely based on maximum novelty score $\mathcal{N}_e(x_i)$.
- *Rejected:* Novelty measures distributional distance from $S=1$, not kilonova identity. Highly novel objects may include unmodeled artifacts, high-$z$ noise, or non-target transients.

### Option C: Complex Learned Non-linear Stacking / Policy Gradient
Train a machine learning model or RL agent to map $(P(\text{KN}), \mathcal{N}_e)$ to binary triggers.
- *Rejected:* Violates ADR 005 simplicity principle. Complex learned policies cannot be characterized transparently, risk overfitting on scarce early training labels, and introduce unconstrained hyperparameters.

### Option D: Monotonic Linear Combination & Sequential Stopping Rule (Chosen)
Combine calibrated classifier confidence and normalized novelty score into a single per-epoch decision score $S_e(x_i)$, and apply a sequential capacity-constrained stopping rule.

---

## Decision

### 1. Confidence Input Selection
Per Step 0 audit (`docs/results/probability_source_selection.md`), the decision policy uses predicted probabilities from the **Uncorrected Baseline Classifier** ($p_{i, \text{KN}, e}$). At the primary decision deadline ($e = 2.0$d), the baseline classifier achieves superior Brier calibration ($BS = 0.6323$ vs $0.7407$ for IPW recalibration) and Reliability ($REL = 0.6156$ vs $0.7243$).

### 2. Combined Per-Epoch Decision Score
For object $i$ at epoch $e \le H$, the combined decision score $S_e(x_i)$ is defined as:

\[
S_e(x_i) = p_{i, \text{KN}, e} + w_{\text{nov}} \cdot \mathcal{N}_{e, \text{norm}}(x_i)
\]

where:
- $p_{i, \text{KN}, e} \in [0, 1]$ is the baseline classifier's predicted probability for Kilonova (PLAsTiCC class 64).
- $\mathcal{N}_{e, \text{norm}}(x_i) = \frac{\mathcal{N}_e(x_i)}{\sigma_{\mathcal{N}, S=1}}$ is the epoch novelty score normalized by the standard deviation of novelty scores on the $S=1$ reference population.
- $w_{\text{nov}} \ge 0$ is a configurable decision weight parameter governing the relative influence of novelty vs. supervised confidence.

### 3. Policy Sanity & Monotonicity
The decision score $S_e(x_i)$ strictly satisfies the **Policy Sanity Constraint**:
- Partial derivative w.r.t. confidence: $\frac{\partial S_e}{\partial p_{i, \text{KN}, e}} = 1 > 0$.
- Partial derivative w.r.t. novelty: $\frac{\partial S_e}{\partial \mathcal{N}_{e, \text{norm}}} = w_{\text{nov}} \ge 0$.

Increasing calibrated confidence while holding novelty fixed will never decrease $S_e$, and increasing novelty while holding confidence fixed will never decrease $S_e$.

### 4. Sequential Stopping Rule
Let $a_{i, e} \in \{0, 1\}$ denote the binary trigger action for object $i$ at epoch $e \in \{0.0, 2.0\}$ days.
1. At epoch $e$, evaluate all objects $i$ that remain untriggered from earlier epochs ($\sum_{e' < e} a_{i, e'} = 0$).
2. Compute $S_e(x_i)$ using observations truncated at $t_{0,i} + e$.
3. Rank candidates in descending order of $S_e(x_i)$.
4. Trigger $a_{i, e} = 1$ for the top candidates satisfying $S_e(x_i) \ge \tau_e$, up to the maximum capacity $K$ available at epoch $e$.
5. If an object is not triggered at any epoch $e \le H=2.0$ days, its final policy decision is $a_i = 0$ (no trigger).

### 5. Decision Weight Rationale
Given classifier confidence has minimal resolution ($RES \approx 0.0001$) while novelty provides extreme-value outlier detection (100% sensitivity to $5\sigma$ shifts), novelty acts as an additive boost to lift rare, out-of-distribution alerts. However, because host-galaxy photo-$z$ novelty alone cannot isolate all target transients, $w_{\text{nov}}$ is kept moderate (default $w_{\text{nov}} = 0.05$) so that high-confidence supervised predictions are preserved when classifier signal improves at later epochs.

### 6. Information Flow Diagram

```
truncated observations (mjd <= t_0 + e)
            ↓
representation (identifiable feature subspace F_e)
            ↓
classifier confidence p_{i, KN, e}
            ↓
novelty score N_e(x_i)
            ↓
combined decision score S_e(x_i) = p_{i, KN, e} + w_nov * N_{e, norm}(x_i)
            ↓
sequential trigger rule (top K untriggered with S_e >= tau_e)
            ↓
utility / regret evaluation (u(a, Y), R_e, MHVER_e)
```

---

## Consequences

- Threshold parameters ($\tau_e, w_{\text{nov}}$) and capacity limits ($K$) are defined in versioned experiment configurations (`configs/decision_policy_v1.yaml`), satisfying the **Threshold Separation Constraint** by avoiding tuning against FULL TRUE deployment labels.
- Evaluated strictly via leakage-safe observation truncation ($mjd \le t_{0,i} + e$).
- Decision metrics reuse ADR 003 definitions ($u(a=1, Y=\text{KN})=+2$, $u(a=1, Y\ne\text{KN})=-1$, $u(a=0,*)=0$, utility regret $R_e$, and missed-high-value-event rate $MHVER_e$).

---

## Addendum 007.1: Rate-Based Capacity Correction for Expanded Population

### Context
When the triage decision policy was frozen in `configs/decision_policy_v1.yaml`, the operational capacity was set to $K = 5$ triggers per epoch. This absolute limit was registered against the preliminary cohort of $N = 12,740$ objects (representing a small $0.75\%$ slice of the survey).
In Phase 6, the evaluation population was expanded to $N = 55,915$ objects (a 4.39x increase in survey scale).

### Justification for Capacity Correction
In a real-world telescope survey operation, capacity $K$ represents a rate of spectroscopic follow-ups per unit of survey area/time, or per candidate alert stream size (resource allocation rate), rather than an absolute, globally fixed number of follow-ups across arbitrarily sized survey volumes.
Maintaining a static $K = 5$ triggers per epoch over a 4.39x larger alert pool artificially suppresses the target recovery rate due to alert pool size growth alone.
For a pure-random selection process:
- At the preliminary scale ($N = 12,740$), the expected random kilonova recovery rate is $2K/N = 10/12,740 \approx 0.0785\%$.
- At the expanded scale ($N = 55,915$), the expected random kilonova recovery rate drops to $2K/N = 10/55,915 \approx 0.0179\%$.

To maintain the same resource allocation rate (i.e. the fraction of alerts followable by the telescope resources) across different survey volumes, the capacity $K$ must scale proportionally with the alert pool size:
\[
K_{\text{corrected}} = \text{round}\left(K_{\text{original}} \times \frac{N_{\text{expanded}}}{N_{\text{original}}}\right) = \text{round}\left(5 \times \frac{55,915}{12,740}\right) = 22
\]

### Decision
For the expanded population evaluation, we introduce a corrected capacity configuration of $K = 22$ triggers per epoch, versioned as `configs/decision_policy_v2.yaml`. The original $K = 5$ baseline results and configuration in `configs/decision_policy_v1.yaml` are preserved for historical reference and direct comparison.

