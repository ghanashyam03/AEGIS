# ADR 008: Small-Sample Evaluation Methodology for Rare Target Events

- **Status:** Accepted for AEGIS evaluation framework
- **Date:** 2026-07-29

## Context

Step 0 audit (`docs/results/kilonova_sample_size_investigation.md`) establishes that the underlying PLAsTiCC survey simulation contains only $N_{\text{KN}} = 133$ kilonova (class 64) objects out of 3.49 million test metadata rows ($P(Y=64) \approx 3.8 \times 10^{-5}$). In the pre-specified 12,740-object evaluation cohort (`plasticc_test_lightcurves_01.csv.gz` slice), there are exactly **$N_{\text{KN}} = 2$ positive kilonova objects** (`489518` and `490807`).

Under standard evaluation protocols (ADR 003), non-parametric percentile bootstrap resampling ($B = 1,000$) was specified for uncertainty estimation. However, when positive event counts are on the order of a handful of objects ($N_{\text{KN}} \le 5$), non-parametric percentile bootstrap methods exhibit severe degenerate behavior:
1. **Quantization & Discretization:** Bootstrap draws from $N=2$ positive objects can yield only 0, 1, or 2 distinct targets, producing step-function percentile bounds rather than continuous distributions.
2. **Deceptive Precision:** Reporting a 95% bootstrap confidence interval on an aggregate rate calculated over 2 objects creates a false appearance of statistical power.
3. **Sensitivity Masking:** Aggregate point estimates hide whether a policy's performance depends entirely on triggering a single specific target object.

---

## Options Considered

### Option A: Standard Percentile Bootstrap (ADR 003 default)
Compute 95% percentile bootstrap CIs via $B = 1,000$ object-level resamples across all metrics.
- *Rejected:* At $N_{\text{KN}} = 2$, bootstrap resampling of the positive class is mathematically ill-posed and uninformative.

### Option B: Asymptotic Normal Approximation
Use Wald intervals ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/N}$).
- *Rejected:* Violates $[0, 1]$ probability boundaries when $\hat{p} \in \{0, 1\}$ or $N$ is small; completely invalid for small counts.

### Option C: Small-Sample Exact Intervals, LOKO Jackknife, and Per-Object Reporting (Chosen)
Combine exact binomial confidence intervals, Leave-One-Kilonova-Out (LOKO) jackknife re-evaluation, and granular per-object itemization.

---

## Decision

For all headline evaluations where the target event count is small ($N_{\text{KN}} \le 10$), AEGIS adopts the following three-part evaluation methodology:

### 1. Exact Binomial Uncertainty for Target Class Rates
For positive-class proportion metrics (e.g. Missed High-Value Event Rate $MHVER_e$, target trigger rate):
- Point estimates MUST be supplemented with **exact Clopper-Pearson 95% binomial confidence intervals** (or Wilson score intervals), defined by the Beta distribution quantiles:
  \[
  \text{CI}_{\text{lower}} = B\left(\frac{\alpha}{2}; k, n - k + 1\right), \quad \text{CI}_{\text{upper}} = B\left(1 - \frac{\alpha}{2}; k + 1, n - k\right)
  \]
  where $k$ is the number of triggered (or missed) targets, $n = N_{\text{KN}}$, and $\alpha = 0.05$.

### 2. Leave-One-Kilonova-Out (LOKO) Jackknife Protocol
For every headline metric (Regret $R_e$, Normalized Regret, $MHVER_e$, Total Utility $U_e$):
- The evaluation MUST be recomputed $N_{\text{KN}}$ times. In iteration $j \in \{1, \dots, N_{\text{KN}}\}$, target object $j$ is omitted from the evaluation population, leaving $N_{\text{KN}} - 1$ positive targets.
- The evaluation report MUST prominently publish the jackknife range $[\min_j, \max_j]$ for every metric alongside the full-sample point estimate.
- This measures and documents the exact sensitivity of headline conclusions to any single target object.

### 3. Granular Per-Object Itemization
Aggregate statistics MUST NOT be presented in isolation. Every headline report MUST include a dedicated **Per-Object Decision Audit Table** explicitly detailing the outcome for each confirmed positive target object ($i \in \{1, \dots, N_{\text{KN}}\}$):
- Object ID, host photometric redshift ($z_{\text{phot}}$), true redshift ($z_{\text{true}}$).
- Decision outcome $a_{i,e} \in \{0, 1\}$ at each epoch $e \in \{0.0, 2.0\}$.
- Decision score $S_e(x_i)$ and population rank at decision epoch.
- First epoch of trigger (if triggered) and individual utility contribution $u(a_i, Y_i)$.

### 4. Non-Target Background Metrics
For non-target background metrics (False Trigger Rate $FTR_e$, False Positive Count $FP_e$, Non-Target Regret), where the non-target population is large ($N_{\text{non-KN}} = 12,738$), standard large-sample statistical metrics and standard 95% confidence bounds remain valid and shall be reported.

---

## Consequences

- Eliminates misleading bootstrap claims for $N_{\text{KN}} = 2$.
- Guarantees full transparency regarding policy performance on individual kilonovae.
- Preserves mathematical rigor without attempting post-hoc sample expansion.

---

# Addendum 008.1: Statistical Methodology Evolution for Expanded Target Populations

- **Status:** Accepted as an extension to ADR 008
- **Date:** 2026-07-29

## Context & Motivation
Following the Step 0 provenance investigation (`docs/results/evaluation_cohort_provenance.md`), the evaluation population was expanded from the preliminary 12,740-object slice ($N_{\text{KN}} = 2$) to the full disjoint survey population containing all **$N_{\text{KN}} = 133$ kilonova objects** and **$N_{\text{SLSN}} = 35,782$ SLSN-I objects**.

With a 66.5-fold increase in target positive event count ($N_{\text{KN}} = 133 \ge 30$), the statistical regime changes fundamentally:
1. **Bootstrap Validity:** Non-parametric percentile bootstrap resampling ($B = 1,000$) is no longer mathematically ill-posed or degenerate. With $N_{\text{KN}} = 133$, bootstrap draws sample a continuous distribution of target outcomes rather than 0, 1, or 2 discrete targets.
2. **Exact Interval Power:** Exact Clopper-Pearson 95% binomial confidence intervals remain mathematically rigorous, but their width contracts from $[0.0250, 1.0000]$ (width 0.9750 at $N=2$) down to $[0.9726, 1.0000]$ (width 0.0274 at $N=133$ for a 100% miss rate).
3. **Jackknife Distribution:** Leave-One-Target-Out jackknife re-evaluation scales from a 2-object per-target audit to an ensemble stability diagnostic across all 133 target objects.

---

## Methodology Decisions for Expanded Evaluation

1. **Dual Uncertainty Reporting:**
   - For positive target rates ($MHVER_e$, Target Trigger Rate), reports MUST provide **exact Clopper-Pearson 95% binomial CIs** as the primary exact bound.
   - For aggregate utility metrics (Total Utility, Regret, Normalized Regret, False Trigger Rate), reports SHALL supplement point estimates with **95% non-parametric percentile bootstrap CIs** ($B = 1,000$ object-level resamples).

2. **Per-Object Audit Aggregation:**
   - While full per-object decision audit tables were explicitly itemized for $N_{\text{KN}} = 2$, itemizing all 133 kilonovae and 35,782 SLSN-I in markdown body text would be unreadable.
   - Headline reports MUST provide summary statistics (mean, min, max score/rank/redshift evolution) across the target population, full JSON per-object audit logs saved as machine-readable artifacts, and explicit top-candidate itemizations.

3. **Historical Continuity:**
   - The original small-sample methodology ($N_{\text{KN}} = 2$) defined in ADR 008 remains an intact, valid historical record of the preliminary phase and is not erased. Addendum 008.1 explicitly supersedes it for all expanded-population findings.

