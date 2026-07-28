# ADR 006: Design and Definition of the Novelty / Distributional-Distance Signal

- **Status:** accepted
- **Date:** 2026-07-28

## Context

Phase 3 and Step 0 audits established key empirical facts regarding early-epoch transient classification:
1. **Lack of Early Discriminative Resolution (Step 0 Diagnostic):** On the FULL TRUE evaluation population ($N=12,740$), the baseline classifier's kilonova probability $P(\text{KN})$ has no discriminative power at initial alert ($e = 0.0$d: ROC-AUC $= 0.5023$, PR-AUC $= 0.0064$) and minimal power at the primary decision deadline ($e = 2.0$d: ROC-AUC $= 0.5360$, PR-AUC $= 0.0016$).
2. **Extreme Data Sparsity & Representation Constrainability (ADR 005 & P6 Audit):** At $e = 0.0$d, 100% of objects have only 1 detection point. Physical light-curve features (rise rates, color pairs, S/N growth) are $>98.5\%\text{--}100\%$ mathematically unconstrained (`docs/results/representation_constrainability.json`). `hostgal_photoz` is the sole 100% constrained feature at alert time.
3. **Selection-Induced Miscalibration (ADR 004):** Post-hoc recalibration fails to fix miscalibration because early light-curve features lack resolution (`docs/results/recalibration_true_population.md`).

Consequently, an early triage policy cannot rely solely on supervised target class confidence. It requires an independent **novelty / distributional-distance signal** to measure how atypical an incoming alert is relative to the known, spectroscopically confirmed population.

---

## Decision

### (a) Epoch-Indexed Identifiable Feature Subspace

The novelty score at epoch $e \in \{0.0, 2.0, 7.0\}$ days shall operate strictly on the feature subspace established as mathematically identifiable by the P6 constrainability audit (`docs/results/representation_constrainability.json`). It shall **not** assume or invent a broader feature space.

- **Epoch $e = 0.0$d Subspace:** Consists of `hostgal_photoz` (100% constrained across all objects). Unconstrained light-curve features are excluded.
- **Epoch $e = 2.0$d Subspace:** Consists of `hostgal_photoz` for all objects, plus any valid/constrained light-curve features present for a given object (e.g. `alert_snr_0`, `snr_growth_rate`, `color_pb2_pb3`, `color_pb3_pb4` when non-NaN).
- **Epoch $e = 7.0$d Subspace:** Consists of `hostgal_photoz` and available constrained light-curve features at $e=7.0$d.

**Handling Heterogeneous Missingness:** Missing or unconstrained features must **never** be silently imputed with global medians/means (which would artificially shrink distances to zero). Instead, distance is evaluated dynamically over the non-NaN identifiable feature subset for each object, normalized by the number of valid dimensions.

### (b) Reference Population ("Known")

The reference population defining "known" space is **strictly the spectroscopically confirmed ($S=1$) labeled population**.

- **Rationale:** At operational decision time, only objects with past successful spectroscopic follow-up ($S=1$) are confirmed and labeled. Using the TRUE ($S=1 + S=0$) population as the reference would leak unselected deployment-time information that an operational policy would never possess.

### (c) Distance / Density Metric

For an object $i$ at epoch $e$ with valid identifiable features $x_{i, \mathcal{V}_i}$ (where $\mathcal{V}_i \subseteq \mathcal{F}_e$ is the set of non-NaN feature indices for object $i$), the novelty score is defined as the **normalized robust standardized Euclidean distance** from the $S=1$ reference population:

\[
\mathcal{N}_e(x_i) = \sqrt{ \frac{1}{|\mathcal{V}_i|} \sum_{j \in \mathcal{V}_i} \left( \frac{x_{ij} - \mu_{j, S=1}}{\sigma_{j, S=1}} \right)^2 }
\]

where $\mu_{j, S=1}$ and $\sigma_{j, S=1}$ are the mean and standard deviation (or robust median and MAD) of feature $j$ estimated strictly on the $S=1$ reference sample.

- **Justification over Complex Metrics:** Full-dimensional covariance inversion (Mahalanobis distance) or high-dimensional Kernel Density Estimation (KDE) fail in this regime due to extreme data sparsity, zero off-diagonal sample covariance for missing features, and high missingness heterogeneity. Standardized distance on the active feature subspace provides a robust, scale-invariant, non-parametric distance measure.

### (d) Held-Out Validation Population (PLAsTiCC Class 15: TDEs)

To evaluate the novelty score without circular reasoning, **PLAsTiCC Class 15 (Tidal Disruption Events)** shall serve as the held-out validation population.

- **Validation Role:** Class 15 was explicitly documented in ADR 002 as considered but excluded from the study set (`STUDY_CLASS_IDS = [64, 90, 95]`). It was never used in feature representation design, classifier training, or calibration audits.
- **Explicit Limitation:** Class 15 is a proxy for genuine astrophysical novelty (an unmodeled real transient class), not literal unprecedented-phenomenon novelty. This limitation must be explicitly acknowledged in all subsequent methodological reports.

---

## Hard Architectural Constraints & Interpretation Boundaries

> [!CAUTION]
> **Interpretation Constraint Enforcement**
> High novelty score **must not** be interpreted as evidence that an object is a kilonova or any specific transient class. A novelty score quantifies distributional distance from the known $S=1$ population, not class membership. Novelty estimation and supervised classification are kept strictly separate.

> [!IMPORTANT]
> **Schema & Isolation Boundaries**
> Class 15 objects must pass through a strictly separate ingestion path. They must **never** be added to `STUDY_CLASS_IDS`, `TRUE_POPULATION_SCHEMA`, or any existing evaluation cohort files.

---

## Consequences

- The novelty module will be implemented under `src/aegis/models/novelty.py` (or `src/aegis/features/novelty.py`).
- Performance will be evaluated via ROC-AUC and PR-AUC as an anomaly detector separating class 15 from study classes ($64, 90, 95$).
- Synthetic extreme-value perturbations of in-distribution objects will serve as a secondary validation check.
