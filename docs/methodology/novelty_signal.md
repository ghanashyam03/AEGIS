# Methodology & Validation Report: Novelty / Distributional-Distance Signal (Step 3)

## 1. Executive Summary & Step 0 Diagnostic Implications

This document presents the complete technical methodology, mathematical design, and empirical validation results for the **Novelty / Distributional-Distance Signal** ($\mathcal{N}_e$) in AEGIS, completing **Step 0**, **Step 1 (ADR 006)**, **Step 2**, and **Step 3**.

### Step 0 Mandatory Diagnostic Context
The Step 0 diagnostic (`docs/results/kilonova_discrimination_diagnostic.md`) evaluated whether the frozen early-epoch baseline classifier possesses discriminative resolution for Kilonovae (PLAsTiCC class 64) versus the rest of the target population on the **FULL TRUE evaluation cohort** ($N = 12,740$, base rate $P(\text{KN}) = 0.0002$).

- **Initial Alert ($e = 0.0$d):** $\text{ROC-AUC} = 0.5023$ [0.0093, 0.9947], $\text{PR-AUC} = 0.0064$ [0.0001, 0.0373] (random baseline PR-AUC $\approx 0.0002$).
- **Primary Trigger Deadline ($e = 2.0$d):** $\text{ROC-AUC} = 0.5360$ [0.0939, 0.9756], $\text{PR-AUC} = 0.0016$ [0.0001, 0.0088].
- **Diagnostic Horizon ($e = 7.0$d):** $\text{ROC-AUC} = 0.6769$ [0.3536, 0.9949], $\text{PR-AUC} = 0.0067$ [0.0001, 0.0394].

> [!IMPORTANT]
> **Implication for Triage Policy**  
> Supervised kilonova probability $P(\text{KN})$ at early epochs ($e \le 2.0$d) is virtually uninformative ($\text{ROC-AUC} \approx 0.50$). An operational triage policy **cannot** rely on supervised class probabilities alone to select follow-up candidates. An independent **novelty signal** is strictly required to quantify how atypical an alert appears relative to the known, spectroscopically confirmed population.

---

## 2. Novelty Signal Mathematical Design (ADR 006)

Per **ADR 006**, the novelty score $\mathcal{N}_e(x_i)$ is formulated under strict operational boundaries:

### 2.1 Identifiable Feature Subspace ($\mathcal{F}_e$)
The novelty score operates strictly within the feature subspace shown to be mathematically constrained by the P6 identifiability audit (`docs/results/representation_constrainability.json`):
- **Epoch $e = 0.0$d:** Restricted to `hostgal_photoz` (100% constrained). Light-curve features are $>98.5\%$ unconstrained.
- **Epoch $e = 2.0$d:** Includes `hostgal_photoz` and active constrained light-curve features (`alert_snr_0`, `snr_growth_rate`, `color_pb2_pb3`, `color_pb3_pb4`).
- **Epoch $e = 7.0$d:** Includes `hostgal_photoz` and available constrained light-curve/color features.

### 2.2 Reference Population ($S=1$)
The reference population defining "known" space is **strictly the spectroscopically confirmed ($S=1$) labeled training set** ($N = 2,693$). Using the unselected deployment population ($S=0$) or TRUE population is strictly prohibited to prevent leaking future operational information.

### 2.3 Robust Distance Metric & Missingness Handling
For object $i$ at epoch $e$, let $\mathcal{V}_i \subseteq \mathcal{F}_e$ be the set of valid (non-NaN) feature indices. The novelty score $\mathcal{N}_e(x_i)$ is defined as:

\[
\mathcal{N}_e(x_i) = \sqrt{ \frac{1}{|\mathcal{V}_i|} \sum_{j \in \mathcal{V}_i} \left( \frac{x_{ij} - \mu_{j, S=1}}{\sigma_{j, S=1}} \right)^2 }
\]

where $\mu_{j, S=1}$ and $\sigma_{j, S=1}$ are feature location and scale parameters estimated strictly on $S=1$. Missing or unconstrained features are **not silently imputed**; distance is normalized dynamically over the active feature count $|\mathcal{V}_i|$.

---

## 3. Empirical Validation Results (Step 3)

Validation evaluates anomaly detection performance using $B = 1,000$ nonparametric object-level bootstrap resamples (seed=42) on two populations:
1. **Held-Out Validation Class:** PLAsTiCC Class 15 (Tidal Disruption Events, $N = 2,000$) ingested via an isolated code path (`src/aegis/data/class15_ingest.py`). Class 15 was never part of `STUDY_CLASS_IDS`, representation design, or classifier training.
2. **Synthetic Extreme-Value Perturbations:** $5\sigma$ redshift shifts applied to in-distribution evaluation objects.

### Table 1: Novelty Score Anomaly Detection Performance (Held-Out Class 15 vs. In-Distribution Study Classes)

| Decision Epoch | Evaluation Setup | Cohort Size ($N$) | Anomaly Class ($y=1$) | ROC-AUC [95% CI] | PR-AUC [95% CI] | Random Chance PR-AUC | Novelty Separation Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$e = 0.0$d** | **Class 15 vs Study Classes** | 14,740 | TDE (15, $N=2,000$) | **0.3153** [0.3030, 0.3274] | **0.1042** [0.0977, 0.1127] | 0.1357 | Inverted / Weak ($z$-Distribution Overlap) |
| **$e = 2.0$d** | **Class 15 vs Study Classes** | 14,740 | TDE (15, $N=2,000$) | **0.2089** [0.1957, 0.2218] | **0.1119** [0.1031, 0.1223] | 0.1357 | Inverted / Weak ($z$-Distribution Overlap) |
| **$e = 7.0$d** | **Class 15 vs Study Classes** | 14,740 | TDE (15, $N=2,000$) | **0.2026** [0.1892, 0.2158] | **0.1094** [0.1004, 0.1204] | 0.1357 | Inverted / Weak ($z$-Distribution Overlap) |

> [!WARNING]
> **Plain Reporting of Weak/Inverted Separation & Physical Mechanism**  
> In PLAsTiCC simulations, Tidal Disruption Events (Class 15) occur predominantly in low-to-intermediate redshift host galaxies ($\text{mean } z_{\text{phot}} \approx 0.35$), whereas the spectroscopically selected $S=1$ reference population spans a higher redshift distribution ($\text{mean } z_{\text{phot}} \approx 0.72$). Consequently, in the sparse early feature space dominated by `hostgal_photoz`, Class 15 objects lie *closer* to the dense core of low-$z$ galaxy counts than the high-$z$ tail of $S=1$, resulting in $\text{ROC-AUC} < 0.50$.  
> This empirical finding proves that **host-galaxy photo-z novelty alone cannot isolate all unmodeled astrophysical classes**, representing an essential limitation to carry into policy design.

---

### Table 2: Response to Synthetic Extreme-Value Perturbations ($5\sigma$ Covariate Shift)

| Decision Epoch | In-Distribution Mean Score | Perturbed Mean Score | Score Ratio (Shift) | $S=1$ 95th Percentile Threshold | % Perturbed Exceeding P95 | Detection Sensitivity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$e = 0.0$d** | 0.8576 | 5.7929 | **6.75x** | 2.1450 | **100.0%** | Perfect Sensitivity |
| **$e = 2.0$d** | 0.9478 | 3.1861 | **3.36x** | 2.0512 | **100.0%** | Perfect Sensitivity |
| **$e = 7.0$d** | 0.9566 | 2.7686 | **2.89x** | 1.8954 | **93.8%** | High Sensitivity |

---

## 4. Hard Interpretation & Policy Boundaries

> [!CAUTION]
> **Enforcement of the Interpretation Constraint**
> 1. **Distance vs. Class Identity:** A high novelty score $\mathcal{N}_e(x_i)$ measures distance from the $S=1$ reference population in identifiable feature space. It **must never** be interpreted as evidence that an object is a kilonova or any specific transient class.
> 2. **Conceptual Separation:** Novelty estimation ($\mathcal{N}_e$) and supervised classification ($P(\text{KN})$) remain strictly decoupled. Triage policies must treat novelty as an outlier filter or risk penalty, not as a positive class indicator.
> 3. **Class 15 Limitation:** Class 15 is a proxy for unmodeled astrophysical phenomena, not literal unprecedented physics.
