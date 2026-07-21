# External Scientific Audit: Data Acquisition & TRUE vs. BIASED Population Pipeline

> **Document ID:** `docs/audits/data_pipeline_audit.md`  
> **Audit Date:** July 21, 2026  
> **Auditor Role:** Independent External Reviewer (Controlled Evaluation Methodology)  
> **Target Package:** AEGIS Data Ingestion & Selection Function Pipeline (`src/aegis/data/`, `configs/data_population.yaml`)  
> **Evaluation Basis:** Architecture Decision Records (ADR 001–004), `docs/problem_statement.md`, `docs/architecture.md`  
> **Supporting Artifacts:** `docs/results/data_audit/` (tables, metrics, plots)

---

## Executive Summary

This independent scientific audit evaluated the completed AEGIS data ingestion pipeline and population construction strategy prior to model development. The audit evaluated six critical dimensions: dataset integrity, selection function pre-observation strictness, distributional shift characterization, class balance dynamics, parameter sensitivity, and scientific limitations.

The AEGIS data pipeline successfully ingests the unblinded PLAsTiCC test metadata release (Zenodo record `2539456`), validates schema constraints via Pandera, and constructs two mathematically rigorous population subsets:
1. **TRUE Population ($N = 1,695,746$)**: Complete simulated astrophysical population restricted to the three study classes (Kilonova: 64, Type Ia Supernova: 90, Superluminous Supernova Type I: 95).
2. **BIASED Population ($N = 650,359$)**: Spectroscopically labeled subset constructed via a deterministic, seeded Bernoulli selection function based on a literature-motivated logistic proxy ($p_{\rm spec}(z)$).

---

## 1. Dataset Integrity Verification

The audit performed byte-for-byte and ID-level tracking across all four pipeline stages (`RAW`, `INTERIM`, `TRUE`, `BIASED`).

### Summary Integrity Table

| Pipeline Stage | Total Object Count | Unique `object_id` Count | Duplicate IDs | Class Scope | Manifest Verification |
|---|---|---|---|---|---|
| **RAW** (`data/raw/`) | 3,492,890 | 3,492,890 | 0 | All PLAsTiCC classes | `manifest.json` (SHA-256 verified) |
| **INTERIM** (`data/interim/`) | 3,492,890 | 3,492,890 | 0 | All PLAsTiCC classes | `manifest.json` (SHA-256 verified) |
| **TRUE** (`data/processed/`) | 1,695,746 | 1,695,746 | 0 | Study classes (64, 90, 95) | `manifest_true.json` (SHA-256 verified) |
| **BIASED** (`data/processed/`) | 650,359 | 650,359 | 0 | Study classes (64, 90, 95) | `manifest_biased.json` (SHA-256 verified) |

### Specific Verification Findings

1. **Completeness of TRUE Population**:  
   Filtering `INTERIM` for `true_target` $\in \{64, 90, 95\}$ yields exactly $1,695,746$ objects ($133$ KN + $1,659,831$ SN Ia + $35,782$ SLSN-I). The `TRUE` population contains 100% of the study-class objects present in the raw source without any missing or dropped rows.

2. **Strict Subset Contract ($\text{BIASED} \subset \text{TRUE}$)**:  
   Let $\mathcal{S}_{\rm TRUE}$ and $\mathcal{S}_{\rm BIASED}$ be the sets of unique `object_id` values. Verification confirms:
   $$\mathcal{S}_{\rm BIASED} \subset \mathcal{S}_{\rm TRUE} \quad \text{and} \quad |\mathcal{S}_{\rm BIASED}| < |\mathcal{S}_{\rm TRUE}|$$
   Specifically, $|\mathcal{S}_{\rm BIASED}| = 650,359 < 1,695,746$.

3. **No Reverse Leakage**:  
   The set difference $\mathcal{S}_{\rm BIASED} \setminus \mathcal{S}_{\rm TRUE} = \emptyset$. Zero objects exist in `BIASED` that were not present in `TRUE`.

4. **Object ID Stability**:  
   `object_id` values are stored consistently as 64-bit integers (`int64`) across all stages. No floating-point conversion, truncation, or re-indexing occurred.

---

## 2. Selection Function Audit

The AEGIS selection function (ADR 004) models spectroscopic follow-up probability $p_{\rm spec}$ as a function of host-galaxy photometric redshift $z_{\rm phot}$:

$$p_{\rm spec}(z_{\rm phot}) = p_{\rm floor} + \frac{p_{\rm bright} - p_{\rm floor}}{1 + \exp\!\left(\dfrac{z_{\rm phot} - z_{50}}{w_z}\right)}$$

Configured defaults: $p_{\rm floor} = 0.10$, $p_{\rm bright} = 0.80$, $z_{50} = 0.50$, $w_z = 0.15$, ${\rm seed} = 42$.

### Pre-Spectroscopic Observable Leakage Inspection

| Feature | Category | Included in $p_{\rm spec}$ Calculation? | Leakage Risk Assessment |
|---|---|---|---|
| `hostgal_photoz` | Pre-trigger / Host metadata | **YES** (sole input) | **PASS** — Observable prior to spectroscopic follow-up. |
| `hostgal_photoz_err` | Host metadata | NO | **PASS** — Retained in metadata, unused in selection. |
| `target` | Competition placeholder | NO | **PASS** — Unused (all zeros in unblinded dataset). |
| `true_target` | True astrophysical class label | NO | **PASS** — Strictly excluded from selection formula. |
| `hostgal_specz` | Host spectroscopic redshift | NO | **PASS** — Strictly excluded (contains post-followup info). |
| Light curve fluxes (`tflux_*`) | Alert time-series | NO | **PASS** — Selection is evaluated purely from host photo-$z$. |
| Simulation truth fields (`true_z`, etc.) | Post-event truth | NO | **PASS** — Strictly excluded from selection formula. |

> [!NOTE]
> **Audit Confirmation:** The selection function is strictly pre-spectroscopic and class-blind. It operates without access to transient labels, light curve data, or spectroscopic truth.

### Selection Function Visualizations

The following supporting plots were generated under `docs/results/data_audit/`:

1. **Selection Function Curve** (`selection_prob_vs_photoz.png`):  
   Visualizes theoretical $p_{\rm spec}(z)$, illustrating the smooth transition from $p_{\rm bright} = 0.80$ at low redshift to $p_{\rm floor} = 0.10$ at high redshift, crossing the midpoint $p = 0.45$ at $z_{50} = 0.50$.
   
   ![Selection Probability vs Photo-z](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/data_audit/selection_prob_vs_photoz.png)

2. **Empirical Retained Fraction vs. Theory** (`retained_fraction_vs_photoz.png`):  
   Compares the empirical binned selection fraction in redshift bins of width $\Delta z = 0.05$ against the analytical $p_{\rm spec}(z)$ curve. The empirical points closely trace the theoretical logistic curve, confirming correct Bernoulli sampling.
   
   ![Retained Fraction vs Photo-z](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/data_audit/retained_fraction_vs_photoz.png)

3. **Selection Probability Histogram** (`selection_prob_histogram.png`):  
   Shows the distribution of assigned $p_{\rm spec}$ probabilities across the TRUE population. The distribution has a mean of $0.384$ and median of $0.347$, reflecting the population density weighted by redshift.
   
   ![Selection Probability Histogram](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/data_audit/selection_prob_histogram.png)

---

## 3. Distributional Bias Characterization

To quantify the selection bias introduced between `TRUE` and `BIASED`, we evaluated distance and divergence metrics across all available numerical features: Two-Sample Kolmogorov-Smirnov (KS) test, Earth Mover's Distance (EMD / Wasserstein-1), and Jensen-Shannon (JS) divergence (50-bin PDF).

### Quantitative Distributional Shift Metrics

| Feature | TRUE Mean | BIASED Mean | Mean Shift ($\Delta$) | Median Shift | KS Statistic | KS $p$-value | Earth Mover Distance (EMD) | JS Divergence |
|---|---|---|---|---|---|---|---|---|
| `hostgal_photoz` | 0.6280 | 0.4867 | -0.1414 | -0.1180 | 0.212653 | $< 10^{-300}$ | 0.141358 | 0.180257 |
| `distmod` | 42.5225 | 41.9026 | -0.6199 | -0.5980 | 0.212664 | $< 10^{-300}$ | 0.619885 | 0.179878 |
| `true_z` | 0.5770 | 0.4787 | -0.0982 | -0.0980 | 0.189779 | $< 10^{-300}$ | 0.098241 | 0.159119 |
| `true_distmod` | 42.3824 | 41.8948 | -0.4876 | -0.4980 | 0.189787 | $< 10^{-300}$ | 0.487607 | 0.159222 |
| `hostgal_photoz_err` | 0.1437 | 0.1643 | +0.0206 | -0.0010 | 0.037974 | $< 10^{-300}$ | 0.021714 | 0.046000 |
| `hostgal_specz` | -8.7573 | -8.6574 | +0.0999 | 0.0000 | 0.010805 | $4.27 \times 10^{-48}$ | 0.100140 | 0.025208 |
| `mwebv` | 0.0777 | 0.0853 | +0.0076 | +0.0030 | 0.022804 | $8.83 \times 10^{-213}$ | 0.007603 | 0.023370 |
| `ra` | 171.1108 | 171.1215 | +0.0107 | -0.1757 | 0.013670 | $9.93 \times 10^{-77}$ | 2.315647 | 0.019932 |
| `decl` | -26.0552 | -26.0158 | +0.0394 | +0.1644 | 0.002256 | 0.0167 | 0.069382 | 0.006836 |
| `ddf_bool` | 0.0075 | 0.0041 | -0.0034 | 0.0000 | 0.003372 | $4.54 \times 10^{-5}$ | 0.003372 | 0.015776 |

### Feature Shift Highlights

- **Strongest Shift ($\text{KS} \approx 0.213$, $\text{JS} \approx 0.180$)**: `hostgal_photoz` and `distmod`. Selection heavily suppresses high-redshift objects ($z > 0.50$), reducing the mean host photo-$z$ from $0.6280$ to $0.4867$ and shifting distance modulus by $-0.62$ mag.
- **Moderate Shift ($\text{KS} \approx 0.190$)**: True astrophysical redshift `true_z` and `true_distmod`. Conforms expected correlation between host photo-$z$ and true transient redshift.
- **Minimal/Negligible Shift ($\text{JS} < 0.02$)**: Sky coordinates (`ra`, `decl`), extinction (`mwebv`), and DDF indicator (`ddf_bool`). Confirms selection function is spatial-blind and extinction-blind.

### Feature Shift Visualizations

1. **Host Photo-$z$ Shift** (`hostgal_photoz_comparison.png`):  
   Shows clear leftward shift in redshift distribution from TRUE (blue) to BIASED (orange).
   
   ![Host Photo-z Comparison](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/data_audit/hostgal_photoz_comparison.png)

2. **Distance Modulus Shift** (`distmod_comparison.png`):  
   Demonstrates truncation of faint/distant host galaxies ($\mu > 43$).
   
   ![Distance Modulus Comparison](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/data_audit/distmod_comparison.png)

3. **Host Photo-$z$ Uncertainty** (`hostgal_photoz_err_comparison.png`):  
   Illustrates secondary distortion in photo-$z$ uncertainty profile.
   
   ![Host Photo-z Uncertainty Comparison](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/data_audit/hostgal_photoz_err_comparison.png)

4. **Milky Way Extinction** (`mwebv_comparison.png`):  
   Confirms extinction profile remains unshifted across selection.
   
   ![Milky Way Extinction Comparison](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/data_audit/mwebv_comparison.png)

---

## 4. Class Balance Analysis

The selection function yields substantial differences in retention rates across the three study classes:
- **Kilonova (KN / Class 64)**: Retained at **58.65%** ($78 / 133$)
- **Type Ia Supernova (SN Ia / Class 90)**: Retained at **38.85%** ($644,876 / 1,659,831$)
- **Superluminous SN (SLSN-I / Class 95)**: Retained at **15.11%** ($5,405 / 35,782$)

### Audit Inquiry: Is Class Retention Bias Driven by Implementation or Physics/Data?

To answer this, we analyzed the intrinsic redshift distributions of each transient class prior to selection:

| Class ID | Class Name | TRUE Count | BIASED Count | Retention Rate | TRUE Mean $z_{\rm phot}$ | TRUE Median $z_{\rm phot}$ | Fraction with $z_{\rm phot} < z_{50} (0.50)$ | Expected Mean $p_{\rm spec}$ |
|---|---|---|---|---|---|---|---|---|
| **64** | Kilonova (KN) | 133 | 78 | **58.65%** | 0.4063 | 0.1400 | **86.47%** | 0.6538 |
| **90** | Type Ia Supernova (SN Ia) | 1,659,831 | 644,876 | **38.85%** | 0.6110 | 0.5730 | **37.76%** | 0.3887 |
| **95** | Superluminous SN (SLSN-I) | 35,782 | 5,405 | **15.11%** | 1.4175 | 1.3850 | **5.36%** | 0.1510 |

### Findings

1. **Root Cause**: The retention disparity is **100% driven by the intrinsic redshift distribution of the simulated classes** in PLAsTiCC, combined with the redshift-dependent selection function.
   - **Kilonovae (Class 64)** are low-redshift transients in the simulation ($86.5\%$ at $z < 0.50$). Consequently, they fall into the high-$p_{\rm spec}$ regime ($p \approx 0.65\text{--}0.80$).
   - **SLSN-I (Class 95)** are rare, extremely luminous transients detected out to high redshifts (mean $z = 1.42$, only $5.4\%$ at $z < 0.50$). They fall into the asymptotic floor regime ($p_{\rm spec} \to p_{\rm floor} = 0.10$).
   - **SN Ia (Class 90)** span the intermediate range (mean $z = 0.61$), yielding an average retention near the global population mean ($38.85\%$).

2. **Class-Blind Verification**:  
   Code inspection confirms `logistic_spec_probability` takes *only* `photoz` as input. Class labels (`true_target`) are **never passed to or evaluated by the selection function**. The class imbalance shift is an emergent property.

### Class Redshift & Retention Mechanics Visualization

![Class Redshift Distributions](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/data_audit/class_redshift_distributions.png)

---

## 5. Sensitivity Analysis

To evaluate how choices of $p_{\rm floor}, p_{\rm bright}, z_{50}, w_z$ impact population balance without altering baseline pipeline code, we computed expected retention rates across 13 parameter variations:

### Parameter Sensitivity Evaluation Matrix

| Scenario | $p_{\rm floor}$ | $p_{\rm bright}$ | $z_{50}$ | $w_z$ | Overall Expected Retention | KN Retention | SN Ia Retention | SLSN-I Retention | KN / SN Ia Ratio | SLSN / SN Ia Ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | **0.10** | **0.80** | **0.50** | **0.15** | **38.37%** | **65.38%** | **38.87%** | **15.10%** | **1.68** | **0.39** |
| `p_floor=0.00` | 0.00 | 0.80 | 0.50 | 0.15 | 32.43% | 63.29% | 33.00% | 5.83% | 1.92 | 0.18 |
| `p_floor=0.05` | 0.05 | 0.80 | 0.50 | 0.15 | 35.40% | 64.34% | 35.94% | 10.46% | 1.79 | 0.29 |
| `p_floor=0.20` | 0.20 | 0.80 | 0.50 | 0.15 | 44.32% | 67.47% | 44.75% | 24.37% | 1.51 | 0.54 |
| `p_bright=0.60` | 0.10 | 0.60 | 0.50 | 0.15 | 30.27% | 49.56% | 30.62% | 13.64% | 1.62 | 0.45 |
| `p_bright=0.95` | 0.10 | 0.95 | 0.50 | 0.15 | 44.45% | 77.25% | 45.06% | 16.19% | 1.71 | 0.36 |
| `z_50=0.30` | 0.10 | 0.80 | 0.30 | 0.15 | 24.38% | 54.28% | 24.64% | 12.42% | 2.20 | 0.50 |
| `z_50=0.40` | 0.10 | 0.80 | 0.40 | 0.15 | 30.86% | 60.88% | 31.23% | 13.59% | 1.95 | 0.44 |
| `z_50=0.60` | 0.10 | 0.80 | 0.60 | 0.15 | 46.34% | 68.22% | 46.97% | 16.97% | 1.45 | 0.36 |
| `z_50=0.80` | 0.10 | 0.80 | 0.80 | 0.15 | 60.86% | 70.92% | 61.69% | 21.94% | 1.15 | 0.36 |
| `w_z=0.05` | 0.10 | 0.80 | 0.50 | 0.05 | 36.44% | 70.52% | 36.92% | 13.89% | 1.91 | 0.38 |
| `w_z=0.10` | 0.10 | 0.80 | 0.50 | 0.10 | 37.42% | 68.73% | 37.91% | 14.34% | 1.81 | 0.38 |
| `w_z=0.25` | 0.10 | 0.80 | 0.50 | 0.25 | 39.80% | 59.16% | 40.29% | 17.22% | 1.47 | 0.43 |

### Key Sensitivity Insights

1. **Midpoint $z_{50}$ is the Primary Control Knob**: Moving $z_{50}$ from $0.50 \to 0.30$ reduces overall retention from $38.4\% \to 24.4\%$, amplifying the relative excess of Kilonovae over SN Ia (ratio increases from $1.68 \to 2.20$).
2. **$p_{\rm floor}$ Governs High-$z$ Sample Size**: Setting $p_{\rm floor} = 0.00$ severely starves the SLSN-I sample ($5.83\%$ retention, $\sim 2,086$ objects left), whereas $p_{\rm floor} = 0.20$ quadruples SLSN-I retention ($24.37\%$).
3. **Recommended Sensitivity Sweep Plan for Model Evaluation Phase**:
   - Primary axis: $z_{50} \in [0.35, 0.65]$ in steps of $0.05$.
   - Secondary axis: $p_{\rm floor} \in [0.05, 0.20]$.

---

## 6. Limitations & Modeling Assumptions

| Assumption | Rationale | Scientific Consequence | Expected Effect on Calibration Experiments |
|---|---|---|---|
| **1. Logistic Functional Form** | Smooth approximation of flux-limited / magnitude-limited selection threshold. | Real spectroscopic follow-up selection exhibits step-function cutoffs in magnitude and target-dependent prioritization. | Softens the selection boundary compared to hard magnitude cuts. Calibration models will see smooth probability gradients. |
| **2. Host Photo-$z$ Proxy** | Direct apparent magnitude is unavailable in metadata without light curve parsing. | Ignores transient intrinsic luminosity variations (bright transients at high $z$ might be selected in reality but are penalized by host $z$). | May over-penalize high-redshift superluminous supernovae (SLSN-I) relative to a true magnitude-limited survey. |
| **3. Class-Blind Selection** | Prevents artificial bias injection based on simulation labels. | Real follow-up programs actively select targets based on preliminary classification (e.g., targeting rare Kilonovae). | Calibration models will be tested against selection bias that does NOT prefer rare classes, providing a clean baseline. |
| **4. Small Kilonova Sample ($N=133$)** | Inherent simulation limit of PLAsTiCC unblinded test dataset. | Extreme statistical variance in Kilonova performance metrics ($78$ objects in `BIASED`). | High variance in KN calibration curves and confidence intervals. Stratified sampling will be mandatory in model training. |

---

## 7. Research Readiness Assessment

### Checklist Verification

- **✓ Is the pipeline scientifically consistent with the research question?**  
  **YES.** The pipeline cleanly isolates the `TRUE` vs. `BIASED` populations, providing a controlled testbed to measure calibration degradation under spectroscopic follow-up selection.

- **✓ Is the proxy selection function clearly separated from published facts?**  
  **YES.** Manifests, code docstrings, and `docs/data/selection_function.md` prominently feature the mandatory disclaimer: `MODELING ASSUMPTION: logistic proxy, not a measured survey selection function`.

- **✓ Are any assumptions insufficiently documented?**  
  **NO.** All parameters, formulas, schema validation constraints, and file provenance records are fully documented in `docs/data/` and versioned in `configs/data_population.yaml`.

- **✓ Is there evidence that the induced bias is large enough to make calibration experiments meaningful?**  
  **YES.** The selection induces significant statistical shifts ($\text{KS} = 0.213$, $\text{JS} = 0.180$, mean redshift shift $\Delta z = -0.1414$), and distorts class balance ($15.1\%$ SLSN-I retention vs. $58.6\%$ KN retention). This creates a realistic challenge for model calibration algorithms.

- **✓ What weaknesses should be addressed before feature extraction & model development?**  
  The small Kilonova sample size ($N=133$ in TRUE, $N=78$ in BIASED) requires careful metric handling (e.g., bootstrapping, exact confidence bounds).

---

## Final Audit Findings & Readiness Score

### Strengths
1. **Mathematical Rigor & Reproducibility**: 100% deterministic, seeded, independently rerunnable stage pipeline with SHA-256 manifest chains.
2. **Pre-Spectroscopic Strictness**: Zero leakage of post-event, light-curve, or true label information into selection calculations.
3. **Substantial Induced Bias**: Demonstrable distributional shift ($\text{KS} = 0.213$) providing a robust benchmark for post-hoc calibration methods.

### Weaknesses
1. **Kilonova Sample Starvation**: Only 133 Kilonova objects exist in the entire TRUE population.
2. **Proxy Simplicity**: Selection relies solely on host photo-$z$ rather than full multi-band peak apparent magnitude.

### Major Risks
- High statistical uncertainty on Kilonova evaluation metrics due to $N=78$ in BIASED population.

### Recommended Changes Before Model Development
1. Enforce stratified splitting during dataset partitioning for model training/validation.
2. Implement bootstrap resampling for evaluation metrics to provide honest confidence intervals on class 64 (KN).

---

### Overall Readiness Score

$$\mathbf{9.5 \,/\, 10}$$

> **Conclusion:** The AEGIS data acquisition and TRUE vs. BIASED population construction pipeline is **APPROVED** for research progression. The pipeline is scientifically sound, fully reproducible, strictly pre-spectroscopic, and verified against all data integrity contracts.
