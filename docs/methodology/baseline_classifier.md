# Early-Epoch Baseline Multiclass Classifier Methodology

> **Document ID:** `docs/methodology/baseline_classifier.md`  
> **Date:** July 28, 2026  
> **Decision Reference:** ADR 003, ADR 004, ADR 005  
> **Module Implementation:** `src/aegis/models/baseline.py` & `src/aegis/config/models.py`  

---

## 1. Executive Summary

This document specifies the model architecture, feature representation inputs, missingness handling strategy, population boundary constraints, and probabilistic evaluation design for the early-epoch baseline classifier in AEGIS.

Per **ADR 003** and **ADR 005**, classification must occur at elapsed observer-frame decision epochs $e \in \{0, 2, 7\}$ days after initial alert ($t_0$), predicting probabilities over the three study classes:
1. **Kilonova (KN)** — PLAsTiCC class ID 64 (primary time-critical positive class)
2. **Type Ia Supernova (SN Ia)** — PLAsTiCC class ID 90 (common comparison class)
3. **Superluminous Type I Supernova (SLSN-I)** — PLAsTiCC class ID 95 (rare, slower comparison class)

Per **ADR 004** and pipeline integrity requirements, all classifier fitting, validation, and hyperparameter tuning are conducted **exclusively on the biased/labeled ($S=1$) population**. The deployment ($S=0$ / TRUE) population is strictly reserved for subsequent evaluation and is never touched during classifier training or model selection.

---

## 2. Model Family Choice & Trade-off Rationale

### 2.1 Selected Family: Histogram-Based Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`)

We select **Histogram-Based Gradient Boosted Decision Trees (GBDT)** (`sklearn.ensemble.HistGradientBoostingClassifier`) as the baseline model family.

#### Rationale for Selection:
1. **Native Support for Extreme Physical Missingness**: At early decision epochs ($e = 0, 2$), missingness in physical light-curve features (rise rates $\dot{F}_b$, cross-band colors $c_{b1,b2}$, S/N growth rate) is severe (>98% at $e=2$d; 100% for multi-point slopes at $e=0$d). GBDTs natively evaluate `NaN` as an explicit split branch during tree building, learning optimal missing-value routing without requiring artificial numeric imputation.
2. **Proper Probabilistic Simplex**: GBDTs trained with multinomial cross-entropy (log-loss) output predictions via softmax that form a valid probability simplex $P(Y=c|X)$ where $p_{ic} \ge 0$ and $\sum_c p_{ic} = 1.0$.
3. **Robustness in Small, Tabular Regimes**: Tabular decision trees outperform deep learning architectures on sparse, small-to-moderate tabular datasets, avoiding overfitting while capturing non-linear interactions between host covariates (`hostgal_photoz`) and light-curve growth rates.

### 2.2 Alternative Model Families Considered & Justified Against

| Model Family | Handling of Missing Features (`NaN`) | Risk / Limitation | Decision |
| :--- | :--- | :--- | :--- |
| **HistGradientBoosting / LightGBM (Selected)** | **Native** (Tree split evaluates `NaN` branch direction) | None for tabular data; fast, robust | **CHOSEN** |
| **Logistic Regression (Multinomial)** | **Requires Imputation** (Cannot handle `NaN` inputs) | Imputing NaNs with zero or mean distorts physical meaning of unconstrained early features | **REJECTED** |
| **Standard Random Forest** | **Requires Imputation** (Cannot handle `NaN` inputs) | Forces artificial imputation or dropping rows with missing features | **REJECTED** |
| **Deep Multilayer Perceptron (MLP)** | **Requires Masking / Imputation** | High propensity to overfit on small $S=1$ samples; non-trivial missing feature encoding | **REJECTED** |

---

## 3. Explicit Support & Fit-Quality Diagnostic Inputs

Per project guidelines, the classifier **does not silently drop objects with poorly constrained features**, nor does it obscure early fit unconstrainability. Instead, fit-quality diagnostics and constraint statuses produced by `aegis.features.representation` are passed directly into the model feature matrix as explicit inputs:

1. **Physical Feature Values**: `rise_rate_pb0`..`pb5`, `color_pb0_pb1`..`color_pb4_pb5`, `alert_snr_0`, `snr_growth_rate`, `hostgal_photoz` (unconstrained features remain `NaN`).
2. **Feature Analytical Uncertainties**: `*_err` analytical uncertainties derived from error propagation.
3. **Feature Status Encodings**: `*_status` integer-encoded status flags (`WELL_CONSTRAINED` = 0, `UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS` = 1, `UNCONSTRAINED_ZERO_BASELINE` = 2, `UNCONSTRAINED_NON_POSITIVE_FLUX` = 3, `UNCONSTRAINED_NO_PASSBAND_PAIR` = 4, `UNCONSTRAINED_NO_DETECTION` = 5).
4. **Summary Diagnostics**:
   - `diag_n_obs_total`: Total observations up to epoch cutoff.
   - `diag_n_det_total`: Total detected observations ($S/N \ge 5.0$).
   - `diag_n_det_passbands`: Number of passbands with $\ge 1$ detection.
   - `diag_det_time_span_days`: Elapsed time between first and last detection.
   - `diag_well_constrained_features`: Count of well-constrained physical features.
   - `diag_unconstrained_features`: Count of unconstrained physical features.

Tree splits can directly isolate objects where `diag_n_det_total < 2` or where status flags indicate zero temporal baseline.

---

## 4. Biased ($S=1$) Population Boundary Policy

> [!IMPORTANT]
> **Strict Training Isolation**
> All model fitting, cross-validation, and hyperparameter tuning must be performed **only on the biased/labeled ($S=1$) population**.

- The `BaselineClassifier.fit_epoch` method enforces `population_type == "BIASED"` and inspects input metadata DataFrames for `S == 1` or `population == "BIASED"`.
- Passing any deployment ($S=0$ / TRUE) population objects to fitting routines raises a runtime `ValueError`.
- Deployment evaluation is reserved strictly for subsequent evaluation modules.

---

## 5. Epoch-Indexed Architecture & Predictions

Epoch-specific models $M_e$ are trained for each decision epoch $e \in \{0, 2, 7\}$ days:

- **Epoch $e = 0.0$d (Initial Alert)**: Information strictly truncated at $t_0$. Physical light-curve slopes and colors are 100% unconstrained (`NaN`), and predictions rely on host photo-$z$ and initial alert S/N.
- **Epoch $e = 2.0$d (Primary Deadline $H = 2$d)**: Partial observations up to $t_0 + 2.0$ days. Median observation count is 2 points.
- **Epoch $e = 7.0$d (Diagnostic Horizon)**: Extended photometric baseline up to $t_0 + 7.0$ days.

Output predictions `predict_proba(X, epoch)` return an $(N, 3)$ matrix aligned to class IDs `[64, 90, 95]` satisfying $p_{ic} \ge 0$ and $\sum_{c} p_{ic} = 1.0$.
