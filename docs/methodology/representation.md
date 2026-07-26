# Early Light-Curve Feature Representation & Identifiability Methodology

> **Document ID:** `docs/methodology/representation.md`  
> **Date:** July 25, 2026  
> **Decision Reference:** ADR 005 (Option b: Low-Parameter Physically Motivated Representation)  
> **Module Implementation:** `src/aegis/features/representation.py` & `src/aegis/config/features.py`  

---

## 1. Executive Summary

This document specifies the canonical mathematical formulas, error propagation derivations, citations, support diagnostics, and empirical constrainability findings for the early light-curve feature representation used by AEGIS.

Per **ADR 005**, full multi-parameter physical forward modeling ($p_{\text{free}} \ge 4\text{--}6$) is mathematically unidentifiable under early alert sparsity ($N_{\text{det}} = 1\text{--}2$). We strictly implement **Option (b)**: a low-parameter representation ($p_{\text{free}} \le 1$) consisting of early flux-rise rates, single-epoch cross-band colors, alert S/N growth statistics, and pre-alert host galaxy photo-$z$ covariates.

---

## 2. Mathematical Definitions & Error Propagation

### 2.1 Passband Early Flux-Rise Rate ($\dot{F}_b$)

The early flux-rise rate $\dot{F}_b \equiv \frac{\mathrm{d}F_b}{\mathrm{d}t}$ serves as a physical proxy for expanding fireball velocity ($F_b \propto t^2$ or linear expansion; Arnett 1982, Riess et al. 1999, Miller et al. 2020).

#### Case 1: Two-Point Fit ($N_{\text{det}, b} = 2$)
For two detected observations $(t_1, F_1, \sigma_{F1})$ and $(t_2, F_2, \sigma_{F2})$ in passband $b$ with $\Delta t = t_2 - t_1 > 0$:

$$\dot{F}_b = \frac{F_2 - F_1}{t_2 - t_1}$$

**Analytical Error Propagation:**

$$\sigma_{\dot{F}_b} = \frac{\sqrt{\sigma_{F_1}^2 + \sigma_{F_2}^2}}{t_2 - t_1}$$

#### Case 2: Weighted Least Squares Fit ($N_{\text{det}, b} > 2$)
For $N > 2$ detected observations, fit linear model $F(t) = a + \dot{F}_b \cdot t$ with weights $w_i = 1 / \sigma_{Fi}^2$:

$$\dot{F}_b = \frac{\left(\sum w_i\right) \left(\sum w_i t_i F_i\right) - \left(\sum w_i t_i\right) \left(\sum w_i F_i\right)}{\Delta}$$

where $\Delta = \left(\sum w_i\right) \left(\sum w_i t_i^2\right) - \left(\sum w_i t_i\right)^2$.

**Analytical Error Propagation:**

$$\sigma_{\dot{F}_b} = \sqrt{\frac{\sum w_i}{\Delta}}$$

**Support Diagnostic:**
- `n_det`: Number of detected observations in passband $b$.
- `dt_days`: Time baseline $t_{\text{max}} - t_{\text{min}}$.
- `chi2_red`: Reduced $\chi^2 = \frac{1}{N-2} \sum w_i (F_i - (a + \dot{F}_b t_i))^2$.
- `condition_number`: Condition number of design matrix $X = [\mathbf{1}, \mathbf{t}]$.

---

### 2.2 Single-Epoch Cross-Band Color ($c_{b1, b2}$)

Single-epoch cross-band color $c_{b1, b2} \equiv m_{b1} - m_{b2}$ serves as a physical proxy for photosphere effective temperature and opacity (Kasen et al. 2017, Metzger 2019). It is evaluated for detected passband pairs $(b1, b2)$ observed within a maximum temporal separation $\Delta t \le 0.5$ days (ADR 005).

$$c_{b1, b2} = m_{b1} - m_{b2} = -2.5 \log_{10}\left(\frac{F_{b1}}{F_{b2}}\right)$$

**Analytical Error Propagation:**

$$\sigma_{c_{b1, b2}} = \frac{2.5}{\ln 10} \sqrt{\left(\frac{\sigma_{F_{b1}}}{F_{b1}}\right)^2 + \left(\frac{\sigma_{F_{b2}}}{F_{b2}}\right)^2}$$

**Support Diagnostic:**
- `dt_days`: $|t_{b1} - t_{b2}| \le 0.5$d.
- `flux_b1`, `flux_b2`: Measured fluxes (must satisfy $F_{b1} > 0, F_{b2} > 0$).

---

### 2.3 Alert Signal-to-Noise Statistics ($\text{S/N}_0$, $\Delta \text{S/N} / \Delta t$)

Alert S/N statistics measure optical depth evolution and power-law heating growth (Piro 2015).

1. **Initial Alert S/N ($\text{S/N}_0$):**
   $$\text{S/N}_0 = \frac{F(t_0)}{\sigma_F(t_0)}, \quad \sigma_{\text{S/N}_0} = 1.0$$
2. **S/N Growth Rate ($\Delta \text{S/N} / \Delta t$):**
   $$\frac{\Delta \text{S/N}}{\Delta t} = \frac{\text{S/N}(t_{\text{latest}}) - \text{S/N}(t_0)}{t_{\text{latest}} - t_0}, \quad \sigma_{\frac{\Delta \text{S/N}}{\Delta t}} = \frac{\sqrt{2}}{t_{\text{latest}} - t_0}$$

---

### 2.4 Pre-Alert Host Galaxy Photo-$z$ (`hostgal_photoz`)

Host galaxy photometric redshift acts as an alert-time contextual covariate (Kessler et al. 2019, ADR 005). Spectroscopic redshift (`hostgal_specz`) and distance modulus (`distmod`) remain strictly forbidden (ADR 003).

---

## 3. Missingness & Unconstrained Feature Handling Policy

> [!IMPORTANT]
> **No Heuristic Imputation**: Features with insufficient observations ($N_{\text{det}} < 2$ for rates, no pair within 0.5d for colors) strictly return `value = NaN` and `uncertainty = NaN`. Downstream models must consume structured `FeatureStatus` flags rather than imputed or default fallback numbers.

### Feature Status Taxonomy (`FeatureStatus`)
- `WELL_CONSTRAINED`: Feature satisfies all parameter identifiability and support criteria.
- `UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS`: $N_{\text{det}} < 2$ for rate fitting.
- `UNCONSTRAINED_ZERO_BASELINE`: $\Delta t = 0.0$ days (cannot compute rate).
- `UNCONSTRAINED_NON_POSITIVE_FLUX`: $F \le 0$ (magnitude/color undefined).
- `UNCONSTRAINED_NO_PASSBAND_PAIR`: No detected pair within $\Delta t \le 0.5$d.
- `UNCONSTRAINED_NO_DETECTION`: $N_{\text{det}} = 0$.

---

## 4. Empirical Constrainability Findings across Real TRUE Population

We benchmarked the representation across the real TRUE population at elapsed decision epochs $e = 0$ days (alert epoch) and $e = 2$ days (primary decision deadline $H = 2$ days). Results are recorded in `docs/results/representation_constrainability.json`.

### 4.1 Constrainability Summary Table (% Well-Constrained Objects)

| Transient Class | Epoch $e$ | Total Objects | `hostgal_photoz` (%) | `alert_snr_0` (%) | `snr_growth_rate` (%) | Rise Rates $\dot{F}_b$ (%) | Cross-Band Colors (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kilonova (KN 64)** | **0d** | 133 | **100.0%** | **1.5%** | **0.0%** | **0.0%** | **0.0%** |
| **Kilonova (KN 64)** | **2d** | 133 | **100.0%** | **1.5%** | **1.5%** | **0.0%** | **1.5%** |
| **SN Ia (90)** | **0d** | 1,659,831 | **100.0%** | **0.8%** | **0.0%** | **0.0%** | **0.0%** |
| **SN Ia (90)** | **2d** | 1,659,831 | **100.0%** | **0.8%** | **0.6%** | **0.01%** | **0.5%** |
| **SLSN-I (95)** | **0d** | 35,782 | **100.0%** | **0.3%** | **0.0%** | **0.0%** | **0.0%** |
| **SLSN-I (95)** | **2d** | 35,782 | **100.0%** | **0.3%** | **0.2%** | **0.01%** | **0.2%** |

---

## 5. Scientific Findings & Problem Difficulty

1. **Total Sparsity at Alert ($e = 0$ days):** At initial detection ($e = 0$), 100% of objects across all study classes have **0.0% well-constrained rise rates, colors, or S/N growth rates**. Every initial alert possesses exactly 1 detected point ($N_{\text{det}} = 1$) in 1 passband with zero temporal baseline ($\Delta t = 0$). Only `hostgal_photoz` is available.
2. **Severe Unconstrainability at Primary Deadline ($e = 2$ days):** At $H = 2$ days, **>98% of objects remain unconstrained** across all photometric light-curve features. For Kilonovae (class 64), 98.5% of objects cannot constrain a multi-point slope or cross-band color pair due to having only 1 observation or no overlapping passband pair within 0.5 days.
3. **Implication for Downstream Classification:** High missingness is a fundamental empirical property of real-time survey cadence, not a pipeline flaw. Downstream classifiers must be designed explicitly for extreme missingness (e.g. tree models handling NaN flags or masked encodings) rather than assuming dense light-curve coverage.
