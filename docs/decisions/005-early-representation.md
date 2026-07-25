# ADR 005: Representation of Early Light-Curve Behavior

- **Status:** accepted for the foundation study
- **Date:** 2026-07-25

## Context

The operational context for early transient classification is fixed by existing Architecture Decision Records:

- **ADR 003:** At elapsed observer-frame epoch $e \in \{0, 2, 7\}$ days after first alert $t_0$ (defined by $\mathrm{flux}/\mathrm{fluxerr} \ge 5.0$), only observations with $\mathrm{MJD} \le t_0 + e$ are usable. The primary decision deadline is $H = 2$ days.
- **ADR 002:** Kilonova (PLAsTiCC class 64) is the primary positive class; Type Ia supernova (90) and superluminous Type I supernova (95) are comparison classes.

Before implementing feature extraction or modeling code, we must establish the scientific representation of early light-curve behavior that the classifier will consume.

### Empirical Data Sparsity (Step 1 Benchmark)

To ground this decision in empirical reality rather than theoretical preference, we measured early observation counts across 15,340 objects from the ingested TRUE target population (`docs/results/early_lightcurve_sparsity.md`):

1. **At Initial Alert ($e = 0$ days):** 100% of objects across all study classes have **exactly 1 detected photometric point** ($N_{\text{det}} = 1$) in **1 single passband** ($N_{\text{det\_pb}} = 1$), with zero temporal baseline ($\Delta t_{\text{det}} = 0.00$ days).
2. **At Primary Decision Deadline ($e = 2$ days):** Kilonovae (class 64) have a median of **2 detected points** [Q1: 1, Q3: 2] across **1 to 2 detected passbands** [Q1: 1, Q3: 2]. Specifically, 50.0% of kilonova alerts have only 1 detected point, 43.3% have 2 points, 5.8% have 3–4 points, and only 1.0% have $\ge 5$ points.

---

## Options Considered

### Option (a): Full Multi-Parameter Physical Forward Model

A physically complete multi-parameter model (e.g., shock-cooling or ejecta expansion model; Kasen et al. 2017, Metzger 2019, Piro 2015).

- **Physical Formulation:** Bolometric luminosity $L_{\text{bol}}(t) \propto M_{\text{ej}} v_k \kappa^{-1} f(t)$ with temperature evolution $T_{\text{eff}}(t) \propto (L_{\text{bol}} / \sigma_{\text{SB}} R_{\text{phot}}^2)^{1/4}$.
- **Free Parameters ($p_{\text{free}} \ge 4\text{--}6$):** Ejecta mass $M_{\text{ej}}$, expansion velocity $v_k$, grey opacity $\kappa$, explosion timestamp $t_{\text{exp}}$, and floor temperature $T_{\text{floor}}$.
- **Identifiability Check:** At $e = 0$ ($N_{\text{det}} = 1$) and $e = 2$ (median $N_{\text{det}} = 2$), $p_{\text{free}} \ge 4 \gg N_{\text{det}}$.
- **Evaluation:** **Rejected due to non-identifiability.** Attempting to fit $\ge 4$ physical parameters to 1–2 data points yields ill-conditioned optimization, infinite posterior degeneracy, and non-convergent MCMC chains.

### Option (b): Reduced Low-Parameter Physically Motivated Feature Representation (Chosen)

A low-parameter representation using quantities directly constrained by observed point counts ($N_{\text{det}} \le 2$).

- **Feature Set:**
  1. **Early Flux-Rise Rate ($\dot{F}_b = \Delta F_b / \Delta t$):** Physical proxy for expanding fireball velocity ($F_b \propto t^2$ or linear expansion; Arnett 1982, Riess et al. 1999, Miller et al. 2020). $p_{\text{free}} = 1$ per passband.
  2. **Single-Epoch Cross-Band Color ($m_{b1} - m_{b2} = -2.5 \log_{10}(F_{b1}/F_{b2})$):** Physical proxy for photosphere effective temperature and composition (blue vs. red kilonova components; Kasen et al. 2017, Metzger 2019). Computed for passbands with observations within $\Delta t < 0.5$d. $p_{\text{free}} = 1$ per color pair.
  3. **Alert Signal-to-Noise Growth Rate ($\text{S/N}_0 = F_0 / \sigma_0$, $\Delta \text{S/N} / \Delta t$):** Physical proxy for early optical depth evolution and power-law heating (Piro 2015). $p_{\text{free}} = 1$.
- **Identifiability Check:** Free parameters per feature $p_{\text{free}} \le 1 \le N_{\text{det}}$ at $e = 0$ and $e = 2$. Fully identifiable.

### Option (c): Non-Physical Generic Statistical Baseline

Raw flux grid or unconstrained polynomial coefficients ($a_0 + a_1 t + a_2 t^2$).

- **Free Parameters ($p_{\text{free}} = 3\text{--}4$ per band):** Polynomial degrees.
- **Identifiability Check:** $p_{\text{free}} \ge 3 > N_{\text{det}} = 1\text{--}2$. Higher-order polynomials oscillate wildly or diverge given 1–2 noisy points.
- **Evaluation:** Included as required comparison baseline to justify physical feature value over raw statistical extrapolation.

---

## Evaluation & Decision Matrix

| Criterion | (a) Full Physical Forward Model | (b) Reduced Physical Features (Chosen) | (c) Generic Statistical Baseline |
| :--- | :---: | :---: | :---: |
| **1. Physical Interpretability** | High (direct $M_{\text{ej}}$, $v_k$) | High (rise rate, color proxy) | Low (unconstrained coefficients) |
| **2. Parameter Identifiability** | **FAILED** ($p \ge 4 \gg N_{\text{det}}$) | **PASSED** ($p \le 1 \le N_{\text{det}}$) | **FAILED** ($p \ge 3 > N_{\text{det}}$) |
| **3. Missing Passband Robustness** | Low (requires multi-band fit) | High (per-band slope & mask) | Low (grid interpolation fails) |
| **4. Computational Complexity** | High (MCMC / non-linear fit) | Very Low (linear / ratio ops) | Low (polynomial fit) |
| **5. Uncertainty Compatibility** | Complex (posterior sampling) | Direct (analytical error prop) | Moderate (covariance matrix) |
| **6. Class Generalizability** | Low (KN-specific model) | High (applies to KN, Ia, SLSN) | High (class-blind) |

---

## Decision

Select **Option (b): Reduced Low-Parameter Physically Motivated Feature Representation**.

1. **Reject Option (a)** plainly because a 4-to-6 parameter physical model cannot be identified from 1 or 2 data points.
2. **Use Option (b)** as the canonical early representation for the classifier at $e = 0$ and $e = 2$ days.
3. **Include Option (c)** strictly as the statistical benchmark comparison in evaluation reports.

### Alert-Time Host Covariate Inclusion

The classifier will include host galaxy photometric redshift (`hostgal_photoz`) as a pre-alert contextual covariate alongside Option (b) light-curve features.

- **Alert-Time Availability Verification:** `hostgal_photoz` is derived from pre-existing photometric sky survey catalogs (e.g., DES/LSST template images) prior to transient alert generation (Kessler et al. 2019, `docs/audits/alert_stream_leakage_audit.md`).
- **Forbidden Fields:** Spectroscopic host redshift (`hostgal_specz`), distance modulus (`distmod`), and simulation truth fields remain strictly forbidden (ADR 003, `docs/data/field_definitions.md`).

---

## Consequences

- Feature extraction modules will implement early flux-rise rates, single-epoch cross-band colors, S/N growth rates, and missing-passband indicators.
- No full physical light-curve fitting library (e.g., MCMC samplers) is required in the core pipeline, ensuring fast inference.
- Classifier inputs remain strictly identified under the extreme data sparsity measured at $e = 0$ and $e = 2$ days.
