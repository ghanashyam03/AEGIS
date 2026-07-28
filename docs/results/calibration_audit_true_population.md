# True-Population Calibration Audit Report (Finding #1)

## 1. Executive Summary

This report delivers the quantitative **True-Population Calibration Audit** for the frozen early-epoch baseline classifier in AEGIS, satisfying **ADR 003** and **ADR 005**. 

The primary objective is to measure and characterize the probabilistic calibration failure introduced by spectroscopic selection bias ($S=1$) when the uncorrected baseline model is deployed on the full **TRUE target population** ($S=1$ test set $N = 2,693$; $S=0$ unselected deployment set $N = 10,047$; FULL TRUE evaluation cohort $N = 12,740$) across elapsed observer-frame decision epochs $e \in \{0.0, 2.0, 7.0\}$ days.

In compliance with project directives, this audit is **strictly observational**: no bias correction, importance weighting, or recalibration has been applied.

### Key Quantitative Findings:
1. **Severe Brier Score Degradation ($BS$):** At initial alert ($e = 0.0$d), the multiclass Brier score degrades from $0.3412$ [0.3164, 0.3645] on the spectroscopically selected $S=1$ test set to $0.7211$ [0.7045, 0.7365] on the $S=0$ deployment set (**2.11x degradation**). On the FULL TRUE population, $BS = 0.6408$ [0.6272, 0.6549]. At the primary deadline ($e = 2.0$d), $BS = 0.6323$ [0.6177, 0.6471] on the FULL TRUE population.
2. **Reliability-Dominated Failure ($REL$):** Murphy Brier score decomposition ($BS = REL - RES + UNC$) proves that calibration failure is overwhelmingly driven by the **Reliability term ($REL$)**. At $e = 2.0$d, $REL$ increases from $0.3184$ [0.2933, 0.3425] on $S=1$ to $0.6953$ [0.6794, 0.7117] on $S=0$ (**2.18x shift**), while Resolution remains minimal ($RES \approx 0.0001$).
3. **Classwise ECE Inflation:** Mean Expected Calibration Error (ECE) on the FULL TRUE population is $26.12\%$ [25.66%, 26.61%] at $e = 0.0$d, $24.88\%$ [24.36%, 25.39%] at $e = 2.0$d, and $22.87\%$ [22.39%, 23.36%] at $e = 7.0$d.

---

## 2. Quantitative Calibration Metrics & Brier Decomposition

Metrics are computed using 10 equal-width probability bins on $[0, 1]$ per ADR 003. Uncertainty bounds are 95% percentile confidence intervals derived from $B = 1,000$ nonparametric object-level bootstrap resamples with seed=42.

### Table 1: Multiclass Brier Score & Murphy Decomposition Across Epochs & Populations

| Decision Epoch | Evaluation Population | Cohort Size ($N$) | Multiclass Brier Score $BS$ [95% CI] | Reliability $REL$ [95% CI] | Resolution $RES$ [95% CI] | Uncertainty $UNC$ [95% CI] | Mean ECE (%) [95% CI] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$e = 0.0$d** | **S=1 Test Set** | 2,693 | 0.3412 [0.3164, 0.3645] | 0.3268 [0.3022, 0.3503] | 0.0001 [0.0000, 0.0004] | 0.0133 [0.0074, 0.0199] | 14.53% [13.63%, 15.39%] |
| **$e = 0.0$d** | **S=0 Deployment** | 10,047 | 0.7211 [0.7045, 0.7365] | 0.7036 [0.6868, 0.7197] | 0.0001 [0.0000, 0.0001] | 0.0162 [0.0127, 0.0195] | 29.23% [28.66%, 29.74%] |
| **$e = 0.0$d** | **FULL TRUE Pop.** | 12,740 | **0.6408** [0.6272, 0.6549] | **0.6240** [0.6098, 0.6383] | 0.0001 [0.0000, 0.0001] | 0.0156 [0.0125, 0.0187] | **26.12%** [25.66%, 26.61%] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$e = 2.0$d** | **S=1 Test Set** | 2,693 | 0.3322 [0.3072, 0.3554] | 0.3184 [0.2933, 0.3425] | 0.0002 [0.0000, 0.0007] | 0.0133 [0.0074, 0.0199] | 13.21% [12.30%, 14.08%] |
| **$e = 2.0$d** | **S=0 Deployment** | 10,047 | 0.7128 [0.6967, 0.7285] | 0.6953 [0.6794, 0.7117] | 0.0001 [0.0001, 0.0002] | 0.0162 [0.0127, 0.0195] | 28.01% [27.48%, 28.55%] |
| **$e = 2.0$d** | **FULL TRUE Pop.** | 12,740 | **0.6323** [0.6177, 0.6471] | **0.6156** [0.6007, 0.6305] | 0.0001 [0.0001, 0.0002] | 0.0156 [0.0125, 0.0187] | **24.88%** [24.36%, 25.39%] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$e = 7.0$d** | **S=1 Test Set** | 2,693 | 0.2964 [0.2733, 0.3189] | 0.2825 [0.2596, 0.3054] | 0.0002 [0.0000, 0.0007] | 0.0133 [0.0074, 0.0199] | 11.95% [11.08%, 12.81%] |
| **$e = 7.0$d** | **S=0 Deployment** | 10,047 | 0.6433 [0.6280, 0.6583] | 0.6260 [0.6104, 0.6420] | 0.0001 [0.0001, 0.0003] | 0.0162 [0.0127, 0.0195] | 25.80% [25.27%, 26.34%] |
| **$e = 7.0$d** | **FULL TRUE Pop.** | 12,740 | **0.5700** [0.5568, 0.5838] | **0.5534** [0.5401, 0.5672] | 0.0001 [0.0001, 0.0002] | 0.0156 [0.0125, 0.0187] | **22.87%** [22.39%, 23.36%] |

> [!NOTE]
> **Murphy Identity Verification**  
> For all populations and decision epochs, the binned Brier score satisfies $BS_{\text{binned}} = REL - RES + UNC$ to numerical precision ($< 10^{-6}$). Total Brier score $BS$ differs from $BS_{\text{binned}}$ only by the within-bin variance of predicted probabilities.

---

## 3. Classwise Expected Calibration Error (ECE)

Classwise ECE is computed using 10 equal-width probability bins on $[0, 1]$. Empty bins are omitted from summation, and bin counts are explicitly tracked.

### Table 2: Classwise ECE Breakdown by Class and Population ($e = 2.0$d Primary Deadline)

| Class ID | Target Class Name | S=1 Test Set ECE [95% CI] | S=0 Deployment ECE [95% CI] | FULL TRUE Population ECE [95% CI] | Equal-Width Non-Empty Bins |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **64** | **Kilonova (KN)** | 0.25% [0.12%, 0.40%] | 0.15% [0.12%, 0.17%] | **0.17%** [0.13%, 0.20%] | Bins 1–3 ($P \le 0.30$) |
| **90** | **Type Ia Supernova (SN Ia)** | 19.82% [18.45%, 21.12%] | 42.01% [41.22%, 42.82%] | **37.32%** [36.54%, 38.08%] | All 10 Bins ($0 \le P \le 1.0$) |
| **95** | **Superluminous SN (SLSN-I)** | 19.57% [18.22%, 20.86%] | 41.87% [41.05%, 42.68%] | **37.15%** [36.37%, 37.91%] | All 10 Bins ($0 \le P \le 1.0$) |
| **Mean** | **All Study Classes** | **13.21%** [12.30%, 14.08%] | **28.01%** [27.48%, 28.55%] | **24.88%** [24.36%, 25.39%] | - |

---

## 4. Visualizations & Figure Walkthroughs

### Figure 6: Reliability Diagrams Across Decision Epochs
![Reliability Diagrams](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig6_true_pop_reliability_diagrams.png)

**Figure 6 Description:**
- Equal-frequency adaptive reliability curves for Kilonova (class 64) across decision epochs $e \in \{0.0, 2.0, 7.0\}$ days.
- Comparing $S=1$ spectroscopically selected test set (blue circles) against $S=0$ unselected deployment set (red squares).
- Demonstrates severe probability overconfidence on $S=0$ objects due to selection shift.

### Figure 7: Murphy Brier Score Decomposition
![Brier Decomposition](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig7_brier_decomposition_by_population.png)

**Figure 7 Description:**
- Breakdown of Murphy Reliability term $REL$ across decision epochs and populations with 95% bootstrap confidence error bars.
- Confirms that calibration error $REL$ accounts for over 97% of total Brier score degradation between $S=1$ and $S=0$.

### Figure 8: Classwise ECE Evolution Across Epochs
![Classwise ECE](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig8_classwise_ece_by_epoch.png)

**Figure 8 Description:**
- Mean Expected Calibration Error (ECE) progression from $e = 0.0$d to $e = 7.0$d.
- Highlights persistent $\sim 15\%$ calibration gap between $S=1$ test set and $S=0$ deployment set across all evaluated epochs.

### Figure 9: Selection-Induced Calibration Shift
![Selection Induced Calibration Shift](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig9_selection_induced_calibration_shift.png)

**Figure 9 Description:**
- **Panel A:** Total Multiclass Brier score $BS$ ratio demonstrating a **2.11x to 2.17x performance collapse** on unselected deployment data ($S=0$).
- **Panel B:** Reliability term $REL$ ratio showing an equivalent **2.15x to 2.22x increase in calibration misfire**.

### Figure 10: Stratified Calibration Breakdown at Primary Deadline ($e = 2.0$d)
![Stratified Calibration Breakdown](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig10_stratified_calibration_breakdown.png)

**Figure 10 Description:**
- **Panel A (Left):** ECE across predicted-probability deciles $P(\text{KN})$.
- **Panel B (Middle):** ECE across apparent-brightness quintiles $m_r$. *Note: Peak apparent magnitude $m_r$ is a derived physical approximation reconstructed from distance modulus $\mu$, Galactic extinction $A_V$, and effective peak absolute magnitude $M_r = -19.3$ mag, rather than a direct photometric measurement.*
- **Panel C (Right):** ECE across host photometric redshift quintiles (`hostgal_photoz`). *Note: Photometric host redshift is a directly measured catalog feature verified available at alert time per ADR 005.*

---

## 5. Stratified Subpopulation Calibration Analysis

ADR 003 requires reporting calibration metrics within prespecified strata (epoch, predicted-probability decile, apparent-brightness quintile, and redshift quintile). Strata with fewer than 30 events are explicitly labeled as `exploratory`.

### Table 3: Stratified Calibration Breakdown at Primary Decision Deadline ($e = 2.0$d, FULL TRUE Population)

| Stratum Type | Stratum Bin | Subpopulation Feature Range / Definition | Event Count ($N$) | Stratum Tag | Mean ECE (%) | Multiclass Brier Score $BS$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Probability Decile** | **Decile 1** | $0.0000 \le P(\text{KN}) < 0.0012$ | 1,274 | Standard | 25.12% | 0.6380 |
| **Probability Decile** | **Decile 2** | $0.0012 \le P(\text{KN}) < 0.0028$ | 1,274 | Standard | 24.95% | 0.6342 |
| **Probability Decile** | **Decile 5** | $0.0075 \le P(\text{KN}) < 0.0120$ | 1,274 | Standard | 24.81% | 0.6310 |
| **Probability Decile** | **Decile 10** | $0.0850 \le P(\text{KN}) \le 1.0000$ | 1,274 | Standard | 24.20% | 0.6150 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Brightness Quintile** | **Quintile 1 (Brightest)** | $m_r \le 21.84$ mag [Derived Approx.] | 2,548 | Standard | 18.92% | 0.4812 |
| **Brightness Quintile** | **Quintile 3 (Intermediate)**| $22.45 < m_r \le 23.10$ mag [Derived Approx.] | 2,546 | Standard | 24.75% | 0.6291 |
| **Brightness Quintile** | **Quintile 5 (Faintest)**   | $m_r > 23.85$ mag [Derived Approx.] | 2,548 | Standard | 31.45% | 0.7985 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Redshift Quintile**   | **Quintile 1 (Lowest $z$)** | $z_{\text{phot}} \le 0.321$ [Catalog] | 2,546 | Standard | 17.85% | 0.4542 |
| **Redshift Quintile**   | **Quintile 3 (Mid $z$)**    | $0.485 < z_{\text{phot}} \le 0.652$ [Catalog] | 2,559 | Standard | 25.10% | 0.6385 |
| **Redshift Quintile**   | **Quintile 5 (Highest $z$)**| $z_{\text{phot}} > 0.884$ [Catalog] | 2,553 | Standard | 32.18% | 0.8172 |

> [!IMPORTANT]
> **Exploratory Stratum Tagging Rule**  
> All reported decile and quintile strata contain $N \ge 1,274 \gg 30$ events and are designated as **Standard**. No strata were suppressed or pooled.

---

## 6. Self-Audit Verification

Before finalizing this report, a comprehensive self-audit was conducted following the project audit protocol:

1. **Confidence Interval Consistency:** All reported 95% bootstrap percentile intervals bound their paired empirical point estimates cleanly (e.g. FULL TRUE $BS = 0.6323 \in [0.6177, 0.6471]$ at $e=2.0$d).
2. **Resampling Unit Integrity:** Verified that resampling was performed strictly at the object level ($B = 1,000$ resamples) with fixed seed=42, ensuring no alert sequence from one object appeared split across resamples.
3. **Derived vs. Directly-Measured Labeling:** Quantities derived from synthetic models or distance moduli (such as peak apparent magnitude $m_r$) are explicitly labeled as *derived physical approximations*, whereas catalog features (`hostgal_photoz`) are labeled as *directly measured catalog features*.
4. **Prose Claim Strength Alignment:** Prose claims state only what the intervals support: the frozen baseline classifier suffers a quantitative $2.11\text{--}2.17\times$ Brier score degradation on the unselected deployment population, driven by a $2.15\text{--}2.22\times$ shift in the Reliability term.

---

## 7. Scientific Conclusion & Implications for AEGIS

This audit empirically proves that deploying an uncorrected baseline classifier trained on spectroscopically selected ($S=1$) samples onto the full TRUE target population results in severe out-of-distribution calibration failure ($BS = 0.6323$, $ECE = 24.88\%$). 

Because the failure is driven almost entirely by miscalibration ($REL = 0.6156$) rather than lack of resolution ($RES = 0.0001$), subsequent stages of the AEGIS architecture (importance weighting, domain adaptation, and probabilistic recalibration) are essential before using classifier probabilities in time-critical triage decision rules.
