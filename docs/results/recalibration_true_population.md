# Selection-Aware Recalibration & Residual Gap Audit Report (Closing Phase 3)

## 1. Executive Summary

This report delivers the quantitative audit for **Selection-Aware Recalibration** applied to the frozen early-epoch baseline classifier in AEGIS, closing **Phase 3** (calibration-decay phase).

The objective of this prompt is to implement, validate, and characterize the limits of an importance-weighted recalibration method (IPW Platt scaling) derived from the empirical selection proxy model ($p_{\text{spec}}(z)$ per ADR 004), and to report **honestly whatever residual calibration gap cannot be corrected**.

### Key Quantitative Findings:
1. **Positivity & Overlap Diagnostic:** Identified 697 objects (**5.47% of FULL TRUE population**, $N=12,740$) in the high-redshift violation region ($z > 1.50$), where selection probability approaches $p_{\text{floor}} = 0.10$ and $S=1$ support density is near-zero. In compliance with ADR 001, these objects are tagged as `uncorrectable` and extrapolation is explicitly masked.
2. **Weight Distribution & Balance:** Importance weights $w_i = 1 / p_{\text{spec}}(z_i)$ on $S=1$ ($N=2,693$) span $[1.2990, 10.0000]$ with a median of $3.5586$, 95th percentile of $9.7557$, $CV = 0.6281$, and Effective Sample Size $ESS = 1,931.09$ ($71.7\%$ efficiency). Weighting achieves near-perfect covariate balance for `hostgal_photoz`, reducing Standardized Mean Difference ($SMD$) from **-0.5793** (unweighted) to **-0.0039** (weighted).
3. **Honest Residual Gap Finding:** Despite achieving near-perfect covariate balance on the selection feature, **post-hoc IPW recalibration fails to resolve early-epoch miscalibration on the FULL TRUE population**. At the primary decision deadline ($e = 2.0$d), FULL TRUE Brier score shifts from $0.6323$ [0.6177, 0.6471] (baseline) to $0.7407$ [0.7258, 0.7555] (recalibrated), Reliability shifts from $0.6156$ [0.6007, 0.6305] to $0.7243$ [0.7092, 0.7394], and mean ECE shifts from $24.88\%$ [24.36%, 25.39%] to $26.24\%$ [25.71%, 26.77%].
4. **Root Cause Mechanism:** Post-hoc recalibration cannot create discriminative resolution ($RES \approx 0.0001$) where early light-curve features contain insufficient physical signal ($e \le 2$d). Platt scaling fitted on low-information $S=1$ logits overfits the 99.16% SN Ia training base rate, causing extreme overconfidence when evaluated on deployment target populations.

---

## 2. Positivity & Weight Diagnostics

### Table 1: Positivity / Overlap Diagnostic Summary ($z_{\text{cutoff}} = 1.50$)

| Diagnostic Attribute | Empirical Value / Definition |
| :--- | :--- |
| **Affected Object Count** | **697 objects** (in evaluation cohort $N=12,740$) |
| **Percentage of TRUE Population** | **5.47%** |
| **Affected Feature Range** | `hostgal_photoz` $\in (1.50, 2.992]$ ($p_{\text{spec}} \le 0.12$) |
| **Affected Probability Deciles** | Decile 1 ($P \le 0.0012$), Decile 2 ($P \le 0.0028$), Faintest Quintile ($m_r > 23.85$) |
| **Action Taken** | **Flagged as uncorrectable & extrapolation masked** |

### Table 2: Importance Weight Distribution & Covariate Balance ($S=1$ Population)

| Metric / Diagnostic | Unweighted $S=1$ | Weighted $S=1$ (IPW) | Target TRUE Population |
| :--- | :---: | :---: | :---: |
| **Minimum Weight ($w_{\text{min}}$)** | 1.0000 | 1.2990 | - |
| **Median Weight ($w_{\text{med}}$)** | 1.0000 | 3.5586 | - |
| **95th Percentile Weight ($w_{\text{p95}}$)** | 1.0000 | 9.7557 | - |
| **Maximum Weight ($w_{\text{max}}$)** | 1.0000 | 10.0000 | - |
| **Coefficient of Variation ($CV$)** | 0.0000 | **0.6281** | - |
| **Effective Sample Size ($ESS$)** | 2,693 | **1,931.09** ($71.7\%$ efficiency) | 12,740 |
| **`hostgal_photoz` Mean** | 0.7183 | **0.9279** | 0.9293 |
| **`hostgal_photoz` SMD** | **-0.5793** | **-0.0039** (Balanced) | 0.0000 |
| **`distmod` SMD** | **-0.4589** | **-0.0015** (Balanced) | 0.0000 |
| **`mwebv` SMD** | **+0.0706** | **+0.0009** (Balanced) | 0.0000 |

---

## 3. Recalibration Performance vs Uncorrected Baseline

All metrics are evaluated across $e \in \{0.0, 2.0, 7.0\}$ days using 1,000 object-level bootstrap resamples ($B=1,000$, seed=42).

### Table 3: Performance Comparison Across Decision Epochs (FULL TRUE Population, $N=12,740$)

| Decision Epoch | Metric | Uncorrected Baseline [95% CI] | IPW Recalibrated [95% CI] | Absolute Change $\Delta$ [95% CI] | Relative Change $\% \Delta$ | Statistically Supported? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$e = 0.0$d** | **Brier Score $BS$** | 0.6408 [0.6272, 0.6549] | 0.7411 [0.7262, 0.7560] | +0.1003 [+0.0862, +0.1145] | -15.65% (Worse) | Yes ($p < 0.001$) |
| **$e = 0.0$d** | **Reliability $REL$** | 0.6240 [0.6098, 0.6383] | 0.7245 [0.7093, 0.7397] | +0.1005 [+0.0863, +0.1147] | -16.12% (Worse) | Yes ($p < 0.001$) |
| **$e = 0.0$d** | **Mean ECE (%)** | 26.12% [25.66%, 26.61%] | 26.99% [26.47%, 27.50%] | +0.87% [+0.35%, +1.38%] | -3.34% (Worse) | Yes ($p < 0.01$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$e = 2.0$d** | **Brier Score $BS$** | **0.6323** [0.6177, 0.6471] | **0.7407** [0.7258, 0.7555] | **+0.1084** [+0.0939, +0.1228] | **-17.14%** (Worse) | **Yes** ($p < 0.001$) |
| **$e = 2.0$d** | **Reliability $REL$** | **0.6156** [0.6007, 0.6305] | **0.7243** [0.7092, 0.7394] | **+0.1087** [+0.0942, +0.1232] | **-17.67%** (Worse) | **Yes** ($p < 0.001$) |
| **$e = 2.0$d** | **Mean ECE (%)** | **24.88%** [24.36%, 25.39%] | **26.24%** [25.71%, 26.77%] | **+1.36%** [+0.84%, +1.89%] | **-5.48%** (Worse) | **Yes** ($p < 0.001$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$e = 7.0$d** | **Brier Score $BS$** | 0.5700 [0.5568, 0.5838] | 0.7065 [0.6917, 0.7212] | +0.1365 [+0.1220, +0.1510] | -23.96% (Worse) | Yes ($p < 0.001$) |
| **$e = 7.0$d** | **Reliability $REL$** | 0.5534 [0.5401, 0.5672] | 0.6906 [0.6757, 0.7054] | +0.1372 [+0.1226, +0.1517] | -24.81% (Worse) | Yes ($p < 0.001$) |
| **$e = 7.0$d** | **Mean ECE (%)** | 22.87% [22.39%, 23.36%] | 24.52% [24.01%, 25.04%] | +1.65% [+1.14%, +2.17%] | -7.22% (Worse) | Yes ($p < 0.001$) |

---

## 4. Visualizations & Figure Walkthroughs

### Figure 11: Reliability Curves (Baseline vs Recalibrated)
![Reliability Diagrams](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig11_recalibration_reliability_diagrams.png)

**Figure 11 Description:** Reliability curves comparing uncorrected baseline (blue circles) vs IPW recalibrated (red squares) probabilities on FULL TRUE data across decision epochs. Demonstrates that Platt recalibration increases probability distortion.

### Figure 12: Brier Score & REL Comparison
![Brier Decomposition](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig12_recalibration_brier_decomposition.png)

**Figure 12 Description:** Panel A shows Brier score $BS$ trajectory; Panel B shows Reliability $REL$ trajectory. Highlights persistent $+0.10$ to $+0.14$ error inflation across all epochs.

### Figure 13: Positivity & Overlap Diagnostic
![Positivity Diagnostic](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig13_positivity_diagnostic_overlap.png)

**Figure 13 Description:** Selection probability $p_{\text{spec}}(z)$ curve showing positivity violation boundary at $z = 1.50$. Objects above $z=1.50$ (697 objects / 5.47% of TRUE) are masked from extrapolation.

### Figure 14: Covariate Balance (SMDs)
![Covariate Balance SMDs](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig14_covariate_balance_smd.png)

**Figure 14 Description:** Standardized Mean Differences before (blue) and after (green) IPW weighting. Confirms that weighting successfully balances selection features ($|SMD| < 0.01 \ll 0.10$).

---

## 5. Characterization of Uncorrectable Strata

The following strata are identified as **uncorrectable** under selection-aware recalibration:

1. **High-Redshift Subpopulation ($z > 1.50$):** 697 objects (5.47% of TRUE) suffer from positivity violations ($P(S=1 \mid z) \approx 0.10$). Importance weighting cannot recover sample density where $S=1$ object counts are sparse ($N < 10$).
2. **Faint Magnitude Strata ($m_r > 23.85$ mag):** Faint transients in Quintile 5 have negligible $S=1$ representation due to flux-limited follow-up selection, leaving recalibrator parameters unconstrained in faint regimes.
3. **Uninformative Early Photometry ($e \le 2.0$d):** Across all strata, early light curves contain minimal physical information ($RES \approx 0.0001$). Recalibration cannot create resolution where early features provide no discriminative power.

---

## 6. Self-Audit Verification

1. **Confidence Interval Consistency:** All 95% bootstrap CIs bound point estimates cleanly (e.g. $BS_{\text{recal}} = 0.7407 \in [0.7258, 0.7555]$ at $e=2.0$d).
2. **Resampling Unit:** Resampling was performed strictly at the object level ($B=1,000$, seed=42).
3. **Extrapolation Masking:** Confirmed that positivity violation objects ($z > 1.50$) are masked from extrapolation per ADR 001.
4. **Honest Reporting:** No claims of full resolution are made; the residual gap and performance degradation are reported exactly as empirically measured.

---

## 7. Scientific Conclusion & Transition to Phase 4

Phase 3 proves empirically that **post-hoc recalibration cannot resolve selection-induced miscalibration when early light-curve features lack physical resolution**. 

Because post-hoc probability scaling fails, **Phase 4 (Triage Policy & Decision Framework)** must incorporate **probabilistic uncertainty and novelty detection** alongside class confidence rather than relying on fixed probability thresholding.
