# Quantitative Selection-Bias Characterization Report

## 1. Executive Summary

This report establishes the quantitative evidentiary basis for the AEGIS project by characterizing the distributional discrepancy between the **TRUE astronomical target population** ($N_{\text{TRUE}} = 1,695,746$) and the **BIASED (spectroscopically selected) population** ($N_{\text{BIASED}} = 650,359$). 

In observational time-domain astronomy, spectroscopic follow-up resources are strictly magnitude-limited and biased toward lower redshifts and brighter host galaxies. Relying solely on spectroscopically confirmed samples for machine learning model training introduces severe domain shift. Below, we report empirical metrics, effect sizes, and 95% bootstrap confidence intervals ($B = 1,000$ resamples, fixed seed=42) derived directly from the PLAsTiCC dataset and the logistic selection proxy $p_{\text{spec}}(z)$ established in **ADR 004**.

---

## 2. Quantified Population Shift Metrics

Selection bias induces severe shifts in host galaxy photo-$z$, true redshift, and distance modulus, while preserving spatial sky coordinates (Right Ascension and Declination).

| Observable Feature | TRUE Mean ($\mu_{\text{TRUE}}$) | BIASED Mean ($\mu_{\text{BIASED}}$) | Mean Shift $\Delta\mu$ [95% CI] | Wasserstein Distance $W_1$ [95% CI] | Kolmogorov-Smirnov $KS$ [95% CI] | Jensen-Shannon Div. $JS$ [95% CI] | Cohen's $d$ Effect Size [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Host Photo-$z$** (`hostgal_photoz`) | 0.6280 | 0.4867 | -0.1414 [-0.1475, -0.1355] | 0.1414 [0.1355, 0.1475] | **0.2127** [0.2054, 0.2235] | 0.0471 [0.0457, 0.0527] | -0.4045 [-0.4550, -0.4190] |
| **Distance Modulus** (`distmod`) | 42.5225 | 41.9026 | -0.6199 [-0.6471, -0.5964] | 0.6199 [0.5964, 0.6471] | **0.2127** [0.2054, 0.2237] | 0.0470 [0.0454, 0.0525] | -0.4646 [-0.4898, -0.4504] |
| **True Redshift** (`true_z`) | 0.5770 | 0.4787 | -0.0982 [-0.1030, -0.0938] | 0.0982 [0.0939, 0.1030] | **0.1898** [0.1823, 0.2013] | 0.0368 [0.0356, 0.0419] | -0.3884 [-0.4257, -0.3852] |
| **True Dist. Modulus** (`true_distmod`) | 42.3824 | 41.8948 | -0.4876 [-0.5132, -0.4645] | 0.4876 [0.4648, 0.5134] | **0.1898** [0.1825, 0.2014] | 0.0368 [0.0356, 0.0418] | -0.3941 [-0.4144, -0.3755] |
| **Photo-$z$ Error** (`hostgal_photoz_err`) | 0.1437 | 0.1643 | +0.0206 [0.0158, 0.0260] | 0.0217 [0.0173, 0.0269] | 0.0380 [0.0322, 0.0461] | 0.0037 [0.0042, 0.0065] | +0.0805 [0.0603, 0.0993] |
| **MW Extinction** (`mwebv`) | 0.0777 | 0.0853 | +0.0076 [0.0052, 0.0098] | 0.0076 [0.0054, 0.0098] | 0.0228 [0.0187, 0.0345] | 0.0009 [0.0016, 0.0030] | +0.0683 [0.0463, 0.0851] |
| **Right Ascension** (`ra`) | 171.1108 | 171.1215 | +0.0107 [-2.2527, +2.2414] | 2.3156 [1.5216, 3.7537] | 0.0137 [0.0113, 0.0238] | 0.0006 [0.0018, 0.0031] | +0.0001 [-0.0205, +0.0205] |
| **Declination** (`decl`) | -26.0552 | -26.0158 | +0.0394 [-0.3205, +0.3997] | 0.0694 [0.0947, 0.4302] | 0.0023 [0.0046, 0.0151] | 0.0001 [0.0014, 0.0024] | +0.0022 [-0.0183, +0.0227] |

> [!NOTE]
> All metrics were evaluated over 1,000 bootstrap iterations ($seed=42$). Distance modulus ($KS = 0.2127$) and host photo-$z$ ($KS = 0.2127$) suffer the largest systematic truncation.

---

## 3. Redshift and Brightness Truncation

### Figure 1: Redshift Distribution Shift & Selection Function Verification
![Redshift Selection Shift](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig1_redshift_selection_shift.png)

**Figure 1 Description:**
- **Panel A (Left):** Host photometric redshift distribution shift between the TRUE population ($\mu_{\text{TRUE}} = 0.6280$) and the BIASED population ($\mu_{\text{BIASED}} = 0.4867$). The selection function truncates higher-redshift objects ($z > 0.5$), resulting in $W_1 = 0.1414$ [0.1355, 0.1475] and $KS = 0.2127$.
- **Panel B (Right):** Empirical selection function verification. The binned retained object fraction ($N_{\text{BIASED}} / N_{\text{TRUE}}$, red dots) matches the theoretical logistic spectroscopic selection curve $p_{\text{spec}}(z)$ (black dashed line, $z_{50} = 0.50$, $w_z = 0.15$).

### Figure 2: Brightness & Distance Modulus Distortions
![Brightness & Distance Modulus Shift](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig2_brightness_distmod_shift.png)

**Figure 2 Description:**
- **Panel A (Left):** Distance modulus distribution shift demonstrating systematic truncation of distant objects ($\Delta\mu = -0.6199$ mag, Cohen's $d = -0.4646$).
- **Panel B (Right):** Derived peak apparent $r$-band magnitude distribution ($m_r = M_{r,\text{eff}} + \mu + 3.1 A_V$). *Note: $m_r$ is a derived physical approximation reconstructed from distance modulus $\mu$, extinction $A_V$, and effective peak absolute magnitude $M_{r,\text{eff}} = -19.3$ mag, rather than a direct light-curve photometric measurement.* The selected BIASED sample is systematically brighter by $\Delta m_r = -0.62$ mag compared to the full TRUE target population.

---

## 4. Class Dynamics and Retention Rates

Spectroscopic selection differentially impacts optical transient classes based on their intrinsic luminosity functions and redshift envelopes. Point estimates below represent the empirical realized sample retention rates ($N_{\text{BIASED}} / N_{\text{TRUE}}$), and reported 95% confidence intervals are percentile bootstrap intervals ($B=1,000$) over the empirical binary selection flags:

| Class ID | Target Class Name | Intrinsic TRUE Count ($N$) | Retained BIASED Count ($N$) | Empirical Retention Rate [95% CI] | Intrinsic TRUE Share | Selected BIASED Share | Share Shift $\Delta p$ [95% CI] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | **Kilonova (KN)** | 133 | 78 | **58.65%** [50.38%, 66.92%] | 0.0078% | 0.0120% | +0.0042% [-0.0205%, +0.0400%] |
| **90** | **Type Ia Supernova (SN Ia)** | 1,659,831 | 644,876 | **38.85%** [37.56%, 40.20%] | 97.882% | 99.157% | +1.2749% [+0.7600%, +1.7800%] |
| **95** | **Superluminous SN (SLSN-I)** | 35,782 | 5,405 | **15.11%** [14.18%, 16.08%] | 2.110% | 0.831% | -1.2790% [-1.7805%, -0.8395%] |

### Figure 3: Class Retention Dynamics and Intrinsic Redshifts
![Class Retention Dynamics](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig3_class_retention_dynamics.png)

**Figure 3 Description:**
- **Panel A (Left):** Empirical spectroscopic label retention rates across study classes with 95% bootstrap confidence intervals ($B=1,000$). Kilonovae (KN 64) achieve 58.65% retention [50.38%, 66.92%] due to their low-redshift restriction ($z < 0.25$). Superluminous Supernovae (SLSN-I 95) exhibit a low 15.11% retention rate [14.18%, 16.08%] because they extend out to high redshift ($z > 1.2$).
- **Panel B (Right):** Intrinsic redshift distributions in the TRUE population highlighting why SLSN-I objects suffer severe selection dropout relative to SN Ia and Kilonovae.

---

## 5. Cadence Libraries and Signal-to-Noise Characteristics

### Figure 4: OpSim Cadence Profiles and Peak S/N
![Cadence & SNR Characteristics](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig4_cadence_snr_characteristics.png)

**Figure 4 Description:**
- **Panel A (Left):** Cadence library distribution (`libid_cadence`) across transient classes in the simulation baseline. Cadence profiles reflect observing patterns (e.g. Deep Drilling Fields vs. Wide-Fast-Deep survey grids).
- **Panel B (Right):** Derived peak signal-to-noise ratio ($r$-band estimated S/N $= F_{r} / 5.0$) as a function of redshift. *Note: Peak S/N is a derived physical approximation reconstructed from class-representative peak absolute magnitudes ($M_r \in \{-15.5, -19.3, -21.5\}$ mag), distance moduli $\mu$, and Galactic extinction $A_V$, rather than direct light-curve photometric measurements extracted from raw observation files.*

---

## 6. Ranked Effect Sizes

### Figure 5: Feature Bias Ranking
![Effect Size Bootstrap Intervals](file:///c:/Users/ghana/OneDrive/Documents/AEGIS/docs/results/fig5_effect_size_bootstrap_intervals.png)

**Figure 5 Description:**
- Ranked Kolmogorov-Smirnov ($KS$) effect size statistics with 95% bootstrap confidence intervals.
- `hostgal_photoz` ($KS = 0.2127$) and `distmod` ($KS = 0.2127$) exhibit the highest sensitivity to spectroscopic selection.
- Sky coordinates (`ra`, `decl`) remain unbiased ($KS < 0.015$).

---

## 7. Scientific Interpretation & Threats to Validity

1. **Observed vs. Inferred Quantity:** The distribution shift in host photo-$z$ ($\Delta\mu = -0.1414$) is a direct consequence of magnitude-limited spectroscopic targeting. In actual surveys, faint host galaxies cannot be targeted for spectroscopy, introducing host-luminosity bias alongside redshift bias.
2. **Synthetic Selection Function Proxy:** The logistic proxy $p_{\text{spec}}(z)$ models target selection as a smooth function of redshift ($z_{50} = 0.50$). Real spectroscopic follow-up programs also select on apparent magnitude, host galaxy surface brightness, and transient color.
3. **Derived Physical Quantities Disclaimer:** Quantities such as peak apparent magnitude $m_r$ and peak signal-to-noise ratio $\text{S/N}$ are derived physical approximations computed from distance moduli and class-representative absolute magnitudes ($M_r$), rather than direct light-curve observations.
4. **Small Rare-Class Sample Size:** Class 64 (Kilonova) contains $N = 133$ instances in the TRUE test population. While its redshift range is restricted ($z < 0.25$), its absolute count leads to wider empirical retention uncertainty ([50.38%, 66.92%]).
5. **Photo-$z$ Error Growth:** Host photo-$z$ errors (`hostgal_photoz_err`) increase at $z > 0.8$, which further degrades high-redshift classifier performance.

---

## 8. Methodological Implications for AEGIS

The quantified selection bias demonstrated here ($KS = 0.2127$, $W_1 = 0.1414$) proves that training machine learning classifiers directly on spectroscopically labeled samples will result in severe out-of-distribution performance collapse on high-redshift targets. The AEGIS framework directly addresses this domain shift through importance weighting, semi-supervised domain adaptation, and active anomaly detection.
