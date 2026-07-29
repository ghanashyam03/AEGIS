# SLSN-I Sample Size Investigation Report (Step 0 Generalization Check)

## 1. Executive Summary

This report documents the Step 0 sample-size investigation for Superluminous Supernovae Type I (SLSN-I, PLAsTiCC class 95) in the TRUE evaluation population, establishing the exact positive event count prior to running the retargeted generalization diagnostic.

### Key Audit Finding:
Unlike kilonovae (class 64), which yielded only $N_{\text{KN}} = 2$ positive objects in the 12,740-object evaluation cohort, **SLSN-I (class 95)** possesses a larger sample footprint:
- Total PLAsTiCC raw test metadata population: **3,492,890 objects**, containing **35,782 SLSN-I objects** ($1.0244\%$).
- Pre-specified study class population (ADR 004, classes 64, 90, 95): **1,695,746 objects**, containing **35,782 SLSN-I objects** ($2.1101\%$).
- Evaluation light-curve cohort (`plasticc_test_lightcurves_01.csv.gz` slice): **12,740 study class objects**, containing **98 SLSN-I objects** ($0.7692\%$).

Per Step 0 instructions, while $N_{\text{SLSN}} = 98$ provides $49\times$ more target events than the kilonova sample ($N=2$), SLSN-I remains a minor class ($0.77\%$) within the evaluation stream. The evaluation methodology will incorporate exact Clopper-Pearson binomial confidence intervals (ADR 008) and jackknife stability checks to ensure statistical rigor.

---

## 2. Population Provenance and Inheritance Trace

The dataset hierarchy and inheritance chain for SLSN-I (class 95) across pipeline stages are as follows:

```
PLAsTiCC Raw Test Metadata (Zenodo 2539456)
  └── Total Objects: 3,492,890
      ├── Type Ia Supernovae (class 90): 1,659,831
      ├── Non-Study Classes (42, 92, 62, 88, 16, 65, 52, 67, 15, etc.): 1,797,144
      ├── Kilonovae (class 64): 133 (0.0038%)
      └── SLSN-I (class 95): 35,782 (1.0244%)
            │
            ▼
TRUE Population (ADR 004: Restricted to Study Classes 64, 90, 95)
  └── Total Objects: 1,695,746
      ├── Type Ia Supernovae (class 90): 1,659,831 (97.8824%)
      ├── Kilonovae (class 64): 133 (0.0078%)
      └── SLSN-I (class 95): 35,782 (2.1101%)
            │
            ▼
Evaluation Population (Intersection with `plasticc_test_lightcurves_01.csv.gz`)
  └── Total Objects: 12,740
      ├── Type Ia Supernovae (class 90): 12,640 (99.2151%)
      ├── Kilonovae (class 64): 2 (0.0157%)
      └── SLSN-I (class 95): 98 (0.7692%)
```

---

## 3. Comparative Class Balance Across Pipeline Stages

### Table 1: Target Class Breakdown Across Dataset Pipeline Stages

| Pipeline Stage | Total Objects ($N$) | SLSN-I Count ($N_{95}$) | Kilonova Count ($N_{64}$) | SN Ia Count ($N_{90}$) | SLSN-I Base Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw PLAsTiCC Test Metadata** | 3,492,890 | 35,782 | 133 | 1,659,831 | 1.0244% |
| **S=1 Training Set (`plasticc_train`)** | 7,848 | 175 | 102 | 2,313 | 2.2299% |
| **TRUE Population (`true_population.csv.gz`)** | 1,695,746 | 35,782 | 133 | 1,659,831 | 2.1101% |
| **BIASED Population (`biased_population.csv.gz`)** | 650,359 | 5,405 | 78 | 644,876 | 0.8311% |
| **Evaluation Cohort (`lightcurves_01` slice)** | 12,740 | 98 | 2 | 12,640 | 0.7692% |

---

## 4. Methodological Implications for Generalization Evaluation

1. **Increased Statistical Power relative to Kilonovae:** With $N_{\text{SLSN}} = 98$ positive objects, point estimates for $MHVER$ and policy trigger counts have substantially tighter confidence bounds than the $N_{\text{KN}} = 2$ kilonova evaluation.
2. **Scope of Generalization Check:** SLSN-I objects are intrinsically brighter and evolve on slower timescales (decaying over months) compared to kilonovae (which fade on day timescales). Testing whether early triage decisions ($e \le 2.0$d) succeed on SLSN-I evaluates policy behavior on a slower, brighter transient class.
3. **Evaluation Protocol Requirements:**
   - Report exact Clopper-Pearson 95% binomial confidence intervals (ADR 008) for SLSN-I $MHVER$.
   - Perform leave-one-out / jackknife stability checks across the $N=98$ objects to report min/max metric bounds.
   - Itemize candidate scores and trigger outcomes for the highest-confidence and highest-novelty SLSN-I objects alongside aggregate summary statistics.
