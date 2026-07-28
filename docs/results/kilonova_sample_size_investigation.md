# Kilonova Sample Size Investigation Report (Step 0)

## 1. Executive Summary

This report documents the Step 0 investigation into the total count of Kilonova (PLAsTiCC class 64) objects present in the underlying simulated population and traces how the evaluation population inherits from it.

### Key Finding:
The low kilonova count is a **genuine physical property of the underlying PLAsTiCC survey simulation**. Kilonovae are exceptionally rare transient events in PLAsTiCC:
- Total PLAsTiCC raw test metadata population: **3,492,890 objects**, containing **133 kilonovae** (class 64 base rate $P(Y=64) \approx 3.81 \times 10^{-5}$, or $0.0038\%$).
- Pre-specified study class population (ADR 004, classes 64, 90, 95): **1,695,746 objects**, containing **133 kilonovae** ($0.0078\%$).
- Evaluation light-curve cohort (`plasticc_test_lightcurves_01.csv.gz` slice): **12,740 study class objects**, containing exactly **2 kilonovae** (object IDs `489518` and `490807`).

Per Step 0 mandate, this low count is documented plainly as a **first-order limitation of the study**. No synthetic or manufactured samples have been added, and all downstream evaluation statistics will be strictly formatted for small-sample precision.

---

## 2. Population Provenance and Inheritance Trace

The dataset hierarchy and inheritance chain from raw PLAsTiCC files to the evaluation cohort are structured as follows:

```
PLAsTiCC Raw Test Metadata (Zenodo 2539456)
  └── Total Objects: 3,492,890
      ├── Type Ia Supernovae (class 90): 1,659,831
      ├── Non-Study Classes (42, 92, 62, 88, 16, 65, 52, 67, 15, etc.): 1,797,144
      ├── Superluminous Supernovae Type I (class 95): 35,782
      └── Kilonovae (class 64): 133 (0.0038%)
            │
            ▼
TRUE Population (ADR 004: Restricted to Study Classes 64, 90, 95)
  └── Total Objects: 1,695,746
      ├── Type Ia Supernovae (class 90): 1,659,831 (97.88%)
      ├── Superluminous Supernovae Type I (class 95): 35,782 (2.11%)
      └── Kilonovae (class 64): 133 (0.0078%)
            │
            ▼
Evaluation Population (Intersection with `plasticc_test_lightcurves_01.csv.gz`)
  └── Total Objects: 12,740
      ├── Type Ia Supernovae (class 90): 12,497 (98.09%)
      ├── Superluminous Supernovae Type I (class 95): 241 (1.89%)
      └── Kilonovae (class 64): 2 (0.0157%)
            ├── Object ID 489518 (z_phot = 0.3807, true_z = 0.4287)
            └── Object ID 490807 (z_phot = 0.1388, true_z = 0.1345)
```

---

## 3. Detailed Object Audit

### Table 1: Target Class Balance Across Dataset Pipeline Stages

| Pipeline Stage | Total Objects ($N$) | Kilonova Count ($N_{64}$) | SN Ia Count ($N_{90}$) | SLSN-I Count ($N_{95}$) | Kilonova Proportion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw PLAsTiCC Test Metadata** | 3,492,890 | 133 | 1,659,831 | 35,782 | 0.0038% |
| **S=1 Training Set (`plasticc_train`)** | 7,848 | 102 | 2,313 | 175 | 1.300% |
| **TRUE Population (`data/processed/true_population.csv.gz`)** | 1,695,746 | 133 | 1,659,831 | 35,782 | 0.0078% |
| **BIASED Population (`data/processed/biased_population.csv.gz`)** | 650,359 | 78 | 644,876 | 5,405 | 0.0120% |
| **Evaluation Cohort (`lightcurves_01` slice)** | 12,740 | 2 | 12,497 | 241 | 0.0157% |

### Confirmed Kilonova Evaluation Objects:
1. **Object ID `489518`**: $z_{\text{phot}} = 0.3807$, $z_{\text{true}} = 0.4287$, $\text{distmod} = 41.713$.
2. **Object ID `490807`**: $z_{\text{phot}} = 0.1388$, $z_{\text{true}} = 0.1345$, $\text{distmod} = 39.006$.

---

## 4. Methodological Conclusion and Protocol Constraints

1. **No Pipeline Defects:** The low count of 2 kilonova objects in the 12,740-object evaluation cohort and 133 objects in the full 3.49M test metadata is not caused by any filtering bug, join defect, or data corruption. It reflects the exact simulated frequency of kilonovae in the PLAsTiCC release.
2. **First-Order Limitation:** This small sample size ($N_{\text{KN}} = 2$ in the evaluation cohort) is a fundamental constraint.
3. **Evaluation Protocol Requirements:** Standard percentile bootstrap confidence intervals on aggregate rates (e.g. MHVER) break down when positive counts are on the order of $N=2$. Per Step 1 and Step 2 mandates, the evaluation protocol MUST employ:
   - **Exact / Small-Sample Intervals:** Exact Clopper-Pearson or Wilson score intervals for positive-class rates.
   - **Leave-One-Kilonova-Out Jackknife Analysis:** Explicit per-object evaluation showing how headline results change when each positive kilonova is excluded.
   - **Per-Object Granular Reporting:** Explicit itemization of individual decisions ($a_{i,e}$) for each confirmed kilonova object under all evaluated policy variants.
