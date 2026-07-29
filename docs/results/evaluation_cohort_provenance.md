# Evaluation Cohort Provenance & Disjoint Population Audit (Step 0)

## 1. Executive Summary

This report documents the Step 0 provenance investigation into the **12,740-object evaluation cohort** used in the preliminary headline evaluation (`docs/results/headline_evaluation_true_population.md`) and SLSN-I generalization diagnostic (`docs/results/slsn_generalization_check.md`). 

The investigation confirms:
1. **Provenance:** The 12,740-object cohort was produced by selecting all study-class objects (`STUDY_CLASSES = [64, 90, 95]`, per ADR 004) whose light curves are contained in `plasticc_test_lightcurves_01.csv.gz`—the first partition of 11 unblinded PLAsTiCC test light-curve release files (Zenodo record 2539456).
2. **Strict Disjointness:** The evaluation cohort is a **strict subset** of the TRUE population (`data/processed/true_population.csv.gz`) and has **zero overlap ($N_{\text{overlap}} = 0$)** with any object used in baseline classifier training, novelty reference-population ($S=1$) construction, or naive threshold calibration.
3. **Full Disjoint Pool Size:** The full disjoint pool consists of all **1,695,746 objects** in the TRUE survey population metadata (`data/processed/true_population.csv.gz`). None of these 1.695M objects were used in any model fitting or hyperparameter selection. This pool contains **133 kilonovae** (class 64) and **35,782 SLSN-I** (class 95)—providing a **66.5x increase** in kilonova sample size over the 2-kilonova evaluation slice.

---

## 2. Detailed Provenance Investigation

### (a) Pipeline Process & Cohort Definition
The PLAsTiCC test dataset release (Zenodo record 2539456) comprises 3,492,890 total simulated light curves partitioned across 11 gzipped CSV files (`plasticc_test_lightcurves_01.csv.gz` through `11.csv.gz`) alongside a single unified metadata table (`plasticc_test_metadata.csv.gz`).

Per **ADR 004**, the study population (TRUE population) is defined by filtering `plasticc_test_metadata.csv.gz` to the pre-specified target and background study classes:
- **Class 64:** Kilonovae ($N = 133$)
- **Class 90:** Type Ia Supernovae ($N = 1,659,831$)
- **Class 95:** Superluminous Supernovae Type I ($N = 35,782$)
- **Total TRUE Population:** $N = 1,695,746$ objects.

The 12,740-object preliminary evaluation cohort resulted from restricting light-curve feature extraction to the single locally cached file `plasticc_test_lightcurves_01.csv.gz`. Intersecting `plasticc_test_lightcurves_01.csv.gz` with `true_population.csv.gz` yielded:
- **Class 64 (KN):** 2 objects (0.0157% of slice, IDs `489518` and `490807`)
- **Class 95 (SLSN-I):** 98 objects (0.7692% of slice)
- **Class 90 (SN Ia):** 12,640 objects (99.2151% of slice)
- **Total Slice Cohort:** 12,740 objects (0.7513% of the TRUE population).

---

### (b) Strict Subsetting & Zero-Leakage Audit

All model components, representation extractors, novelty reference distributions, and policy parameters in AEGIS strictly obey the $S=1$ follow-up boundary:
- **Baseline Classifier Training:** Trained exclusively on `plasticc_train_metadata.csv.gz` / `plasticc_train_lightcurves.csv.gz` ($N = 7,848$ total raw objects, $N = 2,590$ study-class objects: 102 class 64, 2,313 class 90, 175 class 95).
- **Novelty Reference Population ($S=1$):** Built exclusively from the $S=1$ training set feature distributions at each decision epoch ($e \in \{0.0, 2.0\}$ days).
- **Naive Threshold Calibration:** Calibrated strictly on the $S=1$ training set predictions to match operational quota without observing test labels.
- **Frozen Policy Parameters:** Configuration (`configs/decision_policy_v1.yaml`) locked pre-registered values ($w_{\text{nov}} = 0.05, \tau = 0.001, K = 5$).

#### Programmatic Overlap Audit:
- **Training Object IDs ($S=1$) vs. TRUE Population Object IDs:** **0 overlapping objects** ($0 / 1,695,746$).
- **Training Object IDs ($S=1$) vs. 12,740 Evaluation Cohort IDs:** **0 overlapping objects** ($0 / 12,740$).

In the PLAsTiCC dataset architecture, training set object IDs and test set object IDs were generated as completely disjoint simulated catalog draws. Therefore, any subset of `true_population.csv.gz` is guaranteed to be strictly disjoint from every training and calibration asset.

---

### (c) Full Disjoint Pool Quantities

Because zero objects from `true_population.csv.gz` were consumed during model training, novelty reference construction, or hyperparameter selection, **every single object in the TRUE population belongs to the unpeeked, disjoint evaluation pool**.

| Population Level | Total Objects ($N$) | Kilonovae ($N_{64}$) | SLSN-I ($N_{95}$) | Type Ia SN ($N_{90}$) | Kilonova Base Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S=1 Training Set (Model Fit & Ref Pop)** | 2,590 | 102 | 175 | 2,313 | 3.9382% |
| **Preliminary Cohort (`lightcurves_01` slice)** | 12,740 | 2 | 98 | 12,640 | 0.0157% |
| **Full Disjoint TRUE Population** | **1,695,746** | **133** | **35,782** | **1,659,831** | **0.0078%** |
| **Expansion Factor (Full vs. Prelim)** | **133.1x** | **66.5x** | **365.1x** | **131.3x** | — |

---

## 3. Conclusion & Expansion Mandate

The 12,740-object cohort was an arbitrary light-curve file partitioning slice containing only $0.75\%$ of TRUE objects and only $2$ out of $133$ kilonovae. 

Because the full $1,695,746$-object TRUE population is $100\%$ disjoint from all training and calibration data, expanding the evaluation population to encompass all $133$ kilonovae and all available SLSN-I objects is mathematically valid, statistically necessary, and does not violate any zero-leakage guarantee.
