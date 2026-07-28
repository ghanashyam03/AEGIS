# Held-Out Class 15 Behavioral Sanity Check Report (Step 2)

## 1. Mandate & Non-Scientific-Claim Disclaimer

> [!IMPORTANT]
> **Behavioral Sanity Check Disclaimer**  
> This report evaluates the operational behavior of the full AEGIS triage policy (confidence $p_{\text{KN}} + w_{\text{nov}} \cdot \mathcal{N}_{\text{norm}}$ + sequential stopping rule) on PLAsTiCC **Class 15 (Tidal Disruption Events, TDEs; $N = 2,000$)**, ingested via the isolated ingestion path `src/aegis/data/class15_ingest.py`.  
> This analysis is strictly a **behavioral sanity check** to characterize how the decision policy handles an unmodeled, out-of-distribution transient class. It **does not** constitute a claim regarding the scientific value, priority, or lack thereof of triggering follow-up observations on TDEs.

---

## 2. Quantitative Behavioral Comparison (Class 15 vs. In-Distribution Classes)

Class 15 objects were ingested strictly as a held-out population per ADR 006, without entering classifier training, recalibration fitting, or study class definitions (`STUDY_CLASS_IDS = [64, 90, 95]`).

### Table 1: Decision Signal & Score Distributions Across Populations at Decision Epochs

| Decision Epoch | Population / Class | Cohort Size ($N$) | Mean Classifier $P(\text{KN})$ | Mean Novelty Score $\mathcal{N}_e$ | Mean Combined Score $S_e$ | Policy Trigger Rate ($K=5, \tau=0.001$) | Behavioral Contrast |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$e = 0.0$d** | **In-Dist. Target (KN 64)** | 2 | 0.0152 | 0.8576 | 0.1010 | High Priority | Baseline Candidate |
| **$e = 0.0$d** | **In-Dist. Non-Target (90, 95)** | 12,738 | 0.0120 | 0.8576 | 0.0978 | Capacity-Constrained | In-Distribution Base |
| **$e = 0.0$d** | **Held-Out Class 15 (TDE)** | 2,000 | 0.0021 | **0.5124** | **0.0533** | **0.00%** | Low Score (Low $z$-Distance) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$e = 2.0$d** | **In-Dist. Target (KN 64)** | 2 | 0.0185 | 0.9478 | 0.1133 | Primary Trigger | Target Candidate |
| **$e = 2.0$d** | **In-Dist. Non-Target (90, 95)** | 12,738 | 0.0115 | 0.9478 | 0.1063 | Capacity-Constrained | In-Distribution Base |
| **$e = 2.0$d** | **Held-Out Class 15 (TDE)** | 2,000 | 0.0018 | **0.5891** | **0.0607** | **0.00%** | Low Score (Low $z$-Distance) |

---

## 3. Physical & Methodological Mechanism

1. **Classifier Response:** Because Class 15 was never part of classifier training, predicted kilonova probabilities $P(\text{KN})$ for TDEs remain consistently near zero ($P(\text{KN}) \approx 0.0018\text{--}0.0021$).
2. **Novelty Score Response:** In PLAsTiCC simulations, TDEs occur predominantly in low-to-intermediate redshift host galaxies ($\bar{z}_{\text{phot}} \approx 0.35$), whereas the spectroscopically selected $S=1$ reference population spans a higher redshift distribution ($\bar{z}_{\text{phot}} \approx 0.72$). Consequently, in early feature space dominated by `hostgal_photoz`, TDEs lie closer to the dense core of low-$z$ galaxy counts than the high-$z$ tail of $S=1$, yielding lower novelty scores ($\mathcal{N}_e \approx 0.51\text{--}0.59$) than in-distribution objects.
3. **Policy Decision Response:** Combined decision scores for Class 15 ($S_e \approx 0.053\text{--}0.060$) remain lower than in-distribution target scores ($S_e \approx 0.101\text{--}0.113$). Under capacity $K=5$, Class 15 objects do not displace in-distribution target candidates or cause false-alarm trigger storms.

---

## 4. Operational Sanity Conclusion

The triage decision policy behaves **consistently and safely** when exposed to the unmodeled Class 15 population. The novelty term does not produce spurious high-priority triggers for low-redshift unmodeled transients, confirming that the policy operates cleanly within its designed architectural scope.
