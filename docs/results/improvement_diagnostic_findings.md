# AEGIS Model Quality and Early Triage Improvement Findings

This document summarizes the results of the systematic investigation (Steps 0–4) into the kilonova recovery rate on the expanded target population ($N = 55,915$, containing $N_{\text{KN}} = 133$ kilonovae).

---

## 1. Step 0: Discrimination at Scale

We re-measured the discriminative power of the baseline classifier on the expanded population using nonparametric bootstrap ($B = 1,000$, seed=42) for 95% confidence intervals.

### Table 1: Discrimination Metrics on Expanded Population

| Epoch | Target Class | ROC-AUC [95% CI] | PR-AUC [95% CI] | Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **$e = 0.0$d** | **Kilonova (64)** | 0.8164 [0.7673, 0.8644] | 0.1152 [0.0718, 0.1812] | Strong early signal |
| | **SLSN-I (95)** | 0.8867 [0.8840, 0.8896] | 0.9392 [0.9373, 0.9413] | Highly discriminative |
| **$e = 2.0$d** | **Kilonova (64)** | 0.8461 [0.8016, 0.8888] | 0.1882 [0.1232, 0.2716] | Strong signal at deadline |
| | **SLSN-I (95)** | 0.8931 [0.8905, 0.8959] | 0.9424 [0.9405, 0.9444] | Highly discriminative |
| **$e = 7.0$d** | **Kilonova (64)** | 0.9107 [0.8806, 0.9393] | 0.2987 [0.2229, 0.3832] | Very strong signal |
| | **SLSN-I (95)** | 0.8941 [0.8914, 0.8970] | 0.9434 [0.9415, 0.9455] | Highly discriminative |

### Finding:
- **Strong Early Signal**: At scale, the baseline classifier demonstrates strong discriminative power for kilonovae at early epochs ($e=0.0$d ROC-AUC = 0.8164, $e=2.0$d ROC-AUC = 0.8461), contradicting the low-power preliminary cohort estimate of near-chance discrimination (ROC-AUC = 0.5360).
- **Signal is Real**: This replaces the preliminary low-power estimate and confirms that a real physical signal exists in the features at $e \le 2$ days.

---

## 2. Step 1: Capacity Semantics Investigation

- **Selection Mechanism**: The decision policy uses a ranking-based selection mechanism: it ranks untriggered eligible candidates (where combined score $S_e \ge \tau_e = 0.001$) and selects the top $K$ at each epoch.
- **Resource Scaling**: The capacity $K=5$ was pre-registered against the preliminary $N=12,740$ cohort. Under this fixed capacity, a pure-random selection process would achieve a recovery rate of $2K/N \approx 0.0785\%$. Over the expanded $N=55,915$ cohort (a 4.39x increase), the random recovery rate drops to $\approx 0.0179\%$.
- **Rate-Corrected Capacity**: Scaling $K$ to the expanded cohort size to maintain the same resource allocation rate gives $K_{\text{corrected}} = 22$.

### Table 2: Sequential Policy Results under Corrected Capacity ($K=22$, $H=2.0$d)

| Decision Configuration | Total Triggers ($FP + TP$) | Kilonova Triggers ($TP$) | Recovery Rate ($1 - MHVER$) | Missed Target Rate ($MHVER$) [95% CP CI] | False Triggers ($FP$) | FTR [95% CP CI] | Policy Utility | Oracle Utility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | 1 | 1 | 0.8% | 99.2% [95.9%, 100.0%] | 0 | 0.0000% | 2.0 | 44.0 |
| **Frozen Policy** | 44 | 19 | 14.3% | 85.7% [78.6%, 91.2%] | 25 | 0.0448% [0.0290%, 0.0659%] | 13.0 | 44.0 |
| **Novelty Ablation** | 44 | 19 | 14.3% | 85.7% [78.6%, 91.2%] | 25 | 0.0448% [0.0290%, 0.0659%] | 13.0 | 44.0 |

### Finding:
- **Triage Performance Improvement**: Correcting $K$ from 5 to 22 increases kilonova recovery from 5.26% (7 triggered kilonovae) to 14.3% (19 triggered kilonovae) for the frozen policy.
- **Naive Baseline Blocked**: The naive baseline is severely blocked by its high S=1 calibrated threshold ($\tau_{\text{naive}} = 0.979504$), triggering only 1 kilonova and failing to utilize the available capacity slots. The ranking-based policy with low score threshold ($\tau = 0.001$) successfully utilizes capacity.

---

## 3. Step 2: Novelty Rank-Impact Diagnostic

We directly compared the set of object IDs triggered under the frozen fused policy against the novelty-ablation policy ($w_{\text{nov}} = 0.00$) at $K=22$.
- **Triggered Sets**: Both policies triggered the exact same set of 44 objects:
  `{82688355, 89366565, 51722662, 69302706, 89526483, 93838743, 15447864, 53181529, 8071229, 78004281, ...}`
- **Symmetric Difference Size**: **0**

### Finding:
- The additive novelty term, at its current weight of $w_{\text{nov}} = 0.05$, **never changes a single triage decision** in this data regime.

---

## 4. Step 3: Classifier Headroom Investigation

We ran nested cross-validation ($5 \times 3$ folds) on the S=1 training population to select optimal hyperparameters, and evaluated the resulting tuned HistGradientBoostingClassifier against the expanded population.

### Table 3: Hyperparameter Tuning Impact on Kilonova ROC-AUC

| Epoch | S=1 Nested CV ROC-AUC | Best Hyperparameters (S=1) | Default AUC (Expanded) | Tuned AUC (Expanded) | Difference |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **$e = 0.0$d** | 0.9619 +/- 0.0211 | `learning_rate=0.05, max_iter=50, max_depth=3, min_samples_leaf=50, l2_regularization=1.0` | 0.8164 | 0.7839 | -0.0325 |
| **$e = 2.0$d** | 0.9695 +/- 0.0153 | `learning_rate=0.1, max_iter=50, max_depth=3, min_samples_leaf=50, l2_regularization=1.0` | 0.8461 | 0.8646 | +0.0185 |
| **$e = 7.0$d** | 0.9756 +/- 0.0181 | `learning_rate=0.1, max_iter=100, max_depth=3, min_samples_leaf=20, l2_regularization=1.0` | 0.9107 | 0.9220 | +0.0112 |

### Finding:
- **No Substantial Headroom**: Hyperparameter tuning yields very minor AUC differences on the expanded population (-3.25%, +1.85%, +1.12% respectively). The default hyperparameters are highly competitive and close to the model performance ceiling.

---

## 5. Step 4: Data-Completeness Stratification

We evaluated kilonova discrimination separately on subsets of objects at $e = 2.0$d with more complete early observations.

### Table 4: Stratification Results at e = 2.0d

| Completeness Stratum | Total Size ($N$) | Kilonova Count ($N_{\text{KN}}$) | ROC-AUC [95% CI] | PR-AUC [95% CI] |
| :--- | :---: | :---: | :---: | :---: |
| **Full Population** | 55,915 | 133 | 0.8461 [0.8016, 0.8888] | 0.1882 [0.1232, 0.2716] |
| **$N_{\text{det}} > 1$** | 13,629 | 93 | 0.8346 [0.7738, 0.8944] | 0.2606 [0.1766, 0.3684] |
| **$N_{\text{pb}} > 1$** | 9,509 | 67 | 0.8558 [0.7862, 0.9186] | 0.3276 [0.2137, 0.4505] |
| **Both** | 9,509 | 67 | 0.8558 [0.7862, 0.9186] | 0.3276 [0.2137, 0.4505] |

### Finding:
- **No Dilution from Pooling**: The discriminative power (ROC-AUC ~ 0.84–0.85) is statistically consistent across all strata. Pooling sparse alerts and well-observed alerts does not dilute the early classification signal.
- **PR-AUC Density Effect**: PR-AUC increases in the well-observed subsets because the base rate of kilonovae increases (from 0.24% in the full population to 0.70% in the $N_{\text{pb}} > 1$ subset), representing a physical selection effect rather than a change in classifier discriminative resolution.

---

## 6. Architectural Decision & Conclusion

1. **Adopt Corrected Capacity**: We adopt the rate-corrected capacity ($K=22$) and document this decision in Addendum 007.1. This represents the physically correct way to scale spectroscopic resources to alert stream size.
2. **Preserve Classifier v1**: We preserve the baseline classifier version 1 without retuning. The hyperparameter search (Step 3) showed that retuning does not yield a consistent or meaningful improvement over the default settings, and the default model remains extremely competitive.
3. **No Information Ceiling**: The low recovery rate in triage is not due to a "near-chance information ceiling" (ROC-AUC at $e \le 2$d is high: ~0.82–0.85). Rather, the primary bottle-neck is the **extreme scarcity of follow-up slots relative to the search volume** (only 44 trigger slots for 133 target kilonovae hidden in 55,915 alerts). The Sequential Decision Policy successfully elevates kilonova concentration, achieving a 180x density boost in target kilonovae among the triggered alerts.
