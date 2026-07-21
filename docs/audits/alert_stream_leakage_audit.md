# Formal Leakage Audit: Alert-Stream Simulation & Observation Truncation Harness

> **Document ID:** `docs/audits/alert_stream_leakage_audit.md`  
> **Audit Date:** July 21, 2026  
> **Target Module:** AEGIS Observation Truncation Harness (`src/aegis/data/observation.py`, `src/aegis/data/schema.py`)  
> **Verification Basis:** ADR 003, `docs/problem_statement.md`, `docs/architecture.md`  
> **Test Harness:** `tests/test_observation_properties.py` (Hypothesis property tests), `tests/test_observation_regression.py`

---

## 1. Executive Summary

This formal audit evaluates the **AEGIS Alert-Stream Simulation and Incremental Observation Harness** (`src/aegis/data/observation.py`). The primary scientific requirement of this harness is the absolute prevention of **future-information leakage** (look-ahead bias) during epoch-by-epoch transient classification at elapsed observer-frame epochs $e \in \{0, 2, 7\}$ days.

The audit verified by mathematical definition and automated property testing that:
1. Truncation operates strictly on observation timestamps ($\text{MJD} \le T_{\rm cutoff}$).
2. First alert detection timestamp $t_0$ is computed chronologically from first detection ($\text{S/N} \ge 5.0$ or `detected_bool == 1`) and never from peak flux or post-event statistics.
3. All global light-curve statistics and simulation truth labels are stripped or rejected.
4. Cadence, measurement noise, and sampling gaps are preserved without artificial interpolation.
5. Append-only consistency is guaranteed: extending the observation horizon from $T_1$ to $T_2 > T_1$ preserves all $T_1$ historical observations without modification.

---

## 2. Field Classification & Leakage Risk Assessment

Every field present in raw or truncated light-curve observation tables has been enumerated and classified into one of three risk tiers:

| Field Name | Data Type | Field Category | Audit Classification | Justification & Safeguards |
|---|---|---|---|---|
| `object_id` | `int64` | Primary Key | **SAFE** | Static unique object identifier assigned at simulation/ingestion time. Contains zero temporal or physical state information. |
| `mjd` | `float64` | Timestamp | **SAFE** | Modified Julian Date of observation. Truncation enforces $\text{mjd} \le T_{\rm cutoff}$. Invariant to observations occurring at $t > T_{\rm cutoff}$. |
| `passband` | `int64` | Instrument Metadata | **SAFE** | LSST filter band integer ($0 \dots 5 \equiv u, g, r, i, z, Y$). Timestamp-indexed physical measurement setting. |
| `flux` | `float64` | Photometric Measurement | **SAFE** | Calibrated flux measured at timestamp `mjd`. Raw single-epoch measurement; un-influenced by future photometry. |
| `flux_err` | `float64` | Uncertainty | **SAFE** | Measurement uncertainty on `flux`. Raw single-epoch noise estimate. Enforced $\ge 0$ and finite. |
| `detected_bool` | `int64` | Single-Frame Alert Flag | **SAFE** | Binary detection status ($1$ if S/N $\ge 5.0$, else $0$). Evaluated independently per observation frame. |
| `hostgal_photoz` | `float64` | Pre-alert Catalog Feature | **CONDITIONALLY SAFE** | Photometric redshift of host galaxy. Safe IF AND ONLY IF photo-$z$ is measured from pre-transient host catalog imagery. |
| `hostgal_photoz_err` | `float64` | Pre-alert Catalog Feature | **CONDITIONALLY SAFE** | Uncertainty on `hostgal_photoz`. Same constraint as `hostgal_photoz`. |
| `mwebv`, `ra`, `decl` | `float64` | Pre-alert Catalog Metadata | **SAFE** | Spatial coordinates and Milky Way extinction from dust maps. Available prior to transient alert. |
| `hostgal_specz` | `float64` | Host Spectroscopic Redshift | **FORBIDDEN / OPEN QUESTION** | Spectroscopic redshift is frequently obtained post-alert/follow-up. Excluded from alert-time model input until pre-alert catalog presence is verified. |
| `peak_mjd` | `float64` | Global Summary Statistic | **FORBIDDEN** | Requires full light curve to determine peak date. Direct future leakage. Automatically stripped by `truncate_light_curve_at_mjd`. |
| `peak_flux` | `float64` | Global Summary Statistic | **FORBIDDEN** | Maximum flux across full light curve. Direct future leakage. Automatically stripped by `truncate_light_curve_at_mjd`. |
| `total_obs_count` | `int64` | Global Summary Statistic | **FORBIDDEN** | Total number of observations in full light curve. Direct future leakage. Automatically stripped by `truncate_light_curve_at_mjd`. |
| `max_snr` | `float64` | Global Summary Statistic | **FORBIDDEN** | Peak S/N over full light curve. Direct future leakage. Automatically stripped. |
| `final_flux` | `float64` | Global Summary Statistic | **FORBIDDEN** | Late-time fading flux level. Direct future leakage. Automatically stripped. |
| `true_target` / `target` | `int64` | Astrophysical Class Label | **FORBIDDEN** | Ground truth transient class. Held-out label for evaluation only. Automatically stripped if present in observation frame. |
| `true_z`, `true_distmod` | `float64` | Simulation Truth Parameters | **FORBIDDEN** | True redshift and distance modulus. Held-out simulation truth. Automatically stripped. |

---

## 3. Mathematical & Structural Invariants

### 3.1 Cutoff Bound Guarantee
For any light curve DataFrame $\mathcal{L}$ and cutoff MJD $T_{\rm cutoff}$, the truncated output $\mathcal{O}(T_{\rm cutoff}) = \text{truncate}(\mathcal{L}, T_{\rm cutoff})$ satisfies:
$$\forall r \in \mathcal{O}(T_{\rm cutoff}), \quad r.\text{mjd} \le T_{\rm cutoff}$$

### 3.2 Append-Only Consistency
For any two cutoff times $T_1 < T_2$:
$$\mathcal{O}(T_1) = \{ r \in \mathcal{O}(T_2) \mid r.\text{mjd} \le T_1 \}$$
Historical observation rows in $\mathcal{O}(T_1)$ are identical in row count, column values, and ordering to the prefix of $\mathcal{O}(T_2)$.

### 3.3 Future Mutation Invariance
Let $\mathcal{L}'$ be a modified version of light curve $\mathcal{L}$ where any observation $r$ with $r.\text{mjd} > T_{\rm cutoff}$ is altered, corrupted, or appended. Then:
$$\text{truncate}(\mathcal{L}, T_{\rm cutoff}) \equiv \text{truncate}(\mathcal{L}', T_{\rm cutoff})$$
Future observations have zero mathematical influence on the truncated output as of $T_{\rm cutoff}$.

### 3.4 Chronological Initial Alert $t_0$
First alert $t_0$ is defined per ADR 003 as:
$$t_0 = \min \{ \text{mjd} \mid \text{flux} / \text{flux\_err} \ge 5.0 \lor \text{detected\_bool} = 1 \}$$
If no qualifying observation exists, $t_0 = \varnothing$. At epoch $e$ days since first alert, cutoff MJD is $T_{\rm cutoff} = t_0 + e$. $t_0$ depends strictly on observations up to $t_0$ and is invariant to all observations at $t > t_0$.

---

## 4. Automated Test Verification Summary

The leakage-safety properties are proven by automated property-based tests using `hypothesis` (`tests/test_observation_properties.py`) and concrete regression tests (`tests/test_observation_regression.py`):

| Test Name | Verified Invariant | Status |
|---|---|---|
| `test_property_cutoff_bound` | Every returned row satisfies $\text{mjd} \le T_{\rm cutoff}$ across 50 random cutoff points. | **PASS** |
| `test_property_future_invariance` | Mutating/adding future rows at $t > T_{\rm cutoff}$ leaves output at $T_{\rm cutoff}$ 100% identical. | **PASS** |
| `test_property_append_only_consistency` | $\text{truncate}(T_1)$ equals prefix of $\text{truncate}(T_2)$ for $T_1 < T_2$. | **PASS** |
| `test_property_monotonicity` | $\text{len}(\text{truncate}(T_1)) \le \text{len}(\text{truncate}(T_2))$ for $T_1 \le T_2$. | **PASS** |
| `test_property_cadence_preservation` | Output timestamps and fluxes are exact subsets of source data without fake grid interpolation. | **PASS** |
| `test_property_forbidden_fields_stripped` | All forbidden global fields (`peak_mjd`, `peak_flux`, etc.) are stripped automatically. | **PASS** |
| `test_property_t0_chronological_invariance` | $t_0$ calculation is invariant to post-$t_0$ light curve changes. | **PASS** |
| `test_property_as_of_epoch_sequences` | Epoch sequences at $e \in \{0, 2, 7\}$ days preserve append-only prefix integrity. | **PASS** |
| `test_kilonova_epoch_truncation` | Concrete regression on rapidly evolving Kilonova (class 64) profile at $e=0, 2, 7$ days. | **PASS** |
| `test_no_qualifying_detection_handling` | Returns empty observation frame gracefully when S/N $< 5.0$. | **PASS** |
| `test_multi_object_batch_truncation` | Multi-object batch truncation with heterogeneous $t_0$ alert times. | **PASS** |
| `test_invalid_schema_rejection` | Invalid DataFrames (NaN flux, negative flux_err) fail loud via Pandera. | **PASS** |

---

## 5. Audit Conclusion

The AEGIS observation truncation harness (`src/aegis/data/observation.py`) is **certified leakage-safe**. It strictly isolates pre-cutoff observations, strips global summary statistics, preserves real cadence and noise, and enforces append-only consistency. Downstream feature extractors and classifiers operating on outputs of this module cannot access post-cutoff information by construction.
