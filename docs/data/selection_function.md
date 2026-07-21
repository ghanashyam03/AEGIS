# Spectroscopic Follow-Up Selection Function

> [!CAUTION]
> **MODELING ASSUMPTION — NOT AN ESTABLISHED SELECTION FUNCTION**
>
> The selection function documented on this page is a proxy constructed for
> the AEGIS study.  It is **not** the selection function of PLAsTiCC, DES,
> Rubin Observatory, or any historical survey programme.  The functional form
> and **all numeric parameters** were chosen to reflect the qualitative
> direction of real spectroscopic follow-up biases as documented in the
> literature; they are not fitted to any real survey data.  They must not be
> cited as an established result.

---

## Purpose

The AEGIS study measures the effect of spectroscopic follow-up selection bias
on classifier calibration.  The selection function defined here produces the
**BIASED population** — the simulated subset of the TRUE population that would
realistically have received a spectroscopic label — from the TRUE population,
which is the full PLAsTiCC test metadata restricted to the three study classes.

See [ADR 004](../decisions/004-proxy-spectroscopic-selection.md) for the full
decision rationale.

---

## Formula (ADR 004)

For object $i$ with host-galaxy photometric redshift $z_{\rm phot}$:

$$
p_{\rm spec}(z_{\rm phot})
= p_{\rm floor}
  + \frac{p_{\rm bright} - p_{\rm floor}}
         {1 + \exp\!\left(\dfrac{z_{\rm phot} - z_{50}}{w_z}\right)}
$$

This is a logistic (sigmoid) function that:

- Decreases **monotonically** from $p_{\rm bright}$ at $z \to 0$ to
  $p_{\rm floor}$ at $z \to \infty$
- Has its midpoint $(p_{\rm floor} + p_{\rm bright})/2$ at $z = z_{50}$
- Has transition width controlled by $w_z$ (smaller = sharper)

The proxy is **class-blind**: the same curve applies to kilonova, SN Ia, and
SLSN-I identically.  The Bernoulli outcome for each object is independent,
drawn from `numpy.random.default_rng(seed)`.

---

## Parameter Values

The parameters below are read from `configs/data_population.yaml` and are
validated by `SelectionConfig` (Pydantic, frozen model).  Changing any value
requires updating the config and re-running the full pipeline.

| Parameter | Value | Meaning |
|---|---|---|
| `p_floor` | 0.10 | Asymptotic inclusion probability at very high redshift |
| `p_bright` | 0.80 | Asymptotic inclusion probability at very low redshift |
| `z_50` | 0.50 | Redshift at the midpoint (45% inclusion probability) |
| `w_z` | 0.15 | Logistic scale (transition width in redshift units) |
| `seed` | 42 | NumPy `default_rng` seed for reproducible Bernoulli draws |

> [!NOTE]
> **Why these values?**
> The general direction — brighter/closer targets are spectroscopically
> confirmed at higher rates than fainter/more distant ones — is established in
> the literature (see below).  The specific numeric values were chosen to
> produce a biased population that:
> (a) retains a majority of nearby objects ($p_{\rm bright} = 0.80$),
> (b) retains only a small minority of distant objects ($p_{\rm floor} = 0.10$),
> (c) transitions in the $z \approx 0.3$–$0.7$ range typical of photometric
>     surveys, and
> (d) keeps a meaningful number of each study class in the biased population.
>
> They were **not** optimized after inspecting the test labels or class balance.

---

## Literature Motivation

The logistic form expresses the well-established direction of real follow-up
selection, motivated by two published analyses:

1. **Vincenzi et al. (2021)** — DES-SN spectroscopic redshift efficiency:
   The DES 5-year supernova cosmology analysis modeled the spectroscopic-redshift
   efficiency as a selection effect that shapes the redshift distribution and
   biases samples toward host galaxies that are easier to observe spectroscopically
   (brighter, lower redshift).
   [arXiv:2012.07180](https://arxiv.org/abs/2012.07180)

2. **Kessler et al. (2019)** — DES-SN spectroscopic efficiency vs. peak magnitude:
   DES-SN used a smooth, parametric spectroscopic efficiency function of peak
   apparent magnitude to characterize how selection probability varies with
   source brightness.
   [FERMILAB-PUB-20-016-AE](https://lss.fnal.gov/archive/2020/pub/fermilab-pub-20-016-ae.pdf)

> [!IMPORTANT]
> The **logistic functional form**, its use of **host photo-z** rather than peak
> apparent magnitude, and **all numeric parameters** are modeling assumptions
> specific to AEGIS.  The two papers above motivate the direction of the bias
> but do not prescribe this particular parameterization.

---

## Implementation

The selection function is implemented in
[`src/aegis/data/population.py`](../../src/aegis/data/population.py):

```python
def logistic_spec_probability(photoz, p_floor, p_bright, z_50, w_z):
    exponent = (photoz - z_50) / w_z
    return p_floor + (p_bright - p_floor) / (1.0 + np.exp(exponent))
```

Per-object uniform variates are drawn from `numpy.random.default_rng(seed)`.
An object is included in the BIASED population when its variate is strictly
less than $p_{\rm spec}(z_{\rm phot})$.

Objects with missing or non-finite `hostgal_photoz` fail schema validation
before reaching this stage.  No missing value is silently converted into a
selection decision.

---

## Class Balance: TRUE vs. BIASED Population

> [!NOTE]
> The numbers in this table were computed from the actual ingested PLAsTiCC
> test metadata using the pipeline parameters above.  They must not be edited
> by hand; re-run the pipeline and copy from `data/processed/manifest_biased.json`.

<!-- CLASS_BALANCE_TABLE_START -->
<!-- Computed by running the full pipeline on 2026-07-21:                        -->
<!-- uv run python scripts/ingest_population.py --config configs/data_population.yaml --stage all -->
<!-- Verified against data/processed/manifest_biased.json                        -->

| Population | Class 64 (KN) | Class 90 (SN Ia) | Class 95 (SLSN-I) | Total |
|---|---|---|---|---|
| **TRUE** | 133 | 1,659,831 | 35,782 | **1,695,746** |
| **BIASED** | 78 | 644,876 | 5,405 | **650,359** |
| **Retention rate** | 58.6% | 38.9% | 15.1% | **38.4%** |

> [!IMPORTANT]
> The class imbalance is **pre-existing in the simulation**: KN (class 64) has only
> 133 objects in the full test population (vs. 1.66 M SN Ia). This is not an
> artifact of the selection function — the selection function *further* suppresses
> the rare SLSN-I class (15.1% retention) relative to SN Ia (38.9%), because
> SLSN-I tend to be at higher redshifts. KN retention (58.6%) is higher than
> average because the 133 simulated kilonova are distributed across a wide redshift
> range and many fall below z_50.

<!-- CLASS_BALANCE_TABLE_END -->

---

## Reproducibility

Every pipeline run writes `data/processed/manifest_biased.json` containing:

- The full selection parameter set (p_floor, p_bright, z_50, w_z, seed)
- The selection formula (human-readable string)
- SHA-256 checksums of the TRUE and BIASED population files
- Per-class and overall selection summary (counts and retention rates)
- The `"MODELING ASSUMPTION"` disclaimer string

Two runs from the same seed produce byte-identical BIASED populations
(verified by `test_determinism_same_seed` in `tests/test_population.py`).

---

## Limitations

1. **Photo-z is a proxy for observational difficulty**, not a complete
   description.  Real follow-up selection also depends on cadence, weather,
   host surface brightness, human prioritization, and telescope allocation
   — none of which are captured here.

2. **The proxy is class-blind**.  Real follow-up programmes may prioritize
   specific classes based on pre-selection from alert brokers.  A class-aware
   selection would require additional modeling assumptions not grounded in the
   released PLAsTiCC simulation.

3. **Results are specific to this proxy**.  Studies using a different selection
   function (e.g., empirically calibrated) may obtain different bias magnitudes
   while using the same TRUE/BIASED interface and manifest contract.

4. **Not a validation of any real survey**.  This proxy is explicitly labeled
   as a documented assumption.  It must not be described as recovering the
   PLAsTiCC, DES, or Rubin selection function.
