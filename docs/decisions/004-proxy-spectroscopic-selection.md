# ADR 004: Construct the labeled population from the released survey population

- **Status:** accepted; supersedes ADR 001's population-construction decision
- **Date:** 2026-07-21

## Context

The study now requires a mathematically strict relationship: every labeled
object must also be an object in the true survey population. The released
PLAsTiCC train and test files cannot supply that relationship. They are
separately simulated, prescaled samples; training-file object identifiers are
not a subset of the released test-population object identifiers. Consequently,
treating training-file membership as the label indicator would falsely describe
two independently generated releases as an object-level follow-up selection.

## Decision

The **TRUE population** is the complete released PLAsTiCC test metadata
population, restricted only to the prespecified study classes: kilonova (64),
Type Ia supernova (90), and superluminous Type I supernova (95). The released
training files are not an input to population construction and must not be
called the biased population.

The **BIASED population** is a deterministic, seeded Bernoulli subset of TRUE.
Its inclusion probability is a proxy for successful spectroscopic follow-up:

\[
p_{\rm spec}(z_{\rm phot}) = p_{\rm floor} +
(p_{\rm bright}-p_{\rm floor})
\left[1+\exp\left(\frac{z_{\rm phot}-z_{50}}{w_z}\right)\right]^{-1}.
\]

The pipeline will use the released `hostgal_photoz` as \(z_{\rm phot}\), draw
one uniform variate per object from a NumPy generator with a configured seed,
and include the object when \(u<p_{\rm spec}\). Objects with missing or
non-finite photo-z fail ingestion validation; no missing value is silently
converted into a selection decision.

The functional form expresses the established direction of real follow-up
selection: spectroscopic success becomes less likely for fainter, more distant
targets. DES-SN analyses model spectroscopic-redshift efficiency as a selection
effect that shapes the redshift distribution and biases samples toward easier
host spectra ([Vincenzi et al. 2021](https://arxiv.org/abs/2012.07180)); DES-SN
also used a smooth spectroscopic efficiency versus peak apparent magnitude
([Kessler et al. 2019](https://lss.fnal.gov/archive/2020/pub/fermilab-pub-20-016-ae.pdf)).

The logistic form, its use of host photo-z rather than measured peak magnitude,
and **all numeric parameters** \(p_{\rm floor},p_{\rm bright},z_{50},w_z\) are
explicit **modeling assumptions**, not a measured PLAsTiCC or DES selection
function. They are versioned in the Pydantic-validated data configuration and
recorded in every processed manifest. The proxy is class-blind by construction;
it uses no target label.

## Consequences

- The strict-subset and no-reverse-leakage properties can be tested directly.
- This study measures the effect of a transparent, literature-motivated proxy,
  not the selection function of Rubin, DES, or any historical programme.
- Because photo-z is a proxy for observational difficulty, it cannot capture
  cadence, weather, host surface brightness, human prioritization, or
  telescope-allocation effects. It must not be described as a survey-specific
  truth.
- Future work may replace this proxy with a survey-specific, empirically
  validated selection model while retaining the same TRUE/BIASED interface and
  manifest contract.
