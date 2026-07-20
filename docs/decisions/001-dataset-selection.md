# ADR 001: Use PLAsTiCC as the first controlled study dataset

- **Status:** accepted for the foundation study
- **Date:** 2026-07-21

## Context

The study needs an alert-stream-like simulated dataset with irregular, noisy
multi-band light curves and a labeled subset that is deliberately unlike the
deployment population. The labeled subset is the proxy for a spectroscopically
followed-up sample; it must support a reproducible analysis of selection-induced
miscalibration rather than merely a generic domain-shift experiment.

## Options considered

### PLAsTiCC (chosen)

The [PLAsTiCC data-set paper](https://arxiv.org/abs/1810.00001) describes an
LSST-like photometric time-series challenge, and the accompanying simulations
paper documents use of the LSST Operations Simulator and SNANA to produce
observed fluxes and uncertainties in ugrizy bands
([Kessler et al. 2019](https://arxiv.org/abs/1903.11756)). Its released training
sample is a simulated spectroscopic-classification subset of the much larger
detected population; that paper describes the selection construction. This makes
the observed training-versus-test mismatch a suitable, documented operational
proxy for the bias introduced by spectroscopic follow-up.

The unblinded release is public at [Zenodo record 2539456](https://zenodo.org/records/2539456),
with train metadata, train light curves, test metadata, and partitioned test
light curves. On 2026-07-21, `curl -L -I` against the public train-metadata
download URL returned HTTP 200 and `Content-Length: 370350` in this environment.
The full release is sizeable (the record lists 7.5 GB of files), so it is not
committed and future acquisition will be scripted with checksums and a manifest.

### ELAsTiCC / ELAsTiCC2 (deferred alternative)

The [DESC ELAsTiCC portal](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/)
describes alert-stream simulations with millions of objects and tens of millions
of alerts. It is closer to an end-to-end streaming setting, and its broker
training sets were intentionally compositionally different from the stream.
However, some direct/API truth-table access is described as requiring a DESC TOM
account, while the full public training archive is 7.4 GiB. At foundation time,
we did not verify a complete, account-free local download and truth-join workflow.
It is therefore unsuitable as the first reproducible baseline, though it is the
planned scale-up and external-validity check once access is confirmed.

## Decision

Use PLAsTiCC for the first study. Treat membership in its released training set
as the spectroscopic-label indicator \(S=1\), and the released test population as
the deployment population \(S=0\). Do not claim this is a literal record of a
single telescope's historical target-selection program: it is a published
simulation of a plausible spectroscopic sample.

The later data layer will reconstruct an *empirical* selection model
\(\hat{s}(x)=P(S=1\mid x)\) from covariates available at the decision epoch
(photometry-derived summaries, sky/host fields only where released, and epoch).
It will report overlap/positivity diagnostics and will not extrapolate a
selection correction into covariate regions with no labeled support. The
published simulator description is the provenance for the label mechanism; the
empirical model is an analysis aid, not a claim to recover hidden operational
decisions exactly.

## Consequences

- Results answer a controlled LSST-like simulation question, not a measured
  performance estimate for Rubin operations.
- The release's class taxonomy includes the desired rare, rapid transient
  population, but small labeled counts may make class- and epoch-stratified
  intervals wide; uncertainty reporting is mandatory.
- Dataset acquisition, parsing, and any selection-model code are intentionally
  out of scope for this foundation commit.
