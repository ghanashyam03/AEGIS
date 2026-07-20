# Problem statement

## Research question

Under realistic spectroscopic follow-up selection bias, how reliable and
well-calibrated is early-epoch classification of scientifically time-critical
astronomical transients, and can a decision framework that accounts for both
calibrated class confidence and novelty produce safer, more efficient
follow-up-triggering decisions than a fixed-confidence-threshold baseline?

## Required findings

The completed study will report two quantitative findings, including uncertainty
and negative or mixed results where applicable:

1. The magnitude and location of early-classification miscalibration under the
   documented selection shift, measured by the metrics and strata fixed in
   [ADR 003](decisions/003-definitions-and-metrics.md).
2. Whether a bias- and novelty-aware policy improves follow-up triage over the
   validation-selected fixed-confidence baseline at matched capacity, measured
   by normalized utility regret and missed-high-value-event rate.

The foundation study uses the public PLAsTiCC simulation as documented in
[ADR 001](decisions/001-dataset-selection.md), with kilonovae as the primary
time-critical class as documented in
[ADR 002](decisions/002-case-study-classes.md).

## Non-goals

- This is not an operational Rubin Observatory alert broker or telescope
  scheduler.
- It does not estimate the true selection function of any historical or future
  observing programme.
- It does not claim clinical-style safety guarantees, astrophysical discovery
  rates, or generalization to real survey operations from simulation alone.
- It does not optimize scientific utility weights after inspecting test labels.
- It does not implement a data pipeline, feature extractor, model, calibrator,
  novelty detector, or decision policy in this repository foundation.

## Scope boundary

The next implementation phase may begin only after a versioned data manifest,
an alert-time covariate-availability audit, and an experiment protocol are added.
The decisions in `docs/decisions/` are the controlling definitions for that
phase unless superseded by a new ADR.
