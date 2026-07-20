# ADR 003: Fix the early-epoch and evaluation definitions before modelling

- **Status:** accepted for the foundation study
- **Date:** 2026-07-21

## Context

Calibration and triage claims are easy to make ambiguously. The protocol must
fix what information is available, what counts as early, the population on which
metrics are calculated, and the utility used for the decision comparison.

## Options considered

### Fixed-confidence accuracy and aggregate calibration only

Accuracy, a confidence threshold, and one global ECE are simple to communicate.
They were rejected as the primary protocol because accuracy hides rare-target
failures, a global ECE can average away the exact covariate/epoch regions harmed
by selection, and a confidence cutoff has no explicit scarcity or novelty term.

### Epoch-indexed calibration and utility (chosen)

Evaluate only with the prefix of measurements available at each epoch and report
both proper scoring and decision outcomes. This preserves the operational
question while distinguishing probability quality from the value of a trigger.

## Decision

### Operational time and information

For an object \(i\), define \(t_{0,i}\) as the MJD of its first detection with
\(\mathrm{flux}/\mathrm{fluxerr}\ge5\). At elapsed observer-frame epoch
\(e\in\{0,2,7\}\) days, the classifier may use only rows with
\(\mathrm{MJD}\le t_{0,i}+e\), plus metadata demonstrably available at alert
time. Objects with no qualifying detection are outside the early-alert
evaluation population. The primary deadline is \(H=2\) days; 7 days is a
prespecified diagnostic horizon, not a replacement deadline.

An event is **time-critical** in this project if its target-class trigger has a
strict deadline \(H\), and the utility of a successful follow-up after \(H\) is
zero. This is a decision definition, chosen for a kilonova study; it does not
claim that all late data are scientifically worthless.

### Calibration metrics

For class \(c\), epoch \(e\), and evaluation set \(D\), let \(p_{ic,e}\) be the
predicted probability and \(y_{ic}=1[Y_i=c]\).

- **Multiclass Brier score:**
  \[BS_e=|D|^{-1}\sum_i\sum_c(p_{ic,e}-y_{ic})^2.\]
  Report the Murphy reliability, resolution, and uncertainty decomposition
  using a prespecified 10-bin equal-width probability partition, with empty bins
  omitted and bin counts reported.
- **Classwise ECE:** with the same bins \(B_m\),
  \[ECE_{c,e}=\sum_m |B_m|/|D|\;|\operatorname{mean}_{i\in B_m}p_{ic,e}-\operatorname{mean}_{i\in B_m}y_{ic}|.\]
  Report it for kilonova and every comparison class, plus adaptive-bin
  reliability diagrams (minimum 50 objects/bin where feasible).
- Report every metric separately for \(S=1\) and \(S=0\), and within
  prespecified strata: epoch, predicted-probability decile, apparent-brightness
  quintile, and redshift quintile when a released redshift field is valid at the
  intended decision time. Strata with fewer than 30 events are labeled
  exploratory rather than pooled or silently suppressed.

Uncertainty is a nonparametric object-level bootstrap, 1,000 replicates,
percentile 95% intervals. The resampling unit is the object, so no alert prefix
from an object appears in a different resample.

### Decision metrics

At epoch \(e\le H\), a policy chooses \(a_i\in\{0,1\}\) (do not trigger,
trigger). Under the pre-registered reference utility
\(u(a,Y)=2\,1[a=1,Y=KN]-1\,1[a=1,Y\ne KN]\), a true kilonova trigger gains 2
units and any non-kilonova trigger costs 1 unit. No trigger has utility zero.
The coefficient ratio is a transparent reference cost ratio, not a claimed
telescope-time valuation; sensitivity analyses will repeat ratios 1:1, 2:1, and
5:1 before any conclusion about superiority.

For a fixed capacity \(K\) per epoch, the oracle triggers the \(K\) available
objects with largest realized utility (all kilonovae first, then arbitrary
non-targets only if capacity remains). Define utility regret as
\[R_e=U_e(\text{oracle})-U_e(\text{policy}),\]
with \(U_e=\sum_i u(a_i,Y_i)\). Report normalized regret
\(R_e/\max(1,U_e(\text{oracle})-U_e(\text{no-trigger}))\).

The **missed-high-value-event rate** is
\[MHVER_e=\frac{\sum_i1[Y_i=KN,\,a_i=0]}{\sum_i1[Y_i=KN]},\]
calculated only among objects alertable by \(e\). The deadline headline is the
union-of-opportunities version: a kilonova is missed if it receives no trigger
at any evaluated epoch \(e\le H\).

The naive baseline is the best validation-selected fixed cutoff on calibrated
kilonova confidence, with the same capacity constraint and no novelty term. The
proposed policy must use calibrated confidence and a novelty score defined before
testing. Both are assessed on the identical held-out deployment population.

## Consequences

- Capacity \(K\) must be specified in a versioned experiment configuration
  before final evaluation; the first report will show a prespecified capacity
  curve rather than choose a favorable value after seeing the test labels.
- Redshift/host variables are excluded unless their alert-time availability is
  explicitly verified for the selected release.
- These are evaluation definitions only. No classifier, calibrator, novelty
  score, or policy implementation is introduced by this decision.
