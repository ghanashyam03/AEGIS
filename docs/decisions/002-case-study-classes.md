# ADR 002: Make kilonovae the time-critical class

- **Status:** accepted for the foundation study
- **Date:** 2026-07-21

## Context

The primary class must be scientifically valuable specifically when identified
early, must exist in the selected dataset, and must permit meaningful comparison
against common and slower populations rather than an artificial binary task.

## Options considered

### Kilonovae (chosen primary class)

PLAsTiCC model class 64 is a neutron-star-merger kilonova; the released model
table identifies it explicitly ([model reveal table](https://plasticc.org/wp-content/uploads/2019/01/plasticc_modelreveal_2versions.pdf)).
Kilonova spectra and colours evolve on day timescales: spectra of AT 2017gfo
changed rapidly during its first four days
([Nicholl et al. 2017](https://arxiv.org/abs/1710.05853)), and models predict
rapid fading to very faint magnitudes on roughly ten-day timescales
([Tanaka 2016](https://arxiv.org/abs/1605.07235)). This directly connects early
photometric classification to scarce, perishable spectroscopic follow-up.

### Tidal disruption events

PLAsTiCC includes TDEs (class 15), and they are scientifically valuable. They
were not chosen because their characteristic evolution is generally less tightly
coupled to a two-day trigger deadline, weakening the project's test of genuinely
early decisions.

### Type Ia supernovae

PLAsTiCC includes SNe Ia (class 90). They are abundant and important for
cosmology, but the primary scientific value here is not normally confined to the
same rapidly vanishing early window. They are more informative as a comparison
population and a common contaminant than as the central time-critical target.

## Decision

Use kilonova (PLAsTiCC class ID 64) as the primary positive class. Use Type Ia
supernova (90) as the common comparison class and superluminous Type I supernova
(95) as the rare, slower-evolving comparison class. The latter IDs and labels are
published in the [PLAsTiCC model reveal](https://plasticc.org/wp-content/uploads/2019/01/plasticc_modelreveal_2versions.pdf).

All initial decision experiments will be multiclass. The headline safety metric
will nevertheless be target-specific: a kilonova missed before the deadline is
not offset by correct classification of a non-target.

## Consequences

- The initial result is a kilonova-triggering case study, not evidence for every
  fast transient or multi-messenger observing program.
- Type Ia and SLSN-I analyses are prespecified generalization checks, not
  post-hoc substitutes if the kilonova result is weak.
- PLAsTiCC's kilonova simulations are a limited model family; results must state
  this external-validity limitation.
