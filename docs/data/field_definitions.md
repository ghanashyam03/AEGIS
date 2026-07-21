# PLAsTiCC Test Metadata: Field Definitions

This document defines every field from `plasticc_test_metadata.csv.gz` that is
used in AEGIS population construction. The field names, types, and descriptions
are taken from the [PLAsTiCC data-set paper (Kessler et al. 2019)](https://arxiv.org/abs/1903.11756)
and the [PLAsTiCC model reveal table](https://plasticc.org/wp-content/uploads/2019/01/plasticc_modelreveal_2versions.pdf).

## Fields Used in AEGIS

| Field | Type | Units | Description |
|---|---|---|---|
| `object_id` | int | — | Unique integer identifier for each simulated object. Primary key; must be unique across the entire table. |
| `hostgal_photoz` | float | dimensionless | Photometric redshift of the simulated host galaxy. Used as the proxy for distance/observational difficulty in the selection function. Must be ≥ 0 and finite (non-NaN, non-inf). |
| `hostgal_photoz_err` | float | dimensionless | Estimated uncertainty on `hostgal_photoz`. Must be ≥ 0. Retained in validated tables but not used directly in selection calculations. |
| `true_target` | int | — | **Unblinded class label** from the PLAsTiCC model reveal. All AEGIS class-membership checks and filtering operate on this column. Only the three pre-registered study classes are retained in the TRUE population. |

## Study Class IDs (ADR 002)

| `target` value | Class name | Notes |
|---|---|---|
| 64 | Kilonova (KN) | Primary positive class; neutron-star-merger transient. Rapid fading on day–week timescales. |
| 90 | Type Ia supernova (SN Ia) | Common comparison class. Cosmologically important; slower evolution than KN. |
| 95 | Superluminous Type I supernova (SLSN-I) | Rare, slower-evolving comparison class. |

## Fields Present in the Source File but Not Used in Population Construction

The following fields appear in the raw CSV but are not required by any pipeline
stage and are passed through transparently (`strict=False` schema):

| Field | Description |
|---|---|
| `target` | Competition placeholder column — all zeros in the released test set; the class labels were withheld during the PLAsTiCC challenge. **Not used by AEGIS.** |
| `ra` | Right ascension of the object (degrees) |
| `decl` | Declination of the object (degrees) |
| `ddf_bool` | Whether the object falls in a Deep Drilling Field (1) or Wide-Fast-Deep (0) |
| `hostgal_specz` | Spectroscopic redshift of the host galaxy (often −9 for unavailable) |
| `distmod` | Distance modulus |
| `mwebv` | Milky Way E(B−V) reddening at the object location |
| `true_*` | Additional simulation truth columns (true redshift, peak magnitude, submodel index, etc.) |

> [!IMPORTANT]
> Alert-time covariate availability has not yet been audited for this release.
> Fields like `hostgal_specz`, `distmod`, and `true_*` columns must **not** be used
> as classifier inputs until a covariate-availability audit is completed (see
> ADR 003 and `docs/problem_statement.md`).

## Schema Validation Rules

The pipeline enforces the following validation rules (defined in
`src/aegis/data/schema.py`):

| Rule | Applied at stage |
|---|---|
| `object_id` is integer, non-null, unique | RAW and TRUE |
| `hostgal_photoz` is float, ≥ 0, finite (no NaN/inf) | RAW and TRUE |
| `hostgal_photoz_err` is float, ≥ 0 | RAW and TRUE |
| `true_target` is integer, non-null | RAW |
| `true_target` ∈ {64, 90, 95} | TRUE population only |

Any row violating these rules raises `pandera.errors.SchemaErrors` immediately.
There is no silent fallback or imputation.
