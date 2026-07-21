# AEGIS

**A**lert **E**valuation for **G**eneralizable, **I**nformed **S**pectroscopy.

AEGIS is a research framework for studying whether early classification of
astronomical transients remains calibrated when spectroscopic labels are
selection-biased, and whether follow-up triggers that combine calibrated class
confidence with novelty can make safer, more efficient decisions than a fixed
confidence threshold.

## Status

The data ingestion pipeline is implemented and tested. The repository now contains:

- **Ingestion pipeline** (`src/aegis/data/`): three independently re-runnable stages —
  raw download → schema-validated interim → TRUE population (class-filtered) →
  BIASED population (logistic proxy selection function applied). Each stage writes a
  JSON manifest with SHA-256 checksums, row counts, and class balance.
- **Config-driven** (`configs/data_population.yaml`): all pipeline parameters are
  Pydantic-validated (`SelectionConfig`, `PopulationConfig`). Hardcoded constants
  do not exist in the pipeline.
- **Schema validation** (`src/aegis/data/schema.py`): pandera enforces required
  fields, finite photo-z values, unique object IDs, and study-class membership.
  Malformed rows fail loudly.
- **Selection function** (`src/aegis/data/population.py`): logistic proxy for
  spectroscopic follow-up selection (ADR 004). Explicitly documented as a
  modeling assumption, not a measured survey selection function.
  See [`docs/data/selection_function.md`](docs/data/selection_function.md).
- **Test suite** (`tests/`): 36 tests — schema validation, Pydantic config
  validation, logistic function properties, strict-subset contract, no-leakage
  contract, determinism (same seed → identical output), and manifest completeness.
- **Documentation** (`docs/data/`): dataset provenance and license, field
  definitions, and selection function formula with its explicit "MODELING
  ASSUMPTION" labeling and literature motivation.

The next phase (feature extraction, classifier training, calibration, and decision
policy) has not yet begun. The pipeline produces data files in `data/processed/`
which are the input to that phase.

## Research question

Under realistic spectroscopic follow-up selection bias, how reliable and
well-calibrated is early-epoch classification of scientifically time-critical
astronomical transients, and can a decision framework that accounts for both
calibrated class confidence and novelty produce safer, more efficient
follow-up-triggering decisions than a fixed-confidence-threshold baseline?

The first controlled study will use the public LSST-like PLAsTiCC simulation and
will focus on kilonova triggers. The rationale, access verification, limitations,
and alternative considered are in
[ADR 001](docs/decisions/001-dataset-selection.md) and
[ADR 002](docs/decisions/002-case-study-classes.md).

## What the completed study will measure

- Early, classwise calibration under the selected-versus-deployment population
  shift, including Brier score decomposition and expected calibration error.
- Matched-capacity follow-up quality versus a fixed-confidence baseline, using
  oracle-normalized utility regret and the missed-high-value-event rate.

The exact operational-time definition, equations, stratification, uncertainty
procedure, and utility sensitivity plan are fixed in
[ADR 003](docs/decisions/003-definitions-and-metrics.md).

## Install and verify

Prerequisites: Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --all-groups --locked
uv run ruff format --check
uv run ruff check
uv run mypy src
uv run pytest
uv run pre-commit run --all-files
```

To enable the same checks before commits:

```powershell
uv run pre-commit install
```

The GitHub Actions workflow runs the equivalent formatting, linting, type-check,
and test commands on pushes and pull requests.

## Repository map

| Path | Contents |
| --- | --- |
| `docs/problem_statement.md` | Fixed question, required findings, and non-goals |
| `docs/decisions/` | Architecture decision records (ADR 001–004) |
| `docs/data/` | Dataset provenance, field definitions, selection function formula |
| `docs/architecture.md` | Planned component boundaries and reproducibility rules |
| `configs/data_population.yaml` | Pydantic-validated pipeline configuration (classes, paths, selection params) |
| `src/aegis/config/data.py` | `PopulationConfig`, `SelectionConfig`, `load_population_config` |
| `src/aegis/data/schema.py` | `RAW_METADATA_SCHEMA`, `TRUE_POPULATION_SCHEMA`, validation functions |
| `src/aegis/data/ingest.py` | `download_raw_metadata`, `validate_to_interim`, `build_true_population` |
| `src/aegis/data/population.py` | `logistic_spec_probability`, `apply_selection_function` |
| `src/aegis/data/manifest.py` | `sha256sum`, `class_balance`, `selection_summary`, `write_manifest` |
| `scripts/ingest_population.py` | CLI entry point; `--stage raw|interim|true|biased|all` |
| `tests/` | 36 tests: schema, config, logistic function, population contract |
| `data/` | Ignored downloaded and derived artifacts; see its README |

## Data policy

Raw and derived data must not be committed. The public source, checked access
route, and planned acquisition constraints are documented in ADR 001. Future
runs must record source checksums, parser versions, split seeds, and evaluation
configuration before they are interpreted.

## Development standards

The project uses uv, Ruff, mypy, pytest with coverage, Pydantic v2, pre-commit,
and GitHub Actions. Use Conventional Commit messages. Any change to a methodology
decision requires a new ADR or an explicit superseding update that preserves the
reasoning and consequences.

## License

AEGIS is distributed under the [MIT License](LICENSE).
