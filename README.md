# AEGIS

**A**lert **E**valuation for **G**eneralizable, **I**nformed **S**pectroscopy.

AEGIS is a research framework for studying whether early classification of
astronomical transients remains calibrated when spectroscopic labels are
selection-biased, and whether follow-up triggers that combine calibrated class
confidence with novelty can make safer, more efficient decisions than a fixed
confidence threshold.

## Status

This is the reproducible research foundation. It includes the project structure,
quality toolchain, dataset and class-selection decisions, formal metrics, and
architecture. It intentionally contains no data pipeline, feature extractor,
model, calibrator, novelty detector, or trigger-policy implementation yet.

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
| `docs/decisions/` | Architecture decision records for data, classes, and metrics |
| `docs/architecture.md` | Planned component boundaries and reproducibility rules |
| `src/aegis/` | Reserved Python package interfaces by concern |
| `tests/` | Package smoke tests; future tests live alongside their components |
| `configs/` | Versioned experiment configurations (currently empty) |
| `scripts/` | Reproducible operational entry points (currently empty) |
| `notebooks/` | Exploratory notebooks (currently empty) |
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
