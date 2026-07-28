# AEGIS

**A**lert **E**valuation for **G**eneralizable, **I**nformed **S**pectroscopy.

AEGIS is a research framework for studying whether early classification of
astronomical transients remains calibrated when spectroscopic labels are
selection-biased, and whether follow-up triggers that combine calibrated class
confidence with novelty can make safer, more efficient decisions than a fixed
confidence threshold.

## Status

The data ingestion pipeline, observation truncation harness, early light-curve representation, baseline classifier, True-population calibration audit (Finding #1), and **Selection-Aware Recalibration (closing Phase 3)** are implemented and verified. The repository contains:

- **Ingestion pipeline** (`src/aegis/data/`): three independently re-runnable stages —
  raw download → schema-validated interim → TRUE population (class-filtered) →
  BIASED population (logistic proxy selection function applied). Each stage writes a
  JSON manifest with SHA-256 checksums, row counts, and class balance.
- **Observation truncation harness** (`src/aegis/data/observation.py`): alert-stream simulation
  that truncates light curves to "as-of-day-$N$" partial observation sequences ($e \in \{0, 2, 7\}$ days)
  guaranteeing no future-information leakage by construction.
- **Config-driven** (`configs/data_population.yaml`): all pipeline parameters are
  Pydantic-validated (`SelectionConfig`, `PopulationConfig`, `BaselineClassifierConfig`). Hardcoded constants
  do not exist in the pipeline.
- **Early Representation** (`src/aegis/features/representation.py`): low-parameter, strictly identifiable
  early light-curve feature extraction (rise rates, single-epoch colors, S/N growth) complying with ADR 005.
- **Baseline Classifier** (`src/aegis/models/baseline.py`): epoch-indexed `HistGradientBoostingClassifier`
  trained strictly on the spectroscopically selected ($S=1$) population.
- **True-Population Calibration Audit (Finding #1)** ([`docs/results/calibration_audit_true_population.md`](docs/results/calibration_audit_true_population.md)):
  quantifies probabilistic calibration on the full TRUE deployment population ($S=0$ and FULL TRUE) across decision epochs $e \in \{0, 2, 7\}$ days.
  Establishes a quantitative **$2.11\text{--}2.17\times$ Brier score degradation** ($BS = 0.6323$ vs $0.3322$ at $e=2.0$d) driven overwhelmingly by calibration misfire ($REL = 0.6156$ vs $0.3184$).
- **Selection-Aware Recalibration (Phase 3 Closed)** ([`docs/results/recalibration_true_population.md`](docs/results/recalibration_true_population.md)):
  implements IPW Platt recalibration (`src/aegis/recalibration/`), weight diagnostics ($ESS = 1,931.1$, $CV = 0.6281$), covariate balance SMDs (reducing photo-$z$ SMD from $-0.5793$ to $-0.0039$), and positivity diagnostics (697 high-$z$ objects / 5.47% of TRUE masked from extrapolation). Empirically proves that post-hoc recalibration fails to close the residual gap ($BS = 0.7407$ vs $0.6323$ at $e=2.0$d) due to intrinsic early-epoch information limitations ($RES \approx 0.0001$).
- **Triage Decision Framework & Pre-Registration Freeze (Phase 5 Closed)** ([`docs/methodology/decision_policy.md`](docs/methodology/decision_policy.md)):
  implements sequential decision policy (`src/aegis/decision/`), reference utility ($u_{\text{tp}}=+2, u_{\text{fp}}=-1, u=0$), oracle recovery, utility regret ($R_e$), and Missed High-Value Event Rate ($MHVER_e$). Establishes ADR 007 combined score $S_e(x_i) = p_{i,\text{KN},e} + w_{\text{nov}} \cdot \mathcal{N}_{e,\text{norm}}(x_i)$. Formally **locks pre-registered configuration** (`configs/decision_policy_v1.yaml` version `v1.0.0-frozen`, seed=42, $K=5$, 2:1 cost ratio, $w_{\text{nov}}=0.05$, $\tau_e=0.001$, `uncorrected_baseline` probability source) ahead of Phase 6 evaluation.
- **Test suite** (`tests/`): 103 tests — schema validation, Pydantic config
  validation, logistic function properties, strict-subset contract, hypothesis property-based
  leakage tests, baseline classifier fitting, selection-aware recalibration engine, positivity diagnostics, decision utility metrics, leakage safety assertions, score monotonicity, and $3\times3\times3$ grid sensitivity tests.
- **Documentation & Audits** (`docs/audits/`, `docs/results/`, `docs/methodology/`): formal data pipeline audit (`docs/audits/data_pipeline_audit.md`), alert-stream leakage audit (`docs/audits/alert_stream_leakage_audit.md`), selection-bias characterization report ([`docs/results/selection_bias_characterization.md`](docs/results/selection_bias_characterization.md)), True-population calibration audit report ([`docs/results/calibration_audit_true_population.md`](docs/results/calibration_audit_true_population.md)), recalibration methodology ([`docs/methodology/recalibration.md`](docs/methodology/recalibration.md)), recalibration audit report ([`docs/results/recalibration_true_population.md`](docs/results/recalibration_true_population.md)), probability source selection ([`docs/results/probability_source_selection.md`](docs/results/probability_source_selection.md)), ADR 007 decision policy ([`docs/decisions/007-decision-policy.md`](docs/decisions/007-decision-policy.md)), decision sensitivity analysis ([`docs/results/decision_sensitivity_analysis.md`](docs/results/decision_sensitivity_analysis.md)), Class 15 decision behavior check ([`docs/results/class15_decision_behavior.md`](docs/results/class15_decision_behavior.md)), and decision policy methodology report ([`docs/methodology/decision_policy.md`](docs/methodology/decision_policy.md)).

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
| `docs/audits/` | Formal pipeline audit and alert-stream leakage audit |
| `docs/architecture.md` | Component boundaries, reproducibility rules, and truncation harness |
| `configs/data_population.yaml` | Pydantic-validated pipeline configuration |
| `src/aegis/config/data.py` | `PopulationConfig`, `SelectionConfig`, `load_population_config` |
| `src/aegis/data/schema.py` | `RAW_METADATA_SCHEMA`, `TRUE_POPULATION_SCHEMA`, `OBSERVATION_SCHEMA` |
| `src/aegis/data/ingest.py` | `download_raw_metadata`, `validate_to_interim`, `build_true_population` |
| `src/aegis/data/population.py` | `logistic_spec_probability`, `apply_selection_function` |
| `src/aegis/data/observation.py` | Observation truncation, $t_0$ calculation, epoch sequence generation |
| `src/aegis/data/manifest.py` | `sha256sum`, `class_balance`, `selection_summary`, `write_manifest` |
| `src/aegis/recalibration/` | IPW weights, weight diagnostics ($ESS$), covariate balance ($SMD$), positivity diagnostic engine |
| `scripts/ingest_population.py` | CLI entry point; `--stage raw|interim|true|biased|all` |
| `scripts/analyze_selection_bias.py` | Reproducible quantitative selection bias analyzer ($B=1,000$ bootstrap CIs) |
| `scripts/evaluate_true_population_calibration.py` | True-population calibration audit harness ($B=1,000$ bootstrap CIs) |
| `scripts/plot_calibration_audit_figures.py` | Calibration audit figure generator (reliability diagrams, Brier decomposition, strata) |
| `scripts/evaluate_recalibration.py` | Selection-aware recalibration evaluation harness ($B=1,000$ bootstrap CIs) |
| `scripts/plot_recalibration_figures.py` | Recalibration figure generator (reliability, Brier reduction, positivity, SMDs) |
| `docs/methodology/recalibration.md` | Formal recalibration methodology, causal assumptions, and residual error decomposition |
| `docs/results/selection_bias_characterization.md` | Quantified selection-bias characterization report & publication figures |
| `docs/results/calibration_audit_true_population.md` | True-population calibration audit report (Finding #1) & figures |
| `docs/results/recalibration_true_population.md` | Selection-aware recalibration audit report & residual gap characterization |
| `tests/` | 63 tests: schema, config, logistic function, leakage, baseline model, recalibration, regression tests |
| `data/` | Ignored downloaded and derived artifacts |


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
