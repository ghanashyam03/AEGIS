"""Validated configuration for population-data ingestion."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator, model_validator


class DatasetConfig(BaseModel):
    """Immutable source identity for the released metadata table."""

    model_config = ConfigDict(frozen=True)

    name: str
    url: HttpUrl
    filename: str
    license: str

    @field_validator("filename")
    @classmethod
    def filename_is_basename(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("filename must not contain a directory")
        return value


class PathsConfig(BaseModel):
    """Repository-relative artifact locations."""

    model_config = ConfigDict(frozen=True)

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path


class SelectionConfig(BaseModel):
    """Logistic proxy for spectroscopic follow-up selection (ADR 004).

    MODELING ASSUMPTION
    -------------------
    The functional form and all numeric defaults are a literature-motivated
    proxy, NOT a measured survey selection function.  See
    docs/data/selection_function.md for full justification and caveats.

    Parameters
    ----------
    seed:
        Integer seed for numpy.random.default_rng, making Bernoulli draws
        deterministic and reproducible from a fixed value.
    p_floor:
        Asymptotic inclusion probability for very distant (high-z) objects.
    p_bright:
        Asymptotic inclusion probability for very nearby (low-z) objects.
    z_50:
        Photo-z value at which p_spec equals the midpoint (p_floor+p_bright)/2.
    w_z:
        Logistic scale parameter; smaller values produce a sharper transition.
    """

    model_config = ConfigDict(frozen=True)

    seed: int
    p_floor: float
    p_bright: float
    z_50: float
    w_z: float

    @model_validator(mode="after")
    def validate_probability_ordering(self) -> SelectionConfig:
        if not 0 < self.p_floor < 1:
            raise ValueError("p_floor must be in (0, 1)")
        if not 0 < self.p_bright <= 1:
            raise ValueError("p_bright must be in (0, 1]")
        if self.p_bright <= self.p_floor:
            raise ValueError("p_bright must be strictly greater than p_floor")
        if self.z_50 <= 0:
            raise ValueError("z_50 must be positive")
        if self.w_z <= 0:
            raise ValueError("w_z must be positive")
        return self


class PopulationConfig(BaseModel):
    """Configuration required before any source data are read."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetConfig
    paths: PathsConfig
    classes: dict[str, int]
    selection: SelectionConfig

    @field_validator("classes")
    @classmethod
    def expected_class_mapping(cls, value: dict[str, int]) -> dict[str, int]:
        expected = {"kilonova": 64, "sn_ia": 90, "slsn_i": 95}
        if value != expected:
            raise ValueError(f"classes must be exactly {expected}")
        return value


def load_population_config(path: Path) -> PopulationConfig:
    """Load a YAML configuration and validate its complete schema."""

    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return PopulationConfig.model_validate(payload)
