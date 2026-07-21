"""Data ingestion, population construction, and observation truncation harness."""

from aegis.data.observation import (
    generate_as_of_epoch_sequences,
    get_first_detection_mjd,
    truncate_light_curve_at_epoch,
    truncate_light_curve_at_mjd,
)
from aegis.data.schema import (
    OBSERVATION_FORBIDDEN_FIELDS,
    OBSERVATION_SCHEMA,
    RAW_METADATA_SCHEMA,
    TRUE_POPULATION_SCHEMA,
    validate_observation_frame,
    validate_raw_metadata,
    validate_true_population,
)

__all__ = [
    "OBSERVATION_FORBIDDEN_FIELDS",
    "OBSERVATION_SCHEMA",
    "RAW_METADATA_SCHEMA",
    "TRUE_POPULATION_SCHEMA",
    "generate_as_of_epoch_sequences",
    "get_first_detection_mjd",
    "truncate_light_curve_at_epoch",
    "truncate_light_curve_at_mjd",
    "validate_observation_frame",
    "validate_raw_metadata",
    "validate_true_population",
]
