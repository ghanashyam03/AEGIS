"""Configuration for early light-curve feature extraction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class FeatureConfig(BaseModel):
    """Validated configuration for early light-curve feature extraction (ADR 005).

    Parameters
    ----------
    passbands : tuple[int, ...]
        List of passband IDs to consider (default: (0, 1, 2, 3, 4, 5)).
    detection_snr_threshold : float
        S/N threshold for detection (default: 5.0).
    max_color_dt_days : float
        Maximum allowed observer-frame time difference in days between passband
        observations to compute a single-epoch cross-band color (default: 0.5d).
    min_points_for_slope : int
        Minimum number of detected observations in a passband required to fit a rise
        rate (default: 2).
    min_points_for_snr_rate : int
        Minimum number of total detected observations required to compute S/N growth
        rate (default: 2).
    """

    model_config = ConfigDict(frozen=True)

    passbands: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    detection_snr_threshold: float = 5.0
    max_color_dt_days: float = 0.5
    min_points_for_slope: int = 2
    min_points_for_snr_rate: int = 2

    @field_validator("detection_snr_threshold", "max_color_dt_days")
    @classmethod
    def must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Thresholds and window durations must be positive.")
        return value

    @field_validator("min_points_for_slope", "min_points_for_snr_rate")
    @classmethod
    def must_be_at_least_two(cls, value: int) -> int:
        if value < 2:
            raise ValueError(
                "Fitting rates requires at least 2 points to avoid zero degrees of "
                "freedom."
            )
        return value
