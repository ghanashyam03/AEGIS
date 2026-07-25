"""Feature extraction interfaces and early representation (ADR 005)."""

from aegis.config.features import FeatureConfig
from aegis.features.representation import (
    FeatureStatus,
    RepresentationResult,
    SingleFeatureResult,
    extract_early_representation,
)

__all__ = [
    "FeatureConfig",
    "FeatureStatus",
    "RepresentationResult",
    "SingleFeatureResult",
    "extract_early_representation",
]
