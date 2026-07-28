"""Configuration for baseline classifier training and evaluation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class BaselineClassifierConfig(BaseModel):
    """Validated configuration for baseline early-epoch classifier (S=1 training).

    Parameters
    ----------
    epochs : tuple[float, ...]
        Elapsed decision epochs in days after first detection
        (default: (0.0, 2.0, 7.0)).
    study_classes : tuple[int, ...]
        PLAsTiCC class IDs for study target and comparison classes
        (default: (64, 90, 95)). 64: Kilonova, 90: SN Ia, 95: SLSN-I.
    learning_rate : float
        Learning rate for gradient boosting (default: 0.05).
    max_iter : int
        Maximum number of boosting iterations / trees (default: 100).
    max_depth : int | None
        Maximum depth of individual trees (default: 5).
    min_samples_leaf : int
        Minimum number of samples required at a leaf node (default: 20).
    l2_regularization : float
        L2 regularization parameter (default: 1.0).
    random_seed : int
        Random seed for model fitting reproducibility (default: 42).
    """

    model_config = ConfigDict(frozen=True)

    epochs: tuple[float, ...] = (0.0, 2.0, 7.0)
    study_classes: tuple[int, ...] = (64, 90, 95)
    learning_rate: float = 0.05
    max_iter: int = 100
    max_depth: int | None = 5
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_seed: int = 42

    @field_validator("learning_rate", "l2_regularization")
    @classmethod
    def must_be_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Learning rate and L2 regularization must be positive.")
        return value

    @field_validator("max_iter", "min_samples_leaf")
    @classmethod
    def must_be_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Iteration count and min_samples_leaf must be positive.")
        return value
