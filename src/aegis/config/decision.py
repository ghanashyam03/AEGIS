"""Configuration for triage decision policy and utility evaluation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class DecisionPolicyConfig(BaseModel):
    """Configuration for sequential triage decision policy and reference utility.

    Parameters
    ----------
    epochs : tuple[float, ...]
        Observer-frame decision epochs in days (default: (0.0, 2.0, 7.0)).
    primary_deadline : float
        Primary decision deadline epoch H in days (default: 2.0).
    target_class : int
        PLAsTiCC class ID for high-value target event (default: 64 for Kilonova).
    capacity_per_epoch : int
        Maximum number of trigger actions K per decision epoch (default: 5).
    novelty_weight : float
        Decision weight w_nov for normalized novelty score in combined score
        S_e = p_KN + w_nov * N_norm (default: 0.05).
    decision_threshold : float
        Minimum score threshold tau_e required to trigger an alert (default: 0.001).
    u_tp : float
        Reference utility gain for triggering high-value event (default: +2.0).
    u_fp : float
        Reference utility cost for triggering non-target event (default: -1.0).
    """

    model_config = ConfigDict(frozen=True)

    epochs: tuple[float, ...] = (0.0, 2.0, 7.0)
    primary_deadline: float = 2.0
    target_class: int = 64
    capacity_per_epoch: int = 5
    novelty_weight: float = 0.05
    decision_threshold: float = 0.001
    u_tp: float = 2.0
    u_fp: float = -1.0

    @field_validator("capacity_per_epoch")
    @classmethod
    def must_be_positive_capacity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Capacity per epoch must be at least 1.")
        return value

    @field_validator("novelty_weight")
    @classmethod
    def must_be_non_negative_weight(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("Novelty weight must be non-negative.")
        return value

    @field_validator("u_tp")
    @classmethod
    def must_be_positive_utp(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("True positive utility u_tp must be positive.")
        return value

    @field_validator("u_fp")
    @classmethod
    def must_be_negative_ufp(cls, value: float) -> float:
        if value >= 0.0:
            raise ValueError("False positive utility u_fp must be negative.")
        return value
