"""Tests for SelectionConfig Pydantic validation (ADR 004)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aegis.config.data import SelectionConfig


def _valid_kwargs() -> dict[str, object]:
    return {"seed": 42, "p_floor": 0.10, "p_bright": 0.80, "z_50": 0.5, "w_z": 0.15}


def test_valid_selection_config_accepted() -> None:
    cfg = SelectionConfig(**_valid_kwargs())
    assert cfg.p_floor == 0.10
    assert cfg.p_bright == 0.80


def test_p_bright_must_exceed_p_floor() -> None:
    with pytest.raises(
        ValidationError, match="p_bright must be strictly greater than p_floor"
    ):
        SelectionConfig(**{**_valid_kwargs(), "p_bright": 0.10})


def test_p_bright_equal_p_floor_rejected() -> None:
    with pytest.raises(ValidationError):
        SelectionConfig(**{**_valid_kwargs(), "p_bright": 0.10, "p_floor": 0.10})


def test_w_z_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="w_z must be positive"):
        SelectionConfig(**{**_valid_kwargs(), "w_z": 0.0})


def test_w_z_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        SelectionConfig(**{**_valid_kwargs(), "w_z": -0.1})


def test_z_50_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="z_50 must be positive"):
        SelectionConfig(**{**_valid_kwargs(), "z_50": 0.0})


def test_p_floor_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="p_floor must be in"):
        SelectionConfig(**{**_valid_kwargs(), "p_floor": 0.0})


def test_p_bright_must_not_exceed_one() -> None:
    with pytest.raises(ValidationError, match="p_bright must be in"):
        SelectionConfig(**{**_valid_kwargs(), "p_bright": 1.1})


def test_seed_is_stored_exactly() -> None:
    cfg = SelectionConfig(**{**_valid_kwargs(), "seed": 12345})
    assert cfg.seed == 12345


def test_config_is_frozen() -> None:
    cfg = SelectionConfig(**_valid_kwargs())
    with pytest.raises((ValidationError, TypeError)):
        cfg.seed = 0  # type: ignore[misc]
