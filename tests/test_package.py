"""Smoke tests for the package scaffold."""

import aegis


def test_package_is_importable() -> None:
    assert "AEGIS" in aegis.__doc__
