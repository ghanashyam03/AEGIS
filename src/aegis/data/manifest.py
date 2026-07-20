"""Manifest utilities for immutable, independently rerunnable data stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256sum(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_balance(frame: pd.DataFrame) -> dict[str, int]:
    """Return JSON-safe target counts, sorted by class identifier."""

    return {
        str(label): int(count)
        for label, count in frame["target"].value_counts().sort_index().items()
    }


def selection_summary(
    true_df: pd.DataFrame,
    biased_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute per-class selection rates and overall retention fraction.

    Parameters
    ----------
    true_df:
        TRUE population DataFrame (must contain ``object_id`` and ``target``).
    biased_df:
        BIASED population DataFrame (strict subset of true_df by object_id).

    Returns
    -------
    dict with keys:
        ``true_total``, ``biased_total``, ``overall_retention``,
        ``per_class``: dict mapping class-id string → {true, biased, retention}.
    """
    true_counts = class_balance(true_df)
    biased_counts = class_balance(biased_df)
    per_class: dict[str, dict[str, Any]] = {}
    for cls_id, true_n in true_counts.items():
        biased_n = biased_counts.get(cls_id, 0)
        per_class[cls_id] = {
            "true": true_n,
            "biased": biased_n,
            "retention": round(biased_n / true_n, 6) if true_n > 0 else None,
        }
    true_total = len(true_df)
    biased_total = len(biased_df)
    return {
        "true_total": true_total,
        "biased_total": biased_total,
        "overall_retention": round(biased_total / true_total, 6)
        if true_total > 0
        else None,
        "per_class": per_class,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write canonical JSON so stage metadata can be compared byte-for-byte."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
