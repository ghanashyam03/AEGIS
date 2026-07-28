"""Baseline multiclass classifier trained on the BIASED (S=1) population.

This module implements a properly probabilistic multiclass classifier over the three
study classes (kilonova: 64, SNIa: 90, SLSN-I: 95) evaluated at each observer-frame
epoch e in {0, 2, 7} days per ADR 003 & ADR 005.

Models are trained strictly on the BIASED (S=1) population without touching or
anticipating evaluation on the TRUE (S=0) deployment population.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
)

from aegis.config.models import BaselineClassifierConfig
from aegis.features.representation import FeatureStatus, RepresentationResult

STATUS_ENCODING: dict[str, int] = {
    FeatureStatus.WELL_CONSTRAINED.value: 0,
    FeatureStatus.UNCONSTRAINED_INSUFFICIENT_OBSERVATIONS.value: 1,
    FeatureStatus.UNCONSTRAINED_ZERO_BASELINE.value: 2,
    FeatureStatus.UNCONSTRAINED_NON_POSITIVE_FLUX.value: 3,
    FeatureStatus.UNCONSTRAINED_NO_PASSBAND_PAIR.value: 4,
    FeatureStatus.UNCONSTRAINED_NO_DETECTION.value: 5,
}


def representation_results_to_dataframe(
    results: list[RepresentationResult],
) -> pd.DataFrame:
    """Convert a list of RepresentationResult objects into a model feature matrix.

    This function explicitly includes:
    - Raw physical feature values (with NaNs for unconstrained features).
    - Feature analytical uncertainties.
    - Encoded feature status indicators.
    - Summary support & fit-quality diagnostics (n_obs_total, n_det_total, etc.).

    Parameters
    ----------
    results : list[RepresentationResult]
        List of representation results for light curves.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame containing all features, uncertainties, status codes,
        and diagnostics as numeric columns suitable for model fitting.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        flat = r.to_dict()
        row: dict[str, Any] = {}
        for k, v in flat.items():
            if k in ("object_id", "epoch"):
                row[k] = v
            elif k.endswith("_status"):
                row[k] = STATUS_ENCODING.get(str(v), -1)
            else:
                row[k] = float(v) if v is not None else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


class BaselineClassifier:
    """Epoch-indexed baseline probabilistic multiclass classifier.

    Uses Histogram-Based Gradient Boosted Decision Trees (HistGradientBoosting)
    to handle tabular features with intrinsic missingness (NaNs) and explicit
    fit-quality diagnostics. Models are trained strictly on the BIASED (S=1)
    population.
    """

    def __init__(self, config: BaselineClassifierConfig | None = None) -> None:
        if config is None:
            config = BaselineClassifierConfig()
        self.config = config
        self.models_: dict[float, HistGradientBoostingClassifier] = {}
        self.feature_names_: dict[float, list[str]] = {}
        self.classes_ = np.array(self.config.study_classes, dtype=int)

    def _validate_population_boundary(
        self,
        population_type: str,
        meta_df: pd.DataFrame | None = None,
    ) -> None:
        """Assert that training/selection uses only the BIASED (S=1) population."""
        if population_type.upper() != "BIASED":
            raise ValueError(
                f"Invalid population_type '{population_type}'. BaselineClassifier "
                "must be trained EXCLUSIVELY on the BIASED (S=1) population per "
                "ADR 004 & task specification. TRUE (S=0) data is strictly forbidden."
            )

        if meta_df is not None:
            if "population" in meta_df.columns:
                non_biased = meta_df[
                    meta_df["population"].astype(str).str.upper() != "BIASED"
                ]
                if not non_biased.empty:
                    raise ValueError(
                        f"Found {len(non_biased)} non-BIASED population objects "
                        "in training set. TRUE (S=0) population data must not be "
                        "touched during classifier fitting."
                    )
            if "S" in meta_df.columns:
                non_biased_s = meta_df[meta_df["S"] != 1]
                if not non_biased_s.empty:
                    raise ValueError(
                        f"Found {len(non_biased_s)} objects with S != 1 in "
                        "training set. Baseline classifier must only train on S=1 data."
                    )

    def fit_epoch(
        self,
        X: pd.DataFrame | npt.NDArray[Any],
        y: npt.NDArray[Any] | pd.Series[Any],
        epoch: float,
        population_type: str = "BIASED",
        meta_df: pd.DataFrame | None = None,
    ) -> BaselineClassifier:
        """Fit an epoch-specific model on the BIASED (S=1) population.

        Parameters
        ----------
        X : pd.DataFrame | npt.NDArray
            Feature matrix (physical values, uncertainties, status codes, diagnostics).
        y : npt.NDArray | pd.Series
            Target class IDs (must be subset of study_classes: 64, 90, 95).
        epoch : float
            Elapsed decision epoch (e.g. 0.0, 2.0, 7.0).
        population_type : str, default "BIASED"
            Must be strictly "BIASED".
        meta_df : pd.DataFrame | None, default None
            Optional metadata DataFrame for population boundary validation.

        Returns
        -------
        self
        """
        self._validate_population_boundary(population_type, meta_df)

        if isinstance(X, pd.DataFrame):
            feature_cols = [c for c in X.columns if c not in ("object_id", "epoch")]
            X_mat = X[feature_cols].to_numpy(dtype=float)
            self.feature_names_[epoch] = feature_cols
        else:
            X_mat = np.asarray(X, dtype=float)
            self.feature_names_[epoch] = [f"feature_{i}" for i in range(X_mat.shape[1])]

        y_arr = np.asarray(y, dtype=int)

        invalid_classes = set(y_arr) - set(self.config.study_classes)
        if invalid_classes:
            raise ValueError(
                f"Target y contains classes {invalid_classes} outside study classes "
                f"{self.config.study_classes}."
            )

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            random_state=self.config.random_seed,
        )

        model.fit(X_mat, y_arr)
        self.models_[epoch] = model
        return self

    def predict_proba(
        self,
        X: pd.DataFrame | npt.NDArray[Any],
        epoch: float,
    ) -> npt.NDArray[Any]:
        """Predict class probabilities for objects at a specified epoch.

        Parameters
        ----------
        X : pd.DataFrame | npt.NDArray
            Feature matrix for evaluation objects.
        epoch : float
            Elapsed decision epoch (must have been fitted via fit_epoch).

        Returns
        -------
        npt.NDArray
            Array of shape (N, len(study_classes)) with class probabilities.
            Probabilities satisfy p_ic >= 0 and sum to 1.0 across classes.
        """
        if epoch not in self.models_:
            raise KeyError(
                f"No fitted model found for epoch {epoch}. "
                f"Available fitted epochs: {list(self.models_.keys())}."
            )

        model = self.models_[epoch]

        if isinstance(X, pd.DataFrame):
            feature_cols = self.feature_names_.get(epoch, [])
            if feature_cols:
                X_mat = X[feature_cols].to_numpy(dtype=float)
            else:
                cols = [c for c in X.columns if c not in ("object_id", "epoch")]
                X_mat = X[cols].to_numpy(dtype=float)
        else:
            X_mat = np.asarray(X, dtype=float)

        raw_probs = model.predict_proba(X_mat)

        model_classes = list(model.classes_)
        aligned_probs = np.zeros((len(X_mat), len(self.classes_)), dtype=float)

        for i, cls_id in enumerate(self.classes_):
            if cls_id in model_classes:
                idx = model_classes.index(cls_id)
                aligned_probs[:, i] = raw_probs[:, idx]

        row_sums = aligned_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        aligned_probs = aligned_probs / row_sums

        return aligned_probs
