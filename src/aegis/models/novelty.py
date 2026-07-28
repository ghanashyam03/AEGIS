"""Novelty and distributional-distance signal module for AEGIS.

Implements the novelty detector according to ADR 006:
- Reference population: strictly the spectroscopically confirmed (S=1) population.
- Identifiable feature subspace: strictly quantities established as identifiable
  at epoch e by representation constrainability audit
  (P6 / docs/results/representation_constrainability.json).
- Distance metric: robust standardized Euclidean distance over valid (non-NaN)
  feature dimensions per object, explicitly handling heterogeneous missingness without
  silent imputation.
- Interpretation: A novelty score measures distance from the S=1 reference distribution,
  NOT class membership or kilonova identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# Default identifiable feature subspaces per epoch (ADR 006 & P6 audit)
EPOCH_IDENTIFIABLE_FEATURES: dict[float, list[str]] = {
    0.0: ["hostgal_photoz"],
    2.0: [
        "hostgal_photoz",
        "alert_snr_0",
        "snr_growth_rate",
        "color_pb2_pb3",
        "color_pb3_pb4",
    ],
    7.0: [
        "hostgal_photoz",
        "alert_snr_0",
        "snr_growth_rate",
        "color_pb0_pb1",
        "color_pb1_pb2",
        "color_pb2_pb3",
        "color_pb3_pb4",
        "color_pb4_pb5",
        "rise_rate_pb1",
        "rise_rate_pb2",
        "rise_rate_pb3",
    ],
}


class NoveltyDetector:
    """Standardized distributional-distance novelty detector.

    Fits per-feature location and scale statistics strictly on the spectroscopically
    confirmed (S=1) reference population, then computes normalized standardized
    Euclidean distances over active non-NaN feature subsets for evaluation objects.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X_s1: pd.DataFrame | npt.NDArray[Any],
        feature_names: list[str] | None = None,
    ) -> NoveltyDetector:
        """Fit feature reference statistics on the S=1 labeled population.

        Parameters
        ----------
        X_s1 : pd.DataFrame | npt.NDArray
            Feature matrix for S=1 reference objects.
        feature_names : list[str] | None
            Column names if X_s1 is a NumPy array.

        Returns
        -------
        self
        """
        if isinstance(X_s1, pd.DataFrame):
            cols = list(X_s1.columns)
            df = X_s1
        else:
            if feature_names is None:
                cols = [f"feat_{i}" for i in range(X_s1.shape[1])]
            else:
                cols = feature_names
            df = pd.DataFrame(X_s1, columns=cols)

        self.feature_names_ = cols
        self.means_ = {}
        self.stds_ = {}

        for col in cols:
            vals = df[col].to_numpy(dtype=float)
            valid_vals = vals[np.isfinite(vals)]
            if len(valid_vals) > 0:
                mean_val = float(np.mean(valid_vals))
                std_val = float(np.std(valid_vals, ddof=1))
                if std_val < self.eps:
                    std_val = self.eps
            else:
                mean_val = 0.0
                std_val = 1.0

            self.means_[col] = mean_val
            self.stds_[col] = std_val

        self.is_fitted_ = True
        return self

    def score_samples(
        self,
        X_eval: pd.DataFrame | npt.NDArray[Any],
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """Compute per-object novelty score on evaluation data.

        Handles heterogeneous missingness by computing standardized Euclidean
        distance strictly over non-NaN features available for each object:

        N_i = sqrt( 1/|V_i| * sum_{j in V_i} ((x_{ij} - mu_j) / sigma_j)^2 )

        Parameters
        ----------
        X_eval : pd.DataFrame | npt.NDArray
            Evaluation feature matrix.
        feature_names : list[str] | None
            Column names if X_eval is a NumPy array.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing non-negative novelty scores.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "NoveltyDetector must be fitted before calling score_samples."
            )

        if isinstance(X_eval, pd.DataFrame):
            df = X_eval
            cols = list(X_eval.columns)
        else:
            if feature_names is None:
                cols = self.feature_names_
            else:
                cols = feature_names
            df = pd.DataFrame(X_eval, columns=cols)

        n_samples = len(df)
        scores = np.zeros(n_samples, dtype=float)

        # Build feature matrices
        m_list = [self.means_.get(c, 0.0) for c in cols]
        s_list = [self.stds_.get(c, 1.0) for c in cols]

        means_arr = np.array(m_list, dtype=float)
        stds_arr = np.array(s_list, dtype=float)

        vals_mat = df[cols].to_numpy(dtype=float)

        for i in range(n_samples):
            row = vals_mat[i]
            valid_mask = np.isfinite(row)
            n_valid = int(np.sum(valid_mask))

            if n_valid == 0:
                scores[i] = 0.0
            else:
                z_sq = (
                    (row[valid_mask] - means_arr[valid_mask]) / stds_arr[valid_mask]
                ) ** 2
                scores[i] = float(np.sqrt(np.mean(z_sq)))

        return scores


def compute_epoch_novelty_scores(
    df_s1_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    epoch: float,
) -> np.ndarray:
    """Compute per-object novelty scores at a specific decision epoch.

    Selects the identifiable feature subspace for the epoch per ADR 006,
    fits the NoveltyDetector on S=1 reference features, and scores df_eval.

    Parameters
    ----------
    df_s1_train : pd.DataFrame
        DataFrame of features for S=1 reference training set.
    df_eval : pd.DataFrame
        DataFrame of features for evaluation population.
    epoch : float
        Decision epoch (e.g. 0.0, 2.0, 7.0).

    Returns
    -------
    np.ndarray
        Array of novelty scores.
    """
    identifiable_candidates = EPOCH_IDENTIFIABLE_FEATURES.get(epoch, ["hostgal_photoz"])
    # Restrict to columns present in both frames
    avail_cols = [
        c
        for c in identifiable_candidates
        if c in df_s1_train.columns and c in df_eval.columns
    ]

    if not avail_cols:
        # Fallback to hostgal_photoz if available
        avail_cols = [
            c
            for c in ["hostgal_photoz"]
            if c in df_s1_train.columns and c in df_eval.columns
        ]

    detector = NoveltyDetector()
    detector.fit(df_s1_train[avail_cols])
    return detector.score_samples(df_eval[avail_cols])
