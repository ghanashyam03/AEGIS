"""Selection-aware recalibration and positivity diagnostics module for AEGIS."""

from aegis.recalibration.selection_recalibration import (
    PositivityDiagnostic,
    SelectionAwareRecalibrator,
    compute_covariate_balance,
    compute_selection_weights,
    diagnose_positivity_overlap,
)

__all__ = [
    "PositivityDiagnostic",
    "SelectionAwareRecalibrator",
    "compute_covariate_balance",
    "compute_selection_weights",
    "diagnose_positivity_overlap",
]
