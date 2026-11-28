"""Evaluation metrics, calibration, and selective prediction for GeoShift."""

from geoshift.evaluation.calibration import (
    DEFAULT_COVERAGES,
    apply_temperature,
    compute_brier_score,
    compute_expected_calibration_error,
    compute_negative_log_likelihood,
    compute_selective_prediction_curve,
    fit_temperature,
)
from geoshift.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_generalization_gap,
    compute_per_class_metrics,
)

__all__ = [
    "DEFAULT_COVERAGES",
    "apply_temperature",
    "compute_brier_score",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_expected_calibration_error",
    "compute_generalization_gap",
    "compute_negative_log_likelihood",
    "compute_per_class_metrics",
    "compute_selective_prediction_curve",
    "fit_temperature",
]