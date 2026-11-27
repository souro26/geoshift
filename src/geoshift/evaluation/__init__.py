"""Evaluation metrics and domain-generalization analysis for GeoShift."""

from geoshift.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_generalization_gap,
    compute_per_class_metrics,
)

__all__ = [
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_generalization_gap",
    "compute_per_class_metrics",
]