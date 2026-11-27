"""Tests for GeoShift classification, confusion-matrix, and gap metrics."""

import numpy as np
import pytest

from geoshift.evaluation import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_generalization_gap,
    compute_per_class_metrics,
)


def test_compute_confusion_matrix_counts() -> None:
    """Confusion matrix should align rows=true and cols=predicted."""

    y_true = [0, 0, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 2, 0, 2]

    matrix = compute_confusion_matrix(y_true, y_pred, num_classes=3)

    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 2],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(matrix, expected)


def test_compute_classification_metrics_summary() -> None:
    """Overall accuracy, macro/weighted F1, and per-class metrics should match."""

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 0, 2, 2]
    class_names = ["bag", "car", "dog"]

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)

    assert metrics["num_samples"] == 6
    assert metrics["accuracy"] == pytest.approx(5 / 6, rel=1e-5)
    assert metrics["per_class"]["bag"]["support"] == 2
    assert metrics["per_class"]["bag"]["recall"] == pytest.approx(1.0)
    assert metrics["per_class"]["bag"]["precision"] == pytest.approx(2 / 3, rel=1e-5)
    assert metrics["per_class"]["car"]["recall"] == pytest.approx(0.5)
    assert metrics["per_class"]["dog"]["f1"] == pytest.approx(1.0)
    assert len(metrics["confusion_matrix"]) == 3


def test_compute_per_class_metrics_handles_zero_support() -> None:
    """Classes absent from y_true should return zero metrics without division errors."""

    per_class = compute_per_class_metrics(
        y_true=[0, 0],
        y_pred=[0, 0],
        class_names=["bag", "car"],
    )
    assert per_class["car"] == {
        "support": 0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_compute_generalization_gap() -> None:
    """Generalization gap should report both absolute and relative drops."""

    gap = compute_generalization_gap(in_dist_accuracy=0.80, ood_accuracy=0.60)
    assert gap["absolute_gap"] == pytest.approx(0.20, rel=1e-5)
    assert gap["relative_drop"] == pytest.approx(0.25, rel=1e-5)


def test_invalid_inputs_raise_value_error() -> None:
    """Mismatched lengths, empty arrays, and out-of-range indices should fail."""

    with pytest.raises(ValueError, match="empty"):
        compute_confusion_matrix([], [], num_classes=2)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_confusion_matrix([0, 1], [0], num_classes=2)

    with pytest.raises(ValueError, match="outside"):
        compute_confusion_matrix([0, 2], [0, 1], num_classes=2)

    with pytest.raises(ValueError, match="num_classes"):
        compute_confusion_matrix([0, 1], [0, 1], num_classes=1)