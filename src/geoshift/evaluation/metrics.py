"""Classification, per-class, confusion-matrix, and generalization-gap metrics."""

from typing import Sequence

import numpy as np


def _validate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D sequences.")

    if y_true_arr.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}."
        )

    return y_true_arr, y_pred_arr


def compute_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a [num_classes, num_classes] confusion matrix (rows=true, cols=pred)."""

    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2.")

    y_true_arr, y_pred_arr = _validate_predictions(y_true, y_pred)

    if np.any((y_true_arr < 0) | (y_true_arr >= num_classes)):
        raise ValueError("y_true contains class indices outside [0, num_classes).")
    if np.any((y_pred_arr < 0) | (y_pred_arr >= num_classes)):
        raise ValueError("y_pred contains class indices outside [0, num_classes).")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true_arr, y_pred_arr), 1)
    return matrix


def compute_per_class_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """Compute per-class support, accuracy (recall), precision, and F1."""

    num_classes = len(class_names)
    matrix = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)

    results: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(class_names):
        tp = int(matrix[idx, idx])
        support = int(matrix[idx, :].sum())
        predicted_count = int(matrix[:, idx].sum())

        precision = tp / predicted_count if predicted_count > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        results[str(name)] = {
            "support": support,
            "accuracy": round(recall, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    return results


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """Compute overall accuracy, macro/weighted F1, per-class metrics, and confusion matrix."""

    y_true_arr, y_pred_arr = _validate_predictions(y_true, y_pred)
    num_classes = len(class_names)
    matrix = compute_confusion_matrix(y_true_arr, y_pred_arr, num_classes=num_classes)
    per_class = compute_per_class_metrics(y_true_arr, y_pred_arr, class_names=class_names)

    total = int(y_true_arr.size)
    overall_accuracy = float((y_true_arr == y_pred_arr).mean())

    supports = np.array([per_class[str(name)]["support"] for name in class_names], dtype=np.float64)
    precisions = np.array([per_class[str(name)]["precision"] for name in class_names], dtype=np.float64)
    recalls = np.array([per_class[str(name)]["recall"] for name in class_names], dtype=np.float64)
    f1s = np.array([per_class[str(name)]["f1"] for name in class_names], dtype=np.float64)

    present_mask = supports > 0
    macro_precision = float(precisions[present_mask].mean()) if present_mask.any() else 0.0
    macro_recall = float(recalls[present_mask].mean()) if present_mask.any() else 0.0
    macro_f1 = float(f1s[present_mask].mean()) if present_mask.any() else 0.0
    weighted_f1 = float((f1s * supports).sum() / supports.sum()) if supports.sum() > 0 else 0.0

    return {
        "num_samples": total,
        "accuracy": round(overall_accuracy, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def compute_generalization_gap(
    in_dist_accuracy: float,
    ood_accuracy: float,
) -> dict[str, float]:
    """Compute absolute and relative accuracy drop from in-distribution to OOD."""

    absolute_gap = float(in_dist_accuracy) - float(ood_accuracy)
    relative_drop = (
        absolute_gap / float(in_dist_accuracy) if in_dist_accuracy > 0 else 0.0
    )

    return {
        "in_dist_accuracy": round(float(in_dist_accuracy), 6),
        "ood_accuracy": round(float(ood_accuracy), 6),
        "absolute_gap": round(absolute_gap, 6),
        "relative_drop": round(relative_drop, 6),
    }