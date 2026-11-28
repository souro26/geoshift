"""Calibration (temperature scaling, ECE, Brier, NLL) and selective prediction."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from geoshift.config import load_config

_CALIBRATION_CONFIG = load_config("evaluation.yml")["calibration"]
DEFAULT_COVERAGES = tuple(_CALIBRATION_CONFIG["coverages"])
DEFAULT_N_BINS = _CALIBRATION_CONFIG["n_bins"]
DEFAULT_TEMPERATURE_MAX_ITERATIONS = _CALIBRATION_CONFIG[
    "temperature_max_iterations"
]


def _validate_probs_and_labels(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probs_arr = np.asarray(probs, dtype=np.float64)
    y_true_arr = np.asarray(y_true, dtype=np.int64)

    if probs_arr.ndim != 2:
        raise ValueError("probs must be a 2D array of shape [N, num_classes].")
    if y_true_arr.ndim != 1:
        raise ValueError("y_true must be a 1D array of shape [N].")
    if probs_arr.shape[0] == 0:
        raise ValueError("probs and y_true cannot be empty.")
    if probs_arr.shape[0] != y_true_arr.shape[0]:
        raise ValueError(
            f"Sample count mismatch: probs {probs_arr.shape[0]} vs y_true {y_true_arr.shape[0]}."
        )
    if np.any((y_true_arr < 0) | (y_true_arr >= probs_arr.shape[1])):
        raise ValueError("y_true contains class indices outside [0, num_classes).")

    return probs_arr, y_true_arr


def compute_expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Compute Expected Calibration Error (ECE) over equal-width confidence bins."""

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be an integer >= 1.")

    probs_arr, y_true_arr = _validate_probs_and_labels(probs, y_true)
    confidences = probs_arr.max(axis=1)
    predictions = probs_arr.argmax(axis=1)
    accuracies = (predictions == y_true_arr).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = float(len(y_true_arr))

    for idx in range(n_bins):
        low = bin_boundaries[idx]
        high = bin_boundaries[idx + 1]
        if idx == 0:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences > low) & (confidences <= high)

        bin_count = int(in_bin.sum())
        if bin_count > 0:
            avg_conf = float(confidences[in_bin].mean())
            avg_acc = float(accuracies[in_bin].mean())
            ece += (bin_count / n_samples) * abs(avg_acc - avg_conf)

    return float(ece)


def compute_brier_score(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute multi-class Brier score (mean squared probability error)."""

    probs_arr, y_true_arr = _validate_probs_and_labels(probs, y_true)
    one_hot = np.zeros_like(probs_arr)
    one_hot[np.arange(len(y_true_arr)), y_true_arr] = 1.0
    return float(np.mean(np.sum((probs_arr - one_hot) ** 2, axis=1)))


def compute_negative_log_likelihood(
    logits: torch.Tensor,
    y_true: torch.Tensor,
) -> float:
    """Compute mean cross-entropy (negative log-likelihood) from logits."""

    if logits.ndim != 2 or y_true.ndim != 1:
        raise ValueError("logits must be 2D and y_true must be 1D.")
    if logits.shape[0] == 0 or logits.shape[0] != y_true.shape[0]:
        raise ValueError("logits and y_true must be non-empty with matching lengths.")

    return float(F.cross_entropy(logits.float(), y_true.long()).item())


def fit_temperature(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    max_iter: int = DEFAULT_TEMPERATURE_MAX_ITERATIONS,
) -> float:
    """Fit a positive scalar temperature on in-distribution validation logits."""

    if val_logits.ndim != 2 or val_labels.ndim != 1:
        raise ValueError("val_logits must be 2D and val_labels must be 1D.")
    if val_logits.shape[0] == 0 or val_logits.shape[0] != val_labels.shape[0]:
        raise ValueError("val_logits and val_labels must be non-empty with matching lengths.")

    logits = val_logits.detach().float()
    labels = val_labels.detach().long()

    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted_temp = float(torch.exp(log_temp).clamp(min=1e-3, max=100.0).item())
    return fitted_temp


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by a positive temperature."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    return logits.float() / float(temperature)


def compute_selective_prediction_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> list[dict[str, float | int]]:
    """Compute accuracy at decreasing confidence-ranked coverage levels."""

    probs_arr, y_true_arr = _validate_probs_and_labels(probs, y_true)
    confidences = probs_arr.max(axis=1)
    predictions = probs_arr.argmax(axis=1)
    correct = (predictions == y_true_arr).astype(np.float64)

    order = np.argsort(-confidences, kind="mergesort")
    sorted_confs = confidences[order]
    sorted_correct = correct[order]
    n_samples = len(y_true_arr)

    curve: list[dict[str, float | int]] = []
    for cov in coverages:
        if not (0.0 < cov <= 1.0):
            raise ValueError(f"Coverage values must be in (0, 1], got {cov}.")

        k = max(1, min(n_samples, int(round(cov * n_samples))))
        acc = float(sorted_correct[:k].mean())
        threshold = float(sorted_confs[k - 1])
        curve.append(
            {
                "target_coverage": round(float(cov), 4),
                "actual_coverage": round(k / n_samples, 6),
                "num_accepted": k,
                "confidence_threshold": round(threshold, 6),
                "accuracy": round(acc, 6),
            }
        )

    return curve