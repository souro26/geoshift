"""Tests for temperature scaling, calibration metrics, and selective prediction."""

import numpy as np
import pytest
import torch

from geoshift.evaluation import (
    apply_temperature,
    compute_brier_score,
    compute_expected_calibration_error,
    compute_negative_log_likelihood,
    compute_selective_prediction_curve,
    fit_temperature,
)


def test_perfect_calibration_ece_and_brier_are_zero() -> None:
    """Exact one-hot confident predictions should yield 0 ECE and 0 Brier score."""

    probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y_true = np.array([0, 1], dtype=np.int64)

    assert compute_expected_calibration_error(probs, y_true, n_bins=10) == pytest.approx(0.0)
    assert compute_brier_score(probs, y_true) == pytest.approx(0.0)


def test_temperature_scaling_reduces_overconfident_nll() -> None:
    """Fitting temperature on overconfident logits should increase T and lower NLL."""

    logits = torch.tensor(
        [
            [6.0, 0.0],
            [6.0, 0.0],
            [0.0, 6.0],
            [6.0, 0.0],  # wrong with very high confidence
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    raw_nll = compute_negative_log_likelihood(logits, labels)
    temperature = fit_temperature(logits, labels)
    scaled_logits = apply_temperature(logits, temperature)
    scaled_nll = compute_negative_log_likelihood(scaled_logits, labels)

    assert temperature > 1.0
    assert scaled_nll < raw_nll


def test_selective_prediction_curve_improves_accuracy_at_lower_coverage() -> None:
    """Rejecting low-confidence errors should raise accuracy at lower coverage."""

    probs = np.array(
        [
            [0.95, 0.05],  # correct, highest conf
            [0.10, 0.90],  # correct, high conf
            [0.60, 0.40],  # wrong, low conf
            [0.55, 0.45],  # wrong, lowest conf
        ],
        dtype=np.float64,
    )
    y_true = np.array([0, 1, 1, 1], dtype=np.int64)

    curve = compute_selective_prediction_curve(
        probs,
        y_true,
        coverages=(1.0, 0.5),
    )

    assert len(curve) == 2
    assert curve[0]["target_coverage"] == 1.0
    assert curve[0]["accuracy"] == pytest.approx(0.5)
    assert curve[1]["target_coverage"] == 0.5
    assert curve[1]["accuracy"] == pytest.approx(1.0)


def test_invalid_calibration_inputs_raise_value_error() -> None:
    """Invalid n_bins, temperature, coverage, or shapes should raise ValueError."""

    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    y_true = np.array([0, 1])

    with pytest.raises(ValueError, match="n_bins"):
        compute_expected_calibration_error(probs, y_true, n_bins=0)

    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(torch.zeros((2, 2)), temperature=0.0)

    with pytest.raises(ValueError, match="Coverage"):
        compute_selective_prediction_curve(probs, y_true, coverages=(1.5,))