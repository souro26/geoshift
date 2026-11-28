"""Tests for GeoShiftPredictor inference, calibration, and abstention."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from geoshift.inference import GeoShiftPredictor


def _create_mock_fold_dir(
    fold_dir: Path,
    threshold: float = 0.85,
    temperature: float = 0.92,
) -> Path:
    fold_dir.mkdir(parents=True, exist_ok=True)
    head = nn.Linear(1280, 3)

    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "class_names": ["bag", "car", "dog"],
        },
        fold_dir / "logits_and_head.pt",
    )

    reliability = {
        "temperature": temperature,
        "ood_selective_prediction": [
            {
                "target_coverage": 0.90,
                "confidence_threshold": threshold,
                "accuracy": 0.94,
            }
        ],
    }
    (fold_dir / "reliability.json").write_text(
        json.dumps(reliability),
        encoding="utf-8",
    )
    return fold_dir


def test_from_fold_dir_and_predict_image(tmp_path: Path) -> None:
    """Predictor should load artifacts, apply temperature, and return top-K results."""

    fold_dir = _create_mock_fold_dir(
        tmp_path / "heldout_Africa",
        threshold=0.9999,
        temperature=0.95,
    )
    predictor = GeoShiftPredictor.from_fold_dir(
        fold_dir=fold_dir,
        backbone_name="efficientnet_b0",
        target_coverage=0.90,
        pretrained_backbone=False,
        device=torch.device("cpu"),
    )

    image = Image.new("RGB", (128, 128), color=(120, 40, 200))
    result = predictor.predict_image(image, top_k=2)

    assert result["predicted_class"] in {"bag", "car", "dog"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["abstain"] is True
    assert result["temperature"] == pytest.approx(0.95)
    assert result["confidence_threshold"] == pytest.approx(0.9999)
    assert len(result["top_predictions"]) == 2
    assert (
        result["top_predictions"][0]["probability"]
        >= result["top_predictions"][1]["probability"]
    )

    info = predictor.get_model_info()
    assert info["backbone"] == "efficientnet_b0"
    assert info["heldout_region"] == "Africa"
    assert info["num_classes"] == 3


def test_abstention_disabled_when_threshold_is_zero(tmp_path: Path) -> None:
    """Predictions above threshold should not be flagged for abstention."""

    fold_dir = _create_mock_fold_dir(tmp_path / "heldout_Europe", threshold=0.0)
    predictor = GeoShiftPredictor.from_fold_dir(
        fold_dir=fold_dir,
        backbone_name="efficientnet_b0",
        pretrained_backbone=False,
        device=torch.device("cpu"),
    )

    result = predictor.predict_image(Image.new("L", (64, 64), color=128))
    assert result["abstain"] is False
    assert len(result["top_predictions"]) == 3


def test_predictor_invalid_inputs_and_missing_artifact(tmp_path: Path) -> None:
    """Missing artifacts, non-PIL images, and invalid top_k should raise errors."""

    with pytest.raises(FileNotFoundError, match="Missing fold artifact"):
        GeoShiftPredictor.from_fold_dir(tmp_path / "missing_fold")

    fold_dir = _create_mock_fold_dir(tmp_path / "heldout_Africa")
    predictor = GeoShiftPredictor.from_fold_dir(
        fold_dir=fold_dir,
        backbone_name="efficientnet_b0",
        pretrained_backbone=False,
        device=torch.device("cpu"),
    )

    with pytest.raises(TypeError, match="PIL.Image.Image"):
        predictor.predict_image("not_an_image")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="top_k"):
        predictor.predict_image(Image.new("RGB", (32, 32)), top_k=0)