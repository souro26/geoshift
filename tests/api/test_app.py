"""Tests for the GeoShift FastAPI inference service."""

import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from geoshift.api import create_app
from geoshift.inference import GeoShiftPredictor
from geoshift.models import build_classifier


def _build_test_client() -> TestClient:
    model = build_classifier(
        backbone_name="efficientnet_b0",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )
    predictor = GeoShiftPredictor(
        model=model,
        class_names=["bag", "car", "dog"],
        backbone_name="efficientnet_b0",
        heldout_region="Africa",
        temperature=0.914,
        confidence_threshold=0.5,
        target_coverage=0.90,
        device=torch.device("cpu"),
    )
    return TestClient(create_app(predictor=predictor))


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_model_info_endpoints() -> None:
    """Health, UI, and model-info endpoints should return expected metadata."""

    client = _build_test_client()

    home_res = client.get("/")
    assert home_res.status_code == 200
    assert "GeoShift" in home_res.text

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json() == {"status": "ok", "model_loaded": True}

    info_res = client.get("/model-info")
    assert info_res.status_code == 200
    info = info_res.json()
    assert info["backbone"] == "efficientnet_b0"
    assert info["heldout_region"] == "Africa"
    assert info["num_classes"] == 3


def test_predict_endpoint_with_valid_image() -> None:
    """Valid image upload should return calibrated top-K predictions."""

    client = _build_test_client()
    response = client.post(
        "/predict?top_k=2",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] in {"bag", "car", "dog"}
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["abstain"], bool)
    assert len(payload["top_predictions"]) == 2


def test_predict_endpoint_rejects_invalid_or_empty_files() -> None:
    """Empty or non-image uploads should fail cleanly with HTTP 400."""

    client = _build_test_client()

    empty_res = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert empty_res.status_code == 400

    corrupt_res = client.post(
        "/predict",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert corrupt_res.status_code == 400