"""Model loading and single-image inference with calibration and abstention."""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from geoshift.config import load_config
from geoshift.data.transforms import build_eval_transform
from geoshift.models import build_classifier

_INFERENCE_CONFIG = load_config("inference.yml")
_MODEL_CONFIG = load_config("models.yml")


def _extract_threshold_for_coverage(
    reliability_payload: dict[str, Any],
    target_coverage: float,
) -> float | None:
    """Look up the confidence threshold for the closest target coverage."""

    curve = reliability_payload.get("ood_selective_prediction", [])
    for point in curve:
        if abs(float(point["target_coverage"]) - target_coverage) < 1e-6:
            return float(point["confidence_threshold"])
    return None


class GeoShiftPredictor:
    """Predict GeoDE object classes with calibrated confidence and abstention."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        backbone_name: str,
        heldout_region: str | None = None,
        temperature: float = _INFERENCE_CONFIG["fallback_temperature"],
        confidence_threshold: float | None = None,
        target_coverage: float | None = None,
        device: torch.device | None = None,
    ) -> None:
        if not class_names:
            raise ValueError("class_names cannot be empty.")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = list(class_names)
        self.backbone_name = backbone_name
        self.heldout_region = heldout_region
        self.temperature = float(temperature)
        self.confidence_threshold = (
            float(confidence_threshold) if confidence_threshold is not None else None
        )
        self.target_coverage = target_coverage
        self.transform = build_eval_transform()

    @classmethod
    def from_fold_dir(
        cls,
        fold_dir: str | Path,
        backbone_name: str = _MODEL_CONFIG["default_backbone"],
        target_coverage: float = _INFERENCE_CONFIG["target_coverage"],
        pretrained_backbone: bool = _MODEL_CONFIG["pretrained"],
        device: torch.device | None = None,
    ) -> "GeoShiftPredictor":
        """Load a predictor from a saved linear-probe fold directory."""

        fold_path = Path(fold_dir)
        artifact_path = fold_path / "logits_and_head.pt"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing fold artifact: {artifact_path}")

        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
        head_state = payload["head_state_dict"]
        class_names = list(payload["class_names"])

        has_dropout_prefix = any(k.startswith("1.") for k in head_state.keys())
        model = build_classifier(
            backbone_name=backbone_name,
            num_classes=len(class_names),
            pretrained=pretrained_backbone,
            freeze_backbone=_MODEL_CONFIG["freeze_backbone"],
            dropout=(
                _MODEL_CONFIG["inference_head_dropout"]
                if has_dropout_prefix
                else _MODEL_CONFIG["dropout"]
            ),
        )
        model.classifier.load_state_dict(head_state)

        temperature = _INFERENCE_CONFIG["fallback_temperature"]
        confidence_threshold = None
        reliability_path = fold_path / "reliability.json"
        if reliability_path.exists():
            reliability = json.loads(reliability_path.read_text(encoding="utf-8"))
            temperature = float(reliability.get("temperature", 1.0))
            confidence_threshold = _extract_threshold_for_coverage(
                reliability,
                target_coverage=target_coverage,
            )

        heldout_region = fold_path.name.replace("heldout_", "")

        return cls(
            model=model,
            class_names=class_names,
            backbone_name=backbone_name,
            heldout_region=heldout_region,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            target_coverage=target_coverage,
            device=device,
        )

    def predict_image(
        self,
        image: Image.Image,
        top_k: int = _INFERENCE_CONFIG["top_k"],
    ) -> dict[str, Any]:
        """Run calibrated inference on a PIL image and flag low-confidence inputs."""

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image instance.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor).float()
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1).squeeze(0).cpu()

        k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probs, k=k)

        top_predictions = [
            {
                "label": self.class_names[int(idx)],
                "probability": round(float(prob), 6),
            }
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
        ]

        predicted_class = top_predictions[0]["label"]
        confidence = top_predictions[0]["probability"]
        abstain = (
            confidence < self.confidence_threshold
            if self.confidence_threshold is not None
            else False
        )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "abstain": abstain,
            "confidence_threshold": self.confidence_threshold,
            "temperature": self.temperature,
            "top_predictions": top_predictions,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return model, calibration, and selective-prediction metadata."""

        return {
            "backbone": self.backbone_name,
            "heldout_region": self.heldout_region,
            "num_classes": len(self.class_names),
            "temperature": self.temperature,
            "target_coverage": self.target_coverage,
            "confidence_threshold": self.confidence_threshold,
            "device": str(self.device),
        }