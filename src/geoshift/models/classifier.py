from typing import Callable

import torch
from torch import nn
from torchvision import models

from geoshift.config import load_config

_MODEL_CONFIG = load_config("models.yml")
_DATA_CONFIG = load_config("data.yml")


def _build_resnet50(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    backbone = models.resnet50(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, in_features


def _build_convnext_tiny(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    backbone = models.convnext_tiny(weights=weights)
    in_features = backbone.classifier[2].in_features
    backbone.classifier[2] = nn.Identity()
    return backbone, in_features


def _build_efficientnet_b0(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    backbone = models.efficientnet_b0(weights=weights)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return backbone, in_features


def _build_swin_t(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.Swin_T_Weights.DEFAULT if pretrained else None
    backbone = models.swin_t(weights=weights)
    in_features = backbone.head.in_features
    backbone.head = nn.Identity()
    return backbone, in_features


_BACKBONE_BUILDERS: dict[str, Callable[[bool], tuple[nn.Module, int]]] = {
    "resnet50": _build_resnet50,
    "convnext_tiny": _build_convnext_tiny,
    "efficientnet_b0": _build_efficientnet_b0,
    "swin_t": _build_swin_t,
}

SUPPORTED_BACKBONES = tuple(_MODEL_CONFIG["supported_backbones"])
_unknown_backbones = set(SUPPORTED_BACKBONES) - set(_BACKBONE_BUILDERS)
if _unknown_backbones:
    raise ValueError(
        "Configured backbones have no implementation: "
        f"{sorted(_unknown_backbones)}"
    )


class GeoClassifier(nn.Module):
    """Image classifier wrapping a torchvision backbone and linear head."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = _MODEL_CONFIG["pretrained"],
        freeze_backbone: bool = _MODEL_CONFIG["freeze_backbone"],
        dropout: float = _MODEL_CONFIG["dropout"],
    ) -> None:
        super().__init__()

        if backbone_name not in _BACKBONE_BUILDERS:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from: {SUPPORTED_BACKBONES}"
            )

        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")

        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = dropout

        self.backbone, self.feature_dim = _BACKBONE_BUILDERS[backbone_name](
            pretrained
        )

        if dropout > 0.0:
            self.classifier: nn.Module = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.feature_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    @property
    def backbone_frozen(self) -> bool:
        """Return whether the feature-extractor backbone is frozen."""

        return self._backbone_frozen

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters and keep normalization layers in eval mode."""

        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for end-to-end fine-tuning."""

        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode: bool = True) -> "GeoClassifier":
        """Set training mode while keeping frozen backbone layers in eval mode."""

        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of [B, 3, H, W] images."""

        features = self.backbone(images)
        return self.classifier(features)

    def get_parameter_counts(self) -> dict[str, int]:
        """Return total, trainable, and frozen parameter counts."""

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


def build_classifier(
    backbone_name: str = _MODEL_CONFIG["default_backbone"],
    num_classes: int = _DATA_CONFIG["dataset"]["num_classes"],
    pretrained: bool = _MODEL_CONFIG["pretrained"],
    freeze_backbone: bool = _MODEL_CONFIG["freeze_backbone"],
    dropout: float = _MODEL_CONFIG["dropout"],
) -> GeoClassifier:
    """Construct a GeoClassifier for GeoDE experiments."""

    return GeoClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )