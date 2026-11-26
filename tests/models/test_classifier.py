"""Tests for GeoClassifier and backbone builders."""

import pytest
import torch
from torch import nn

from geoshift.models import SUPPORTED_BACKBONES, GeoClassifier, build_classifier


@pytest.mark.parametrize("backbone_name", SUPPORTED_BACKBONES)
def test_forward_output_shape_for_all_backbones(backbone_name: str) -> None:
    """Each supported backbone should output [batch_size, num_classes] logits."""

    model = build_classifier(
        backbone_name=backbone_name,
        num_classes=40,
        pretrained=False,
        freeze_backbone=True,
    )
    model.eval()

    inputs = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (2, 40)
    assert torch.isfinite(logits).all()


def test_freeze_backbone_keeps_head_trainable_and_backbone_in_eval() -> None:
    """Freezing backbone should lock backbone weights and preserve eval mode."""

    model = build_classifier(
        backbone_name="resnet50",
        num_classes=10,
        pretrained=False,
        freeze_backbone=True,
    )
    model.train(True)

    counts = model.get_parameter_counts()
    assert model.backbone_frozen is True
    assert counts["frozen"] > 0
    assert counts["trainable"] > 0
    assert counts["total"] == counts["trainable"] + counts["frozen"]

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert model.backbone.training is False
    assert model.classifier.training is True


def test_unfreeze_backbone_enables_full_finetuning() -> None:
    """Unfreezing should make all parameters trainable and allow train mode."""

    model = GeoClassifier(
        backbone_name="resnet50",
        num_classes=10,
        pretrained=False,
        freeze_backbone=True,
    )
    model.unfreeze_backbone()
    model.train(True)

    counts = model.get_parameter_counts()
    assert model.backbone_frozen is False
    assert counts["frozen"] == 0
    assert counts["trainable"] == counts["total"]
    assert model.backbone.training is True


def test_dropout_head_construction() -> None:
    """Positive dropout should wrap the classifier head in a Sequential."""

    model = build_classifier(
        backbone_name="resnet50",
        num_classes=5,
        pretrained=False,
        dropout=0.2,
    )
    assert isinstance(model.classifier, nn.Sequential)
    assert isinstance(model.classifier[0], nn.Dropout)
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].out_features == 5


def test_invalid_arguments_raise_value_error() -> None:
    """Invalid backbone, class count, or dropout should raise ValueError."""

    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_classifier(backbone_name="unknown_net", pretrained=False)

    for bad_classes in (0, 1, True):
        with pytest.raises(ValueError, match="num_classes"):
            build_classifier(
                backbone_name="resnet50",
                num_classes=bad_classes,
                pretrained=False,
            )

    for bad_dropout in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="dropout"):
            build_classifier(
                backbone_name="resnet50",
                num_classes=10,
                pretrained=False,
                dropout=bad_dropout,
            )