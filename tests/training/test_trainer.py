"""Tests for the GeoShift training and validation loop."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geoshift.models import build_classifier
from geoshift.training import evaluate, fit, set_seed, train_one_epoch


class SyntheticImageDictDataset(Dataset):
    """Small deterministic dataset matching GeoDEIterableDataset's dict output."""

    def __init__(self, num_samples: int = 12, num_classes: int = 3, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.arange(num_samples) % num_classes
        self.images = torch.randn(num_samples, 3, 224, 224, generator=generator)
        for idx in range(num_samples):
            self.images[idx, :, :32, :32] += float(self.labels[idx]) * 2.0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.images[index],
            "label": int(self.labels[index].item()),
        }


def test_fit_creates_checkpoints_and_history(tmp_path: Path) -> None:
    """fit() should train, validate, and persist checkpoints and history JSON."""

    set_seed(42)
    train_loader = DataLoader(
        SyntheticImageDictDataset(num_samples=8, num_classes=3, seed=1),
        batch_size=4,
        shuffle=False,
    )
    val_loader = DataLoader(
        SyntheticImageDictDataset(num_samples=4, num_classes=3, seed=2),
        batch_size=4,
        shuffle=False,
    )

    model = build_classifier(
        backbone_name="efficientnet_b0",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,
    )
    criterion = nn.CrossEntropyLoss()

    result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        use_amp=False,
    )

    best_ckpt = Path(result["best_checkpoint_path"])
    last_ckpt = Path(result["last_checkpoint_path"])
    history_file = Path(result["history_path"])

    assert best_ckpt.exists()
    assert last_ckpt.exists()
    assert history_file.exists()
    assert len(result["history"]) == 2
    assert result["best_epoch"] in (1, 2)

    history_payload = json.loads(history_file.read_text(encoding="utf-8"))
    assert history_payload["best_epoch"] == result["best_epoch"]
    assert len(history_payload["history"]) == 2

    loaded = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    assert loaded["backbone_name"] == "efficientnet_b0"
    assert loaded["num_classes"] == 3
    assert "model_state_dict" in loaded


def test_max_batches_limits_samples_processed(tmp_path: Path) -> None:
    """max_batches should cap the number of batches per epoch."""

    loader = DataLoader(
        SyntheticImageDictDataset(num_samples=10, num_classes=3, seed=3),
        batch_size=2,
        shuffle=False,
    )
    model = build_classifier(
        backbone_name="efficientnet_b0",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,
    )
    criterion = nn.CrossEntropyLoss()

    train_metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        max_batches=2,
    )
    val_metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        max_batches=3,
    )

    assert train_metrics["num_samples"] == 4
    assert val_metrics["num_samples"] == 6


def test_invalid_arguments_and_empty_loader_raise_value_error(tmp_path: Path) -> None:
    """Invalid epochs, max_batches, or empty loaders should raise ValueError."""

    empty_loader = DataLoader(
        SyntheticImageDictDataset(num_samples=0, num_classes=3),
        batch_size=2,
    )
    non_empty_loader = DataLoader(
        SyntheticImageDictDataset(num_samples=4, num_classes=3),
        batch_size=2,
    )
    model = build_classifier(
        backbone_name="efficientnet_b0",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,
    )
    criterion = nn.CrossEntropyLoss()

    with pytest.raises(ValueError, match="epochs"):
        fit(
            model=model,
            train_loader=non_empty_loader,
            val_loader=non_empty_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
            epochs=0,
            output_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="max_batches"):
        train_one_epoch(
            model=model,
            loader=non_empty_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
            max_batches=0,
        )

    with pytest.raises(ValueError, match="zero samples"):
        evaluate(
            model=model,
            loader=empty_loader,
            criterion=criterion,
            device=torch.device("cpu"),
        )