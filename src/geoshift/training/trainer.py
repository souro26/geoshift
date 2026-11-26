"""PyTorch training and validation loop for GeoShift classifiers."""

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    """Write JSON atomically via a temporary file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _save_checkpoint_atomic(payload: dict[str, Any], output_path: Path) -> None:
    """Write a PyTorch checkpoint atomically via a temporary file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Train model for one epoch (or up to max_batches) and return metrics."""

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1 when provided.")

    model.train()
    amp_enabled = use_amp and device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        preds = logits.detach().argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training loader yielded zero samples.")

    elapsed = time.perf_counter() - start_time
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_samples": total_samples,
        "elapsed_seconds": round(elapsed, 3),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Evaluate model on a DataLoader and return loss and accuracy."""

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1 when provided.")

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Evaluation loader yielded zero samples.")

    elapsed = time.perf_counter() - start_time
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_samples": total_samples,
        "elapsed_seconds": round(elapsed, 3),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    use_amp: bool = True,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Run multi-epoch training, validation, checkpointing, and history logging."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    best_val_accuracy = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"
    history_path = output_dir / "training_history.json"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=amp_enabled,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=amp_enabled,
            max_batches=max_val_batches,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "train_samples": train_metrics["num_samples"],
            "train_seconds": train_metrics["elapsed_seconds"],
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_samples": val_metrics["num_samples"],
            "val_seconds": val_metrics["elapsed_seconds"],
        }
        history.append(epoch_record)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "backbone_name": getattr(model, "backbone_name", None),
            "num_classes": getattr(model, "num_classes", None),
        }

        _save_checkpoint_atomic(checkpoint, last_checkpoint_path)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            _save_checkpoint_atomic(checkpoint, best_checkpoint_path)

        _save_json_atomic(
            {
                "best_epoch": best_epoch,
                "best_val_accuracy": round(best_val_accuracy, 6),
                "history": history,
            },
            history_path,
        )

    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "history_path": str(history_path),
    }