"""Train and evaluate frozen-backbone linear probes across geographic holdouts."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from geoshift.evaluation import (
    compute_classification_metrics,
    compute_generalization_gap,
)
from geoshift.training import set_seed


ALL_REGIONS = (
    "Africa",
    "Americas",
    "EastAsia",
    "Europe",
    "SouthEastAsia",
    "WestAsia",
)

DEFAULT_FEATURES_PATH = Path("outputs/features/resnet50_acbae318.pt")
DEFAULT_MANIFEST_DIR = Path("data/manifests/geographic")
DEFAULT_OUTPUT_ROOT = Path("outputs/experiments")
DEFAULT_REVISION = "acbae31823b22f84c3bbb11585d37d50aabaac44"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for frozen-feature linear probe experiments."""

    parser = argparse.ArgumentParser(
        description="Run frozen-backbone linear probes on cached GeoDE features."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to cached feature .pt file.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory containing heldout_<Region>.csv manifests.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        choices=list(ALL_REGIONS),
        help="Single held-out region to run (omit to run all 6 regions).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per fold (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for linear probe training (default: 256).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout before the linear classification layer (default: 0.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="Expected pinned GeoDE revision.",
    )
    return parser.parse_args()


def _predict_logits(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference over a feature DataLoader and return (logits, labels)."""

    head.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            logits = head(batch_features)
            all_logits.append(logits.cpu())
            all_labels.append(batch_labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def run_single_fold(
    region: str,
    features: torch.Tensor,
    index_to_row: dict[int, int],
    manifest_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    device: torch.device,
) -> dict:
    """Train and evaluate a linear probe for one held-out geographic region."""

    set_seed(seed)
    manifest = pd.read_csv(manifest_path)

    class_names = sorted(manifest["label"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    split_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split_name in ("train", "validation", "test"):
        split_df = manifest[manifest["split"] == split_name]
        row_positions = [index_to_row[int(idx)] for idx in split_df["source_index"]]
        split_x = features[row_positions]
        split_y = torch.tensor(
            [class_to_idx[label] for label in split_df["label"]],
            dtype=torch.long,
        )
        split_tensors[split_name] = (split_x, split_y)

    train_loader = DataLoader(
        TensorDataset(*split_tensors["train"]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(*split_tensors["validation"]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(*split_tensors["test"]),
        batch_size=batch_size,
        shuffle=False,
    )

    feature_dim = int(features.shape[1])
    if dropout > 0.0:
        head: nn.Module = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        ).to(device)
    else:
        head = nn.Linear(feature_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = 0
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_y.size(0)
            total_loss += float(loss.detach().item()) * bs
            total_correct += int((logits.detach().argmax(dim=1) == batch_y).sum().item())
            total_samples += bs

        scheduler.step()

        val_logits, val_labels = _predict_logits(head, val_loader, device)
        val_preds = val_logits.argmax(dim=1)
        val_acc = float((val_preds == val_labels).float().mean().item())
        train_acc = total_correct / total_samples

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / total_samples, 6),
                "train_accuracy": round(train_acc, 6),
                "val_accuracy": round(val_acc, 6),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    assert best_state is not None
    head.load_state_dict(best_state)

    val_logits, val_labels = _predict_logits(head, val_loader, device)
    test_logits, test_labels = _predict_logits(head, test_loader, device)

    val_metrics = compute_classification_metrics(
        y_true=val_labels.numpy(),
        y_pred=val_logits.argmax(dim=1).numpy(),
        class_names=class_names,
    )
    test_metrics = compute_classification_metrics(
        y_true=test_labels.numpy(),
        y_pred=test_logits.argmax(dim=1).numpy(),
        class_names=class_names,
    )
    gap_metrics = compute_generalization_gap(
        in_dist_accuracy=val_metrics["accuracy"],
        ood_accuracy=test_metrics["accuracy"],
    )

    fold_dir = output_dir / f"heldout_{region}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "head_state_dict": best_state,
            "class_names": class_names,
            "val_logits": val_logits,
            "val_labels": val_labels,
            "test_logits": test_logits,
            "test_labels": test_labels,
        },
        fold_dir / "logits_and_head.pt",
    )

    fold_report = {
        "heldout_region": region,
        "best_epoch": best_epoch,
        "generalization_gap": gap_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(fold_report, file, indent=2)

    return {
        "region": region,
        "best_epoch": best_epoch,
        "train_samples": int(split_tensors["train"][0].shape[0]),
        "val_samples": int(split_tensors["validation"][0].shape[0]),
        "test_samples": int(split_tensors["test"][0].shape[0]),
        "in_dist_val_accuracy": val_metrics["accuracy"],
        "in_dist_macro_f1": val_metrics["macro_f1"],
        "ood_test_accuracy": test_metrics["accuracy"],
        "ood_macro_f1": test_metrics["macro_f1"],
        "absolute_gap": gap_metrics["absolute_gap"],
        "relative_drop": gap_metrics["relative_drop"],
    }


def main() -> None:
    args = parse_args()

    if not args.features_path.exists():
        raise FileNotFoundError(f"Features file not found: {args.features_path}")

    payload = torch.load(args.features_path, map_location="cpu", weights_only=True)
    if payload["revision"] != args.revision:
        raise ValueError(
            f"Feature revision ({payload['revision']}) does not match "
            f"expected revision ({args.revision})."
        )

    features: torch.Tensor = payload["features"].float()
    source_indices: list[int] = payload["source_indices"].tolist()
    index_to_row = {int(src_idx): row_idx for row_idx, src_idx in enumerate(source_indices)}

    backbone_name = payload["backbone"]
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"linear_probe_{backbone_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regions_to_run = [args.region] if args.region else list(ALL_REGIONS)

    print("GeoShift | Frozen-Backbone Linear Probe")
    print(
        f"Backbone: {backbone_name} | Samples: {features.shape[0]:,} | "
        f"Dim: {features.shape[1]} | Device: {device} | Epochs: {args.epochs}"
    )
    print()

    fold_summaries: list[dict] = []
    for region in regions_to_run:
        manifest_path = args.manifest_dir / f"heldout_{region}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        summary = run_single_fold(
            region=region,
            features=features,
            index_to_row=index_to_row,
            manifest_path=manifest_path,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=args.seed,
            device=device,
        )
        fold_summaries.append(summary)
        print(
            f"{region:14} | "
            f"In-Dist Val: {summary['in_dist_val_accuracy']:.4f} | "
            f"OOD Test: {summary['ood_test_accuracy']:.4f} | "
            f"Gap: {summary['absolute_gap']:+.4f} | "
            f"OOD Macro-F1: {summary['ood_macro_f1']:.4f}"
        )

    mean_val = float(np.mean([s["in_dist_val_accuracy"] for s in fold_summaries]))
    mean_ood = float(np.mean([s["ood_test_accuracy"] for s in fold_summaries]))
    mean_gap = float(np.mean([s["absolute_gap"] for s in fold_summaries]))

    aggregate = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": backbone_name,
        "dataset_revision": args.revision,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "mean_in_dist_val_accuracy": round(mean_val, 6),
        "mean_ood_test_accuracy": round(mean_ood, 6),
        "mean_absolute_gap": round(mean_gap, 6),
        "folds": fold_summaries,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=2)

    print("\nSUMMARY")
    print(f"Mean In-Dist Val Acc: {mean_val:.4f}")
    print(f"Mean OOD Test Acc:    {mean_ood:.4f}")
    print(f"Mean Gen Gap:         {mean_gap:+.4f}")
    print(f"Saved results to:     {summary_path}")


if __name__ == "__main__":
    main()