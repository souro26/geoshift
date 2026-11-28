"""Train and evaluate a frozen or fine-tuned GeoShift baseline on one held-out region."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from geoshift.config import load_config, project_path
from geoshift.data.dataset import GeoDEIterableDataset
from geoshift.data.transforms import build_eval_transform, build_train_transform
from geoshift.models import SUPPORTED_BACKBONES, build_classifier
from geoshift.training import evaluate, fit, set_seed

_DATA_CONFIG = load_config("data.yml")
_MANIFEST_SPLITS = _DATA_CONFIG["splitting"]["splits"]
_PATH_CONFIG = load_config("paths.yml")
_TRAINING_CONFIG = load_config("training.yml")
_MODEL_CONFIG = load_config("models.yml")
_BASELINE_CONFIG = _TRAINING_CONFIG["baseline"]
_DATALOADER_CONFIG = _TRAINING_CONFIG["dataloader"]
DEFAULT_MANIFEST_DIR = project_path(_PATH_CONFIG["data"]["manifests"])
DEFAULT_OUTPUT_ROOT = project_path(_PATH_CONFIG["outputs"]["runs"])
DEFAULT_REVISION = _DATA_CONFIG["dataset"]["revision"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for baseline training."""

    parser = argparse.ArgumentParser(
        description="Train a GeoShift baseline on one geographic holdout manifest."
    )
    parser.add_argument(
        "--region",
        type=str,
        default=_BASELINE_CONFIG["region"],
        choices=_DATA_CONFIG["dataset"]["regions"],
        help="Held-out test region.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=_MODEL_CONFIG["default_backbone"],
        choices=list(SUPPORTED_BACKBONES),
        help="Pretrained torchvision backbone.",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Fine-tune the entire backbone instead of freezing it.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_BASELINE_CONFIG["epochs"],
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_BASELINE_CONFIG["batch_size"],
        help="Batch size for DataLoaders.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=_BASELINE_CONFIG["learning_rate"],
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=_BASELINE_CONFIG["weight_decay"],
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=_BASELINE_CONFIG["dropout"],
        help="Dropout before classifier linear layer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_BASELINE_CONFIG["seed"],
        help="Random seed.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision (AMP) on CUDA.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=_BASELINE_CONFIG["max_train_batches"],
        help="Optional cap on training batches per epoch (for fast sanity runs).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=_BASELINE_CONFIG["max_val_batches"],
        help="Optional cap on validation batches per epoch.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=_BASELINE_CONFIG["max_test_batches"],
        help="Optional cap on held-out test batches.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory containing heldout_<Region>.csv manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit run output directory.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="Pinned GeoDE Hugging Face revision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest_path = args.manifest_dir / f"heldout_{args.region}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    mode_tag = "finetune" if args.unfreeze else "frozen"
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{args.backbone}_{mode_tag}_heldout_{args.region}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = _BASELINE_CONFIG["use_amp"] and not args.no_amp
    use_amp = use_amp and device.type == "cuda"

    train_dataset = GeoDEIterableDataset(
        manifest=manifest,
        split=_MANIFEST_SPLITS["train"],
        revision=args.revision,
        transform=build_train_transform(),
    )
    val_dataset = GeoDEIterableDataset(
        manifest=manifest,
        split=_MANIFEST_SPLITS["validation"],
        revision=args.revision,
        transform=build_eval_transform(),
    )
    test_dataset = GeoDEIterableDataset(
        manifest=manifest,
        split=_MANIFEST_SPLITS["test"],
        revision=args.revision,
        transform=build_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=_DATALOADER_CONFIG["num_workers"],
        pin_memory=(
            _DATALOADER_CONFIG["pin_memory_on_cuda"] and device.type == "cuda"
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=_DATALOADER_CONFIG["num_workers"],
        pin_memory=(
            _DATALOADER_CONFIG["pin_memory_on_cuda"] and device.type == "cuda"
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=_DATALOADER_CONFIG["num_workers"],
        pin_memory=(
            _DATALOADER_CONFIG["pin_memory_on_cuda"] and device.type == "cuda"
        ),
    )

    num_classes = len(train_dataset.class_to_idx)
    model = build_classifier(
        backbone_name=args.backbone,
        num_classes=num_classes,
        pretrained=_MODEL_CONFIG["pretrained"],
        freeze_backbone=not args.unfreeze,
        dropout=args.dropout,
    ).to(device)

    param_counts = model.get_parameter_counts()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print("GeoShift | Baseline Training")
    print(
        f"Held-out Region: {args.region} | Backbone: {args.backbone} ({mode_tag}) | "
        f"Device: {device} | AMP: {use_amp}"
    )
    print(
        f"Parameters: {param_counts['trainable']:,} trainable / "
        f"{param_counts['total']:,} total"
    )

    fit_result = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        output_dir=output_dir,
        use_amp=use_amp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    best_checkpoint = torch.load(
        fit_result["best_checkpoint_path"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        max_batches=args.max_test_batches,
    )

    in_dist_acc = float(fit_result["best_val_accuracy"])
    ood_acc = float(test_metrics["accuracy"])
    gen_gap = in_dist_acc - ood_acc

    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": args.revision,
        "heldout_region": args.region,
        "backbone": args.backbone,
        "freeze_backbone": not args.unfreeze,
        "device": str(device),
        "use_amp": use_amp,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_test_batches": args.max_test_batches,
        "parameter_counts": param_counts,
        "class_to_idx": train_dataset.class_to_idx,
        "best_epoch": fit_result["best_epoch"],
        "in_dist_val_accuracy": round(in_dist_acc, 6),
        "ood_test_accuracy": round(ood_acc, 6),
        "generalization_gap": round(gen_gap, 6),
        "ood_test_loss": round(float(test_metrics["loss"]), 6),
        "ood_test_samples": int(test_metrics["num_samples"]),
        "best_checkpoint_path": fit_result["best_checkpoint_path"],
    }

    summary_path = output_dir / "baseline_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nBASELINE EVALUATION COMPLETE")
    print(f"Best Epoch:              {fit_result['best_epoch']}")
    print(f"In-Dist Val Accuracy:    {in_dist_acc:.4f}")
    print(f"Held-Out Test Accuracy:  {ood_acc:.4f}")
    print(f"Generalization Gap:      {gen_gap:+.4f}")
    print(f"Summary saved to:        {summary_path}")


if __name__ == "__main__":
    main()