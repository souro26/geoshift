"""Extract and cache frozen-backbone image features from GeoDE in a single pass."""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from geoshift.data.geode import GeoDEStream
from geoshift.data.transforms import build_eval_transform
from geoshift.models import SUPPORTED_BACKBONES, build_classifier


DEFAULT_REVISION = "acbae31823b22f84c3bbb11585d37d50aabaac44"
DEFAULT_OUTPUT_DIR = Path("outputs/features")


def _save_pt_atomic(payload: dict, output_path: Path) -> None:
    """Save a PyTorch payload atomically via a temporary file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for feature extraction."""

    parser = argparse.ArgumentParser(
        description="Extract frozen-backbone embeddings from MLap/GeoDE."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=list(SUPPORTED_BACKBONES),
        help="Pretrained backbone for feature extraction (default: resnet50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for GPU feature extraction (default: 64).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2000,
        help="Save a resumable checkpoint every N samples (default: 2000).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on samples to extract (useful for quick tests).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="Pinned GeoDE Hugging Face revision.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for extracted feature files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing final feature file and checkpoint.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision (AMP) on CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1.")

    rev_prefix = args.revision[:8]
    final_path = args.output_dir / f"{args.backbone}_{rev_prefix}.pt"
    ckpt_path = args.output_dir / f"{args.backbone}_{rev_prefix}_ckpt.pt"

    if final_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Feature file already exists: {final_path}. "
            "Pass --overwrite to regenerate."
        )

    if args.overwrite:
        ckpt_path.unlink(missing_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    transform = build_eval_transform()
    model = build_classifier(
        backbone_name=args.backbone,
        num_classes=40,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    model.eval()

    feature_chunks: list[torch.Tensor] = []
    source_indices: list[int] = []
    labels: list[str] = []
    regions: list[str] = []
    countries: list[str] = []
    start_index = 0

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        feature_chunks.append(ckpt["features"])
        source_indices = ckpt["source_indices"].tolist()
        labels = list(ckpt["labels"])
        regions = list(ckpt["regions"])
        countries = list(ckpt["countries"])
        start_index = len(source_indices)
        print(f"Resuming from checkpoint: {start_index:,} samples already extracted.")

    print("GeoShift | Single-Pass Feature Extraction")
    print(
        f"Backbone: {args.backbone} | Revision: {args.revision} | "
        f"Device: {device} | AMP: {use_amp}"
    )

    stream = GeoDEStream(
        split="train",
        revision=args.revision,
        metadata_only=False,
    )

    batch_tensors: list[torch.Tensor] = []
    batch_indices: list[int] = []
    batch_labels: list[str] = []
    batch_regions: list[str] = []
    batch_countries: list[str] = []

    processed_since_ckpt = 0
    total_start = time.perf_counter()

    def flush_batch() -> None:
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                feats = model.backbone(images)
        feature_chunks.append(feats.float().cpu())
        source_indices.extend(batch_indices)
        labels.extend(batch_labels)
        regions.extend(batch_regions)
        countries.extend(batch_countries)

        batch_tensors.clear()
        batch_indices.clear()
        batch_labels.clear()
        batch_regions.clear()
        batch_countries.clear()

    def build_payload() -> dict:
        return {
            "backbone": args.backbone,
            "revision": args.revision,
            "feature_dim": model.feature_dim,
            "features": torch.cat(feature_chunks, dim=0),
            "source_indices": torch.tensor(source_indices, dtype=torch.long),
            "labels": labels,
            "regions": regions,
            "countries": countries,
        }

    for sample in stream:
        if sample.source_index < start_index:
            continue

        if args.max_samples is not None and len(source_indices) + len(batch_tensors) >= args.max_samples:
            break

        if not isinstance(sample.image, Image.Image):
            raise TypeError(
                f"Expected PIL Image at source_index={sample.source_index}."
            )

        rgb_image = sample.image.convert("RGB")
        batch_tensors.append(transform(rgb_image))
        batch_indices.append(int(sample.source_index))
        batch_labels.append(sample.label)
        batch_regions.append(sample.region)
        batch_countries.append(sample.country)

        if len(batch_tensors) >= args.batch_size:
            flush_batch()
            processed_since_ckpt += args.batch_size

            if processed_since_ckpt >= args.checkpoint_interval:
                _save_pt_atomic(build_payload(), ckpt_path)
                processed_since_ckpt = 0
                elapsed = time.perf_counter() - total_start
                rate = (len(source_indices) - start_index) / elapsed if elapsed > 0 else 0.0
                print(
                    f"Checkpoint saved | Total: {len(source_indices):,} samples | "
                    f"Rate: {rate:.1f} samples/sec"
                )

    flush_batch()

    if not source_indices:
        raise RuntimeError("No features were extracted.")

    final_payload = build_payload()
    _save_pt_atomic(final_payload, final_path)
    ckpt_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - total_start
    print("\nFEATURE EXTRACTION COMPLETE")
    print(f"Samples extracted: {len(source_indices):,}")
    print(f"Feature shape:     {tuple(final_payload['features'].shape)}")
    print(f"Elapsed seconds:   {elapsed:.1f}s")
    print(f"Saved to:          {final_path}")


if __name__ == "__main__":
    main()