"""Extract and cache frozen-backbone image features from GeoDE with O(1) chunked checkpoints."""

import argparse
import shutil
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
        help="Save an incremental O(1) chunk every N samples (default: 2000).",
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
        help="Overwrite any existing final feature file and chunk directory.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision (AMP) on CUDA.",
    )
    return parser.parse_args()


def _load_existing_chunks(chunks_dir: Path) -> tuple[int, int]:
    """Return (next_chunk_id, total_samples_already_saved) from chunk files."""

    if not chunks_dir.exists():
        return 0, 0

    chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))
    if not chunk_files:
        return 0, 0

    total_saved = 0
    for chunk_file in chunk_files:
        chunk = torch.load(chunk_file, map_location="cpu", weights_only=True)
        total_saved += int(chunk["source_indices"].numel())

    return len(chunk_files), total_saved


def _merge_chunks(
    chunks_dir: Path,
    backbone: str,
    revision: str,
    feature_dim: int,
) -> dict:
    """Merge all incremental chunk files into one final feature dictionary."""

    chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise RuntimeError("No feature chunks were saved.")

    feature_chunks: list[torch.Tensor] = []
    source_indices: list[torch.Tensor] = []
    labels: list[str] = []
    regions: list[str] = []
    countries: list[str] = []

    for chunk_file in chunk_files:
        chunk = torch.load(chunk_file, map_location="cpu", weights_only=True)
        feature_chunks.append(chunk["features"])
        source_indices.append(chunk["source_indices"])
        labels.extend(chunk["labels"])
        regions.extend(chunk["regions"])
        countries.extend(chunk["countries"])

    return {
        "backbone": backbone,
        "revision": revision,
        "feature_dim": feature_dim,
        "features": torch.cat(feature_chunks, dim=0),
        "source_indices": torch.cat(source_indices, dim=0),
        "labels": labels,
        "regions": regions,
        "countries": countries,
    }


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
    chunks_dir = args.output_dir / f"{args.backbone}_{rev_prefix}_chunks"

    if final_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Feature file already exists: {final_path}. "
            "Pass --overwrite to regenerate."
        )

    if args.overwrite and chunks_dir.exists():
        shutil.rmtree(chunks_dir)

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

    next_chunk_id, start_index = _load_existing_chunks(chunks_dir)
    if start_index > 0:
        print(
            f"Resuming from {next_chunk_id} chunks "
            f"({start_index:,} samples already extracted)."
        )

    print("GeoShift | Single-Pass Feature Extraction (O(1) Chunked)")
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

    chunk_features: list[torch.Tensor] = []
    chunk_indices: list[int] = []
    chunk_labels: list[str] = []
    chunk_regions: list[str] = []
    chunk_countries: list[str] = []

    total_extracted = start_index
    total_start = time.perf_counter()

    def flush_batch() -> None:
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                feats = model.backbone(images)
        chunk_features.append(feats.float().cpu())
        chunk_indices.extend(batch_indices)
        chunk_labels.extend(batch_labels)
        chunk_regions.extend(batch_regions)
        chunk_countries.extend(batch_countries)

        batch_tensors.clear()
        batch_indices.clear()
        batch_labels.clear()
        batch_regions.clear()
        batch_countries.clear()

    def flush_chunk() -> None:
        nonlocal next_chunk_id
        if not chunk_indices:
            return
        chunk_path = chunks_dir / f"chunk_{next_chunk_id:05d}.pt"
        _save_pt_atomic(
            {
                "features": torch.cat(chunk_features, dim=0),
                "source_indices": torch.tensor(chunk_indices, dtype=torch.long),
                "labels": list(chunk_labels),
                "regions": list(chunk_regions),
                "countries": list(chunk_countries),
            },
            chunk_path,
        )
        next_chunk_id += 1
        chunk_features.clear()
        chunk_indices.clear()
        chunk_labels.clear()
        chunk_regions.clear()
        chunk_countries.clear()

    for sample in stream:
        if sample.source_index < start_index:
            continue

        if args.max_samples is not None and total_extracted >= args.max_samples:
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
        total_extracted += 1

        if len(batch_tensors) >= args.batch_size:
            flush_batch()
            if len(chunk_indices) >= args.checkpoint_interval:
                flush_chunk()
                elapsed = time.perf_counter() - total_start
                rate = (total_extracted - start_index) / elapsed if elapsed > 0 else 0.0
                print(
                    f"Chunk saved (#{next_chunk_id}) | Total: {total_extracted:,} samples | "
                    f"Rate: {rate:.1f} samples/sec"
                )

    flush_batch()
    flush_chunk()

    final_payload = _merge_chunks(
        chunks_dir=chunks_dir,
        backbone=args.backbone,
        revision=args.revision,
        feature_dim=model.feature_dim,
    )
    _save_pt_atomic(final_payload, final_path)
    shutil.rmtree(chunks_dir, ignore_errors=True)

    elapsed = time.perf_counter() - total_start
    print("\nFEATURE EXTRACTION COMPLETE")
    print(f"Samples extracted: {int(final_payload['source_indices'].numel()):,}")
    print(f"Feature shape:     {tuple(final_payload['features'].shape)}")
    print(f"Elapsed seconds:   {elapsed:.1f}s")
    print(f"Saved to:          {final_path}")


if __name__ == "__main__":
    main()