"""Benchmark streaming retrieval and preprocessing throughput on GeoDE."""

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from geoshift.config import load_config, project_path
from geoshift.data.dataset import GeoDEIterableDataset
from geoshift.data.transforms import build_eval_transform

_DATA_CONFIG = load_config("data.yml")
_MANIFEST_SPLITS = _DATA_CONFIG["splitting"]["splits"]
_PATH_CONFIG = load_config("paths.yml")
_BENCHMARK_CONFIG = load_config("evaluation.yml")["benchmark"]
MANIFEST_DIR = project_path(_PATH_CONFIG["data"]["manifests"])
OUTPUT_DIR = project_path(_PATH_CONFIG["outputs"]["benchmarks"])
DATASET_ID = _DATA_CONFIG["dataset"]["id"]
DATASET_REVISION = _DATA_CONFIG["dataset"]["revision"]


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Benchmark the GeoDE streaming data pipeline."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=_BENCHMARK_CONFIG["samples"],
        help="Number of samples to benchmark.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=_BENCHMARK_CONFIG["region"],
        help="Held-out geographic region defining the manifest.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=_BENCHMARK_CONFIG["split"],
        choices=list(_MANIFEST_SPLITS.values()),
        help="Dataset split to benchmark.",
    )
    return parser.parse_args()


def benchmark_pipeline(
    manifest: pd.DataFrame,
    split: str,
    sample_limit: int,
    transform,
) -> dict:
    """Measure retrieval and preprocessing throughput."""

    dataset = GeoDEIterableDataset(
        manifest=manifest,
        split=split,
        revision=DATASET_REVISION,
        transform=transform,
    )
    samples_processed = 0
    start_time = time.perf_counter()

    for _ in dataset:
        samples_processed += 1
        if samples_processed >= sample_limit:
            break

    elapsed_seconds = time.perf_counter() - start_time
    samples_per_second = (
        samples_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )

    return {
        "samples_processed": samples_processed,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "samples_per_second": round(samples_per_second, 3),
    }


def main() -> None:
    """Execute the GeoDE pipeline benchmark."""

    args = parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero.")

    manifest_path = MANIFEST_DIR / f"heldout_{args.region}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    available_samples = int((manifest["split"] == args.split).sum())
    if available_samples == 0:
        raise ValueError(f"No samples available for split '{args.split}'.")

    sample_limit = min(args.samples, available_samples)

    print("GEOSHIFT DATA PIPELINE BENCHMARK")
    print(f"Region: {args.region}")
    print(f"Split: {args.split}")
    print(f"Requested samples: {args.samples}")
    print(f"Samples available: {available_samples}")
    print(f"Samples to process: {sample_limit}")
    print(f"Dataset revision: {DATASET_REVISION}")
    print()

    print("[1/2] Benchmarking image retrieval without transforms...")
    raw_results = benchmark_pipeline(
        manifest=manifest,
        split=args.split,
        sample_limit=sample_limit,
        transform=None,
    )
    print(raw_results)
    print()

    print("[2/2] Benchmarking retrieval with evaluation transforms...")
    transformed_results = benchmark_pipeline(
        manifest=manifest,
        split=args.split,
        sample_limit=sample_limit,
        transform=build_eval_transform(),
    )
    print(transformed_results)
    print()

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "region": args.region,
        "split": args.split,
        "requested_samples": args.samples,
        "raw_pipeline": raw_results,
        "transformed_pipeline": transformed_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        OUTPUT_DIR / f"benchmark_{args.region}_{args.split}_{args.samples}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("BENCHMARK COMPLETE")
    print(f"Raw throughput: {raw_results['samples_per_second']} samples/sec")
    print(
        f"Transformed throughput: {transformed_results['samples_per_second']} samples/sec"
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()