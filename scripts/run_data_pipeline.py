"""Coordinate the end-to-end GeoDE metadata, profiling, and manifest pipeline."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from geoshift.config import load_config, project_path
from geoshift.data.cleaning import clean_geode_metadata
from geoshift.data.geode import GeoDESample
from geoshift.data.ingestion import ingest_geode_metadata
from geoshift.data.manifest import build_geographic_manifests
from geoshift.data.profiling import profile_geode
from geoshift.data.validation import validate_geode_metadata

_DATA_CONFIG = load_config("data.yml")
_PATH_CONFIG = load_config("paths.yml")
DEFAULT_PINNED_REVISION = _DATA_CONFIG["dataset"]["revision"]
DEFAULT_SPLIT = _DATA_CONFIG["dataset"]["split"]
DEFAULT_RAW_DIR = project_path(_PATH_CONFIG["data"]["raw"])
DEFAULT_PROFILE_DIR = project_path(_PATH_CONFIG["data"]["profile"])
DEFAULT_MANIFEST_DIR = project_path(_PATH_CONFIG["data"]["manifests"])
DEFAULT_VALIDATION_FRACTION = _DATA_CONFIG["splitting"]["validation_fraction"]
DEFAULT_RANDOM_SEED = _DATA_CONFIG["splitting"]["seed"]
DEFAULT_CHECKPOINT_INTERVAL = _DATA_CONFIG["ingestion"]["checkpoint_interval"]


def _iter_geode_samples(cleaned_df: pd.DataFrame) -> Iterator[GeoDESample]:
    """Yield lightweight GeoDESample records from cleaned metadata."""

    for source_index, label, region, country in cleaned_df[
        ["source_index", "label", "region", "country"]
    ].itertuples(index=False, name=None):
        yield GeoDESample(
            image=None,
            label=label,
            region=region,
            country=country,
            source_index=int(source_index),
        )


def _save_json_atomic(payload: dict, output_path: Path) -> None:
    """Write JSON atomically via a temporary file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _check_manifest_overwrite(
    manifest_dir: Path,
    regions: list[str],
    overwrite: bool,
) -> None:
    """Prevent accidental overwrite of existing manifest outputs."""

    if overwrite:
        return

    summary_path = manifest_dir / "manifest_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Manifest summary already exists at {summary_path}. "
            "Pass overwrite=True / --overwrite to replace it."
        )

    for region in regions:
        manifest_path = manifest_dir / f"heldout_{region}.csv"
        if manifest_path.exists():
            raise FileExistsError(
                f"Manifest already exists at {manifest_path}. "
                "Pass overwrite=True / --overwrite to replace it."
            )


def run_pipeline(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
    split: str = DEFAULT_SPLIT,
    revision: str | None = DEFAULT_PINNED_REVISION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_RANDOM_SEED,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    overwrite: bool = False,
    reuse_raw: bool = False,
) -> dict:
    """Run ingestion, cleaning, validation, profiling, and manifest generation."""

    raw_dir = Path(raw_dir)
    profile_dir = Path(profile_dir)
    manifest_dir = Path(manifest_dir)

    raw_csv_path = raw_dir / "geode_metadata.csv"
    ingestion_meta_path = raw_dir / "ingestion_metadata.json"

    if reuse_raw:
        if not raw_csv_path.exists() or not ingestion_meta_path.exists():
            raise FileNotFoundError(
                "Cannot reuse raw metadata because "
                f"{raw_csv_path} or {ingestion_meta_path} is missing."
            )

        with ingestion_meta_path.open("r", encoding="utf-8") as file:
            ingestion_metadata = json.load(file)

        if ingestion_metadata.get("split") != split:
            raise ValueError(
                "Existing raw metadata split "
                f"({ingestion_metadata.get('split')}) does not match "
                f"requested split ({split})."
            )

        if (
            revision is not None
            and ingestion_metadata.get("revision") != revision
        ):
            raise ValueError(
                "Existing raw metadata revision "
                f"({ingestion_metadata.get('revision')}) does not match "
                f"requested revision ({revision})."
            )
    else:
        ingestion_metadata = ingest_geode_metadata(
            output_dir=raw_dir,
            split=split,
            revision=revision,
            overwrite=overwrite,
        )

    resolved_revision = ingestion_metadata["revision"]
    dataset_id = ingestion_metadata["dataset_id"]

    raw_df = pd.read_csv(raw_csv_path)
    cleaned_df = clean_geode_metadata(raw_df)
    validation_summary = validate_geode_metadata(cleaned_df)

    profile_report = profile_geode(
        samples=_iter_geode_samples(cleaned_df),
        dataset_id=dataset_id,
        split=split,
        revision=resolved_revision,
        output_dir=profile_dir,
        checkpoint_interval=checkpoint_interval,
    )

    manifests = build_geographic_manifests(
        samples=cleaned_df,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    _check_manifest_overwrite(
        manifest_dir=manifest_dir,
        regions=list(manifests.keys()),
        overwrite=overwrite,
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_summary = {
        "dataset_id": dataset_id,
        "dataset_revision": resolved_revision,
        "total_samples": validation_summary["total_samples"],
        "number_of_classes": validation_summary["number_of_classes"],
        "number_of_regions": validation_summary["number_of_regions"],
        "validation_fraction": validation_fraction,
        "random_seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifests": {},
    }

    for region, manifest_df in manifests.items():
        filename = f"heldout_{region}.csv"
        output_path = manifest_dir / filename
        manifest_df.to_csv(output_path, index=False)

        split_counts = {
            str(key): int(value)
            for key, value in manifest_df["split"].value_counts().items()
        }
        manifest_summary["manifests"][region] = {
            "filename": filename,
            "split_counts": split_counts,
        }

    summary_path = manifest_dir / "manifest_summary.json"
    _save_json_atomic(manifest_summary, summary_path)

    return {
        "ingestion_metadata": ingestion_metadata,
        "validation_summary": validation_summary,
        "profile_report": profile_report,
        "manifest_summary": manifest_summary,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GeoShift data pipeline."""

    parser = argparse.ArgumentParser(
        description="Run the GeoShift metadata, profiling, and manifest pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for raw ingested metadata CSV and provenance JSON.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=DEFAULT_PROFILE_DIR,
        help="Directory for metadata profiling outputs.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory for generated geographic split manifests.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Dataset split to ingest (default: train).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_PINNED_REVISION,
        help="Pinned Hugging Face dataset revision.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Fraction of each class-region group assigned to validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible geographic splits.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Sample interval for profiling checkpoints.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing raw metadata and manifest files.",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse existing raw metadata CSV instead of re-streaming GeoDE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("GeoShift | Data Pipeline")
    print(f"Split: {args.split} | Revision: {args.revision}")

    results = run_pipeline(
        raw_dir=args.raw_dir,
        profile_dir=args.profile_dir,
        manifest_dir=args.manifest_dir,
        split=args.split,
        revision=args.revision,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        overwrite=args.overwrite,
        reuse_raw=args.reuse_raw,
    )

    validation_summary = results["validation_summary"]
    manifest_summary = results["manifest_summary"]

    print(
        f"Validated {validation_summary['total_samples']:,} samples "
        f"({validation_summary['number_of_classes']} classes, "
        f"{validation_summary['number_of_regions']} regions)"
    )

    for region, details in manifest_summary["manifests"].items():
        counts = details["split_counts"]
        print(
            f"{region:15} | "
            f"Train: {counts.get('train', 0):,} | "
            f"Validation: {counts.get('validation', 0):,} | "
            f"Test: {counts.get('test', 0):,}"
        )

    print(f"Outputs saved to: {args.manifest_dir}")


if __name__ == "__main__":
    main()