
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from geoshift.config import load_config, project_path
from geoshift.data.geode import GeoDESample

_PATH_CONFIG = load_config("paths.yml")
_DATA_CONFIG = load_config("data.yml")
_PROFILE_DIR = project_path(_PATH_CONFIG["data"]["profile"])
_CHECKPOINT_INTERVAL = _DATA_CONFIG["ingestion"]["checkpoint_interval"]


def _build_report(
    dataset_id: str,
    split: str,
    revision: str,
    total_samples: int,
    class_counts: Counter,
    region_counts: Counter,
    country_counts: Counter,
    class_region_counts: Counter,
    elapsed: float,
    complete: bool,
) -> dict:
    """Build a structured metadata profiling report."""

    return {
        "dataset_id": dataset_id,
        "split": split,
        "revision": revision,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "complete": complete,
        "total_samples": total_samples,
        "number_of_classes": len(class_counts),
        "number_of_regions": len(region_counts),
        "number_of_countries": len(country_counts),
        "elapsed_seconds": round(elapsed, 2),
        "samples_per_second": round(total_samples / elapsed, 2)
        if elapsed else 0,
        "class_counts": dict(sorted(class_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
        "country_counts": dict(sorted(country_counts.items())),
        "class_region_counts": {
            f"{label}|{region}": count
            for (label, region), count in sorted(
                class_region_counts.items()
            )
        },
    }


def _save_report(report: dict, output_path: Path) -> None:
    """Save a report using a temporary file before replacement."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(output_path)


def profile_geode(
    samples: Iterable[GeoDESample],
    dataset_id: str,
    split: str,
    revision: str,
    output_dir: str | Path = _PROFILE_DIR,
    checkpoint_interval: int = _CHECKPOINT_INTERVAL,
) -> dict:
    """Profile GeoDE metadata and save checkpoint and final reports."""

    if checkpoint_interval < 1:
        raise ValueError(
            "checkpoint_interval must be at least 1."
        )

    output_dir = Path(output_dir)

    checkpoint_path = output_dir / "geode_profile_checkpoint.json"
    final_path = output_dir / "geode_profile.json"

    class_counts = Counter()
    region_counts = Counter()
    country_counts = Counter()
    class_region_counts = Counter()

    total_samples = 0
    start_time = time.perf_counter()

    for sample in samples:
        class_counts[sample.label] += 1
        region_counts[sample.region] += 1
        country_counts[sample.country] += 1
        class_region_counts[
            (sample.label, sample.region)
        ] += 1

        total_samples += 1

        if total_samples % checkpoint_interval == 0:
            elapsed = time.perf_counter() - start_time

            report = _build_report(
                dataset_id,
                split,
                revision,
                total_samples,
                class_counts,
                region_counts,
                country_counts,
                class_region_counts,
                elapsed,
                complete=False,
            )

            _save_report(report, checkpoint_path)

    if total_samples == 0:
        raise ValueError("Cannot profile an empty dataset.")

    elapsed = time.perf_counter() - start_time

    report = _build_report(
        dataset_id,
        split,
        revision,
        total_samples,
        class_counts,
        region_counts,
        country_counts,
        class_region_counts,
        elapsed,
        complete=True,
    )

    _save_report(report, final_path)

    checkpoint_path.unlink(missing_ok=True)

    return report
