
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from geoshift.data.geode import GeoDEStream


OUTPUT_DIR = Path("outputs/data_profile")
CHECKPOINT_PATH = OUTPUT_DIR / "geode_profile_checkpoint.json"
FINAL_PATH = OUTPUT_DIR / "geode_profile.json"

CHECKPOINT_INTERVAL = 1000


def save_report(
    dataset,
    total_samples,
    class_counts,
    region_counts,
    country_counts,
    class_region_counts,
    elapsed,
    complete=False,
):
    report = {
        "dataset_id": dataset.DATASET_ID,
        "split": "train",
        "revision": dataset.get_revision(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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
            for (label, region), count in sorted(class_region_counts.items())
        },
    }

    output_path = FINAL_PATH if complete else CHECKPOINT_PATH

    temporary_path = output_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    temporary_path.replace(output_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("GeoShift | Full GeoDE Metadata Profiling")
    print("=" * 65)

    dataset = GeoDEStream(
        split="train",
        metadata_only=True,
    )

    class_counts = Counter()
    region_counts = Counter()
    country_counts = Counter()
    class_region_counts = Counter()

    total_samples = 0
    start_time = time.perf_counter()

    print(f"Dataset: {dataset.DATASET_ID}")
    print(f"Revision: {dataset.get_revision()}")
    print("\nScanning metadata...\n")

    for sample in dataset:
        class_counts[sample.label] += 1
        region_counts[sample.region] += 1
        country_counts[sample.country] += 1
        class_region_counts[(sample.label, sample.region)] += 1

        total_samples += 1

        if total_samples % CHECKPOINT_INTERVAL == 0:
            elapsed = time.perf_counter() - start_time

            save_report(
                dataset,
                total_samples,
                class_counts,
                region_counts,
                country_counts,
                class_region_counts,
                elapsed,
            )

            print(
                f"Processed: {total_samples:,} | "
                f"Throughput: {total_samples / elapsed:.2f} samples/sec"
            )

    elapsed = time.perf_counter() - start_time

    save_report(
        dataset,
        total_samples,
        class_counts,
        region_counts,
        country_counts,
        class_region_counts,
        elapsed,
        complete=True,
    )

    print("\n" + "=" * 65)
    print("PROFILE COMPLETE")
    print("=" * 65)
    print(f"Total samples: {total_samples:,}")
    print(f"Classes: {len(class_counts)}")
    print(f"Regions: {len(region_counts)}")
    print(f"Countries: {len(country_counts)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Report saved to: {FINAL_PATH}")


if __name__ == "__main__":
    main()
