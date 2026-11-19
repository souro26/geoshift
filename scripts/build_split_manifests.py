
"""Generate geographic split manifests for the GeoDE dataset."""

import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from geoshift.data.geode import GeoDEStream
from geoshift.data.manifest import build_geographic_manifests


output_dir = Path("data/manifests/geographic")
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("GeoShift | Geographic Manifest Generation")

    dataset = GeoDEStream(split="train", metadata_only=True)

    print(f"Dataset: {dataset.DATASET_ID}")
    print(f"Revision: {dataset.get_revision()}")
    print("\nStreaming metadata...\n")

    records = []

    for sample in dataset:
        records.append(
            {
                "source_index": sample.source_index,
                "label": sample.label,
                "region": sample.region,
                "country": sample.country,
            }
        )

        if len(records) % 10000 == 0:
            print(f"Processed: {len(records):,} samples")

    if not records:
        raise RuntimeError("No metadata records were loaded.")

    print(f"\nTotal records: {len(records):,}")
    print("\nBuilding geographic manifests...\n")

    samples = pd.DataFrame.from_records(records)

    manifests = build_geographic_manifests( samples=samples, validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED, )

    summary = {
        "dataset_id": dataset.DATASET_ID,
        "dataset_revision": dataset.get_revision(),
        "total_samples": len(samples),
        "number_of_classes": samples["label"].nunique(),
        "number_of_regions": samples["region"].nunique(),
        "validation_fraction": VALIDATION_FRACTION,
        "random_seed": RANDOM_SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifests": {},
    }

    for region, manifest in manifests.items():
        filename = f"heldout_{region}.csv"
        output_path = output_dir / filename
        manifest.to_csv(output_path, index=False)
        split_counts = ( manifest["split"] .value_counts() .to_dict() )
        summary["manifests"][region] = { "filename": filename, "split_counts": split_counts, }

        print(
            f"{region:15} | "
            f"Train: {split_counts.get('train', 0):,} | "
            f"Validation: {split_counts.get('validation', 0):,} | "
            f"Test: {split_counts.get('test', 0):,}"
        )

    summary_path = output_dir / "manifest_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("MANIFEST GENERATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
