
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from geoshift.config import load_config, project_path
from geoshift.data.geode import GeoDEStream

_DATA_CONFIG = load_config("data.yml")
_RAW_DIR = project_path(load_config("paths.yml")["data"]["raw"])


RAW_COLUMNS = [
    "source_index",
    "object",
    "region",
    "ip_country",
]


def ingest_geode_metadata(
    output_dir: str | Path = _RAW_DIR,
    split: str = _DATA_CONFIG["dataset"]["split"],
    revision: str | None = _DATA_CONFIG["dataset"]["revision"],
    overwrite: bool = False,
) -> dict:
    """Stream GeoDE metadata and persist a raw snapshot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "geode_metadata.csv"
    metadata_path = output_dir / "ingestion_metadata.json"

    if not overwrite and (csv_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            "Ingestion output already exists. "
            "Set overwrite=True to replace it."
        )

    stream = GeoDEStream(
        split=split,
        revision=revision,
        metadata_only=True,
    )

    csv_temp_path = None
    metadata_temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as temp_file:

            csv_temp_path = Path(temp_file.name)

            writer = csv.DictWriter(
                temp_file,
                fieldnames=RAW_COLUMNS,
            )

            writer.writeheader()

            sample_count = 0

            for sample in stream:
                writer.writerow(
                    {
                        "source_index": sample.source_index,
                        "object": sample.label,
                        "region": sample.region,
                        "ip_country": sample.country,
                    }
                )

                sample_count += 1

        if sample_count == 0:
            raise ValueError("GeoDE ingestion returned zero samples.")

        ingestion_metadata = {
            "dataset_id": stream.DATASET_ID,
            "split": split,
            "revision": stream.get_revision(),
            "ingested_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "sample_count": sample_count,
            "columns": RAW_COLUMNS,
            "complete": True,
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as temp_file:

            metadata_temp_path = Path(temp_file.name)

            json.dump(
                ingestion_metadata,
                temp_file,
                indent=4,
            )

        os.replace(csv_temp_path, csv_path)
        csv_temp_path = None

        os.replace(metadata_temp_path, metadata_path)
        metadata_temp_path = None

        return ingestion_metadata

    finally:
        if csv_temp_path is not None:
            csv_temp_path.unlink(missing_ok=True)

        if metadata_temp_path is not None:
            metadata_temp_path.unlink(missing_ok=True)
