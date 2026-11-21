
import csv
import json

from datetime import datetime
from types import SimpleNamespace

import pytest

import geoshift.data.ingestion as ingestion_module
from geoshift.data.ingestion import ingest_geode_metadata


REVISION = "acbae31823b22f84c3bbb11585d37d50aabaac44"


@pytest.fixture
def mock_stream(monkeypatch):
    """Provide a mock GeoDE stream."""

    samples = [
        SimpleNamespace(
            source_index=0,
            label="car",
            region="Africa",
            country="Nigeria",
        ),
        SimpleNamespace(
            source_index=1,
            label="dog",
            region="Europe",
            country="France",
        ),
    ]

    calls = []

    class FakeGeoDEStream:
        DATASET_ID = "MLap/GeoDE"

        def __init__(
            self,
            split="train",
            revision=None,
            metadata_only=False,
        ):
            calls.append(
                {
                    "split": split,
                    "revision": revision,
                    "metadata_only": metadata_only,
                }
            )

            self.samples = samples

        def __iter__(self):
            return iter(self.samples)

        def get_revision(self):
            return REVISION

    monkeypatch.setattr(
        ingestion_module,
        "GeoDEStream",
        FakeGeoDEStream,
    )

    return calls, samples


def test_ingestion_creates_output_files(tmp_path, mock_stream):
    """Ingestion should create the CSV and metadata files."""

    result = ingest_geode_metadata(output_dir=tmp_path)

    csv_path = tmp_path / "geode_metadata.csv"
    metadata_path = tmp_path / "ingestion_metadata.json"

    assert csv_path.exists()
    assert metadata_path.exists()

    assert result["complete"] is True
    assert result["sample_count"] == 2


def test_ingestion_preserves_raw_metadata(tmp_path, mock_stream):
    """CSV should preserve original GeoDE metadata field names and values."""

    ingest_geode_metadata(output_dir=tmp_path)

    csv_path = tmp_path / "geode_metadata.csv"

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == [
        "source_index",
        "object",
        "region",
        "ip_country",
    ]

    assert rows == [
        {
            "source_index": "0",
            "object": "car",
            "region": "Africa",
            "ip_country": "Nigeria",
        },
        {
            "source_index": "1",
            "object": "dog",
            "region": "Europe",
            "ip_country": "France",
        },
    ]


def test_ingestion_records_provenance(tmp_path, mock_stream):
    """Metadata should record the dataset source and revision."""

    result = ingest_geode_metadata(
        output_dir=tmp_path,
        split="train",
        revision=REVISION,
    )

    metadata_path = tmp_path / "ingestion_metadata.json"

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata == result
    assert metadata["dataset_id"] == "MLap/GeoDE"
    assert metadata["split"] == "train"
    assert metadata["revision"] == REVISION
    assert metadata["sample_count"] == 2
    assert metadata["complete"] is True

    datetime.fromisoformat(metadata["ingested_at_utc"])


def test_stream_receives_correct_arguments(tmp_path, mock_stream):
    """Ingestion should request metadata-only streaming."""

    calls, _ = mock_stream

    ingest_geode_metadata(
        output_dir=tmp_path,
        split="train",
        revision=REVISION,
    )

    assert calls == [
        {
            "split": "train",
            "revision": REVISION,
            "metadata_only": True,
        }
    ]


def test_ingestion_rejects_existing_output(tmp_path, mock_stream):
    """Existing output should not be overwritten by default."""

    csv_path = tmp_path / "geode_metadata.csv"
    csv_path.write_text("existing data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ingest_geode_metadata(output_dir=tmp_path)

    assert csv_path.read_text(encoding="utf-8") == "existing data"


def test_ingestion_allows_overwrite(tmp_path, mock_stream):
    """overwrite=True should replace existing ingestion output."""

    csv_path = tmp_path / "geode_metadata.csv"
    metadata_path = tmp_path / "ingestion_metadata.json"

    csv_path.write_text("old data", encoding="utf-8")
    metadata_path.write_text("{}", encoding="utf-8")

    result = ingest_geode_metadata(
        output_dir=tmp_path,
        overwrite=True,
    )

    assert result["complete"] is True

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["object"] == "car"


def test_ingestion_rejects_empty_stream(
    tmp_path,
    mock_stream,
):
    """An empty stream should not produce completed ingestion files."""

    _, samples = mock_stream
    samples.clear()

    with pytest.raises(
        ValueError,
        match="zero samples",
    ):
        ingest_geode_metadata(output_dir=tmp_path)

    assert not (tmp_path / "geode_metadata.csv").exists()
    assert not (tmp_path / "ingestion_metadata.json").exists()


def test_failed_ingestion_does_not_publish_partial_files(
    tmp_path,
    monkeypatch,
):
    """A stream failure should not publish incomplete output."""

    class FailingGeoDEStream:
        DATASET_ID = "MLap/GeoDE"

        def __init__(self, **kwargs):
            pass

        def __iter__(self):
            yield SimpleNamespace(
                source_index=0,
                label="car",
                region="Africa",
                country="Nigeria",
            )

            raise RuntimeError("Simulated stream failure")

        def get_revision(self):
            return REVISION

    monkeypatch.setattr(
        ingestion_module,
        "GeoDEStream",
        FailingGeoDEStream,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated stream failure",
    ):
        ingest_geode_metadata(output_dir=tmp_path)

    assert not (tmp_path / "geode_metadata.csv").exists()
    assert not (tmp_path / "ingestion_metadata.json").exists()

    assert list(tmp_path.iterdir()) == []
