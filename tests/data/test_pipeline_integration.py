"""End-to-end integration tests for the GeoShift data pipeline."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

import geoshift.data.dataset as dataset_module
import geoshift.data.ingestion as ingestion_module
from geoshift.data.dataset import GeoDEIterableDataset
from geoshift.data.transforms import build_eval_transform, build_train_transform


PINNED_REVISION = "acbae31823b22f84c3bbb11585d37d50aabaac44"
REGIONS = [
    ("Africa", "South Africa"),
    ("Americas", "Colombia"),
    ("EastAsia", "Japan"),
    ("Europe", "France"),
    ("SouthEastAsia", "Indonesia"),
    ("WestAsia", "Turkey"),
]
CLASSES = ["car", "dog"]


def _load_run_pipeline_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_data_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_data_pipeline",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_synthetic_records() -> list[dict]:
    """Build 24 synthetic GeoDE records (6 regions x 2 classes x 2 samples)."""

    records = []
    for region, country in REGIONS:
        for label in CLASSES:
            for copy_idx in range(2):
                color = (
                    (len(records) * 37) % 255,
                    (len(records) * 73) % 255,
                    (len(records) * 19 + copy_idx * 40) % 255,
                )
                records.append(
                    {
                        "image": Image.new("RGB", (64, 64), color=color),
                        "object": label,
                        "region": region,
                        "ip_country": country,
                    }
                )
    return records


@pytest.fixture
def mock_geode_stream(monkeypatch):
    """Mock GeoDEStream across both ingestion and PyTorch dataset modules."""

    records = _build_synthetic_records()
    stream_calls = []

    class FakeGeoDEStream:
        DATASET_ID = "MLap/GeoDE"

        def __init__(
            self,
            split: str = "train",
            revision: str | None = None,
            metadata_only: bool = False,
        ) -> None:
            stream_calls.append(
                {
                    "split": split,
                    "revision": revision,
                    "metadata_only": metadata_only,
                }
            )
            self.split = split
            self.revision = revision or PINNED_REVISION
            self.metadata_only = metadata_only
            self.dataset = records

        def __iter__(self):
            for idx, record in enumerate(self.dataset):
                yield SimpleNamespace(
                    image=None if self.metadata_only else record["image"],
                    label=record["object"],
                    region=record["region"],
                    country=record["ip_country"],
                    source_index=idx,
                )

        def get_revision(self) -> str:
            return self.revision

    monkeypatch.setattr(ingestion_module, "GeoDEStream", FakeGeoDEStream)
    monkeypatch.setattr(dataset_module, "GeoDEStream", FakeGeoDEStream)

    return stream_calls, records


def test_end_to_end_pipeline_to_dataloader(tmp_path, mock_geode_stream):
    """Verify full flow from GeoDE ingestion through PyTorch DataLoader."""

    stream_calls, records = mock_geode_stream
    pipeline = _load_run_pipeline_module()

    raw_dir = tmp_path / "data" / "raw"
    profile_dir = tmp_path / "outputs" / "data_profile"
    manifest_dir = tmp_path / "data" / "manifests" / "geographic"

    results = pipeline.run_pipeline(
        raw_dir=raw_dir,
        profile_dir=profile_dir,
        manifest_dir=manifest_dir,
        split="train",
        revision=PINNED_REVISION,
        validation_fraction=0.15,
        seed=42,
        checkpoint_interval=10,
        overwrite=False,
        reuse_raw=False,
    )

    assert len(stream_calls) == 1
    assert stream_calls[0]["metadata_only"] is True
    assert stream_calls[0]["revision"] == PINNED_REVISION

    assert (raw_dir / "geode_metadata.csv").exists()
    assert (raw_dir / "ingestion_metadata.json").exists()
    assert (profile_dir / "geode_profile.json").exists()
    assert not (profile_dir / "geode_profile_checkpoint.json").exists()
    assert (manifest_dir / "manifest_summary.json").exists()

    summary = json.loads(
        (manifest_dir / "manifest_summary.json").read_text(encoding="utf-8")
    )
    assert summary["dataset_revision"] == PINNED_REVISION
    assert summary["total_samples"] == len(records)
    assert summary["number_of_classes"] == len(CLASSES)
    assert summary["number_of_regions"] == len(REGIONS)
    assert len(summary["manifests"]) == len(REGIONS)

    africa_manifest_path = manifest_dir / "heldout_Africa.csv"
    assert africa_manifest_path.exists()
    africa_manifest = pd.read_csv(africa_manifest_path)

    for split_name, transform in [
        ("train", build_train_transform()),
        ("validation", build_eval_transform()),
        ("test", build_eval_transform()),
    ]:
        dataset = GeoDEIterableDataset(
            manifest=africa_manifest,
            split=split_name,
            revision=PINNED_REVISION,
            transform=transform,
        )
        loader = DataLoader(dataset, batch_size=2, num_workers=0)
        batches = list(loader)
        total_yielded = sum(batch["image"].shape[0] for batch in batches)
        assert total_yielded == len(dataset)

        first_batch = batches[0]
        assert first_batch["image"].ndim == 4
        assert first_batch["image"].shape[1:] == (3, 224, 224)
        assert first_batch["image"].dtype == torch.float32
        assert torch.isfinite(first_batch["image"]).all()

    assert results["validation_summary"]["valid"] is True


def test_pipeline_overwrite_and_reuse_raw(tmp_path, mock_geode_stream):
    """Verify overwrite protection and raw CSV reuse without re-streaming."""

    stream_calls, _ = mock_geode_stream
    pipeline = _load_run_pipeline_module()

    raw_dir = tmp_path / "data" / "raw"
    profile_dir = tmp_path / "outputs" / "data_profile"
    manifest_dir = tmp_path / "data" / "manifests" / "geographic"

    pipeline.run_pipeline(
        raw_dir=raw_dir,
        profile_dir=profile_dir,
        manifest_dir=manifest_dir,
        revision=PINNED_REVISION,
        checkpoint_interval=10,
    )
    assert len(stream_calls) == 1

    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(
            raw_dir=raw_dir,
            profile_dir=profile_dir,
            manifest_dir=manifest_dir,
            revision=PINNED_REVISION,
            overwrite=False,
            reuse_raw=False,
        )

    pipeline.run_pipeline(
        raw_dir=raw_dir,
        profile_dir=profile_dir,
        manifest_dir=manifest_dir,
        revision=PINNED_REVISION,
        checkpoint_interval=10,
        overwrite=True,
        reuse_raw=True,
    )
    assert len(stream_calls) == 1

    with pytest.raises(ValueError, match="revision"):
        pipeline.run_pipeline(
            raw_dir=raw_dir,
            profile_dir=profile_dir,
            manifest_dir=manifest_dir,
            revision="different_revision",
            overwrite=True,
            reuse_raw=True,
        )