
"""Tests for the GeoDE PyTorch streaming dataset."""

from types import SimpleNamespace

import pandas as pd
import pytest
from torch.utils.data import DataLoader

from geoshift.data.dataset import GeoDEIterableDataset


@pytest.fixture
def sample_manifest() -> pd.DataFrame:
    """Create a synthetic geographic manifest."""

    return pd.DataFrame(
        [
            {
                "source_index": 0,
                "label": "car",
                "region": "Africa",
                "country": "Nigeria",
                "split": "test",
            },
            {
                "source_index": 1,
                "label": "dog",
                "region": "Africa",
                "country": "Nigeria",
                "split": "test",
            },
            {
                "source_index": 2,
                "label": "car",
                "region": "Europe",
                "country": "France",
                "split": "train",
            },
            {
                "source_index": 3,
                "label": "dog",
                "region": "Europe",
                "country": "France",
                "split": "train",
            },
        ]
    )


class FakeImage:
    """Minimal image substitute for testing."""

    def __init__(self):
        self.converted = False

    def convert(self, mode):
        assert mode == "RGB"
        self.converted = True
        return self


class FakeGeoDEStream:
    """Mock Hugging Face stream."""

    def __init__(self, **kwargs):
        assert kwargs["split"] == "train"
        assert kwargs["metadata_only"] is False
        assert kwargs["revision"] == "test_revision"

        self.dataset = [
            {
                "image": FakeImage(),
                "object": "car",
                "region": "Africa",
                "ip_country": "Nigeria",
            },
            {
                "image": FakeImage(),
                "object": "dog",
                "region": "Africa",
                "ip_country": "Nigeria",
            },
            {
                "image": FakeImage(),
                "object": "car",
                "region": "Europe",
                "ip_country": "France",
            },
            {
                "image": FakeImage(),
                "object": "dog",
                "region": "Europe",
                "ip_country": "France",
            },
        ]


@pytest.fixture
def dataset(monkeypatch, sample_manifest):
    """Create a dataset using a mocked GeoDE stream."""

    monkeypatch.setattr(
        "geoshift.data.dataset.GeoDEStream",
        FakeGeoDEStream,
    )

    return GeoDEIterableDataset(
        manifest=sample_manifest,
        split="test",
        revision="test_revision",
    )


def test_dataset_length(dataset):
    assert len(dataset) == 2


def test_only_selected_split_is_returned(dataset):
    samples = list(dataset)

    assert len(samples) == 2
    assert all(sample["region"] == "Africa" for sample in samples)


def test_labels_are_mapped_consistently(dataset):
    samples = list(dataset)

    labels = {
        sample["label_name"]: sample["label"]
        for sample in samples
    }

    assert labels == {
        "car": 0,
        "dog": 1,
    }


def test_sample_contains_required_metadata(dataset):
    sample = next(iter(dataset))

    assert set(sample) == {
        "image",
        "label",
        "label_name",
        "region",
        "country",
        "source_index",
    }


def test_images_are_converted_to_rgb(dataset):
    samples = list(dataset)

    assert all(
        sample["image"].converted
        for sample in samples
    )


def test_transform_is_applied(monkeypatch, sample_manifest):
    monkeypatch.setattr(
        "geoshift.data.dataset.GeoDEStream",
        FakeGeoDEStream,
    )

    dataset = GeoDEIterableDataset(
        manifest=sample_manifest,
        split="test",
        revision="test_revision",
        transform=lambda image: "transformed",
    )

    samples = list(dataset)

    assert all(
        sample["image"] == "transformed"
        for sample in samples
    )


def test_invalid_split_raises_error(sample_manifest):
    with pytest.raises(ValueError, match="Invalid split"):
        GeoDEIterableDataset(
            manifest=sample_manifest,
            split="invalid",
            revision="test_revision",
        )


def test_missing_revision_raises_error(sample_manifest):
    with pytest.raises(ValueError, match="pinned GeoDE revision"):
        GeoDEIterableDataset(
            manifest=sample_manifest,
            split="test",
            revision="",
        )


def test_empty_selected_split_raises_error(sample_manifest):
    with pytest.raises(ValueError, match="No samples found"):
        GeoDEIterableDataset(
            manifest=sample_manifest,
            split="validation",
            revision="test_revision",
        )


def test_missing_manifest_column_raises_error(sample_manifest):
    invalid_manifest = sample_manifest.drop(columns=["country"])

    with pytest.raises(ValueError, match="missing columns"):
        GeoDEIterableDataset(
            manifest=invalid_manifest,
            split="test",
            revision="test_revision",
        )


@pytest.mark.parametrize("invalid_index", [True, 1.5, "zero"])
def test_invalid_source_index_raises_error(
    sample_manifest,
    invalid_index,
):
    invalid_manifest = sample_manifest.copy()
    invalid_manifest["source_index"] = invalid_manifest[
        "source_index"
    ].astype(object)
    invalid_manifest.loc[0, "source_index"] = invalid_index

    with pytest.raises(ValueError, match="must be integers"):
        GeoDEIterableDataset(
            manifest=invalid_manifest,
            split="test",
            revision="test_revision",
        )


def test_duplicate_source_indices_raise_error(sample_manifest):
    invalid_manifest = sample_manifest.copy()
    invalid_manifest.loc[1, "source_index"] = 0

    with pytest.raises(ValueError, match="duplicate source indices"):
        GeoDEIterableDataset(
            manifest=invalid_manifest,
            split="test",
            revision="test_revision",
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("label", ""),
        ("region", " "),
        ("country", 123),
    ],
)
def test_invalid_metadata_text_raises_error(
    sample_manifest,
    column,
    value,
):
    invalid_manifest = sample_manifest.copy()
    invalid_manifest[column] = invalid_manifest[column].astype(object)
    invalid_manifest.loc[0, column] = value

    with pytest.raises(ValueError, match="invalid text values"):
        GeoDEIterableDataset(
            manifest=invalid_manifest,
            split="test",
            revision="test_revision",
        )


def test_invalid_split_assignment_raises_error(sample_manifest):
    invalid_manifest = sample_manifest.copy()
    invalid_manifest.loc[0, "split"] = "testing"

    with pytest.raises(ValueError, match="invalid split assignments"):
        GeoDEIterableDataset(
            manifest=invalid_manifest,
            split="test",
            revision="test_revision",
        )


def test_multiple_dataloader_workers_are_rejected(
    monkeypatch,
    sample_manifest,
):
    monkeypatch.setattr(
        "geoshift.data.dataset.GeoDEStream",
        FakeGeoDEStream,
    )

    monkeypatch.setattr(
        "geoshift.data.dataset.get_worker_info",
        lambda: SimpleNamespace(id=0, num_workers=2),
    )

    dataset = GeoDEIterableDataset(
        manifest=sample_manifest,
        split="test",
        revision="test_revision",
    )

    with pytest.raises(RuntimeError, match="Multiple DataLoader workers"):
        next(iter(dataset))


def test_dataloader_with_zero_workers(
    monkeypatch,
    sample_manifest,
):
    monkeypatch.setattr(
        "geoshift.data.dataset.GeoDEStream",
        FakeGeoDEStream,
    )

    dataset = GeoDEIterableDataset(
        manifest=sample_manifest,
        split="test",
        revision="test_revision",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    samples = list(loader)

    assert len(samples) == 2
    assert samples[0][0]["label_name"] == "car"
    assert samples[1][0]["label_name"] == "dog"


def test_missing_image_raises_error(
    monkeypatch,
    sample_manifest,
):
    class MissingImageStream:
        def __init__(self, **kwargs):
            self.dataset = [
                {
                    "image": None,
                    "object": "car",
                    "region": "Africa",
                    "ip_country": "Nigeria",
                }
            ]

    monkeypatch.setattr(
        "geoshift.data.dataset.GeoDEStream",
        MissingImageStream,
    )

    dataset = GeoDEIterableDataset(
        manifest=sample_manifest.iloc[[0]],
        split="test",
        revision="test_revision",
    )

    with pytest.raises(RuntimeError, match="Missing image"):
        next(iter(dataset))


def test_stream_exhaustion_raises_error(
    monkeypatch,
    sample_manifest,
):
    class ShortStream:
        def __init__(self, **kwargs):
            self.dataset = [
                {
                    "image": FakeImage(),
                    "object": "car",
                    "region": "Africa",
                    "ip_country": "Nigeria",
                }
            ]

    monkeypatch.setattr(
        "geoshift.data.dataset.GeoDEStream",
        ShortStream,
    )

    dataset = GeoDEIterableDataset(
        manifest=sample_manifest,
        split="test",
        revision="test_revision",
    )

    with pytest.raises(RuntimeError, match="Stream ended"):
        list(dataset)


def test_metadata_mismatch_raises_error(
    monkeypatch,
    sample_manifest,
):
    class MismatchedStream:
        def __init__(self, **kwargs):
            self.dataset = [
                {
                    "image": FakeImage(),
                    "object": "car",
                    "region": "Europe",
                    "ip_country": "France",
                }
            ]

    monkeypatch.setattr(
        "geoshift.data.dataset.GeoDEStream",
        MismatchedStream,
    )

    dataset = GeoDEIterableDataset(
        manifest=sample_manifest.iloc[[0]],
        split="test",
        revision="test_revision",
    )

    with pytest.raises(RuntimeError, match="Metadata mismatch"):
        next(iter(dataset))
