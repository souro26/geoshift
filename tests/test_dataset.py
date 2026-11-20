
"""Tests for the GeoDE PyTorch streaming dataset."""

import pandas as pd
import pytest

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
    with pytest.raises(ValueError):
        GeoDEIterableDataset(
            manifest=sample_manifest,
            split="invalid",
            revision="test_revision",
        )


def test_missing_revision_raises_error(sample_manifest):
    with pytest.raises(ValueError):
        GeoDEIterableDataset(
            manifest=sample_manifest,
            split="test",
            revision="",
        )


def test_empty_selected_split_raises_error(sample_manifest):
    with pytest.raises(ValueError):
        GeoDEIterableDataset(
            manifest=sample_manifest,
            split="validation",
            revision="test_revision",
        )


def test_missing_manifest_column_raises_error(sample_manifest):
    invalid_manifest = sample_manifest.drop(columns=["country"])

    with pytest.raises(ValueError):
        GeoDEIterableDataset(
            manifest=invalid_manifest,
            split="test",
            revision="test_revision",
        )
