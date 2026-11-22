
from types import SimpleNamespace

import pytest

import geoshift.data.geode as geode_module
from geoshift.data.geode import GeoDEStream, GeoDESample


REVISION = "acbae31823b22f84c3bbb11585d37d50aabaac44"


@pytest.fixture
def mock_dataset(monkeypatch):
    """Mock the Hugging Face dataset loader."""

    samples = [
        { "image": "image_1", "object": "car", "region": "Africa", "ip_country": "Nigeria", },
        { "image": "image_2", "object": "dog", "region": "Europe", "ip_country": "France", },
    ]

    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(samples)

    monkeypatch.setattr( geode_module, "load_dataset", fake_load_dataset, )
    return calls


def test_initialization_with_explicit_revision(mock_dataset):
    """Stream should use the supplied dataset revision."""

    stream = GeoDEStream(revision=REVISION)

    assert stream.revision == REVISION
    assert stream.split == "train"
    assert stream.metadata_only is False


def test_revision_resolution_when_not_provided( monkeypatch, mock_dataset, ):
    """Stream should resolve the current revision from Hugging Face."""

    class FakeHfApi:
        def dataset_info(self, repo_id):
            assert repo_id == "MLap/GeoDE"
            return SimpleNamespace(sha=REVISION)

    monkeypatch.setattr( geode_module, "HfApi", FakeHfApi, )
    stream = GeoDEStream()
    assert stream.revision == REVISION


def test_metadata_only_loading(mock_dataset):
    """Metadata-only mode should request only metadata columns."""

    GeoDEStream( revision=REVISION, metadata_only=True)
    _, kwargs = mock_dataset[0]

    assert kwargs["streaming"] is True
    assert kwargs["revision"] == REVISION
    assert kwargs["columns"] == [ "object", "region", "ip_country", ]


def test_full_dataset_loading(mock_dataset):
    """Full mode should load the image column as well."""

    GeoDEStream( revision=REVISION, metadata_only=False, )
    _, kwargs = mock_dataset[0]

    assert kwargs["streaming"] is True
    assert kwargs["revision"] == REVISION
    assert "columns" not in kwargs


def test_sample_iteration(mock_dataset):
    """Stream should convert source records into GeoDESample objects."""

    stream = GeoDEStream(revision=REVISION)
    samples = list(stream)
    assert len(samples) == 2

    assert samples[0] == GeoDESample(
        image="image_1",
        label="car",
        region="Africa",
        country="Nigeria",
        source_index=0,
    )

    assert samples[1] == GeoDESample(
        image="image_2",
        label="dog",
        region="Europe",
        country="France",
        source_index=1,
    )


def test_metadata_only_iteration(mock_dataset):
    """Metadata-only samples should have image=None."""

    stream = GeoDEStream( revision=REVISION, metadata_only=True)
    samples = list(stream)

    assert len(samples) == 2
    assert all(sample.image is None for sample in samples)

    assert samples[0].label == "car"
    assert samples[0].region == "Africa"
    assert samples[0].country == "Nigeria"


def test_source_indices_are_sequential(mock_dataset):
    """Source indices should follow stream ordering."""

    stream = GeoDEStream(revision=REVISION)
    samples = list(stream)

    assert [sample.source_index for sample in samples] == [0, 1]


def test_get_revision(mock_dataset):
    """get_revision should return the pinned revision."""

    stream = GeoDEStream(revision=REVISION)
    assert stream.get_revision() == REVISION


def test_sample_is_immutable():
    """GeoDESample should be a frozen dataclass."""

    sample = GeoDESample(
        image=None,
        label="car",
        region="Africa",
        country="Nigeria",
        source_index=0,
    )

    with pytest.raises(AttributeError):
        sample.label = "dog"