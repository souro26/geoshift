
import json
from types import SimpleNamespace

import pytest

from geoshift.data.profiling import profile_geode

REVISION = "acbae31823b22f84c3bbb11585d37d50aabaac44"


@pytest.fixture
def mock_samples():
    """Provide representative GeoDE metadata samples."""

    return [
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
        SimpleNamespace(
            source_index=2,
            label="car",
            region="Africa",
            country="Kenya",
        ),
        SimpleNamespace(
            source_index=3,
            label="dog",
            region="Africa",
            country="Nigeria",
        ),
    ]


def test_profile_generates_final_report(tmp_path, mock_samples):
    """Profiling should generate a completed JSON report."""

    report = profile_geode(
        samples=mock_samples,
        dataset_id="MLap/GeoDE",
        split="train",
        revision=REVISION,
        output_dir=tmp_path,
    )

    final_path = tmp_path / "geode_profile.json"

    assert final_path.exists()
    assert report["complete"] is True
    assert report["total_samples"] == 4

    with final_path.open(encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report


def test_profile_counts_classes_and_regions(tmp_path, mock_samples):
    """Profiling should calculate correct class and region counts."""

    report = profile_geode(
        mock_samples,
        "MLap/GeoDE",
        "train",
        REVISION,
        output_dir=tmp_path,
    )

    assert report["class_counts"] == {
        "car": 2,
        "dog": 2,
    }

    assert report["region_counts"] == {
        "Africa": 3,
        "Europe": 1,
    }

    assert report["country_counts"] == {
        "France": 1,
        "Kenya": 1,
        "Nigeria": 2,
    }


def test_profile_counts_class_region_combinations(
    tmp_path,
    mock_samples,
):
    """Profiling should calculate class-region combinations."""

    report = profile_geode(
        mock_samples,
        "MLap/GeoDE",
        "train",
        REVISION,
        output_dir=tmp_path,
    )

    assert report["class_region_counts"] == {
        "car|Africa": 2,
        "dog|Africa": 1,
        "dog|Europe": 1,
    }


def test_profile_records_provenance(tmp_path, mock_samples):
    """Report should contain dataset provenance information."""

    report = profile_geode(
        mock_samples,
        "MLap/GeoDE",
        "train",
        REVISION,
        output_dir=tmp_path,
    )

    assert report["dataset_id"] == "MLap/GeoDE"
    assert report["split"] == "train"
    assert report["revision"] == REVISION
    assert report["number_of_classes"] == 2
    assert report["number_of_regions"] == 2
    assert report["number_of_countries"] == 3
    assert report["generated_at_utc"]

    assert report["elapsed_seconds"] >= 0
    assert report["samples_per_second"] >= 0


def test_profile_creates_checkpoint(
    tmp_path,
    mock_samples,
):
    """Profiling should save checkpoints at the configured interval."""

    report = profile_geode(
        mock_samples,
        "MLap/GeoDE",
        "train",
        REVISION,
        output_dir=tmp_path,
        checkpoint_interval=2,
    )

    checkpoint_path = (
        tmp_path / "geode_profile_checkpoint.json"
    )

    # Successful profiling removes the last checkpoint.
    assert not checkpoint_path.exists()

    assert report["total_samples"] == 4


def test_profile_removes_checkpoint_after_completion(
    tmp_path,
    mock_samples,
):
    """Successful profiling should remove stale checkpoint files."""

    checkpoint_path = (
        tmp_path / "geode_profile_checkpoint.json"
    )

    checkpoint_path.write_text(
        '{"complete": false}',
        encoding="utf-8",
    )

    profile_geode(
        mock_samples,
        "MLap/GeoDE",
        "train",
        REVISION,
        output_dir=tmp_path,
    )

    assert not checkpoint_path.exists()
    assert (tmp_path / "geode_profile.json").exists()


def test_profile_rejects_empty_dataset(tmp_path):
    """Profiling should reject an empty stream."""

    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        profile_geode(
            [],
            "MLap/GeoDE",
            "train",
            REVISION,
            output_dir=tmp_path,
        )

    assert not (tmp_path / "geode_profile.json").exists()


def test_profile_rejects_invalid_checkpoint_interval(
    tmp_path,
    mock_samples,
):
    """Checkpoint interval must be a positive integer."""

    with pytest.raises(
        ValueError,
        match="checkpoint_interval",
    ):
        profile_geode(
            mock_samples,
            "MLap/GeoDE",
            "train",
            REVISION,
            output_dir=tmp_path,
            checkpoint_interval=0,
        )


def test_profile_accepts_generator_input(tmp_path, mock_samples):
    """Profiling should process iterable streams without materializing them."""

    def sample_generator():
        yield from mock_samples

    report = profile_geode(
        sample_generator(),
        "MLap/GeoDE",
        "train",
        REVISION,
        output_dir=tmp_path,
    )

    assert report["total_samples"] == 4
