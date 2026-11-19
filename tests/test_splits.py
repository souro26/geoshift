
"""Tests for geographic train, validation, and test splitting."""

import pandas as pd
import pytest

from geoshift.data.splits import build_geographic_split


@pytest.fixture
def sample_metadata() -> pd.DataFrame:
    """Create synthetic metadata for split testing."""

    records = []

    for label in ["car", "dog"]:
        for region in ["Africa", "Europe", "Asia"]:
            for index in range(20):
                records.append(
                    {
                        "source_index": len(records),
                        "label": label,
                        "region": region,
                        "country": (
                            "Nigeria"
                            if region == "Africa"
                            else "France"
                            if region == "Europe"
                            else "Japan"
                        ),
                    }
                )

    return pd.DataFrame(records)


def test_all_three_splits_are_created(sample_metadata):
    result = build_geographic_split(
        sample_metadata,
        heldout_region="Africa",
    )

    assert set(result["split"]) == {
        "train",
        "validation",
        "test",
    }


def test_heldout_region_only_appears_in_test(sample_metadata):
    result = build_geographic_split(
        sample_metadata,
        heldout_region="Africa",
    )

    test_regions = set(
        result.loc[result["split"] == "test", "region"]
    )

    non_test_regions = set(
        result.loc[result["split"] != "test", "region"]
    )

    assert test_regions == {"Africa"}
    assert "Africa" not in non_test_regions


def test_every_class_appears_in_all_splits(sample_metadata):
    result = build_geographic_split(
        sample_metadata,
        heldout_region="Africa",
    )

    expected_classes = set(sample_metadata["label"])

    for split_name in ["train", "validation", "test"]:
        actual_classes = set(
            result.loc[result["split"] == split_name, "label"]
        )

        assert actual_classes == expected_classes


def test_no_source_sample_is_lost_or_duplicated(sample_metadata):
    result = build_geographic_split(
        sample_metadata,
        heldout_region="Africa",
    )

    assert len(result) == len(sample_metadata)

    assert result["source_index"].is_unique

    assert set(result["source_index"]) == set(
        sample_metadata["source_index"]
    )


def test_split_is_reproducible(sample_metadata):
    first = build_geographic_split(
        sample_metadata,
        heldout_region="Africa",
        seed=42,
    )

    second = build_geographic_split(
        sample_metadata,
        heldout_region="Africa",
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_can_change_validation_assignments(
    sample_metadata,
):
    first = build_geographic_split(
        sample_metadata,
        heldout_region="Africa",
        seed=42,
    )

    second = build_geographic_split(
        sample_metadata,
        heldout_region="Africa",
        seed=123,
    )

    first_validation = set(
        first.loc[
            first["split"] == "validation",
            "source_index",
        ]
    )

    second_validation = set(
        second.loc[
            second["split"] == "validation",
            "source_index",
        ]
    )

    assert first_validation != second_validation


def test_invalid_validation_fraction_raises_error(sample_metadata):
    with pytest.raises(ValueError):
        build_geographic_split(
            sample_metadata,
            heldout_region="Africa",
            validation_fraction=1.0,
        )


def test_unknown_region_raises_error(sample_metadata):
    with pytest.raises(ValueError):
        build_geographic_split(
            sample_metadata,
            heldout_region="Antarctica",
        )


def test_empty_dataframe_raises_error():
    empty_metadata = pd.DataFrame(
        columns=[
            "source_index",
            "label",
            "region",
            "country",
        ]
    )

    with pytest.raises(ValueError):
        build_geographic_split(
            empty_metadata,
            heldout_region="Africa",
        )


def test_duplicate_source_indices_raise_error(sample_metadata):
    duplicated = pd.concat(
        [sample_metadata, sample_metadata.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        build_geographic_split(
            duplicated,
            heldout_region="Africa",
        )


def test_missing_metadata_raises_error(sample_metadata):
    invalid_metadata = sample_metadata.copy()
    invalid_metadata.loc[0, "country"] = None

    with pytest.raises(ValueError):
        build_geographic_split(
            invalid_metadata,
            heldout_region="Africa",
        )
