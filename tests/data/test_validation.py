
import pandas as pd
import pytest

from geoshift.data.validation import validate_geode_metadata


@pytest.fixture
def valid_metadata():
    """Provide valid canonical GeoDE metadata."""

    return pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3, 4, 5],
            "label": ["car", "dog", "chair", "truck", "car", "dog"],
            "region": [
                "Africa",
                "Europe",
                "EastAsia",
                "WestAsia",
                "Americas",
                "SouthEastAsia",
            ],
            "country": [
                "Nigeria",
                "France",
                "Japan",
                "Turkey",
                "Brazil",
                "Thailand",
            ],
        }
    )


def test_valid_metadata_passes_validation(valid_metadata):
    """Valid metadata should return a successful validation summary."""

    result = validate_geode_metadata(valid_metadata)

    assert result["valid"] is True
    assert result["total_samples"] == 6
    assert result["number_of_classes"] == 4
    assert result["number_of_regions"] == 6
    assert result["number_of_countries"] == 6
    assert result["source_indices_unique"] is True
    assert result["missing_values"] is False


def test_validation_returns_observed_regions(valid_metadata):
    """Validation summary should contain observed regions."""

    result = validate_geode_metadata(valid_metadata)

    assert result["regions"] == [
        "Africa",
        "Americas",
        "EastAsia",
        "Europe",
        "SouthEastAsia",
        "WestAsia",
    ]


def test_validation_rejects_missing_columns():
    """Missing required columns should raise ValueError."""

    samples = pd.DataFrame(
        {
            "source_index": [0],
            "label": ["car"],
            "region": ["Africa"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_geode_metadata(samples)


def test_validation_rejects_empty_dataset():
    """An empty DataFrame should raise ValueError."""

    samples = pd.DataFrame(
        columns=[
            "source_index",
            "label",
            "region",
            "country",
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        validate_geode_metadata(samples)


@pytest.mark.parametrize( "column", ["source_index", "label", "region", "country"], )
def test_validation_rejects_missing_values( valid_metadata, column, ):
    """Missing metadata values should raise ValueError."""

    valid_metadata.loc[0, column] = None

    with pytest.raises(
        ValueError,
        match="missing metadata values",
    ):
        validate_geode_metadata(valid_metadata)


def test_validation_rejects_duplicate_source_indices( valid_metadata, ):
    """Duplicate source indices should raise ValueError."""

    valid_metadata.loc[1, "source_index"] = 0

    with pytest.raises(
        ValueError,
        match="duplicate source_index",
    ):
        validate_geode_metadata(valid_metadata)


@pytest.mark.parametrize( "invalid_index", ["zero", 1.5, True], )
def test_validation_rejects_non_integer_source_indices( valid_metadata, invalid_index, ):
    """Source indices must contain integer values."""

    valid_metadata["source_index"] = valid_metadata[
        "source_index"
    ].astype(object)

    valid_metadata.loc[0, "source_index"] = invalid_index

    with pytest.raises(
        ValueError,
        match="integer values",
    ):
        validate_geode_metadata(valid_metadata)


@pytest.mark.parametrize( "column", ["label", "region", "country"], )
def test_validation_rejects_empty_text_values( valid_metadata, column, ):
    """Empty or whitespace-only text should be rejected."""

    valid_metadata.loc[0, column] = "   "

    with pytest.raises(
        ValueError,
        match="invalid text values",
    ):
        validate_geode_metadata(valid_metadata)


def test_validation_rejects_non_string_metadata( valid_metadata, ):
    """Non-string labels should be rejected."""

    valid_metadata["label"] = valid_metadata[
        "label"
    ].astype(object)

    valid_metadata.loc[0, "label"] = 123

    with pytest.raises(
        ValueError,
        match="invalid text values",
    ):
        validate_geode_metadata(valid_metadata)


def test_validation_rejects_unknown_regions( valid_metadata, ):
    """Regions outside the supported GeoDE regions should be rejected."""

    valid_metadata.loc[0, "region"] = "UnknownRegion"

    with pytest.raises(
        ValueError,
        match="Unknown geographic regions",
    ):
        validate_geode_metadata(valid_metadata)


def test_validation_does_not_modify_input(valid_metadata):
    """Validation should not modify the original DataFrame."""

    original = valid_metadata.copy(deep=True)

    validate_geode_metadata(valid_metadata)

    pd.testing.assert_frame_equal(
        valid_metadata,
        original,
    )
