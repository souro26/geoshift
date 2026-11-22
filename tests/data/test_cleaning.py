
import pandas as pd
import pytest

from geoshift.data.cleaning import clean_geode_metadata


@pytest.fixture
def raw_metadata():
    """Provide representative raw GeoDE metadata."""

    return pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3],
            "object": ["car", "dog", "chair", "truck"],
            "region": ["Africa", "Europe", "EastAsia", "WestAsia"],
            "ip_country": [
                "South_Africa",
                "United_Kingdom",
                "Japan",
                "United_Arab_Emirates",
            ],
        }
    )


def test_cleaning_produces_canonical_schema(raw_metadata):
    """Output should contain the expected canonical columns."""

    cleaned = clean_geode_metadata(raw_metadata)

    assert list(cleaned.columns) == [
        "source_index",
        "label",
        "region",
        "country",
        "raw_object",
        "raw_region",
        "raw_country",
    ]


def test_cleaning_renames_source_columns(raw_metadata):
    """Raw source columns should be mapped to canonical names."""

    cleaned = clean_geode_metadata(raw_metadata)

    assert "object" not in cleaned.columns
    assert "ip_country" not in cleaned.columns

    assert cleaned["label"].tolist() == [
        "car",
        "dog",
        "chair",
        "truck",
    ]

    assert cleaned["country"].tolist() == [
        "South Africa",
        "United Kingdom",
        "Japan",
        "United Arab Emirates",
    ]


def test_cleaning_normalizes_country_aliases(raw_metadata):
    """Known country aliases should be standardized."""

    cleaned = clean_geode_metadata(raw_metadata)

    assert cleaned.loc[0, "country"] == "South Africa"
    assert cleaned.loc[1, "country"] == "United Kingdom"
    assert cleaned.loc[3, "country"] == "United Arab Emirates"


def test_cleaning_preserves_raw_values(raw_metadata):
    """Original source values should remain available."""

    cleaned = clean_geode_metadata(raw_metadata)

    assert cleaned["raw_country"].tolist() == [
        "South_Africa",
        "United_Kingdom",
        "Japan",
        "United_Arab_Emirates",
    ]

    assert cleaned["raw_object"].tolist() == raw_metadata["object"].tolist()
    assert cleaned["raw_region"].tolist() == raw_metadata["region"].tolist()


def test_cleaning_preserves_source_indices(raw_metadata):
    """Cleaning should not modify source indices."""

    cleaned = clean_geode_metadata(raw_metadata)

    assert cleaned["source_index"].tolist() == [0, 1, 2, 3]


def test_cleaning_strips_surrounding_whitespace():
    """Text fields should have surrounding whitespace removed."""

    raw = pd.DataFrame(
        {
            "source_index": [0],
            "object": [" car "],
            "region": [" Africa "],
            "ip_country": [" South_Africa "],
        }
    )

    cleaned = clean_geode_metadata(raw)

    assert cleaned.loc[0, "label"] == "car"
    assert cleaned.loc[0, "region"] == "Africa"
    assert cleaned.loc[0, "country"] == "South Africa"

    # Raw values must remain unchanged.
    assert cleaned.loc[0, "raw_object"] == " car "
    assert cleaned.loc[0, "raw_region"] == " Africa "
    assert cleaned.loc[0, "raw_country"] == " South_Africa "


def test_cleaning_preserves_unknown_country_values():
    """Unknown country values should not be guessed or modified."""

    raw = pd.DataFrame(
        {
            "source_index": [0],
            "object": ["car"],
            "region": ["Africa"],
            "ip_country": ["Unknown_Country"],
        }
    )

    cleaned = clean_geode_metadata(raw)

    assert cleaned.loc[0, "country"] == "Unknown_Country"


def test_cleaning_does_not_modify_input(raw_metadata):
    """Cleaning should leave the original DataFrame unchanged."""

    original = raw_metadata.copy(deep=True)

    clean_geode_metadata(raw_metadata)

    pd.testing.assert_frame_equal(raw_metadata, original)


def test_cleaning_rejects_missing_columns():
    """Missing required source columns should raise ValueError."""

    raw = pd.DataFrame(
        {
            "source_index": [0],
            "object": ["car"],
            "region": ["Africa"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required raw columns",
    ):
        clean_geode_metadata(raw)


def test_cleaning_rejects_empty_dataframe():
    """An empty DataFrame should raise ValueError."""

    raw = pd.DataFrame(
        columns=[
            "source_index",
            "object",
            "region",
            "ip_country",
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        clean_geode_metadata(raw)
