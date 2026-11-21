
from numbers import Integral

import pandas as pd


REQUIRED_COLUMNS = {
    "source_index",
    "label",
    "region",
    "country",
}

VALID_REGIONS = {
    "Africa",
    "Americas",
    "EastAsia",
    "Europe",
    "SouthEastAsia",
    "WestAsia",
}


def validate_geode_metadata(samples: pd.DataFrame) -> dict:
    """Validate the structure and contents of cleaned GeoDE metadata."""

    missing_columns = REQUIRED_COLUMNS - set(samples.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if samples.empty:
        raise ValueError("Cannot validate an empty dataset.")

    metadata = samples[list(REQUIRED_COLUMNS)]

    if metadata.isnull().any().any():
        missing_counts = metadata.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()

        raise ValueError(
            f"Dataset contains missing metadata values: {missing_counts}"
        )

    invalid_indices = samples["source_index"].map(
        lambda value: (
            not isinstance(value, Integral)
            or isinstance(value, bool)
        )
    )

    if invalid_indices.any():
        raise ValueError(
            "source_index must contain integer values."
        )

    if samples["source_index"].duplicated().any():
        duplicate_count = int(
            samples["source_index"].duplicated().sum()
        )

        raise ValueError(
            f"Found {duplicate_count} duplicate source_index values."
        )

    for column in ["label", "region", "country"]:
        invalid_values = samples[column].map(
            lambda value: not isinstance(value, str)
            or not value.strip()
        )

        if invalid_values.any():
            raise ValueError(
                f"Column '{column}' contains invalid text values."
            )

    observed_regions = set(samples["region"].unique())

    invalid_regions = observed_regions - VALID_REGIONS

    if invalid_regions:
        raise ValueError(
            f"Unknown geographic regions: {sorted(invalid_regions)}"
        )

    return {
        "valid": True,
        "total_samples": len(samples),
        "number_of_classes": int(samples["label"].nunique()),
        "number_of_regions": int(samples["region"].nunique()),
        "number_of_countries": int(samples["country"].nunique()),
        "regions": sorted(observed_regions),
        "source_indices_unique": True,
        "missing_values": False,
    }
