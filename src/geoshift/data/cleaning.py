
import pandas as pd

from geoshift.config import load_config

COUNTRY_ALIASES = load_config("data.yml")["cleaning"]["country_aliases"]


RAW_REQUIRED_COLUMNS = {
    "source_index",
    "object",
    "region",
    "ip_country",
}

CANONICAL_COLUMNS = [
    "source_index",
    "label",
    "region",
    "country",
    "raw_object",
    "raw_region",
    "raw_country",
]


def _normalize_text(value):
    """Strip surrounding whitespace from string values."""

    if isinstance(value, str):
        return value.strip()

    return value


def _normalize_country(value):
    """Normalize known country-name aliases."""

    value = _normalize_text(value)

    if not isinstance(value, str):
        return value

    return COUNTRY_ALIASES.get(value, value)


def clean_geode_metadata(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw GeoDE metadata into a canonical representation."""

    missing_columns = RAW_REQUIRED_COLUMNS - set(raw_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required raw columns: {sorted(missing_columns)}"
        )

    if raw_data.empty:
        raise ValueError("Cannot clean an empty dataset.")

    cleaned = raw_data.copy()

    # Preserve original source values for traceability.
    cleaned["raw_object"] = cleaned["object"]
    cleaned["raw_region"] = cleaned["region"]
    cleaned["raw_country"] = cleaned["ip_country"]

    # Convert source column names to canonical names.
    cleaned = cleaned.rename(
        columns={
            "object": "label",
            "ip_country": "country",
        }
    )

    # Normalize surrounding whitespace.
    cleaned["label"] = cleaned["label"].map(_normalize_text)
    cleaned["region"] = cleaned["region"].map(_normalize_text)

    # Normalize country aliases.
    cleaned["country"] = cleaned["country"].map(_normalize_country)

    return cleaned[CANONICAL_COLUMNS].copy()
