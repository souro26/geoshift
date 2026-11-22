
"""Tests for geographic manifest generation and validation."""

import pandas as pd
import pytest

from geoshift.data.manifest import (
    build_geographic_manifests,
    validate_geographic_manifest,
)


@pytest.fixture
def sample_metadata() -> pd.DataFrame:
    """Create synthetic GeoDE metadata for testing."""

    records = []

    for label in ["car", "dog"]:
        for region in ["Africa", "Europe", "EastAsia"]:
            for _ in range(20):
                country = {
                    "Africa": "Nigeria",
                    "Europe": "France",
                    "EastAsia": "Japan",
                }[region]

                records.append(
                    {
                        "source_index": len(records),
                        "label": label,
                        "region": region,
                        "country": country,
                    }
                )

    return pd.DataFrame(records)


def test_generates_manifest_for_every_region(sample_metadata):
    manifests = build_geographic_manifests(sample_metadata)

    assert set(manifests) == {"Africa", "Europe", "EastAsia"}


def test_every_manifest_contains_all_source_samples(sample_metadata):
    manifests = build_geographic_manifests(sample_metadata)

    expected_indices = set(sample_metadata["source_index"])

    for manifest in manifests.values():
        assert set(manifest["source_index"]) == expected_indices


def test_every_manifest_passes_validation(sample_metadata):
    manifests = build_geographic_manifests(sample_metadata)

    for region, manifest in manifests.items():
        validate_geographic_manifest(
            manifest=manifest,
            heldout_region=region,
            expected_samples=sample_metadata,
        )


def test_heldout_region_is_exclusive_to_test(sample_metadata):
    manifests = build_geographic_manifests(sample_metadata)

    for region, manifest in manifests.items():
        test_regions = set(
            manifest.loc[manifest["split"] == "test", "region"]
        )
        non_test_regions = set(
            manifest.loc[manifest["split"] != "test", "region"]
        )

        assert test_regions == {region}
        assert region not in non_test_regions


def test_manifest_rejects_missing_samples(sample_metadata):
    manifests = build_geographic_manifests(sample_metadata)
    manifest = manifests["Africa"].iloc[1:].copy()

    with pytest.raises(ValueError):
        validate_geographic_manifest(
            manifest=manifest,
            heldout_region="Africa",
            expected_samples=sample_metadata,
        )


def test_manifest_rejects_duplicate_samples(sample_metadata):
    manifests = build_geographic_manifests(sample_metadata)
    manifest = manifests["Africa"].copy()

    manifest.loc[1, "source_index"] = manifest.loc[0, "source_index"]

    with pytest.raises(ValueError, match="duplicate"):
        validate_geographic_manifest(
            manifest=manifest,
            heldout_region="Africa",
            expected_samples=sample_metadata,
        )


def test_manifest_rejects_incorrect_heldout_region(sample_metadata):
    manifests = build_geographic_manifests(sample_metadata)
    manifest = manifests["Africa"].copy()

    test_index = manifest.index[manifest["split"] == "test"][0]
    manifest.loc[test_index, "region"] = "Europe"

    with pytest.raises(ValueError):
        validate_geographic_manifest(
            manifest=manifest,
            heldout_region="Africa",
            expected_samples=sample_metadata,
        )


def test_empty_dataset_raises_error():
    empty_metadata = pd.DataFrame(
        columns=["source_index", "label", "region", "country"]
    )

    with pytest.raises(ValueError):
        build_geographic_manifests(empty_metadata)


@pytest.mark.parametrize(
    "column, incorrect_value",
    [
        ("label", "truck"),
        ("region", "Europe"),
        ("country", "Brazil"),
    ],
)
def test_manifest_rejects_modified_source_metadata(
    sample_metadata,
    column,
    incorrect_value,
):
    manifests = build_geographic_manifests(sample_metadata)
    manifest = manifests["Africa"].copy()

    row_index = manifest.index[0]
    manifest.loc[row_index, column] = incorrect_value

    with pytest.raises(ValueError, match="metadata"):
        validate_geographic_manifest(
            manifest=manifest,
            heldout_region="Africa",
            expected_samples=sample_metadata,
        )


def test_manifest_rejects_expected_samples_with_duplicate_indices(
    sample_metadata,
):
    manifests = build_geographic_manifests(sample_metadata)
    manifest = manifests["Africa"].copy()

    expected = sample_metadata.copy()
    expected.loc[1, "source_index"] = expected.loc[0, "source_index"]

    with pytest.raises(ValueError, match="duplicate"):
        validate_geographic_manifest(
            manifest=manifest,
            heldout_region="Africa",
            expected_samples=expected,
        )


def test_manifest_rejects_expected_samples_missing_required_columns(
    sample_metadata,
):
    manifests = build_geographic_manifests(sample_metadata)
    manifest = manifests["Africa"].copy()

    expected = sample_metadata.drop(columns=["country"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_geographic_manifest(
            manifest=manifest,
            heldout_region="Africa",
            expected_samples=expected,
        )


@pytest.mark.parametrize("invalid_index", [True, 1.5, "zero"])
def test_manifest_rejects_non_integer_source_indices(
    sample_metadata,
    invalid_index,
):
    manifests = build_geographic_manifests(sample_metadata)
    manifest = manifests["Africa"].copy()

    manifest["source_index"] = manifest["source_index"].astype(object)
    manifest.loc[0, "source_index"] = invalid_index

    with pytest.raises(ValueError, match="integers"):
        validate_geographic_manifest(
            manifest=manifest,
            heldout_region="Africa",
            expected_samples=sample_metadata,
        )


def test_build_rejects_non_integer_source_indices(sample_metadata):
    malformed = sample_metadata.copy()
    malformed["source_index"] = malformed["source_index"].astype(object)
    malformed.loc[0, "source_index"] = True

    with pytest.raises(ValueError, match="integers"):
        build_geographic_manifests(malformed)
