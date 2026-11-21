
import pandas as pd

from geoshift.data.splits import build_geographic_split


VALID_SPLITS = {"train", "validation", "test"}

MANIFEST_COLUMNS = [
    "source_index",
    "label",
    "region",
    "country",
    "split",
]

SOURCE_COLUMNS = [
    "source_index",
    "label",
    "region",
    "country",
]


def validate_geographic_manifest(
    manifest: pd.DataFrame,
    heldout_region: str,
    expected_samples: pd.DataFrame,
) -> None:
    """Validate a generated geographic split manifest."""

    missing_manifest_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)

    if missing_manifest_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_manifest_columns)}"
        )

    missing_expected_columns = set(SOURCE_COLUMNS) - set(expected_samples.columns)

    if missing_expected_columns:
        raise ValueError(
            "Expected samples are missing columns: "
            f"{sorted(missing_expected_columns)}"
        )

    if manifest.empty:
        raise ValueError("Manifest cannot be empty.")

    if expected_samples.empty:
        raise ValueError("Expected samples cannot be empty.")

    if manifest[MANIFEST_COLUMNS].isnull().any().any():
        raise ValueError("Manifest contains missing values.")

    if expected_samples[SOURCE_COLUMNS].isnull().any().any():
        raise ValueError("Expected samples contain missing values.")

    if manifest["source_index"].duplicated().any():
        raise ValueError("Manifest contains duplicate source indices.")

    if expected_samples["source_index"].duplicated().any():
        raise ValueError("Expected samples contain duplicate source indices.")

    if set(manifest["split"].unique()) != VALID_SPLITS:
        raise ValueError("Manifest must contain all three split types.")

    expected = expected_samples[SOURCE_COLUMNS].set_index("source_index")
    actual = manifest[SOURCE_COLUMNS].set_index("source_index")

    if set(actual.index) != set(expected.index):
        raise ValueError(
            "Manifest does not contain exactly the expected samples."
        )

    # Compare metadata after aligning rows by source_index.
    actual = actual.sort_index()
    expected = expected.sort_index()

    if not actual.equals(expected):
        raise ValueError(
            "Manifest metadata does not match the expected source samples."
        )

    test_regions = set(
        manifest.loc[manifest["split"] == "test", "region"]
    )

    if test_regions != {heldout_region}:
        raise ValueError(
            "Test split contains an unexpected geographic region."
        )

    non_test_regions = set(
        manifest.loc[manifest["split"] != "test", "region"]
    )

    if heldout_region in non_test_regions:
        raise ValueError(
            "Held-out region appears outside the test split."
        )

    total_classes = set(expected_samples["label"])

    for split_name in VALID_SPLITS:
        split_classes = set(
            manifest.loc[manifest["split"] == split_name, "label"]
        )

        if split_classes != total_classes:
            raise ValueError(
                f"Not all classes are represented in {split_name}."
            )


def build_geographic_manifests(
    samples: pd.DataFrame,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate and validate one manifest per held-out region."""

    missing_columns = set(SOURCE_COLUMNS) - set(samples.columns)

    if missing_columns:
        raise ValueError(
            f"Samples are missing required columns: {sorted(missing_columns)}"
        )

    if samples.empty:
        raise ValueError("Cannot generate manifests from empty data.")

    if samples[SOURCE_COLUMNS].isnull().any().any():
        raise ValueError("Samples contain missing metadata values.")

    if samples["source_index"].duplicated().any():
        raise ValueError("Samples contain duplicate source indices.")

    regions = sorted(samples["region"].unique())

    manifests = {}

    for heldout_region in regions:
        manifest = build_geographic_split(
            samples=samples,
            heldout_region=heldout_region,
            validation_fraction=validation_fraction,
            seed=seed,
        )

        validate_geographic_manifest(
            manifest=manifest,
            heldout_region=heldout_region,
            expected_samples=samples,
        )

        manifests[heldout_region] = manifest[MANIFEST_COLUMNS].copy()

    return manifests
