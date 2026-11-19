import pandas as pd
from geoshift.data.splits import build_geographic_split

valid_splits = {"train", "validation", "test"}

manifest_columns = ["source_index","label","region","country","split"]

def validate_geographic_manifest(manifest: pd.DataFrame, heldout_region: str, expected_samples: pd.DataFrame) -> None:
    """Validate a generated geographic split manifest."""

    missing_columns = set(manifest_columns) - set(manifest.columns)

    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    if manifest.empty:
        raise ValueError("Manifest cannot be empty.")

    if manifest[manifest_columns].isnull().any().any():
        raise ValueError("Manifest contains missing values.")

    if manifest["source_index"].duplicated().any():
        raise ValueError("Manifest contains duplicate source indices.")

    if set(manifest["split"].unique()) != valid_splits:
        raise ValueError("Manifest must contain all three split types.")

    expected_indices = set(expected_samples["source_index"])
    actual_indices = set(manifest["source_index"])

    if actual_indices != expected_indices:
        raise ValueError("Manifest does not contain exactly the expected samples.")

    test_regions = set(manifest.loc[manifest["split"] == "test", "region"])

    if test_regions != {heldout_region}:
        raise ValueError("Test split contains an unexpected geographic region.")

    non_test_regions = set(manifest.loc[manifest["split"] != "test", "region"])

    if heldout_region in non_test_regions:
        raise ValueError("Held-out region appears outside the test split.")

    total_classes = set(expected_samples["label"])

    for split_name in valid_splits:
        split_classes = set(manifest.loc[manifest["split"] == split_name,"label"])

        if split_classes != total_classes:
            raise ValueError(f"Not all classes are represented in {split_name}.")


def build_geographic_manifests(samples: pd.DataFrame, validation_fraction: float = 0.15, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate and validate one manifest per held-out region."""

    if samples.empty:
        raise ValueError("Cannot generate manifests from empty data.")

    regions = sorted(samples["region"].unique())

    manifests = {}

    for heldout_region in regions:
        manifest = build_geographic_split(samples=samples, heldout_region=heldout_region, validation_fraction=validation_fraction, seed=seed)

        validate_geographic_manifest(manifest=manifest, heldout_region=heldout_region, expected_samples=samples)

        manifests[heldout_region] = manifest[manifest_columns].copy()

    return manifests
