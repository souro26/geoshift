import numpy as np
import pandas as pd

required_columns = {"source_index","label","region","country"}

validation_fraction = 0.15

def build_geographic_split(samples: pd.DataFrame, heldout_region: str, validation_fraction: float = validation_fraction, seed: int = 42):
    """Build a geographic holdout split from GeoDE metadata."""
    missing_columns = required_columns - set(samples.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if samples.empty:
        raise ValueError("Cannot split an empty dataset.")

    if samples[list(required_columns)].isnull().any().any():
        raise ValueError("Dataset contains missing metadata values.")

    if samples["source_index"].duplicated().any():
        raise ValueError("source_index values must be unique.")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    manifest = samples[
        ["source_index", "label", "region", "country"]
    ].copy()

    manifest = manifest.reset_index(drop=True)

    available_regions = set(manifest["region"].unique())

    if heldout_region not in available_regions:
        raise ValueError(
            f"Unknown held-out region: {heldout_region}"
        )

    test_mask = manifest["region"] == heldout_region

    if not test_mask.any():
        raise ValueError(
            f"No samples found for region: {heldout_region}"
        )

    manifest["split"] = "train"
    manifest.loc[test_mask, "split"] = "test"

    training_pool = manifest.loc[~test_mask]

    rng = np.random.default_rng(seed)

    for _, group in training_pool.groupby(
        ["label", "region"],
        sort=True,
    ):
        group_size = len(group)

        if group_size < 2:
            raise ValueError(
                "Each class-region group needs at least two samples "
                "to create a validation split."
            )

        validation_size = max(
            1,
            round(group_size * validation_fraction),
        )

        validation_size = min(
            validation_size,
            group_size - 1,
        )

        validation_indices = rng.choice(
            group.index.to_numpy(),
            size=validation_size,
            replace=False,
        )

        manifest.loc[validation_indices, "split"] = "validation"

    # Every class must be represented in all three splits.
    split_classes = manifest.groupby("split")["label"].nunique()
    total_classes = manifest["label"].nunique()

    if len(split_classes) != 3 or (split_classes != total_classes).any():
        raise ValueError(
            "Not every class is represented in all three splits."
        )

    return manifest
