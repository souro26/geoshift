from numbers import Integral

import numpy as np
import pandas as pd

from geoshift.config import load_config

REQUIRED_COLUMNS = { "source_index", "label", "region", "country", }
_SPLIT_CONFIG = load_config("data.yml")["splitting"]
DEFAULT_VALIDATION_FRACTION = _SPLIT_CONFIG["validation_fraction"]
DEFAULT_RANDOM_SEED = _SPLIT_CONFIG["seed"]
MANIFEST_SPLITS = _SPLIT_CONFIG["splits"]


def build_geographic_split(
    samples: pd.DataFrame,
    heldout_region: str,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Build a geographic holdout split from GeoDE metadata."""
    missing_columns = REQUIRED_COLUMNS - set(samples.columns)

    if missing_columns:
        raise ValueError( f"Missing required columns: {sorted(missing_columns)}" )
    if samples.empty:
        raise ValueError("Cannot split an empty dataset.")

    metadata = samples[list(REQUIRED_COLUMNS)]

    if metadata.isnull().any().any():
        raise ValueError("Dataset contains missing metadata values.")

    invalid_indices = samples["source_index"].map(
        lambda value: ( not isinstance(value, Integral) or isinstance(value, bool) )
    )

    if invalid_indices.any():
        raise ValueError( "source_index must contain integer values." )

    if samples["source_index"].duplicated().any():
        raise ValueError( "source_index values must be unique." )

    for column in ["label", "region", "country"]:
        invalid_values = samples[column].map( lambda value: not isinstance(value, str) or not value.strip() )

        if invalid_values.any():
            raise ValueError( f"Column '{column}' contains invalid text values." )

    if not isinstance(validation_fraction, (int, float, np.integer, np.floating)):
        raise ValueError( "validation_fraction must be a numeric value between 0 and 1." )

    if not 0 < validation_fraction < 1:
        raise ValueError( "validation_fraction must be between 0 and 1." )

    manifest = samples[ ["source_index", "label", "region", "country"] ].copy()
    manifest = manifest.reset_index(drop=True)
    available_regions = set(manifest["region"].unique())

    if heldout_region not in available_regions:
        raise ValueError( f"Unknown held-out region: {heldout_region}" )

    test_mask = manifest["region"] == heldout_region
    manifest["split"] = MANIFEST_SPLITS["train"]
    manifest.loc[test_mask, "split"] = MANIFEST_SPLITS["test"]
    training_pool = manifest.loc[~test_mask]
    rng = np.random.default_rng(seed)

    for _, group in training_pool.groupby( ["label", "region"], sort=True, ):
        group_size = len(group)

        if group_size < 2:
            raise ValueError( "Each class-region group needs at least two samples to create a validation split." )

        validation_size = max( 1, round(group_size * validation_fraction), )
        validation_size = min( validation_size, group_size - 1, )
        validation_indices = rng.choice( group.index.to_numpy(), size=validation_size, replace=False, )

        manifest.loc[validation_indices, "split"] = MANIFEST_SPLITS["validation"]

    split_classes = manifest.groupby("split")["label"].nunique()
    total_classes = manifest["label"].nunique()

    if (
        len(split_classes) != len(MANIFEST_SPLITS)
        or (split_classes != total_classes).any()
    ):
        raise ValueError( "Not every class is represented in all three splits." )

    return manifest
