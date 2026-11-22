
from collections.abc import Callable
from numbers import Integral

import pandas as pd
from torch.utils.data import IterableDataset, get_worker_info

from geoshift.data.geode import GeoDEStream


REQUIRED_COLUMNS = {
    "source_index",
    "label",
    "region",
    "country",
    "split",
}

VALID_SPLITS = {"train", "validation", "test"}


class GeoDEIterableDataset(IterableDataset):
    """Stream selected GeoDE samples using a geographic manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        revision: str,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        missing_columns = REQUIRED_COLUMNS - set(manifest.columns)

        if missing_columns:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing_columns)}"
            )

        if manifest.empty:
            raise ValueError("Manifest cannot be empty.")

        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split name: {split}")

        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("A pinned GeoDE revision is required.")

        metadata_columns = [
            "source_index",
            "label",
            "region",
            "country",
            "split",
        ]

        if manifest[metadata_columns].isnull().any().any():
            raise ValueError("Manifest contains missing metadata values.")

        # Validate source indices before using them as dictionary keys.
        invalid_indices = manifest["source_index"].map(
            lambda value: (
                not isinstance(value, Integral)
                or isinstance(value, bool)
            )
        )

        if invalid_indices.any():
            raise ValueError(
                "Manifest source_index values must be integers."
            )

        if manifest["source_index"].duplicated().any():
            raise ValueError(
                "Manifest contains duplicate source indices."
            )

        # Validate metadata text fields.
        for column in ["label", "region", "country"]:
            invalid_values = manifest[column].map(
                lambda value: (
                    not isinstance(value, str)
                    or not value.strip()
                )
            )

            if invalid_values.any():
                raise ValueError(
                    f"Manifest column '{column}' contains invalid text values."
                )

        # Reject unknown split assignments.
        observed_splits = set(manifest["split"].unique())
        invalid_splits = observed_splits - VALID_SPLITS

        if invalid_splits:
            raise ValueError(
                f"Manifest contains invalid split assignments: "
                f"{sorted(invalid_splits)}"
            )

        self.manifest = manifest.copy()
        self.split = split
        self.revision = revision
        self.transform = transform

        self.selected_samples = self.manifest.loc[
            self.manifest["split"] == split
        ].copy()

        if self.selected_samples.empty:
            raise ValueError(f"No samples found for split: {split}")

        self.expected_samples = {
            row.source_index: (
                row.label,
                row.region,
                row.country,
            )
            for row in self.selected_samples.itertuples(index=False)
        }

        # Build the mapping from the entire manifest so that class indices
        # remain consistent across train, validation, and test datasets.
        self.class_to_idx = {
            label: index
            for index, label in enumerate(
                sorted(self.manifest["label"].unique())
            )
        }

    def __len__(self) -> int:
        """Return the number of selected samples."""

        return len(self.selected_samples)

    def __iter__(self):
        """Stream and yield only samples assigned to this split."""

        worker_info = get_worker_info()

        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError(
                "Multiple DataLoader workers are not supported yet. "
                "Use num_workers=0 until stream sharding is implemented."
            )

        stream = GeoDEStream(
            split="train",
            revision=self.revision,
            metadata_only=False,
        )

        pending_indices = set(self.expected_samples)

        for source_index, sample in enumerate(stream.dataset):
            if source_index not in pending_indices:
                continue

            expected_metadata = self.expected_samples[source_index]

            actual_metadata = (
                sample["object"],
                sample["region"],
                sample["ip_country"],
            )

            if actual_metadata != expected_metadata:
                raise RuntimeError(
                    f"Metadata mismatch at source index {source_index}. "
                    "Check the dataset revision and manifest."
                )

            image = sample["image"]

            if image is None:
                raise RuntimeError(
                    f"Missing image at source index {source_index}."
                )

            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            label = expected_metadata[0]

            yield {
                "image": image,
                "label": self.class_to_idx[label],
                "label_name": label,
                "region": expected_metadata[1],
                "country": expected_metadata[2],
                "source_index": source_index,
            }

            pending_indices.remove(source_index)

            if not pending_indices:
                break

        if pending_indices:
            raise RuntimeError(
                "Stream ended before all selected samples were found. "
                f"Missing {len(pending_indices)} samples. "
                "Check the dataset revision and source indices."
            )
