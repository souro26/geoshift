from collections.abc import Callable

import pandas as pd
from torch.utils.data import IterableDataset, get_worker_info

from geoshift.data.geode import GeoDEStream


REQUIRED_COLUMNS = {
    "source_index",
    "label",
    "region",
    "country",
}


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

        if manifest[list(REQUIRED_COLUMNS)].isnull().any().any():
            raise ValueError("Manifest contains missing metadata values.")

        if manifest["source_index"].duplicated().any():
            raise ValueError("Manifest contains duplicate source indices.")

        if "split" not in manifest.columns:
            raise ValueError("Manifest must contain a split column.")

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split name: {split}")

        if not revision:
            raise ValueError("A pinned GeoDE revision is required.")

        self.manifest = manifest.copy()
        self.split = split
        self.revision = revision
        self.transform = transform

        self.selected_samples = self.manifest.loc[
            self.manifest["split"] == split
        ].copy()

        if self.selected_samples.empty:
            raise ValueError(
                f"No samples found for split: {split}"
            )

        self.expected_samples = {
            int(row.source_index): (
                row.label,
                row.region,
                row.country,
            )
            for row in self.selected_samples.itertuples(index=False)
        }

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

            # Access the image only after confirming this sample is selected.
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
            missing_count = len(pending_indices)

            raise RuntimeError(
                f"Stream ended before all selected samples were found. "
                f"Missing {missing_count} samples. "
                "Check the dataset revision and source indices."
            )
