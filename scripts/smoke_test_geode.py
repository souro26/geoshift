
"""Small real-data integration test for the GeoDE pipeline."""

import pandas as pd
import torch

from geoshift.config import load_config
from geoshift.data.dataset import GeoDEIterableDataset
from geoshift.data.geode import GeoDEStream
from geoshift.data.transforms import build_eval_transform

_DATA_CONFIG = load_config("data.yml")
NUM_SAMPLES = _DATA_CONFIG["smoke_test"]["samples"]
IMAGE_SIZE = _DATA_CONFIG["image"]["size"]


def main() -> None:
    print("GeoShift | GeoDE Integration Smoke Test")
    print("\nStep 1: Loading metadata...")

    metadata_stream = GeoDEStream(
        split=_DATA_CONFIG["dataset"]["split"],
        metadata_only=True,
    )
    records = []
    for sample in metadata_stream:
        records.append(
            {
                "source_index": sample.source_index,
                "label": sample.label,
                "region": sample.region,
                "country": sample.country,
                "split": _DATA_CONFIG["splitting"]["splits"]["test"],
            }
        )

        if len(records) >= NUM_SAMPLES:
            break

    if len(records) != NUM_SAMPLES:
        raise RuntimeError( f"Expected {NUM_SAMPLES} metadata records, " f"received {len(records)}." )

    manifest = pd.DataFrame.from_records(records)
    revision = metadata_stream.get_revision()

    print(f"Dataset revision: {revision}")
    print(f"Metadata records: {len(manifest)}")

    print("\nStep 2: Initializing image dataset...")
    dataset = GeoDEIterableDataset(
        manifest=manifest,
        split=_DATA_CONFIG["splitting"]["splits"]["test"],
        revision=revision,
        transform=build_eval_transform(),
    )
    print(f"Selected samples: {len(dataset)}")
    print("\nStep 3: Retrieving and preprocessing images...\n")
    samples = list(dataset)

    if len(samples) != NUM_SAMPLES:
        raise RuntimeError( f"Expected {NUM_SAMPLES} images, received {len(samples)}." )

    for sample in samples:
        image = sample["image"]

        if not isinstance(image, torch.Tensor):
            raise TypeError("Transformed image is not a PyTorch tensor.")

        if image.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError( f"Unexpected image shape: {tuple(image.shape)}" )

        if image.dtype != torch.float32:
            raise TypeError( f"Unexpected image dtype: {image.dtype}" )

        if not torch.isfinite(image).all():
            raise ValueError("Image tensor contains non-finite values.")

        print(
            f"Index: {sample['source_index']} | "
            f"Class: {sample['label_name']} | "
            f"Region: {sample['region']} | "
            f"Country: {sample['country']} | "
            f"Shape: {tuple(image.shape)}"
        )

    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
