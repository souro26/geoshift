
from geoshift.data.geode import GeoDEStream


def main() -> None:
    print("=" * 60)
    print("GeoShift | GeoDE Dataset Inspection")
    print("=" * 60)

    print("\nInitializing metadata-only stream...")

    dataset = GeoDEStream(
        split="train",
        metadata_only=True,
    )

    print(f"\nDataset: {dataset.DATASET_ID}")
    print(f"Revision: {dataset.get_revision()}")

    print("\nDataset features:")
    print(dataset.dataset.features)

    print("\nFirst 5 records:")

    for sample in dataset:
        print(
            f"Index: {sample.source_index} | "
            f"Class: {sample.label} | "
            f"Region: {sample.region} | "
            f"Country: {sample.country}"
        )

        if sample.source_index == 4:
            break

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
