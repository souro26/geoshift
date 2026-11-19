
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi


@dataclass(frozen=True)
class GeoDESample:
    """A single GeoDE sample with geographic metadata."""

    image: Any
    label: str
    region: str
    country: str
    source_index: int


class GeoDEStream:
    """Streaming interface for the GeoDE dataset on Hugging Face."""

    DATASET_ID = "MLap/GeoDE"

    def __init__(self, split: str = "train", revision: str | None = None, metadata_only: bool = False) -> None:
        self.split = split
        self.metadata_only = metadata_only

        if revision is None:
            revision = HfApi().dataset_info(repo_id=self.DATASET_ID).sha

        self.revision = revision

        self.dataset = self._load()

    
    def _load(self):
        """Load GeoDE as a remote streaming dataset."""

        if self.metadata_only:
            return load_dataset(
                self.DATASET_ID,
                split=self.split,
                streaming=True,
                revision=self.revision,
                columns=[
                    "object",
                    "region",
                    "ip_country",
                ],
            )

        return load_dataset(
            self.DATASET_ID,
            split=self.split,
            streaming=True,
            revision=self.revision,
        )


    def __iter__(self) -> Iterator[GeoDESample]:
        """Iterate over GeoDE samples."""

        for index, sample in enumerate(self.dataset):
            yield GeoDESample(
                image=None if self.metadata_only else sample["image"],
                label=sample["object"],
                region=sample["region"],
                country=sample["ip_country"],
                source_index=index,
            )

    def get_revision(self) -> str:
        """Return the pinned dataset revision."""

        return self.revision
