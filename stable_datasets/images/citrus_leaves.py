"""Citrus Leaves image classification dataset."""

import zipfile

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class CitrusLeaves(BaseDatasetBuilder):
    """Dataset builder for citrus leaf disease classification."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://data.mendeley.com/datasets/3f83gxmv57/2",
        assets={
            "train": "https://data.mendeley.com/public-files/datasets/3f83gxmv57/files/53398b67-6f0e-4a67-8384-e2b574b2ebf4/file_downloaded",
        },
        citation="Rauf et al. (2019). Citrus Fruits and Leaves Dataset. DOI: 10.17632/3f83gxmv57.2",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Citrus Leaves contains images of healthy and diseased "
                "citrus leaves across four classification categories."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=["Black spot", "canker", "greening", "healthy"]),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license="CC BY 4.0",
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Read labeled citrus leaf images from the archive."""
        labels = {name.lower(): index for index, name in enumerate(self.info.features["label"].names)}

        with zipfile.ZipFile(data_path, "r") as archive:
            for file_name in sorted(archive.namelist()):
                if not file_name.lower().endswith(".png"):
                    continue

                parts = file_name.replace("\\", "/").split("/")
                if len(parts) < 3 or parts[-3].lower() != "leaves":
                    continue

                label_name = parts[-2].replace("_", " ").lower()
                if label_name not in labels:
                    continue

                image_bytes = archive.read(file_name)
                yield (
                    file_name,
                    {
                        "image": image_bytes,
                        "label": labels[label_name],
                    },
                )
