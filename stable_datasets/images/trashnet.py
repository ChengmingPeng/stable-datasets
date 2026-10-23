import zipfile
from collections import defaultdict

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class TrashNet(BaseDatasetBuilder):
    """TrashNet image classification dataset with six waste classes."""

    VERSION = Version("1.0.1")

    DATA_URL = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"

    SOURCE = DatasetSource(
        homepage="https://github.com/garythung/trashnet",
        assets={
            "train": DownloadInfo(
                url=DATA_URL,
                filename="trashnet-resized.zip",
            ),
            "test": DownloadInfo(
                url=DATA_URL,
                filename="trashnet-resized.zip",
            ),
        },
        citation=("Thung, G. and Yang, M. (2016). Classification of Trash for Recyclability Status. Stanford CS229."),
        license="MIT",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "TrashNet contains images of household waste "
                "in six categories. The original images are "
                "divided into deterministic train/test splits."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(
                        names=[
                            "cardboard",
                            "glass",
                            "metal",
                            "paper",
                            "plastic",
                            "trash",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        class_names = self.info.features["label"].names
        groups = defaultdict(list)

        with zipfile.ZipFile(data_path, "r") as archive:
            for filename in archive.namelist():
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                parts = filename.replace("\\", "/").split("/")
                if len(parts) < 2 or "__MACOSX" in parts or parts[-1].startswith("._"):
                    continue

                label = parts[-2].lower()
                if label in class_names:
                    groups[label].append(filename)

            if not any(groups.values()):
                raise ValueError("No labeled TrashNet images found.")

            for label in class_names:
                label_id = self.info.features["label"].str2int(label)

                for index, filename in enumerate(sorted(groups[label])):
                    is_test = index % 5 == 0

                    if is_test != (split == "test"):
                        continue

                    yield (
                        filename,
                        {
                            "image": archive.read(filename),
                            "label": label_id,
                        },
                    )
