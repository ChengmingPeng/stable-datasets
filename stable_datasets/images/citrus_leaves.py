
"""Citrus Leaves image classification dataset."""

from stable_datasets.schema import DatasetSource, Version
from stable_datasets.utils import BaseDatasetBuilder


class CitrusLeaves(BaseDatasetBuilder):
    """Dataset builder for citrus leaf disease classification."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://data.mendeley.com/datasets/3f83gxmv57/2",
        assets={},
      citation="Rauf et al. (2019). Citrus Fruits and Leaves Dataset. DOI: 10.17632/3f83gxmv57.2",
    )
