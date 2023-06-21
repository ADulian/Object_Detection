import logging

from pathlib import Path
from collections import defaultdict

from datasets.coco_data_sample import CocoDataSample

log = logging.getLogger("lightning")


# --------------------------------------------------------------------------------
class CocoDataset:

    # --------------------------------------------------------------------------------
    def __init__(self,
                 imgs_path: Path,
                 annotations_path: (Path | None) = None):

        self._data_samples = self._load_data(imgs_path=imgs_path, annotations_path=annotations_path)
        log.info("Coco Dataset Initialized")

    # --------------------------------------------------------------------------------
    def _load_data(self,
                   imgs_path: Path,
                   annotations_path: (Path | None) = None) -> list[CocoDataSample]:

        """Load img and annotation data for coco

        Args:
            imgs_path: (Path): Root path to images
            annotations_path: (Path | None): Root path to annotations if file exists

        Returns:
            list[CocoDataSample]: List of Coco data samples

        """

        # Samples
        data_samples = []

        # Img + Annotations
        imgs_data = {}
        annotations_data = defaultdict(list)

        return data_samples

    # --------------------------------------------------------------------------------
    def __len__(self):
        ...

    # --------------------------------------------------------------------------------
    def __getitem__(self):
        ...




