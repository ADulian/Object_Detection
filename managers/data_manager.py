"""
ToDo:
    - Loading COCO dataset
    - Incorporating Lightning Data Module
    - Setup Train/Val/Test Loaders

"""

import logging

import lightning as L

from datasets.coco_dataset import CocoDataset
from utils.io import path_check

from datasets.base_classes import DatasetSplit

log = logging.getLogger("lightning")

# --------------------------------------------------------------------------------
class DataManager(L.LightningDataModule):
    """Data Manager class

    General management of datasets
    """
    # --------------------------------------------------------------------------------
    def __init__(self):
        """Initialize Data Manager
        """
        super().__init__()

        # Datasets
        self._train_set = None
        self._val_set = None
        self._test_set = None

        log.info("Data Manager Initialized")

    # --------------------------------------------------------------------------------
    def prepare_data(self) -> None:
        ...

    # --------------------------------------------------------------------------------
    def setup(self, stage: str = "") -> None:
        """

        Args:
            stage: (str): Processing stage, e.g. "train"

        """

        # Paths
        coco_root = path_check("/media/albert/FastData/Data/data-mscoco")

        train_imgs_path = coco_root / "images/train2017"
        val_imgs_path = coco_root / "images/val2017"
        # test_imgs_path = coco_root / "images/test2017"

        train_annotations_file = coco_root / "annotations/person_keypoints_train2017.json"
        val_annotations_file = coco_root / "annotations/person_keypoints_val2017.json"

        # Datasets
        self._train_set = CocoDataset(dataset_split=DatasetSplit.TRAIN,
                                      imgs_path=train_imgs_path,
                                      annotations_file=train_annotations_file)

        self._val_set = CocoDataset(dataset_split=DatasetSplit.VALIDATION,
                                    imgs_path=val_imgs_path,
                                    annotations_file=val_annotations_file)

    # --------------------------------------------------------------------------------
    def train_dataloader(self) -> None:
        ...

    # --------------------------------------------------------------------------------
    def val_dataloader(self) -> None:
        ...

    # --------------------------------------------------------------------------------
    def test_dataloader(self) -> None:
        ...





