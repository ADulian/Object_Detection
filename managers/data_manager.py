import logging

import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader

from datasets.coco_dataset import CocoDataset
from datasets.base_classes import DatasetSplit
from utils.io import path_check

log = logging.getLogger("lightning")

# --------------------------------------------------------------------------------
class DataManager(L.LightningDataModule):
    """Data Manager class

    General management of datasets
    """
    # --------------------------------------------------------------------------------
    def __init__(self,
                 coco_root_path: (str | Path),
                 train_batch_size: int = 1,
                 val_batch_size: int = 1,
                 test_batch_size: int = 1,
                 shuffle: bool = False,
                 pin_memory: bool = False,
                 num_workers: int = 0) -> None:
        """Initialize Data Manager

        Args:
            coco_root_path: (str | Path): Root path to coco dataset
            train_batch_size: (int): Train batch size
            val_batch_size: (int): Validation batch size
            test_batch_size: (int): Test batch size
            shuffle: (bool): Shuffle data
            pin_memory: (bool): Pin memory
            num_workers: (int): Number of workers

        """
        super().__init__()

        # --- Attr

        # General
        self._coco_root_path = path_check(coco_root_path)

        # Datasets specific
        self._train_set = None
        self._val_set = None
        self._test_set = None

        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._test_batch_size = test_batch_size

        self._shuffle = shuffle

        self._pin_memory = pin_memory
        self._num_workers = num_workers


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
        # coco_root = path_check("/media/albert/FastData/Datasets/COCO")
        # coco_root = path_check("/media/albert/FastData/Datasets/COCO")

        train_imgs_path = self._coco_root_path / "images/train2017"
        val_imgs_path = self._coco_root_path / "images/val2017"
        # test_imgs_path = coco_root / "images/test2017"

        train_annotations_file = self._coco_root_path / "annotations/person_keypoints_train2017.json"
        val_annotations_file = self._coco_root_path / "annotations/person_keypoints_val2017.json"

        # Datasets
        # self._train_set = CocoDataset(dataset_split=DatasetSplit.TRAIN,
        #                               imgs_path=train_imgs_path,
        #                               annotations_file=train_annotations_file)

        self._val_set = CocoDataset(dataset_split=DatasetSplit.VALIDATION,
                                    imgs_path=val_imgs_path,
                                    annotations_file=val_annotations_file)

    # --------------------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        """Get train data loader

        Returns:
            DataLoader: Train data loader object
        """
        return DataLoader(dataset=self._train_set, batch_size=self._train_batch_size, shuffle=self._shuffle,
                          pin_memory=self._pin_memory, num_workers=self._num_workers)

    # --------------------------------------------------------------------------------
    def val_dataloader(self) -> DataLoader:
        """Get val data loader

        Returns:
            DataLoader: Val data loader object
        """
        return DataLoader(dataset=self._val_set, batch_size=self._val_batch_size, shuffle=False,
                          pin_memory=self._pin_memory, num_workers=self._num_workers)

    # --------------------------------------------------------------------------------
    def test_dataloader(self) -> DataLoader:
        """Get test data loader

        Returns:
            DataLoader: Test data loader object
        """
        return DataLoader(dataset=self._test_set, batch_size=self._test_batch_size, shuffle=False,
                          pin_memory=self._pin_memory, num_workers=self._num_workers)





