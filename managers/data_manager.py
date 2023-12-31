import os
import logging
from pathlib import Path
from typing import Type

import lightning as L
from torch.utils.data import DataLoader

from datasets.coco_dataset import CocoDataset
from datasets.base_classes import DatasetSplit
from utils.io import path_check
from models.yolo_v1.yolo_v1_gt_generator import YoloV1GTGenerator

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
                 num_workers: int = 0,
                 use_dev_set: bool = False) -> None:
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
        self._shuffle = shuffle

        self._pin_memory = pin_memory
        self._num_workers = num_workers

        self.num_classes = None
        self._gt_generator = None
        self._use_dev_set = use_dev_set

        # Datasets specific
        self._train_set = None
        self._val_set = None
        self._test_set = None

        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._test_batch_size = test_batch_size

        # Load Dataset
        self._load_data()

        log.info("Data Manager Initialised")

    # --------------------------------------------------------------------------------
    def _load_data(self):
        """Load Datasets

        """
        log.info(f"Loading dataset from {self._coco_root_path}")

        # Datasets
        if self._use_dev_set:
            dev_imgs_path = Path(os.getcwd()) / "dev_dataset/images"
            dev_annotations_file = Path(os.getcwd()) / "dev_dataset/instances_dev2017.json"

            dev_set = CocoDataset(dataset_split=DatasetSplit.TRAIN,
                                  imgs_path=dev_imgs_path,
                                  annotations_file=dev_annotations_file)

            self._train_set = CocoDataset(dataset_split=DatasetSplit.TRAIN,
                                          imgs_path=dev_imgs_path,
                                          annotations_file=dev_annotations_file)

            self._val_set = CocoDataset(dataset_split=DatasetSplit.VALIDATION,
                                        imgs_path=dev_imgs_path,
                                        annotations_file=dev_annotations_file)
        else:
            train_imgs_path = self._coco_root_path / "images/train2017"
            val_imgs_path = self._coco_root_path / "images/val2017"

            train_annotations_file = self._coco_root_path / "annotations/instances_train2017.json"
            val_annotations_file = self._coco_root_path / "annotations/instances_val2017.json"

            self._train_set = CocoDataset(dataset_split=DatasetSplit.TRAIN,
                                          imgs_path=train_imgs_path,
                                          annotations_file=train_annotations_file)

            self._val_set = CocoDataset(dataset_split=DatasetSplit.VALIDATION,
                                        imgs_path=val_imgs_path,
                                        annotations_file=val_annotations_file)

        # Set number of classes
        self.num_classes = len(self._val_set.idx_to_class)

    # --------------------------------------------------------------------------------
    def update(self,
               gt_generator: Type[YoloV1GTGenerator]):
        """Update Data Manager

        Args:
            gt_generator: (Type[YoloV1GTGenerator]): A reference to the ground truth generator
        """

        if self._gt_generator is None:
            self._gt_generator = gt_generator

    # --------------------------------------------------------------------------------
    def prepare_data(self) -> None:
        """Prepare Data
        """

    # --------------------------------------------------------------------------------
    def setup(self, stage: str = "") -> None:
        """Setup Datasets

        Args:
            stage: (str): Processing stage, e.g. "train"

        """

        # Wrap Coco datasets into Ground Truth Generator
        if self._gt_generator is None:
            raise ValueError("Ground truth generator not set. Tip: Ensure to call update before setup.")

        # Type hint
        self._gt_generator: Type[YoloV1GTGenerator]

        if self._train_set is not None:
            self._train_set = self._gt_generator(dataset=self._train_set)

        if self._val_set is not None:
            self._val_set = self._gt_generator(dataset=self._val_set)

        if self._test_set is not None:
            self._test_set = self._gt_generator(dataset=self._test_set)

    # --------------------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        """Get train data loader

        Returns:
            DataLoader: Train data loader object
        """
        if self._train_set is not None:
            return DataLoader(dataset=self._train_set, batch_size=self._train_batch_size, shuffle=self._shuffle,
                              pin_memory=self._pin_memory, num_workers=self._num_workers)
        else:
            raise ValueError("Train set is None!")

    # --------------------------------------------------------------------------------
    def val_dataloader(self) -> DataLoader:
        """Get val data loader

        Returns:
            DataLoader: Val data loader object
        """
        if self._val_set is not None:
            return DataLoader(dataset=self._val_set, batch_size=self._val_batch_size, shuffle=False,
                              pin_memory=self._pin_memory, num_workers=self._num_workers)
        else:
            raise ValueError("Validation set is None!")

    # --------------------------------------------------------------------------------
    def test_dataloader(self) -> DataLoader:
        """Get test data loader

        Returns:
            DataLoader: Test data loader object
        """
        if self._test_set is not None: # Optional
            return DataLoader(dataset=self._test_set, batch_size=self._test_batch_size, shuffle=False,
                              pin_memory=self._pin_memory, num_workers=self._num_workers)
