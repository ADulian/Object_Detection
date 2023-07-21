import os
import numpy as np
from typing import Callable

from torchvision.transforms.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset

from datasets.coco_dataset import CocoDataset
from models.common.tools import get_cfg

# --------------------------------------------------------------------------------
class YoloV1GTGenerator(Dataset):
    """Ground truth generator for YoloV1
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 dataset: CocoDataset,
                 transforms: (list[Callable] | None) = None,
                 transforms_on: bool = True):
        """Initialize generator

        Args:
            dataset: (CocoDataset): Dataset
            transforms: (list[Callable] | None): list of transforms
            transforms_on: (bool): Use transforms
        """

        # Cfg
        self._cfg = get_cfg(root_path=os.path.dirname(os.path.abspath(__file__)))

        # Dataset
        self._dataset = dataset
        self._in_size = self._cfg["in_size"]
        self._num_cells = self._cfg["num_cells"]
        self._cell_size = self._in_size // self._num_cells

        # Transforms
        self._transforms_on = transforms_on
        self._transforms = transforms

    # --------------------------------------------------------------------------------
    @property
    def transforms(self):
        """Get (init if None) transforms
        """
        if self._transforms_on:
            if self._transforms is None:
                self._transforms = Compose(
                    [ToTensor(),
                     Normalize(.5, 0.5)]
                )

        return self._transforms

    # --------------------------------------------------------------------------------
    def __len__(self) -> int:
        """Get length of dataset

        Return:
             int: Length of dataset
        """

        return len(self._dataset)

    # --------------------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """

        Args:
            idx: (int): Index of the sample

        Returns:
            tuple[PIL.Image.Image, CocoDataSample]: An image and a data sample information

        """

        # Get data sample
        img, data_sample = self._dataset[idx]

        # ToDo :: Resize such that aspect ratio is preserverd
        # ToDO :: Bounding box needs to be resized as well (smaller images will get upscaled)
        # Transform img
        if self.transforms is not None:
            img = self.transforms(img)


        # Generate ground truth for each annotation
        num_features = 6 # BBox (4), Confidence (1),  Class (1)
        grid = np.zeros((self._num_cells, self._num_cells, num_features), dtype=float)

        for annotation in data_sample.annotations:
            # Get bbox
            x, y, w, h = annotation.bbox

            # x, y to mid_x, mid_y
            x, y = x + (w / 2), y + (h / 2)

            # Skip if mid falls outside of image boundries
            if x > self._in_size or y > self._in_size:
                continue

            # Compute cell position
            x_cell, y_cell = x // self._cell_size, y // self._cell_size
            x_cell, y_cell = int(min(x_cell, self._num_cells)), int(min(y_cell, self._num_cells))

            # Normalize bbox
            x, y = (x / self._cell_size) - x_cell, (y / self._cell_size) - y_cell
            w, h = w / self._in_size, h / self._in_size

            # Place information in a grid
            grid[x_cell, y_cell] = np.array([x, y, w ,h , 1, annotation.class_id], dtype=float)

        return img, grid
