import os
import numpy as np
from typing import Callable

import torch
from torchvision.transforms.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset

from datasets.coco_dataset import CocoDataset
from models.common.tools import get_cfg
from image_processing.pil_resize_preserve_ratio import PILResizePreserveRatio
from custom_types.bbox import BBoxFormat, BBoxResizePad
from .utils import normalize_bbox

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
        self._preprocess_img = PILResizePreserveRatio(target_size=self._in_size,
                                                      is_square=True)
        self._preprocess_bbox = BBoxResizePad()

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
                     Normalize(0.5, 0.5)]
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

        # Get relevant data
        class_idx = []
        bboxs = []
        for ann in data_sample.annotations:
            class_idx.append(ann.class_idx)
            bboxs.append(ann.bbox)

        # Preprocess image
        img, shape_info = self._preprocess_img(img=img)


        # Preprocess bounding box
        bboxs = self._preprocess_bbox(bboxs=bboxs,
                                      max_height=shape_info.final_height,
                                      max_width=shape_info.final_width,
                                      padding_height=shape_info.padding_height,
                                      padding_width=shape_info.padding_width,
                                      resize_scale=shape_info.resize_scale)

        # Transform img
        if self.transforms is not None:
            img = self.transforms(img)

        # Generate ground truth for each annotation
        num_features = 6 # BBox (4), Confidence (1),  Class (1)
        grid = np.zeros((self._num_cells, self._num_cells, num_features), dtype=float)

        for class_idx, bbox in zip(class_idx, bboxs):
            if bbox is not None:
                # Get bbox
                x, y, w, h = bbox.get_bbox(bbox_format=BBoxFormat.MID_X_MID_Y_WH)

                # Skip if mid falls outside of image boundries
                if x > self._in_size or y > self._in_size:
                    continue

                # Compute cell position
                x_cell, y_cell = x // self._cell_size, y // self._cell_size
                x_cell, y_cell = int(min(x_cell, self._num_cells)), int(min(y_cell, self._num_cells))

                # Normalize bbox
                x, y, w, h = normalize_bbox(mid_x=x, mid_y=y, width=w, height=h,
                                            x_cell=x_cell, y_cell=y_cell,
                                            cell_size=self._cell_size, in_size=self._in_size)

                # Place information in a grid
                grid[x_cell, y_cell] = np.array([x, y, w ,h , 1, class_idx], dtype=float)

        grid = torch.from_numpy(grid)

        if torch.isnan(grid).any():
            raise ValueError("Grid has NaN values")

        if torch.isnan(img).any():
            raise ValueError("Img is NaN")

        return img, grid
