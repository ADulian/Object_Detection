"""Utility module that contains helpful functions specifically for YoloV1
"""

import torch
import numpy as np

from custom_types.exceptions import DimensionLengthError

# --------------------------------------------------------------------------------
def normalize_bbox(mid_x: float, mid_y: float, width: float, height: float,
                   x_cell: int, y_cell: int,
                   cell_size: int, in_size: int):
    """Normalize bbox as defined in YoloV1 paper

    The format of bbox must be [mid-x, mid-y, width, height]

    Args:
        mid_x: (float): Mid x position of bbox
        mid_y: (float): Mid y position of bbox
        width: (float): BBox width
        height: (float): BBox height
        x_cell: (int): X position of bbox in a grid
        y_cell: (int): Y position of bbox in a grid
        cell_size: (int): Size of a cell
        in_size: (int): Size of an input tensor

    Returns:
        tuple[float]: Normalized BBox

    """

    mid_x, mid_y = (mid_x / cell_size) - x_cell, (mid_y / cell_size) - y_cell
    width, height = width / in_size, height / in_size

    return mid_x, mid_y, width, height


# --------------------------------------------------------------------------------
def unnormalize_bbox(mid_x: float, mid_y: float, width: float, height: float,
                     x_cell: int, y_cell: int,
                     cell_size: int, in_size: int):
    """Unnormalize bbox (inverse of normalize function)

    The format of bbox must be [mid-x, mid-y, width, height]

    Args:
        mid_x: (float): Mid x position of bbox
        mid_y: (float): Mid y position of bbox
        width: (float): BBox width
        height: (float): BBox height
        x_cell: (int): X position of bbox in a grid
        y_cell: (int): Y position of bbox in a grid
        cell_size: (int): Size of a cell
        in_size: (int): Size of an input tensor

    Returns:
        tuple[float]: Unnormalized BBox
    """

    mid_x, mid_y = (mid_x + x_cell) * cell_size, (mid_y + y_cell) * cell_size
    width, height = width * in_size, height * in_size

    return mid_x, mid_y, width, height

# --------------------------------------------------------------------------------
def batch_unnormalize_bbox(batch_bbox: (np.ndarray | torch.Tensor),
                           cell_size: int, in_size: int) -> (np.ndarray | torch.Tensor):
    """Unnormalize batch of bboxs (inverse of normalize function)

    Args:
        batch_bbox: (np.ndarray | torch.Tensor): Batch of BBoxs
        cell_size: (int): Size of a cell
        in_size: (int): Size of an input tensor

    Returns:
        (np.ndarray | torch.Tensor): Unnormalized batch of BBoxs

    """

    # Check for shape
    if len(batch_bbox.shape) != 4:
        add_msg = "The correct shape should be: [batch_size, x_cells, y_cells, bbox]"
        raise DimensionLengthError(dim_length=len(batch_bbox.shape), target_dim_length=4, add_msg=add_msg)

    # Cache original type in case it's changed
    org_type = type(batch_bbox)
    if isinstance(batch_bbox, np.ndarray): # To Torch
        batch_bbox = torch.from_numpy(batch_bbox)

    # Device
    device = batch_bbox.device

    # X, Y
    num_x_cells, num_y_cells = batch_bbox.shape[1], batch_bbox.shape[2]
    x_pos = torch.arange(num_x_cells, device=device).reshape(1, num_x_cells, 1, 1)
    y_pos = torch.arange(num_y_cells, device=device).reshape(1, 1, num_y_cells, 1)

    batch_bbox[..., :1] = (batch_bbox[..., :1] + x_pos) * cell_size
    batch_bbox[..., 1:2] = (batch_bbox[..., 1:2] + y_pos) * cell_size

    # W, H
    batch_bbox[..., 2:4] *= in_size

    # Change back to numpy
    if org_type == np.ndarray and type(batch_bbox) != np.ndarray:
        batch_bbox = batch_bbox.numpy()

    return batch_bbox
