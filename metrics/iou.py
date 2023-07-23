import numpy as np
import torch

from custom_types.bbox import BBoxFormat
from utils.math import torch_epsilon

# --------------------------------------------------------------------------------
def to_xyxy(bbox: torch.Tensor,
            bbox_format: BBoxFormat.XYXY = BBoxFormat.XYXY) -> torch.Tensor:
    """Transform bounding box with current format to xyxy
    """

    if bbox_format == BBoxFormat.XYXY:
        return bbox

    elif bbox_format == BBoxFormat.XYWH:
        bottom_right_x, bottom_right_y = bbox[..., 0] + bbox[..., 2], bbox[..., 1] + bbox[...,3]
        bbox[..., 2] = bottom_right_x
        bbox[..., 3] = bottom_right_y

        return bbox

    elif bbox_format == BBoxFormat.MID_X_MID_Y_WH:
        mid_x = bbox[..., 0]
        mid_y = bbox[..., 1]
        half_width = bbox[..., 2] / 2
        half_height = bbox[..., 3] / 2

        top_left_x, top_left_y = mid_x - half_width, mid_y - half_height
        bottom_right_x, bottom_right_y = mid_x + half_width, mid_y + half_height

        return torch.stack([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dim=1)

# --------------------------------------------------------------------------------
def to_torch(bbox: (torch.Tensor | np.ndarray)) -> torch.Tensor:
    """Checks if bbox is valid and then transforms to torch
    """

    # Check if either numpy or torch
    if not isinstance(bbox, (torch.Tensor, np.ndarray)):
        raise ValueError(f"Tensor must be either of type torch.Tensor or np.ndarray. Current type: "
                         f"\nBBox_1: {type(bbox)}")

    # To torch
    if isinstance(bbox, np.ndarray):
        return torch.from_numpy(bbox)

    return bbox

# --------------------------------------------------------------------------------
def to_batch(bbox: torch.Tensor) -> torch.Tensor:
    """
    """

    # Already batched
    if len(bbox.shape) == 2:
        return bbox
    # Single Example
    elif len(bbox.shape) == 1:
        return bbox.unsqueeze(0)
    elif len(bbox.shape) > 2 or len(bbox.shape) < 1:
        raise ValueError(f"Tensor must have either 1 or 2 dimensions. Current shape: {bbox.shape}")

# --------------------------------------------------------------------------------
def iou(bbox_1: (torch.Tensor | np.ndarray),
        bbox_2: (torch.Tensor | np.ndarray),
        bbox_format: BBoxFormat.XYXY = BBoxFormat.XYXY):
    """Computer IoU between two bounding boxes or a batch of bboxes

    Args:
        bbox_1: (torch.Tensor | np.ndarray): First bounding box or a batch of bboxes
        bbox_2: (torch.Tensor | np.ndarray): Second bounding box or a batch of bboxes
        bbox_format: (BBoxFormat.XYXY): Format of bounding boxes

    """

    # To Torch Tensor
    bbox_1 = to_torch(bbox=bbox_1)
    bbox_2 = to_torch(bbox=bbox_2)

    # To Batch
    bbox_1 = to_batch(bbox=bbox_1)
    bbox_2 = to_batch(bbox=bbox_2)

    # To XY_Top_Left, XY_Bottom_Right
    bbox_1 = to_xyxy(bbox=bbox_1, bbox_format=bbox_format)
    bbox_2 = to_xyxy(bbox=bbox_2, bbox_format=bbox_format)

    # Intersection
    top_left_x = torch.max(bbox_1[0], bbox_2[0])
    top_left_y = torch.max(bbox_1[1], bbox_2[1])

    bottom_right_x = torch.min(bbox_1[2], bbox_2[2])
    bottom_right_y = torch.min(bbox_1[3], bbox_2[3])

    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y

    intersection = width * height

    # Union
    area = lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    bbox_1_area = area(bbox_1)
    bbox_2_area = area(bbox_2)

    union = bbox_1_area + bbox_2_area - intersection

    # Intersection over Union
    intersection_over_union = intersection / (union + torch_epsilon(tensor=union))

    return intersection_over_union
