import numpy as np
from enum import Enum
from typing import TypedDict

class BBoxFormat(Enum):
    """Bounding Box Format

    Formats:
        - xyxy - Left upper corner / Right bottom corner
        - xywh - Left upper corner / Width / Height
        - mid_x_mid_y_wh - Mid X/ Mid Y/ Width/ Height
    """
    XYXY = 1
    """
    *-----|
    |     |
    |_____*
    """

    XYWH = 2
    """
    *-----^
    |     ^
    |>>>>>^
    """

    MID_X_MID_Y_WH = 3
    """
    |-----^
    |  *  ^
    |>>>>>^
    """

class BBoxParts(TypedDict):
    top_left: np.ndarray
    bottom_right: np.ndarray
    width_height: np.ndarray
    mid_x_y: np.ndarray

# --------------------------------------------------------------------------------
class BBox:
    """A class that defines bounding box and it's individual parts
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 bbox: np.ndarray,
                 bbox_format: BBoxFormat = BBoxFormat.XYWH):
        """Initialize bounding box object

        """

        self.bbox = bbox
        self.width = 0.0
        self.height = 0.0
        self.top_x = 0.0
        self.top_y = 0.0
        self.bottom_x = 0.0
        self.bottom_y = 0.0
        self.mid_x = 0.0
        self.mid_y = 0.0
        self.bbox_format = bbox_format
        self.bbox_parts = self._set_bbox_parts()

    # --------------------------------------------------------------------------------
    def _set_bbox_parts(self) -> BBoxParts:
        """Set all parts of bounding box

        Returns:
            BBoxParts: A dictionary of bounding box parts
        """

        top_left = np.array([self.top_x, self.top_y], dtype=float)
        bottom_right = np.array([self.bottom_x, self.bottom_y], dtype=float)
        width, height = self.width, self.height
        mid_x_y = np.array([self.mid_x, self.mid_y], dtype=float)

        if self.bbox_format == BBoxFormat.XYXY:
            top_left = self.bbox[:2]
            bottom_right = self.bbox[2:]
            width, height = bottom_right[0] - top_left[0], top_left[1] - bottom_right[1]
            mid_x_y = np.array(bottom_right[0] - (width / 2), bottom_right[1] - (height / 2), dtype=float)

        elif self.bbox_format == BBoxFormat.XYWH:
            top_left = self.bbox[:2]
            width, height = self.bbox[2], self.bbox[3]
            bottom_right = np.array([top_left[0] + width, top_left[1] + height], dtype=float)
            mid_x_y = np.array([bottom_right[0] - (width / 2), bottom_right[1] - (height / 2)], dtype=float)

        elif self.bbox_format == BBoxFormat.MID_X_MID_Y_WH:
            mid_x_y = self.bbox[:2]
            width, height = self.bbox[2], self.bbox[3]
            top_left = np.array([mid_x_y[0] - (width / 2), mid_x_y[1] - (height / 2)], dtype=float)
            bottom_right = top_left = np.array([mid_x_y[0] + (width / 2), mid_x_y[1] + (height / 2)], dtype=float)

        bbox_parts = {"top_left" : top_left,
                      "bottom_right": bottom_right,
                      "width_height": np.array([width, height], dtype=float),
                      "mid_x_y": mid_x_y}

        self.width = width
        self.height = height
        self.top_x = top_left[0]
        self.top_y = top_left[1]
        self.bottom_x = bottom_right[0]
        self.bottom_y = bottom_right[1]
        self.mid_x = mid_x_y[0]
        self.mid_y = mid_x_y[1]

        return bbox_parts

    # --------------------------------------------------------------------------------
    def is_empty(self) -> bool:
        """Returns True if all entries are empty
        """
        return bool(self.bbox.sum())

    # --------------------------------------------------------------------------------
    def get_bbox(self,
                 bbox_format: BBoxFormat = BBoxFormat.XYWH) -> np.ndarray:
        """

        Args:
            bbox_format: (BBoxFormat): Desired Bounding Box format

        Returns:
            np.ndarray: Returns bounding box as an numpy array given a desired format
        """
        if bbox_format == BBoxFormat.XYXY:
            return np.concatenate((self.bbox_parts["top_left"], self.bbox_parts["bottom_right"]), dtype=float)
        elif bbox_format == BBoxFormat.XYWH:
            return np.concatenate((self.bbox_parts["top_left"], self.bbox_parts["width_height"]), dtype=float)
        elif bbox_format == BBoxFormat.MID_X_MID_Y_WH:
            return np.concatenate((self.bbox_parts["mid_x_y"], self.bbox_parts["width_height"]), dtype=float)
