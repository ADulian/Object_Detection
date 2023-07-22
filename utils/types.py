import numpy as np
from enum import Enum
from typing import TypedDict
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
class BBoxParts(TypedDict):
    """Parts of bounding box
    """
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
        """Set all parts of bounding box and update values

        Returns:
            BBoxParts: A dictionary of bounding box parts
        """

        if self.bbox_format == BBoxFormat.XYXY:
            self.top_x, self.top_y = self.bbox[0], self.bbox[1]
            self.bottom_x, self.bottom_y = self.bbox[2], self.bbox[3]
            self.width, self.height = self.bottom_x - self.top_x, self.bottom_y - self.top_y
            self.mid_x, self.mid_y = self.bottom_x - (self.width / 2), self.bottom_x - (self.height / 2)

        elif self.bbox_format == BBoxFormat.XYWH:
            self.top_x, self.top_y = self.bbox[0], self.bbox[1]
            self.width, self.height = self.bbox[2], self.bbox[3]
            self.bottom_x, self.bottom_y = self.top_x + self.width, self.top_y + self.height
            self.mid_x, self.mid_y = self.bottom_x - (self.width / 2), self.bottom_x - (self.height / 2)

        elif self.bbox_format == BBoxFormat.MID_X_MID_Y_WH:
            self.mid_x, self.mid_y = self.bbox[0], self.bbox[1]
            self.width, self.height = self.bbox[2], self.bbox[3]
            self.top_x, self.top_y = self.mid_x - (self.width / 2), self.mid_y - (self.height / 2)
            self.bottom_x, self.bottom_y = self.top_x + self.width, self.top_y + self.height

        bbox_parts = {"top_left" : np.array([self.top_x, self.top_y], dtype=float),
                      "bottom_right": np.array([self.bottom_x, self.bottom_y], dtype=float),
                      "width_height": np.array([self.width, self.height], dtype=float),
                      "mid_x_y": np.array([self.mid_x, self.mid_y], dtype=float)}

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

    # --------------------------------------------------------------------------------
    def clamp_bbox(self,
                   max_width: int,
                   max_height: int):
        """Clamp bounding box with (0,0) - (max_width, max_height)

        Args:
            max_width: (int): Max width boundary
            max_height: (int): Max height boundary

        """

        # Get bounding box in correct formatS
        bbox = self.get_bbox(bbox_format=BBoxFormat.XYXY)

        # Clamp xy1
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])

        # Clamp xy2
        bbox[2] = max(max_width, bbox[2])
        bbox[3] = max(max_height, bbox[2])

        # Update Bounding box
        self.update_bbox(bbox=bbox, bbox_format=BBoxFormat.XYXY)

    # --------------------------------------------------------------------------------
    def update_bbox(self,
                    bbox: np.ndarray,
                    bbox_format: BBoxFormat):
        """Update bounding box information

        Args:
            bbox: (np.ndarray): A new bounding box
            bbox_format: (BBoxFormat): A format of new bounding box

        """

        self.bbox = bbox
        self.bbox_format = bbox_format
        self.bbox_parts = self._set_bbox_parts()

    def draw_box(self, img: Image.Image):
        """Draw a bounding box on the PIL Image

        Args:
            img: (Image.Image): Image object

        """

        if not isinstance(img, Image.Image):
            raise ValueError(f"Can only draw on PIL image, given type: {type(img)}")

        draw = ImageDraw.Draw(img)

        # Define the coordinates of the rectangle (left, top, right, bottom)
        bbox = self.get_bbox(bbox_format=BBoxFormat.XYXY)

        # Draw the rectangle on the image
        draw.rectangle(tuple(bbox), outline="red")
