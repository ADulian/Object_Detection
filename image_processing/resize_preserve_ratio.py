import math

import numpy as np
from PIL import Image

from custom_types.bbox import BBox, BBoxFormat

# --------------------------------------------------------------------------------
class PILResizePreserveRatio:

    # --------------------------------------------------------------------------------
    def __init__(self):
        """Initialie Resize Transform for Pil Images
        """

    # --------------------------------------------------------------------------------
    def __call__(self,
                 target_size: int,
                 img: Image.Image,
                 bboxs: (list[BBox] | None) = None,
                 padding: bool = True,
                 longer_size: bool = True):
        """ Resize objects such that aspect ratio is preserved

        Args:
            target_size: (int): Target size
            img: (Image): Pil Image
            bboxs: (BBox | None): A list of bounding box objects
            padding: (bool): Pad image, if True the image will be a square of size target_size x target)size
            longer_size: (bool): Resize with respect to the longer size. If false then shorter size == target size and
            longer size will be cropped from left to right

        """


        # Transform Image
        if not isinstance(img, Image.Image):
            raise ValueError(f"Can only transform PIL image, given image is of type: {type(img)}")

        # Current Size
        img_width = img.width
        img_height = img.height
        old_ratio = img_width / img_height

        # New Size Scale Ratio
        scale_ratio = max(img_width, img_height) / target_size

        # New Size
        new_width = img_width / scale_ratio
        new_height = img_height / scale_ratio
        new_ratio = new_width / new_height

        # Sanity check
        assert math.isclose(old_ratio, new_ratio), f"Something went wrong! Old: {old_ratio}, New: {new_ratio}"

        # Resize
        img = img.resize((int(new_width), int(new_height)))

        # Pad
        if padding:
            padded_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            padded_img.paste(img, (0,0))
            img = padded_img # Consistency

        # Transform Bounding Box
        if bboxs is not None:
            # Transforms the Bbox object to np bounding boxes and split into two sets of vectors
            np_bboxs = np.stack([bbox.get_bbox(bbox_format=BBoxFormat.XYXY) for bbox in bboxs]).T

            x1y1 = np_bboxs[:2]
            x2y2 = np_bboxs[2:]

            # Scale Matrix
            scale_ratio = 1 / scale_ratio
            scale_matrix = np.array([[scale_ratio, 0],
                                     [0, scale_ratio]], dtype=float)

            # Dot Product
            x1y1 = scale_matrix @ x1y1
            x2y2 = scale_matrix @ x2y2

            # Turn back into BBox
            bboxs = np.concatenate([x1y1, x2y2]).T
            bboxs = [BBox(bbox=bbox, bbox_format=BBoxFormat.XYXY).clamp_bbox(max_width=new_width, max_height=new_height)
                     for bbox in bboxs]

        return img, bboxs