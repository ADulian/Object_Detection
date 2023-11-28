import math

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from custom_types.bbox import BBox, BBoxFormat

# --------------------------------------------------------------------------------
class PILResizePreserveRatio:
    """A simple module for resizing PIL Image
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 target_size: int,
                 is_square: bool,
                 resize_longer_side: bool = True,
                 div_factor: int = 64) -> None:
        """Initialise Resize Transform for PIL Image
        """

        self._target_size = target_size
        self._is_square = is_square
        self._resize_longr_side = resize_longer_side
        self._div_factor = div_factor

    # --------------------------------------------------------------------------------
    def __call__(self,
                 img: PILImage) -> PILImage:
        """Resize an image

        The image preserves the aspect ratio and the sides are padded
        such that each side is divisible by div_factor (64 default)

        Args:
            img: (PILImage): PIL Image

        Returns
            PILImage: Resized PIL Image
        """

        # 1. Cache default shape
        org_width, org_height = img.width, img.height

        # 2. Compute ratio of max side
        ratio = self._target_size / max(org_width, org_height)

        # 3. Compute new size of the image
        new_width, new_height = int(org_width * ratio), int(org_height * ratio)

        # 4. Resize the image
        resized_img = img.resize((new_width, new_height))

        # 5. Compute size of padded image
        if self._is_square:
            # 5.1 Set both sides to have the same target size if the image is meant to be square
            padded_width, padded_height = self._target_size, self._target_size
        else:
            # 5.2 Compute padding to ensure that both Width and Height are divisible by div factor
            padded_width = math.ceil(new_width / self._div_factor) * self._div_factor
            padded_height = math.ceil(new_height / self._div_factor) * self._div_factor

        # 6. Compute padding values (img is placed at the centre so pad /2 from sides
        padding_width = int((padded_width - new_width) / 2)
        padding_height = int((padded_height - new_height) / 2)

        # 7. Create new image with final padded shape
        # and paste resize image onto that w.r.t. padding
        padded_img = Image.new(mode=img.mode, size=(padded_width, padded_height), color=0)
        padded_img.paste(resized_img, (padding_width, padding_height))

        return padded_img

    # --------------------------------------------------------------------------------
    def old_call(self,
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
        if bboxs is not None and bboxs:
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
            bboxs = [BBox(bbox=bbox, bbox_format=BBoxFormat.XYXY) for bbox in bboxs]

            for bbox in bboxs:
                bbox.clamp_bbox(max_width=new_width, max_height=new_height)

        return img, bboxs