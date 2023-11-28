import numpy as np

from custom_types.bbox import BBox, BBoxFormat

# --------------------------------------------------------------------------------
class BBoxResizePad:
    """ A simple class that Resizes and Pads Bounding Box objects
    """

    # --------------------------------------------------------------------------------
    def __call__(self,
                 bboxs: (BBox | list[BBox]),
                 max_height: int = 0,
                 max_width: int = 0,
                 padding_height: int = 0,
                 padding_width: int = 0,
                 resize_scale: float = 1.0) -> (BBox | list[BBox]):

        """ Resize and Pad bounding boxes w.r.t. changes made to an Image

        Args:
            bboxs: (BBox | list[BBox]): BBox object/s
            max_height: (int): Max height for clamping (final height of an Image)
            max_width: (int): Max width for clamping (final width of an Image)
            padding_height: (int): Padding amount added to the top of an Image
            padding_width: (int): Padding amount added to the left of an Image
            resize_scale: (float): Resize scale used for an Image

        Returns:
            (BBox | list[BBox]): Resized and Padded BBox object/s

        """

        # Turn single box into list of bboxs for consistency
        is_single_box = isinstance(bboxs, BBox)
        bboxs = [bboxs] if is_single_box else bboxs

        # 1. Transforms the Bbox object to np bounding boxes and split into two sets of vectors
        # Shape -> N x 4
        # Shape.T -> 4 x N
        np_bboxs = np.stack([bbox.get_bbox(bbox_format=BBoxFormat.XYXY) for bbox in bboxs]).T

        x1y1 = np_bboxs[:2]  # 2 x N
        x2y2 = np_bboxs[2:]  # 2 x N

        # 2. Create Scale Matrix
        # Shape -> 2 x 2
        scale_matrix = np.array([[resize_scale, 0],
                                 [0, resize_scale]], dtype=float)

        # 3. Scale Bbox
        x1y1 = scale_matrix @ x1y1  # [2 x 2] @ [2 x N] = [2 x N]
        x2y2 = scale_matrix @ x2y2  # [2 x 2] @ [2 x N] = [2 x N]

        # 4. Pad boxes
        bboxs = np.concatenate([x1y1, x2y2]).T  # N x 4
        bboxs[:, ::2] += padding_width
        bboxs[:, 1::2] += padding_height

        # 5. Turn back into BBox
        bboxs = [BBox(bbox=bbox, bbox_format=BBoxFormat.XYXY) for bbox in bboxs]

        # 6. Clamp bboxs
        clamp_bboxs = bool(max_height) + bool(max_width)
        if clamp_bboxs == 2:  # Both values are given
            for bbox in bboxs:
                bbox.clamp_bbox(max_width=max_width, max_height=max_height)

        elif clamp_bboxs == 1:  # One of values is given
            raise ValueError(f"Only one max value has been give: max height {max_height}, max width {max_width}")

        return bboxs[0] if is_single_box else bboxs
