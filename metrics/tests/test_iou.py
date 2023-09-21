import numpy as np

from metrics.iou import iou
from custom_types.bbox import BBoxFormat, BBox

# --------------------------------------------------------------------------------
class Test_IoU:
    """Test Intersection over Union on unnormalized and normalized data as well as various box format
    """

    # BBoxs
    BBOX_1_UNNORMALIZED = BBox(bbox=np.array([350, 400, 550, 700], dtype=np.float),
                               bbox_format=BBoxFormat.XYXY)

    BBOX_2_UNNORMALIZED = BBox(bbox=np.array([450, 550, 600, 950], dtype=np.float),
                               bbox_format=BBoxFormat.XYXY)

    # True IoUs
    IOU_UNNORMALIZED = 0.1429


    # --------------------------------------------------------------------------------
    def test_xyxy(self):
        """Test IoU on XYXY bbox
        """

        # Get XYXY bboxs
        bbox_1 = self.BBOX_1_UNNORMALIZED.get_bbox(bbox_format=BBoxFormat.XYXY)
        bbox_2 = self.BBOX_2_UNNORMALIZED.get_bbox(bbox_format=BBoxFormat.XYXY)

        # Get IoU between bboxes
        _iou = iou(bbox_1=bbox_1, bbox_2=bbox_2, bbox_format=BBoxFormat.XYXY).item()

        # Round to 4 decimal places
        _iou = round(_iou, 4)

        assert self.IOU_UNNORMALIZED == _iou

    # --------------------------------------------------------------------------------
    def test_xywh(self):
        ...

    # --------------------------------------------------------------------------------
    def test_mid_x_mid_y_wh(self):
        ...

