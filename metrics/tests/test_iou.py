import numpy as np

from metrics.iou import iou
from custom_types.bbox import BBoxFormat, BBox

# --------------------------------------------------------------------------------
class Test_IoU:
    """Test Intersection over Union on unnormalized and normalized data as well as various box format
    """

    # BBoxs
    BBOX_1 = BBox(bbox=np.array([350, 400, 550, 700], dtype=np.float),
                  bbox_format=BBoxFormat.XYXY)

    BBOX_2 = BBox(bbox=np.array([450, 550, 600, 950], dtype=np.float),
                  bbox_format=BBoxFormat.XYXY)

    # True IoUs
    IOU = 0.1429

    # --------------------------------------------------------------------------------
    def get_iou(self,
                bbox_format: BBoxFormat):

        """Compute IoU between unnormalized bboxes
        """
        bbox_1 = self.BBOX_1.get_bbox(bbox_format=bbox_format)
        bbox_2 = self.BBOX_2.get_bbox(bbox_format=bbox_format)

        # Get IoU between bboxes
        _iou = iou(bbox_1=bbox_1, bbox_2=bbox_2, bbox_format=bbox_format).item()

        # Round to 4 decimal places
        _iou = round(_iou, 4)

        return _iou

    # --------------------------------------------------------------------------------
    def test_xyxy(self):
        """Test IoU on XYXY bboxs
        """

        # Compute XYXY IoU
        _iou = self.get_iou(bbox_format=BBoxFormat.XYXY)

        assert self.IOU == _iou

    # --------------------------------------------------------------------------------
    def test_xywh(self):
        """Test IoU on XYWH bboxs
        """

        # Compute XYWH IoU
        _iou = self.get_iou(bbox_format=BBoxFormat.XYWH)

        assert self.IOU == _iou

    # --------------------------------------------------------------------------------
    def test_mid_x_mid_y_wh(self):
        """Test IoU on Mid-X Mid-Y WH bboxs
        """

        # Compute XYWH IoU
        _iou = self.get_iou(bbox_format=BBoxFormat.MID_X_MID_Y_WH)

        assert self.IOU == _iou
