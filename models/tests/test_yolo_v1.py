import torch

from metrics.iou import iou
from custom_types.bbox import BBoxFormat

# --------------------------------------------------------------------------------
def get_y_test(flatten: bool = True):
    """Generate ground truth test data
    """

    # Ground truth empty tensor
    y = torch.zeros(size=(1, 7, 7, 6), dtype=torch.float)

    # Ground truth data
    bbox_1 = [0.25, 0.35, 0.45, 0.55, 1.0, 3]
    bbox_2 = [0.10, 0.20, 0.30, 0.40, 1.0, 8]

    # Place ground truth data into tensor
    y[0, 2, 2] = torch.tensor(bbox_1)
    y[0, 3, 3] = torch.tensor(bbox_2)

    # Flatten such that shape becomes [-1, features]
    if flatten:
        y = y.view(-1, y.shape[-1])

    return y

# --------------------------------------------------------------------------------
def get_y_hat_test(flatten: bool = True):
    """Generate prediction test data
    """

    # Ground truth empty tensor
    y_hat = torch.zeros(size=(1, 7, 7, 10), dtype=torch.float)

    # Objects in cell
    bbox_in_1 = [0.23, 0.32, 0.40, 0.51, 0.8, 0.1, 0.6, 0.3, 0.5, 0.7] # First box is better fit
    bbox_in_2 = [0.4, 0.4, 0.6, 0.2, 0.5, 0.12, 0.24, 0.33, 0.40, 0.9] # Second box is better fit

    # Objects not in cell

    # Place predictions data into tensor
    y_hat[0, 2, 2] = torch.tensor(bbox_in_1)
    y_hat[0, 3, 3] = torch.tensor(bbox_in_2)

    # Flatten such that shape becomes [-1, features]
    if flatten:
        y_hat = y_hat.view(-1, y_hat.shape[-1])

    return y_hat


# --------------------------------------------------------------------------------
class TestYoloV1Metrics:
    """Test Yolo V1 Metrics
    """

    def test_iou(self):
        """Tets Intersection over Union
        """

        # Get test data
        y_bbox = get_y_test()[:, :4]
        y_hat_bbox = get_y_hat_test()[:, :4]

        # Compute IoU between true and predicted boxes
        computed_iou = iou(bbox_1=y_bbox, bbox_2=y_hat_bbox, bbox_format=BBoxFormat.MID_X_MID_Y_WH)

        #


        y = y_bbox.numpy()
        y_hat = y_hat_bbox.numpy()
        computed_iou = computed_iou.numpy()



        ...


# --------------------------------------------------------------------------------
class TestYoloV1Criterion:
    """Test Yolo V1 Criterion
    """

    # --------------------------------------------------------------------------------
    def test_xy_loss(self):
        """
        """

        y = get_y_test()
        y_hat = get_y_hat_test()

        y_np = y.numpy()
        y_hat_np = y_hat.numpy()

        ...
