import torch

from metrics.iou import iou
from custom_types.bbox import BBoxFormat

# --------------------------------------------------------------------------------
class YoloV1Criterion:
    """YoloV1 Criterion
    """

    # --------------------------------------------------------------------------------
    def __init__(self):
        """Initialize Criterion for YoloV1
        """

        self.w_coords = 5.
        self.w_noobj = 5.

    # --------------------------------------------------------------------------------
    def __call__(self,
                 y: torch.Tensor,
                 y_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss

        Args:
            y: (torch.Tensor): Ground truth data
            y_hat: (torch.Tensor): Model's predictions

        Returns:
            torch.Tensor: Loss tensor
        """

        # Sanity Check
        assert y.shape[:-1] == y_hat.shape[:-1], f"Something went wrong, \nshape-> y: {y.shape}, y_hat: {y_hat.shape}"

        # Cache shape
        batch_size, s, s, n = y.shape
        n_hat = y_hat.shape[-1]

        # Reshape
        y = y.view(batch_size * s * s, n)
        y[:10, 4] = 1.0 # Temp
        y_hat = y_hat.view(batch_size * s * s, n_hat)

        # Mask Obj/NoObj
        mask_obj = (y[:, -2] == 1.0).unsqueeze(-1)
        mask_no_obj = torch.logical_not(mask_obj)

        # Compute IoU between both target boxes and gt box form each cell
        ious_hat_1 = iou(bbox_1=y[:, :4], bbox_2=y_hat[:, :4], bbox_format=BBoxFormat.MID_X_MID_Y_WH)
        ious_hat_2 = iou(bbox_1=y[:, :4], bbox_2=y_hat[:, 5:9], bbox_format=BBoxFormat.MID_X_MID_Y_WH)

        # Compute top predicted box in each cell based on IoU
        ious_hat = torch.tensor((ious_hat_2 > ious_hat_1)) # 1 if 2nd box is True box 0 if 1st

        # BBox + Confidence(Object) (mid_x, mid_y, w, h, p)
        y_hat_bboxs = torch.zeros(*y[:, :5].shape, device=y.device)

        y_hat_bboxs_1_indices = torch.nonzero(ious_hat).squeeze()
        y_hat_bboxs_2_indices = torch.nonzero(torch.logical_not(ious_hat)).squeeze()

        y_hat_bboxs[y_hat_bboxs_1_indices] = y_hat[:, :5][y_hat_bboxs_1_indices]
        y_hat_bboxs[y_hat_bboxs_2_indices] = y_hat[:, 5:10][y_hat_bboxs_2_indices]

        # Classes
        y_hat_classes = y_hat[:, 10:]

        # --- BBox X, Y
        x_y_loss = ((y_hat[:, :2] - y_hat_bboxs[:, :2]) ** 2)
        x_y_loss = (x_y_loss * mask_obj).sum() * self.w_coords

        # --- BBox W, H

        # --- BBox Confidence Obj

        # --- BBox Confidence No Obj

        # --- Classes


        return torch.Tensor([0.0])

    # --------------------------------------------------------------------------------
