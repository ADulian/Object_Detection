import os
import torch

from metrics.iou import iou
from custom_types.bbox import BBoxFormat
from models.common.tools import get_cfg
from models.yolo_v1.utils import batch_unnormalize_bbox

# --------------------------------------------------------------------------------
class YoloV1Criterion:
    """YoloV1 Criterion
    """

    # --------------------------------------------------------------------------------
    def __init__(self):
        """Initialize Criterion for YoloV1
        """

        # Cfg
        self._cfg = get_cfg(root_path=os.path.dirname(os.path.abspath(__file__)))

        # Settings
        self._in_size = self._cfg["in_size"]
        self._num_cells = self._cfg["num_cells"]
        self._cell_size = self._in_size // self._num_cells

        self._w_coords = self._cfg["criterion"]["w_coords"]
        self._w_noobj = self._cfg["criterion"]["w_noobj"]

    # --------------------------------------------------------------------------------
    def _compute_ious(self,
                      y: torch.Tensor,
                      y_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        """ Compute IoUs between ground truth and predictions 2 bboxs
        Args:
            y: (torch.Tensor): Ground truth data
            y_hat: (torch.Tensor): Model's predictions

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 2xTensors with IoUs

        """

        # Get bounding boxes
        bboxs_y = y[..., :4].clone()
        bboxs_y_hat_1 = y_hat[..., :4].clone()
        bboxs_y_hat_2 = y_hat[..., 5:9].clone()

        # Unnormalize
        bboxs_y = batch_unnormalize_bbox(batch_bbox=bboxs_y,
                                         cell_size=self._cell_size, in_size=self._in_size)

        bboxs_y_hat_1 = batch_unnormalize_bbox(batch_bbox=bboxs_y_hat_1,
                                               cell_size=self._cell_size, in_size=self._in_size)

        bboxs_y_hat_2 = batch_unnormalize_bbox(batch_bbox=bboxs_y_hat_2,
                                               cell_size=self._cell_size, in_size=self._in_size)

        # Rehspae bboxs
        bboxs_y = bboxs_y.view(-1, bboxs_y.shape[-1])
        bboxs_y_hat_1 = bboxs_y_hat_1.view(-1, bboxs_y_hat_1.shape[-1])
        bboxs_y_hat_2 = bboxs_y_hat_2.view(-1, bboxs_y_hat_2.shape[-1])

        # Compute IoU
        ious_1 = iou(bbox_1=bboxs_y, bbox_2=bboxs_y_hat_1, bbox_format=BBoxFormat.MID_X_MID_Y_WH)
        ious_2 = iou(bbox_1=bboxs_y, bbox_2=bboxs_y_hat_2, bbox_format=BBoxFormat.MID_X_MID_Y_WH)

        return ious_1, ious_2

    # --------------------------------------------------------------------------------
    def _compare_ious(self,
                      y: torch.Tensor,
                      y_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare IoUs for 2 proposal boxes per prediction

        Args:
            y: (torch.Tensor): Ground truth data
            y_hat: (torch.Tensor): Model's predictions

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 2xTensors with indices of Top Boxes
        """

        # Compute IoUs
        ious_1, ious_2 = self._compute_ious(y=y, y_hat=y_hat)

        # Compute top predicted box in each cell based on IoU
        ious_hat = torch.tensor((ious_1 > ious_2))  # 1 if 2nd box is True box 0 if 1st

        # Get indices of top boxes
        y_hat_bboxs_1_indices = torch.nonzero(ious_hat).squeeze()
        y_hat_bboxs_2_indices = torch.nonzero(torch.logical_not(ious_hat)).squeeze()

        return y_hat_bboxs_1_indices, y_hat_bboxs_2_indices

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
        y = torch.ones(1, 7, 7, 6)
        y_hat = torch.ones(1, 7, 7, 90)
        assert y.shape[:-1] == y_hat.shape[:-1], f"Something went wrong, \nshape-> y: {y.shape}, y_hat: {y_hat.shape}"

        # Sigmoid Y_Hat
        y_hat = torch.sigmoid(y_hat)

        # Cache shape
        batch_size, s, s, n = y.shape
        n_hat = y_hat.shape[-1]

        # Compute and Compare IoUs of predicted boxes
        y_hat_bboxs_1_indices, y_hat_bboxs_2_indices = self._compare_ious(y, y_hat)

        # Reshape
        y = y.view(batch_size * s * s, n)
        y_hat = y_hat.view(batch_size * s * s, n_hat)

        # BBox + Confidence(Object) (mid_x, mid_y, w, h, p)
        y_hat_bboxs = torch.zeros(*y[:, :5].shape, device=y.device)

        y_hat_bboxs[y_hat_bboxs_1_indices] = y_hat[:, :5][y_hat_bboxs_1_indices]
        y_hat_bboxs[y_hat_bboxs_2_indices] = y_hat[:, 5:10][y_hat_bboxs_2_indices]

        # Mask Obj/NoObj
        mask_obj = (y[:, -2] == 1.0).unsqueeze(-1)
        mask_no_obj = torch.logical_not(mask_obj)

        # --- BBox X, Y
        x_y_loss = ((y[:, :2] - y_hat_bboxs[:, :2]) ** 2)
        x_y_loss = (x_y_loss * mask_obj).sum() * self._w_coords

        # --- BBox W, H
        w_h_loss = ((torch.sqrt(y[:, 2:4]) - torch.square(y_hat_bboxs[:, 2:4])) ** 2)
        w_h_loss = (w_h_loss * mask_obj).sum() * self._w_coords

        # --- BBox Confidence
        c_loss = ((y[:, 4:5] - y_hat_bboxs[:, 4:5]) ** 2)

        # Obj
        iou_truth_pred = iou(bbox_1=y[:, :4], bbox_2=y_hat_bboxs[:, :4], bbox_format=BBoxFormat.MID_X_MID_Y_WH)
        c_obj_loss = (c_loss * iou_truth_pred.unsqueeze(-1) * mask_obj).sum()

        # No Obj
        c_no_obj_loss = (c_loss * mask_no_obj).sum() * self._w_noobj

        # --- Classes
        y_hat_classes = y_hat[:, 10:]   # Predictions [-1, num_classes]

        true_classes = y[:, -1].unsqueeze(-1).to(torch.int64) # Indices of true classes [-1, 1]
        src_values = torch.ones([y_hat.shape[0], 1], dtype=torch.float, device=y.device) # Ones [-1, 1]
        y_classes = torch.zeros_like(y_hat_classes, device=y.device) # Target Y Tensor [-1, num_classes]

        # Fill Y Tensor with ones. (src_values) at index dictated by true_classes Tensor
        y_classes.scatter_(dim=-1, index=true_classes, src=src_values)

        # Compute Loss
        cl_loss = ((y_classes - y_hat_classes) ** 2)
        cl_loss = (cl_loss * mask_obj).sum()

        # Combine
        loss = x_y_loss + w_h_loss + c_obj_loss + c_no_obj_loss + cl_loss

        return loss
