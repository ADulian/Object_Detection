import torch

from metrics.iou import iou

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

        # Computer Responsible BBoxes
        y_bboxs = y[:, :4]
        y_hat_bboxs_1 = y_hat[:, :4]
        y_hat_bboxs_2 = y_hat[:, 5:9]

        iou(bbox_1=y_hat_bboxs_1[0], bbox_2=y_hat_bboxs_2[0])


        return torch.Tensor([0.0])

    # --------------------------------------------------------------------------------
