import os

import torch
import torchvision.transforms as T
from PIL.Image import Image as PILImage

from image_processing.pil_resize_preserve_ratio import PILResizePreserveRatio
from models.common.tools import get_cfg


# --------------------------------------------------------------------------------
class YoloV1PostProcessing:
    """YoloV1 Post Processing
    """

    # --------------------------------------------------------------------------------
    def __init__(self):
        """Initialize the post processing
        """

        # Cfg
        self._cfg = get_cfg(root_path=os.path.dirname(os.path.abspath(__file__)))

        # Settings
        self._in_size = self._cfg["in_size"]
        self._num_cells = self._cfg["num_cells"]
        self._div_factor = 64

        # Resize
        self._pil_resize = PILResizePreserveRatio(target_size=self._in_size,
                                                  is_square=True,
                                                  resize_longer_side=True)

    # --------------------------------------------------------------------------------
    def __call__(self,
                 model_out: torch.Tensor):
        """
        """

        # Squeeze batch dim
        model_out = model_out.squeeze(0)

        # Check for shapes
        # Expected [7 x 7 x N]
        num_cells_height, num_cells_width, _ = model_out.shape
        assert num_cells_width == num_cells_height, "Grid must have an equal height and width. " \
                                                    f"Got: Height {num_cells_height}, Width {num_cells_width}"

        assert num_cells_width == self._num_cells, "Incorrect number of output cells " \
                                                   f"Expected {self._num_cells}, Got {num_cells_width}"

        # Transform from Grid space

        ...

    # --------------------------------------------------------------------------------
    def pre_process_img(self,
                        img: PILImage) -> torch.Tensor:
        """Apply appropriate pre-processing onto an Image to prepare it for the Model's forward()

        Image is resized to match target size
        Resized image is turned into Torch.Tensor of shalep [channels, height, width]

        Args:
            img: (PILImage): A PIL Image

        Returns:

        """

        img, shape_info = self._pil_resize(img=img)
        img = T.ToTensor()(img)

        return img

    # --------------------------------------------------------------------------------

