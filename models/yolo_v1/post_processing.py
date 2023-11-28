import os
import math

import torch
import torchvision.transforms as T
from PIL import Image
from PIL.Image import Image as PILImage

from image_processing.resize_preserve_ratio import PILResizePreserveRatio
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
        self._div_factor = 64

        # Resize
        self._pil_resize = PILResizePreserveRatio(target_size=self._in_size,
                                                  is_square=True,
                                                  resize_longer_side=True)

    # --------------------------------------------------------------------------------
    def __call__(self):
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

        img = self._pil_resize(img=img)
        img = T.ToTensor()(img)

        return img
