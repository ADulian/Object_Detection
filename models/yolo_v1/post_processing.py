import os
import math

import torch
import torchvision.transforms as T
from PIL import Image
from PIL.Image import Image as PILImage

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

        img = self.resize_img(img=img)
        img = T.ToTensor()(img)

        return img

    # --------------------------------------------------------------------------------
    def resize_img(self,
                   img: PILImage,
                   resize_longer_side: bool = True) -> PILImage:

        """Resize the img to expected size

        Args:
            img: (PILImage): A PIL Image
            resize_longer_side: (bool): Resize w.r.t. one side
                - True: Resize w.r.t. longer side (shorter side < target size)
                - False: Resize w.r.t. shorter size (longer side > target size)

        Returns:
            PILImage: Resized Image

        """

        # 1. Cache default shape
        org_width, org_height = img.width, img.height

        # 2. Compute ratio of max side
        ratio = self._in_size / max(org_width, org_height)

        # 3. Compute new size of the image
        new_width, new_height = int(org_width * ratio), int(org_height * ratio)

        # 4. Resize the image
        resized_img = img.resize((new_width, new_height))

        # 5. Compute padding to ensure that both Width and Height are divisible by div factor
        padded_width = math.ceil(new_width / self._div_factor) * self._div_factor
        padded_height = math.ceil(new_height / self._div_factor) * self._div_factor

        padding_width = int((padded_width - new_width) / 2)
        padding_height = int((padded_height - new_height) / 2)

        # 6. Create new image with final padded shape
        # and paste resize image onto that w.r.t. padding
        padded_img = Image.new(mode=img.mode, size=(padded_width, padded_height), color=0)
        padded_img.paste(resized_img, (padding_width, padding_height))

        return padded_img
