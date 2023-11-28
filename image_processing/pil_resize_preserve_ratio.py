import math

from PIL import Image
from PIL.Image import Image as PILImage

from image_processing.pil_img_shape_info import PILImageShapeInfo

# --------------------------------------------------------------------------------
class PILResizePreserveRatio:
    """A simple module for resizing PIL Image
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 target_size: int,
                 is_square: bool,
                 resize_longer_side: bool = True,
                 div_factor: int = 64) -> None:
        """Initialise Resize Transform for PIL Image
        """

        self._target_size = target_size
        self._is_square = is_square
        self._resize_longr_side = resize_longer_side
        self._div_factor = div_factor

    # --------------------------------------------------------------------------------
    def __call__(self,
                 img: PILImage) -> tuple[PILImage, PILImageShapeInfo]:
        """Resize an image

        The image preserves the aspect ratio and the sides are padded
        such that each side is divisible by div_factor (64 default)

        Args:
            img: (PILImage): PIL Image

        Returns
            tuple[PILImage, PILImageShapeInfo]: Resized PIL Image and info about how it was resized
        """

        # 1. Cache default shape
        org_width, org_height = img.width, img.height

        # 2. Compute ratio of max side
        resize_scale = self._target_size / max(org_width, org_height)

        # 3. Compute new size of the image
        new_width, new_height = int(org_width * resize_scale), int(org_height * resize_scale)

        # 4. Resize the image
        resized_img = img.resize((new_width, new_height))

        # 5. Compute size of padded image
        if self._is_square:
            # 5.1 Set both sides to have the same target size if the image is meant to be square
            padded_width, padded_height = self._target_size, self._target_size
        else:
            # 5.2 Compute padding to ensure that both Width and Height are divisible by div factor
            padded_width = math.ceil(new_width / self._div_factor) * self._div_factor
            padded_height = math.ceil(new_height / self._div_factor) * self._div_factor

        # 6. Compute padding values (img is placed at the centre so pad /2 from sides
        padding_width = int((padded_width - new_width) / 2)
        padding_height = int((padded_height - new_height) / 2)

        # 7. Create new image with final padded shape
        # and paste resize image onto that w.r.t. padding
        padded_img = Image.new(mode=img.mode, size=(padded_width, padded_height), color=0)
        padded_img.paste(resized_img, (padding_width, padding_height))

        # Create an object with data about image resize and padding
        shape_info = PILImageShapeInfo(original_height=org_height,
                                       original_width=org_width,
                                       final_height=padded_height,
                                       final_width=padded_width,
                                       resize_scale=resize_scale)


        return padded_img, shape_info
