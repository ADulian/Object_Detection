from .model import YoloV1
from .criterion import YoloV1Criterion
from .gt_generator import YoloV1GTGenerator

# --------------------------------------------------------------------------------
class YoloV1Manager:
    """Yolo V1 Manager
    """

    # --------------------------------------------------------------------------------
    def __init__(self):
        """Init Manager
        """

        self.model = ""
        self.criterion = ""
        self.gt_generator = ""
