import logging
from typing import Type

from models.yolo_v1.model import YoloV1
from models.yolo_v1.gt_generator import YoloV1GTGenerator

log = logging.getLogger("lightning")

# --------------------------------------------------------------------------------
class ModelManager:
    """Model Manager Class
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 num_classes: int,
                 lr: float=1e-3):
        """Initialise Model Manager

        Args:
            num_classes: (int): Number of classes for Yolo prediction
            lr: (float): Learning rate
        """

        self.model, self.gt_generator = self._init_model(num_classes=num_classes, lr=lr)

        log.info("Model Manager Initialised")

    # --------------------------------------------------------------------------------
    def _init_model(self,
                    num_classes: int,
                    lr: float=1e-3) -> tuple[YoloV1, Type[YoloV1GTGenerator]]:
        """Initialize model and get reference to ground truth generator
        Args:
            num_classes: (int): Number of classes for Yolo prediction
            lr: (float): Learning rate
        """


        model = YoloV1(num_classes=num_classes, lr=lr)

        log.info(f"{model} Initialised")

        return model, YoloV1GTGenerator

    # --------------------------------------------------------------------------------
    def save_weights(self):
        """Save model's weights
        """
        ...

    # --------------------------------------------------------------------------------
    def load_weights(self):
        """Load model's weights
        """
        ...