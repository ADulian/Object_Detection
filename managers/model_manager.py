import logging
from typing import Type

import torch
import torch.nn as nn

from models.base_classes.model_base import ModelBase
from models.yolo_v1.yolo_v1_model import YoloV1
from models.yolo_v1.yolo_v1_gt_generator import YoloV1GTGenerator

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
                    lr: float=1e-3) -> tuple[ModelBase, Type[YoloV1GTGenerator]]:
        """Initialize model and get reference to ground truth generator
        Args:
            num_classes: (int): Number of classes for Yolo prediction
            lr: (float): Learning rate
        """


        model = YoloV1(num_classes=num_classes, lr=lr)
        model = self.init_weights(model)

        log.info(f"{model} Initialised")

        return model, YoloV1GTGenerator

    # --------------------------------------------------------------------------------
    @staticmethod
    def init_weights(model: ModelBase) -> ModelBase:
        """Initialize weight of the model

        Args:
            model: (ModelBase): model

        Returns:
            ModelBase: model with initialized weights

        """

        log.info("Initializing weights with mean 0.0 and std 1e-4")
        # --- Init weights
        with torch.no_grad():
            for m in model.modules():
                # Conv2D or ConvTranspose2D
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.normal_(m.weight, mean=0.0, std=1e-4)

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                # BatchNorm
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        return model

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