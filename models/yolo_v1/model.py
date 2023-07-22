import os
from typing import Any

import torch
from torch import nn

from models.common.tools import get_layer, get_cfg
from models.base_classes.model_base import ModelBase

# --------------------------------------------------------------------------------
class YoloV1(ModelBase):
    """Yolo V1
    Ref: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 num_classes: int = 20,
                 lr: float = 1e-3):
        """Initialize Yolo

        Args:
            num_classes: (int): Number of classes
            lr: (float): Learning rate
        """
        super().__init__(lr=lr)

        # Cfg
        self._cfg = get_cfg(root_path=os.path.dirname(os.path.abspath(__file__)))

        # Yolo Attribs
        self.num_classes = num_classes
        self.num_bboxes = self._cfg["num_bboxes"]
        self.num_cells = self._cfg["num_cells"]
        self.num_cell_features = (self.num_bboxes * 5 + self.num_classes)

        # Initialize Model
        self.yolo = self._parse_model()
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=(self.num_cells * self.num_cells * 1024),
                                          out_features=4096),
                                nn.LeakyReLU(negative_slope=0.1),
                                nn.Linear(in_features=4096,
                                          out_features=(self.num_cells * self.num_cells * self.num_cell_features)))

    # --------------------------------------------------------------------------------
    def _parse_model(self) -> nn.Sequential:
        """Parse model from config

        Returns:
            nn.Sequential: Parsed Yolo model
        """
        model = []
        in_channels = 3
        cfg_layers = self._cfg["layers"]

        for cfg_layer in cfg_layers:

            # Get layer's kwargs
            kwargs = cfg_layers[cfg_layer]

            # Get layer
            layer = cfg_layer.split("_")[0]
            layer = get_layer(layer=layer)

            # Append and update in channels
            if not layer is nn.MaxPool2d:
                # Append
                kwargs["in_channels"] = in_channels

                # Update
                out_channels = kwargs["out_channels"]
                in_channels = out_channels[-1] if isinstance(out_channels, list) else out_channels

            # Initialize
            layer = layer(**kwargs)

            # Append module
            model.append(layer)

        return nn.Sequential(*model)

    # --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass on the  input tensor

        Args:
            x: (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor

        """

        out = self.fc(self.yolo(x))

        # Reshape so that the output = (cells * cells * num_features)
        out = out.view(out.shape[0], self.num_cells, self.num_cells, self.num_cell_features)

        return out

    # --------------------------------------------------------------------------------
    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int):
        """Training Step

        Args:
            batch: (torch.Tensor): batch of data
            batch_idx: (int): Batch index
        """
        raise NotImplementedError("Training step not implemented")

    # --------------------------------------------------------------------------------
    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int):
        """Validation Step

        Args:
            batch: (torch.Tensor): batch of data
            batch_idx: (int): Batch index
        """
        raise NotImplementedError("Validation step not implemented")

    # --------------------------------------------------------------------------------
    def test_step(self,
                  batch: torch.Tensor,
                  batch_idx: int):
        """Test Step

        Args:
            batch: (torch.Tensor): batch of data
            batch_idx: (int): Batch index
        """
        raise NotImplementedError("Test step not implemented")

    # --------------------------------------------------------------------------------
    def predict_step(self,
                     batch: Any,
                     batch_idx: int,
                     dataloader_idx: int = 0):
        """Prediction Step

        Args:
            batch: (Any): batch of data
            batch_idx: (int): Batch index
            dataloader_idx: (int): Index of the current data loader
        """
        raise NotImplementedError("Prediction step not implemented")

    # --------------------------------------------------------------------------------
    def __str__(self):
        """String representation of the class
        """
        return "Yolo V1"
