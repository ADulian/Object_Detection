import os
import yaml
import torch
import lightning as L

from torch import nn

from utils.io import path_check
from models.common.utils import get_layer

# --------------------------------------------------------------------------------
class YoloV1(L.LightningModule):
    """Yolo V1
    Ref: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 num_bboxes: int = 2,
                 num_classes: int = 20):
        """Initialize Yolo

        Args:
            num_bboxes: (int): Number of bounding boxes per cell
            num_classes: (int): Number of classes
        """
        super().__init__()

        # Yolo Attribs
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes
        self.num_cells = 7
        self.num_cell_features = (self.num_bboxes * 5 + self.num_classes)

        # Initialize Model
        self.yolo = self._parse_model()
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=(self.num_cells * self.num_cells * 1024),
                                          out_features=4096),
                                nn.ReLU(),
                                nn.Linear(in_features=4096,
                                          out_features=(self.num_cells * self.num_cells * self.num_cell_features)))

    # --------------------------------------------------------------------------------
    @staticmethod
    def _get_cfg() -> dict:
        """Get config

        Returns:
            dict: model config
        """

        root_path = path_check(os.path.dirname(os.path.abspath(__file__)))
        cfg = root_path / "config.yaml"

        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)

        return cfg

    # --------------------------------------------------------------------------------
    def _parse_model(self) -> nn.Sequential:
        """Parse model from config

        Returns:
            nn.Sequential: Parsed Yolo model
        """
        model = []
        in_channels = 3
        cfg_layers = self._get_cfg()["layers"]

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
    def _init_fc_layers(self):
        ...

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
    def training_step(self):
        ...

    # --------------------------------------------------------------------------------
    def validation_step(self):
        ...

    # --------------------------------------------------------------------------------
    def test_step(self):
        ...

    # --------------------------------------------------------------------------------
    def prediction_step(self):
        ...
