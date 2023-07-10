import os
import sys
import yaml
import lightning as L

from torch import nn

from utils.io import path_check
from ..common.conv_block import ConvBlock

# --------------------------------------------------------------------------------
class YoloV1(L.LightningModule):
    """Yolo V1
    Ref: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf
    """

    # --------------------------------------------------------------------------------
    def __init__(self):
        """Initialize Yolo
        """
        super().__init__()

        self._cfg = self._get_cfg()
        self.yolo = self._parse_model()

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
        cfg_layers = self._cfg["layers"]

        for cfg_layer in cfg_layers:
            # Initialize Conv Block
            layers = cfg_layers[cfg_layer]
            conv_block = ConvBlock(in_channels=in_channels, layers=layers)

            # Update in channels
            in_channels = conv_block.out_channels

            # Append module
            model.append(conv_block)

        return nn.Sequential(*model)

    # --------------------------------------------------------------------------------
    def forward(self, x):
        ...

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
