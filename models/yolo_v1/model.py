import os
import sys
import yaml
import lightning as L

from utils.io import path_check

from ..common.conv_layers import ConvMaxPoolBlock

# --------------------------------------------------------------------------------
class YoloV1(L.LightningModule):

    # --------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

        self._cfg = self._get_cfg()

        self._parse_model()
        ...

    # --------------------------------------------------------------------------------
    @staticmethod

    def _get_cfg():

        root_path = path_check(os.path.dirname(os.path.abspath(__file__)))
        config_path = root_path / "config.yaml"

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        return cfg

    # --------------------------------------------------------------------------------
    def _parse_model(self):
        layers = self._cfg["layers"]

        for layer in layers:
            module = layer.split("_")[0]
            module = getattr(sys.modules[__name__], module)
            module = module()

            ...

    # --------------------------------------------------------------------------------
    def _init_model(self):
        ...

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
