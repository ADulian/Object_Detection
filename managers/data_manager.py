"""
ToDo:
    - Loading COCO dataset
    - Incorporating Lightning Data Module
    - Setup Train/Val/Test Loaders

"""

import logging

import lightning as L

log = logging.getLogger("lightning")

# --------------------------------------------------------------------------------
class DataManager(L.LightningDataModule):
    """Data Manager class

    General management of datasets
    """
    # --------------------------------------------------------------------------------
    def __init__(self):
        """Initialize Data Manager
        """
        super().__init__()

        log.info("Data Manager Initialized")

    # --------------------------------------------------------------------------------
    def prepare_data(self) -> None:
        ...

    # --------------------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        ...

    # --------------------------------------------------------------------------------
    def train_dataloader(self) -> None:
        ...

    # --------------------------------------------------------------------------------
    def val_dataloader(self) -> None:
        ...

    # --------------------------------------------------------------------------------
    def test_dataloader(self) -> None:
        ...





