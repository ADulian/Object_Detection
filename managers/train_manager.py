import logging

from managers.data_manager import DataManager
from managers.model_manager import ModelManager

log = logging.getLogger("lightning")

# --------------------------------------------------------------------------------
class TrainManager:
    """Train Manager
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 data_manager: DataManager,
                 model_manager: ModelManager):
        """Initialize Train Manager

        Args:
            data_manager: (DataManager): Data Manager
            model_manager: (ModelManager): Model Manager
        """

        self.data_manager = data_manager
        self.model_manager = model_manager

    # --------------------------------------------------------------------------------
    def __call__(self):
        """Main Train function
        """
        ...
