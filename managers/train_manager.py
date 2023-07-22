import logging

import lightning as L

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
                 model_manager: ModelManager,
                 epochs: int = 1,
                 accelerator: str ="auto",
                 devices: int = 1):
        """Initialize Train Manager

        Args:
            data_manager: (DataManager): Data Manager
            model_manager: (ModelManager): Model Manager
            epochs: (int): Number of training epochs
            accelerator: (str): Accelerator type, e.g. gpu, cpu
            devices: (int): Number of devices for training, e.g. number of gpus
        """

        self._data_manager = data_manager
        self._model_manager = model_manager
        self._trainer = L.Trainer(max_epochs=epochs,
                                  accelerator=accelerator,
                                  devices=devices)

    # --------------------------------------------------------------------------------
    def __call__(self):
        """Main Train function
        """

        log.info(f"Fitting the {self._model_manager.model} for {self._trainer.max_epochs} epochs")

        # Update data manager
        self._data_manager.update(gt_generator=self._model_manager.gt_generator)

        # Fit the model
        self._trainer.fit(model=self._model_manager.model,
                          datamodule=self._data_manager)
