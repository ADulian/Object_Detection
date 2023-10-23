import logging

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path

from managers.data_manager import DataManager
from managers.model_manager import ModelManager
from utils.dir import get_output_dir

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
                 devices: int = 1,
                 ckpt_path: str = ""):
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
        self._logger = WandbLogger(project="Dummy")
        self._output_dir = get_output_dir()
        self._ckpt_path = self._check_checkpoint_exists(ckpt_path=Path(ckpt_path))

        self._trainer = L.Trainer(max_epochs=epochs,
                                  logger=self._logger,
                                  callbacks=self._get_callbacks(),
                                  accelerator=accelerator,
                                  devices=devices,
                                  log_every_n_steps=1)

    # --------------------------------------------------------------------------------
    def __call__(self):
        """Main Train function
        """

        log.info(f"Fitting the {self._model_manager.model} for {self._trainer.max_epochs} epochs")

        # Update data manager
        self._data_manager.update(gt_generator=self._model_manager.gt_generator)

        # Watch Model
        self._logger.watch(self._model_manager.model,
                           log_freq=1)

        # Fit the model
        self._trainer.fit(model=self._model_manager.model,
                          datamodule=self._data_manager,
                          ckpt_path=str(self._ckpt_path))

    # ---------------------------------------------------0-----------------------------
    def _get_callbacks(self) -> list:
        """ Initialize Trainer Callbacks

        """
        callbacks = [
            self._set_checkpoint_callback(),
        ]

        return callbacks


    # --------------------------------------------------------------------------------
    def _set_checkpoint_callback(self) -> ModelCheckpoint:
        """Create Model Checkpoint callback


        Returns:
            ModelCheckpoint: Checkpoint callback
        """

        # Define Checkpoint Callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self._output_dir / "checkpoints",
            filename="{epoch}",
            monitor="val_loss",
            save_top_k=1,
            verbose=False,
            save_last=True,
        )

        return checkpoint_callback

    # --------------------------------------------------------------------------------
    def _check_checkpoint_exists(self,
                                 ckpt_path: (Path | str)) -> Path:
        """Check if checkpoint is not empty and if exists

        Args:
            ckpt_path: (Path | str): Checkpoint path

        Returns:
            Path: Checked checkpoint path
        """

        # Check if checkpoint is provided
        if str(ckpt_path):
            # Ensure path is of Path object
            if not isinstance(ckpt_path, Path):
                ckpt_path = Path(ckpt_path)

            # Ensure that ckpt ends with correct suffix
            ckpt_path = ckpt_path.with_suffix(".ckpt")

            # Check if exists
            if not ckpt_path.exists():
                raise ValueError(f"Checkpoint at {ckpt_path} not found!")

        return ckpt_path
