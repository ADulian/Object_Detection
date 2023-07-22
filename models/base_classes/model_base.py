from typing import Any

import torch
import lightning as L

# --------------------------------------------------------------------------------
class ModelBase(L.LightningModule):
    """Base class for Models
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 lr: float = 1e-3):
        """Initialize model base

        Args:
            lr: (float): Learning rate
        """
        super().__init__()

        self.lr = lr

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
    def configure_optimizers(self) -> Any:
        """Configure optimiser
        """

        return torch.optim.Adam(params=self.parameters(),
                                lr=self.lr)