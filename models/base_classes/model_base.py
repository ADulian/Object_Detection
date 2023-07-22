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
    def training_step(self, *args, **kwargs):
        """Training Step
        """
        raise NotImplementedError("Training step not implemented")

    # --------------------------------------------------------------------------------
    def validation_step(self, *args, **kwargs):
        """Validation Step
        """
        raise NotImplementedError("Validation step not implemented")

    # --------------------------------------------------------------------------------
    def test_step(self, *args, **kwargs):
        """Test Step
        """
        raise NotImplementedError("Test step not implemented")

    # --------------------------------------------------------------------------------
    def predict_step(self, *args, **kwargs):
        """Prediction Step
        """
        raise NotImplementedError("Prediction step not implemented")

    # --------------------------------------------------------------------------------
    def configure_optimizers(self) -> Any:
        """Configure optimiser
        """

        return torch.optim.Adam(params=self.parameters(),
                                lr=self.lr)