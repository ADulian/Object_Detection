from typing import Any

import torch
import lightning as L

# --------------------------------------------------------------------------------
class ModelBase(L.LightningModule):
    """Base class for Models
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4):
        """Initialize model base

        Args:
            lr: (float): Learning rate
            weight_decay: (float): Weight decay
        """
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

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

        Advised to use per batch of samples, e.g. a batch of Images
        """
        raise NotImplementedError("Prediction step not implemented")

    # --------------------------------------------------------------------------------
    def inference_step(self, *args, **kwargs):
        """Inference Step

        Advised to use per single sample, e.g. an Image
        """
        raise NotImplementedError("Inference step not implemented")

    # --------------------------------------------------------------------------------
    def configure_optimizers(self) -> Any:
        """Configure optimiser
        """

        return torch.optim.AdamW(params=self.parameters(),
                                 lr=self.lr,
                                 weight_decay=self.weight_decay)