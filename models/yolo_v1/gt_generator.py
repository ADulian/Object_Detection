import os

from torch.utils.data import Dataset

from datasets.coco_dataset import CocoDataset
from models.common.tools import get_cfg

# --------------------------------------------------------------------------------
class YoloV1GTGenerator(Dataset):
    """Ground truth generator for YoloV1
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 dataset: CocoDataset):
        """Initialize generator
        """

        # Cfg
        self._cfg = get_cfg(root_path=os.path.dirname(os.path.abspath(__file__)))

        # Dataset
        self._dataset = dataset

    # --------------------------------------------------------------------------------
    def __len__(self) -> int:
        """Get length of dataset

        Return:
             int: Length of dataset
        """

        return len(self._dataset)

    # --------------------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """

        Args:
            idx: (int): Index of the sample

        Returns:
            tuple[PIL.Image.Image, CocoDataSample]: An image and a data sample information

        """

        # Get data sample
        img, data_sample = self._dataset[idx]

        # Create ground truth data

        ...

