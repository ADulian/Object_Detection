import json
import logging
from pathlib import Path
from collections import defaultdict

import PIL.Image
from torch.utils.data import Dataset

from datasets.coco_data_sample import CocoDataSample
from datasets.base_classes import DatasetSplit
from utils.io import load_image

log = logging.getLogger("lightning")

# --------------------------------------------------------------------------------
class CocoDataset(Dataset):
    """Coco Dataset Wrapper
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 dataset_split: DatasetSplit,
                 imgs_path: Path,
                 annotations_file: (Path | None) = None):
        """Initialize Coco Dataset

        Args:
            dataset_split: (DatasetSplit): Dataset split
            imgs_path: (Path): Root path to images
            annotations_file: (Path | None): Path to annotations file

        """

        self._dataset_split = dataset_split
        self._imgs_path = imgs_path
        self._data_samples, self.idx_to_class, self.class_to_idx = self._load_data(annotations_file=annotations_file)

        split = dataset_split.name[0] + dataset_split.name[1:].lower()
        log.info(f"{split} Coco Dataset Initialised")

    # --------------------------------------------------------------------------------
    @staticmethod
    def _load_data(annotations_file: (Path | None) = None) -> tuple[list[CocoDataSample],
                                                                    dict[int, str],
                                                                    dict[str, int]]:

        """Load img and annotation data for coco

        Args:
            annotations_file: (Path | None): Root path to annotations if file exists

        Returns:
            tuple[list[CocoDataSample], dict[int, str]]: List of Coco data samples, Class mapping

        """

        # Json Data
        with open(annotations_file, "rb") as f:
            data_json = json.load(f)

        # Limit to only subset of categories (Person, Car)
        limit_cat = [1, 3]
        categories = [d for d in data_json["categories"] if d["id"] in limit_cat]

        # Mappings
        idx_to_class = {idx : cat["name"] for idx, cat in enumerate(categories)}
        class_to_idx = {v : k for k, v in idx_to_class.items()}
        cat_to_class_idx = {cat["id"] : [idx, cat["name"]] for idx, cat in enumerate(categories)}

        imgs_data_json, annotations_data_json = data_json["images"], data_json["annotations"]

        # Img + Annotations
        imgs_data = {}
        annotations_data = defaultdict(list)

        # Populate img data
        for img_data in imgs_data_json:
            imgs_data[img_data["id"]] = img_data

        # Populate annotations
        for annotation_data in annotations_data_json:
            annotations_data[annotation_data["image_id"]].append(annotation_data)

        # Initialize data samples
        data_samples = [CocoDataSample(img_data=img_data,
                                       annotation_data=annotations_data[img_key],
                                       cat_to_class_idx=cat_to_class_idx)
                        for img_key, img_data in imgs_data.items()]

        return data_samples, idx_to_class, class_to_idx

    # --------------------------------------------------------------------------------
    def __len__(self) -> int:
        """Get length of dataset

        Return:
             int: Length of dataset
        """
        return len(self._data_samples)

    # --------------------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[PIL.Image.Image, CocoDataSample]:
        """

        Args:
            idx: (int): Index of the sample

        Returns:
            tuple[PIL.Image.Image, CocoDataSample]: An image and a data sample information

        """

        # Get data sample
        data_sample = self._data_samples[idx]

        # Load image
        img_path = Path(self._imgs_path) / data_sample.img_file_name
        img = load_image(path=img_path)

        return img, data_sample
