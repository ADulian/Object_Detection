import json
import logging

from pathlib import Path
from collections import defaultdict

from datasets.coco_data_sample import CocoDataSample

log = logging.getLogger("lightning")


# --------------------------------------------------------------------------------
class CocoDataset:

    # --------------------------------------------------------------------------------
    def __init__(self,
                 imgs_path: Path,
                 annotations_file: (Path | None) = None):

        self._imgs_path = imgs_path
        self._data_samples = self._load_data(annotations_file=annotations_file)

        log.info("Coco Dataset Initialized")

    # --------------------------------------------------------------------------------
    @staticmethod
    def _load_data(annotations_file: (Path | None) = None) -> list[CocoDataSample]:

        """Load img and annotation data for coco

        Args:
            annotations_file: (Path | None): Root path to annotations if file exists

        Returns:
            list[CocoDataSample]: List of Coco data samples

        """

        # Json Data
        with open(annotations_file, "rb") as f:
            data_json = json.load(f)

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
                                       annotation_data=annotations_data[img_key])
                        for img_key, img_data in imgs_data.items()]

        return data_samples

    # --------------------------------------------------------------------------------
    def __len__(self):
        ...

    # --------------------------------------------------------------------------------
    def __getitem__(self):
        ...




