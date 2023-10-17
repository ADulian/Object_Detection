import numpy as np

from custom_types.bbox import BBox, BBoxFormat

# --------------------------------------------------------------------------------
class CocoDataSample:
    """Wrapper for Coco Samples
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 img_data: dict,
                 annotation_data: list[dict],
                 cat_to_class_idx: dict[str, list[int, str]]):

        """Initialize Coco Data Sample

        Args:
            img_data: (dict): Img related data
            annotation_data: (list[dict]): A list of annotations per sample
            cat_to_class_idx: (dict[str, list[int, str]]): A dictionary mapping from cat_id to idx, cat_name

        """

        # Image Data
        self.img_id = img_data["id"]
        self.img_file_name = img_data["file_name"]
        self.img_height = img_data["height"]
        self.img_width = img_data["width"]

        # Annotations Data
        self.annotations = self._get_annotations(annotation_data=annotation_data,
                                                 cat_to_class_idx=cat_to_class_idx)


    # --------------------------------------------------------------------------------
    def get_bboxes(self) -> np.ndarray:
        """Get an array of bounding boxes for all annotations

        Returns:
            np.ndarray: Array of bounding boxes of shape [num_bboxs, 4] where 4 -> [x_min, y_min, width, height]
        """

        return np.stack([annotation.bbox for annotation in self.annotations])

    # --------------------------------------------------------------------------------
    def _get_annotations(self,
                         annotation_data: list[dict],
                         cat_to_class_idx: dict[str, list[int, str]]) -> list:
        """Get annotations if its valid

        if in limited categories then class_idx is not None

        Args:
            annotation_data: (list[dict]): a list of annotation data
            cat_to_class_idx: (dict[str, list[int, str]]): mapping from category id to class idx/name

        Returns:
            list: list of Coco annotations
        """

        annotations = []

        for annotation in annotation_data:
            coco_annotation = CocoAnnotation(annotation=annotation,
                                             cat_to_class_idx=cat_to_class_idx)

            if coco_annotation.class_idx is not None:
                annotations.append(coco_annotation)


        return annotations

# --------------------------------------------------------------------------------
class CocoAnnotation:
    """Wrapper for a single annotation
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 annotation: dict,
                 cat_to_class_idx: dict[str, list[int, str]]):
        """Initialize Coco Annotation

        Args:
            annotation: (dict): A single annotation
            cat_to_class_idx: (dict[str, list[int, str]]): A dictionary mapping from cat_id to idx, cat_name

        """

        self.id = annotation["id"]
        self.image_id = annotation["image_id"]
        self.is_crowd = annotation["iscrowd"]
        self.area = annotation["area"]
        self.cat_id = annotation["category_id"]

        self.class_idx = None
        self.cat_name = None
        class_idx_cat_name = cat_to_class_idx.get(self.cat_id)
        if class_idx_cat_name:
            self.class_idx, self.cat_name = class_idx_cat_name

        self.bbox = BBox(bbox=np.array(annotation["bbox"], dtype=float),
                         bbox_format=BBoxFormat.XYWH)