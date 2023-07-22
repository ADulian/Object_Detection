import numpy as np

from custom_types.bbox import BBox, BBoxFormat

# --------------------------------------------------------------------------------
class CocoDataSample:
    """Wrapper for Coco Samples
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 img_data: dict,
                 annotation_data: list[dict]):

        """Initialize Coco Data Sample

        Args:
            img_data: (dict): Img related data
            annotation_data: (list[dict]): A list of annotations per sample

        """

        # Image Data
        self.img_id = img_data["id"]
        self.img_file_name = img_data["file_name"]
        self.img_height = img_data["height"]
        self.img_width = img_data["width"]

        # Annotations Data
        self.annotations = [CocoAnnotation(annotation=annotation) for annotation in annotation_data]

    # --------------------------------------------------------------------------------
    def get_bboxes(self) -> np.ndarray:
        """Get an array of bounding boxes for all annotations

        Returns:
            np.ndarray: Array of bounding boxes of shape [num_bboxs, 4] where 4 -> [x_min, y_min, width, height]
        """

        return np.stack([annotation.bbox for annotation in self.annotations])

# --------------------------------------------------------------------------------
class CocoAnnotation:
    """Wrapper for a single annotation
    """

    # --------------------------------------------------------------------------------
    def __init__(self, annotation: dict):
        """Initialize Coco Annotation
        """

        self.id = annotation["id"]
        self.image_id = annotation["image_id"]
        self.is_crowd = annotation["iscrowd"]
        self.area = annotation["area"]
        self.class_id = annotation["category_id"]
        self.bbox = BBox(bbox=np.array(annotation["bbox"], dtype=float),
                         bbox_format=BBoxFormat.XYWH)