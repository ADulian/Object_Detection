import argparse

# --------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse arguments

    Returns:
        argparse.Namespace: Parse arguments
    """

    parser = argparse.ArgumentParser(description="Framework for developing, training and testing of object "
                                                 "detection models")

    parser.add_argument("--coco_path", type=str, default="/media/albert/FastData/Datasets/COCO",
                        help="Root path of Coco Dataset")

    return parser.parse_args()