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
    parser.add_argument("--train_bs", type=int, default=1, help="Train batch size")
    parser.add_argument("--val_bs", type=int, default=1, help="Validation batch size")
    parser.add_argument("--test_bs", type=int, default=1, help="Test batch size")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle train dataset")
    parser.add_argument("--pin_memory", type=bool, default=False, help="Pin memory")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")

    return parser.parse_args()