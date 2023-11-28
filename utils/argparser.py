import argparse

# --------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse arguments

    Returns:
        argparse.Namespace: Parse arguments
    """

    parser = argparse.ArgumentParser(description="Framework for developing, training and testing of object "
                                                 "detection models")

    # Datasets
    parser.add_argument("--coco_path", type=str, default="/media/albert/FastData/Datasets/COCO",
                        help="Root path of Coco Dataset")
    parser.add_argument("--train_bs", type=int, default=1, help="Train batch size")
    parser.add_argument("--val_bs", type=int, default=1, help="Validation batch size")
    parser.add_argument("--test_bs", type=int, default=1, help="Test batch size")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle train dataset")
    parser.add_argument("--pin_memory", type=bool, default=False, help="Pin memory")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--use_dev_set", type=bool, default=True, help="Whether to use dev set (few images)")

    # Training
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--accelerator", type=str, default="auto", help="Trainer aceelerator type, e.g. gpu, cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of accelerator device, e.g. number of gpus")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight Decay")
    parser.add_argument("--ckpt_path", type=str, default="", help="Checkpoint Path")

    return parser.parse_args()