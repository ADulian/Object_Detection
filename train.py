import logging

from managers.data_manager import DataManager
from managers.model_manager import ModelManager
from managers.train_manager import TrainManager
from utils.argparser import parse_args

# Setup Logging
log = logging.getLogger("lightning")
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------------
def main():
    """ Main
    """

    from models.yolo_v1.criterion import YoloV1Criterion

    l = YoloV1Criterion()
    import torch
    y = torch.randn(10,7,7,6)
    y_hat = torch.randn(10,7,7,90)
    # l(y=y, y_hat=y_hat)

    # Parse arguments
    args = parse_args()

    # Init Data Manager
    dm = DataManager(coco_root_path=args.coco_path,
                     train_batch_size=args.train_bs,
                     val_batch_size=args.val_bs,
                     test_batch_size=args.test_bs,
                     shuffle=args.shuffle,
                     pin_memory=args.pin_memory,
                     num_workers=args.num_workers)

    # Init Model Manager
    mm = ModelManager(num_classes=dm.num_classes,
                      lr=args.lr)

    # Init Trainer Manager
    tm = TrainManager(data_manager=dm,
                      model_manager=mm,
                      epochs=args.epochs,
                      accelerator=args.accelerator,
                      devices=args.devices)

    # Train
    tm()


# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
