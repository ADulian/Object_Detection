import logging

from managers.data_manager import DataManager
from utils.argparser import parse_args


# --- Temp
from models.yolo_v1.model import YoloV1
from models.yolo_v1.gt_generator import YoloV1GTGenerator

# Setup Logging
log = logging.getLogger("lightning")
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------------
def main():
    """ Main
    """

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

    # Init Model
    model = YoloV1(num_classes=dm.num_classes)

    # Init Target Generator
    gt = YoloV1GTGenerator(dm._val_set)

    for g in gt:
        ...
    # next(iter(gt))



# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
