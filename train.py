import logging

from managers.data_manager import DataManager
from utils.argparser import parse_args

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
    dm.setup()

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
