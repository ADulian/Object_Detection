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
    dm = DataManager(coco_root_path=args.coco_path)
    dm.setup()

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
