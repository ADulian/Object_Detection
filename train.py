import logging

from managers.data_manager import DataManager

# Setup Logging
log = logging.getLogger("lightning")
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------------
def main():
    """ Main
    """


    dm = DataManager()
    dm.setup()

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
