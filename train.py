import logging

from managers.data_manager import DataManager

# Setup Logging
log = logging.getLogger("lightning")
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

# --------------------------------------------------------------------------------
def main():
    """ Main
    """


    dm = DataManager()

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
