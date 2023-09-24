
# --------------------------------------------------------------------------------
class DimensionLengthError(Exception):
    """Exception for handling incorrect length of dimensions in a tensor
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 dim_length: int,
                 target_dim_length: int,
                 add_msg: str = ""):
        """

        Args:
            dim_length: (int): Actual length of dimensions
            target_dim_length: (int): Expected length of dimensions
            add_msg: (str): Optional, additional message
        """
        super().__init__(f"Incorrect length of dimensions! Expected: {target_dim_length}, Actual: {dim_length}. "
                         + add_msg)
