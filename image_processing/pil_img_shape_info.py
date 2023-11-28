

# --------------------------------------------------------------------------------
class PILImageShapeInfo:
    """
    A data structure for storing information about changes made to a PIL Image
    """

    # --------------------------------------------------------------------------------
    def __init__(self,
                 original_height: int,
                 original_width: int,
                 final_height: int = 0,
                 final_width: int = 0,
                 resize_scale: float = 1.) -> None:
        """Initialise Shape Info object

        Args:
            original_height: (int): Original height
            original_width: (int): Original width
            final_height: (int): Final height after resize and padding
            final_width: (int): Final width after resize and padding
            resize_scale: (float): Scale factor used to go from original -> new size


        """

        self._original_height = original_height
        self._original_width = original_width

        self._final_height = final_height
        self._final_width = final_width

        self._resize_scale = resize_scale

    # --------------------------------------------------------------------------------
    @property
    def original_height(self) -> int:
        """ Original height of an Image before applying any operation

        Returns:
            int: Original height of an Image before applying any operation

        """

        return self._original_height

    # --------------------------------------------------------------------------------
    @property
    def original_width(self) -> int:
        """ Original width of an Image before applying any operation

        Returns:
            int: Original width of an Image before applying any operation

        """

        return self._original_width

    # --------------------------------------------------------------------------------
    @property
    def new_height(self) -> int:
        """ Height of an Image after resizing

        Returns:
            int: Height of an Image after resizing

        """

        return int(self._original_height * self.resize_scale)

    # --------------------------------------------------------------------------------
    @property
    def new_width(self) -> int:
        """ Width of an Image after resizing

        Returns:
            int: Width of an Image after resizing

        """

        return int(self._original_width * self.resize_scale)

    # --------------------------------------------------------------------------------
    @property
    def final_height(self) -> int:
        """ Final height of an Image after resizing and padding

        Returns:
            int: Final height of an Image after resizing and padding
        """
        return self._final_height

    # --------------------------------------------------------------------------------
    @property
    def final_width(self) -> int:
        """ Final width of an Image after resizing and padding

        Returns:
            int: Final width of an Image after resizing and padding
        """

        return self._final_width

    # --------------------------------------------------------------------------------
    @property
    def padding_height(self) -> int:
        """Amount of padding applied to height post resizing

        Note that the amount represents half of total padding
        as the assumption is that image is being padded equally on top and bottom (center)

        Returns:
            int: Amount of padding applied to height
        """

        return (self.final_height - self.new_height) // 2

    # --------------------------------------------------------------------------------
    @property
    def padding_width(self) -> int:
        """Amount of padding applied to width post resizing

        Note that the amount represents half of total padding
        as the assumption is that image is being padded equally from left and right (center)

        Returns:
            int: Amount of padding applied to height
        """

        return (self.final_width - self.new_width) // 2

    # --------------------------------------------------------------------------------
    @property
    def resize_scale(self) -> float:
        """Get resize scale between old and new size

        Note that it is assumed that whilst resizing an Image, the aspect ratio
        of both sides remaints equal, thus, resize scale applies to both height and width

        Returns:
            float: A resize scale between old and new size
        """
        return self._resize_scale
