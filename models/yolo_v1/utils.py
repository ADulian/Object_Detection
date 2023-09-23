"""Utility module that contains helpful functions specifically for YoloV1
"""


# --------------------------------------------------------------------------------
def normalize_bbox(mid_x: float, mid_y: float, width: float, height: float,
                   x_cell: int, y_cell: int,
                   cell_size: int, in_size: int):
    """Normalize bbox as defined in YoloV1 paper
    """

    mid_x, mid_y = (mid_x / cell_size) - x_cell, (mid_y / cell_size) - y_cell
    width, height = width / in_size, height / in_size

    return mid_x, mid_y, width, height


# --------------------------------------------------------------------------------
def unnormalize_bbox(mid_x: float, mid_y: float, width: float, height: float,
                     x_cell: int, y_cell: int,
                     cell_size: int, in_size: int):
    """Unnormalize bboc (inverse of normalize function)
    """

    mid_x, mid_y = (mid_x + x_cell) * cell_size, (mid_y + y_cell) * cell_size
    width, height = width * in_size, height * in_size

    return mid_x, mid_y, width, height