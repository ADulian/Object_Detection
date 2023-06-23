from pathlib import Path
from PIL import Image

# --------------------------------------------------------------------------------
def path_check(path: str) -> Path:
    """Ensure path is of Path instance
    Args:
        path: (str): Path

    Returns:
        Path: Path object
    """
    if not isinstance(path, Path):
        path = Path(path)

    return path

# --------------------------------------------------------------------------------
def load_image(path: (str | Path)) -> Image.Image:
    """

    Args:
        path: (str | Path): Image path

    Returns:
        Image.Image: Loaded PIL Image
    """

    path = path_check(path)

    # Check if file exists
    if not path.is_file():
        raise ValueError(f"Image at {path} not found")

    img = Image.open(path)

    return img
