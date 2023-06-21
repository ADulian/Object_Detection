from pathlib import Path

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