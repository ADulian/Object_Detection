import os
import datetime

from pathlib import Path

sub_dir = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")

# ---------------------------------------------------0-----------------------------
def get_output_dir() -> Path:
    """Get global output dir

    The output dir is based on the OBJ_DET_DIR env var
    If env var doesn't exist then the output dir = ./

    Returns:
        Path: Global output dir

    """
    env_var = "OBJ_DET_DIR"
    if env_var in os.environ:
        global_dir = Path(os.environ[env_var]) / "run" / sub_dir

        return global_dir

    return Path("./run") / sub_dir