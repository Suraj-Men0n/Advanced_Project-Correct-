"""Resolve package data paths (dataset/, models/) that work whether a node
is launched from the package's source checkout (CWD-relative, convenient
for local dev) or via `ros2 run`/`ros2 launch` from an arbitrary directory
(falls back to the colcon-installed share directory).

Without this, `sim_bridge.py`'s dataset_dir and `spectral_node.py`'s
checkpoint parameters only resolved correctly when the process happened to
be launched with CWD set to the package source root -- silently spawning
zero fragments (sim_bridge just returns early on an empty glob, no error)
or crashing on FileNotFoundError (spectral_node's torch.load) otherwise.
"""
from __future__ import annotations
import os


def resolve_data_path(relative_path: str) -> str:
    """Return an existing path for `relative_path` (e.g. 'dataset' or
    'models/spectral_cnn.pt'), checking in order:
      1. as given -- absolute, or relative to the process's current
         working directory (this is what makes `python3 -m
         tyre_sorting.sim.env` and running tests from the source checkout
         still work unchanged)
      2. relative to this package's colcon-installed share directory
         (what makes `ros2 run`/`ros2 launch` work from any directory)

    Returns the original (possibly non-existent) relative_path unchanged if
    neither resolves, so callers still get a clear "file not found" error
    pointing at the path they actually asked for, rather than a confusing
    one from deep inside ament_index_python.
    """
    if os.path.exists(relative_path):
        return relative_path
    try:
        from ament_index_python.packages import get_package_share_directory
        share_path = os.path.join(get_package_share_directory('tyre_sorting'), relative_path)
        if os.path.exists(share_path):
            return share_path
    except Exception:
        pass  # ament_index_python unavailable outside a ROS 2 environment -- fine, fall through
    return relative_path
