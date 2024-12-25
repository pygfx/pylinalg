"""
pylinalg

Linear algebra utilities for Python.
"""

from importlib.metadata import version

from .matrix import *
from .misc import *
from .quaternion import *
from .vector import *

__version__ = version("pylinalg")
version_info = tuple(map(int, __version__.split(".")))

del version


__all__ = [
    name for name in globals() if name.startswith(("vec_", "mat_", "quat_", "aabb_"))
]
