"""
pylinalg

Linear algebra utilities for Python.
"""

# flake8: noqa
from importlib.metadata import version

__version__ = version("pylinalg")
version_info = tuple(map(int, __version__.split(".")))


from . import matrix, misc, quaternion, vector
from .matrix import *
from .misc import *
from .quaternion import *
from .vector import *

__all__ = matrix.__all__ + misc.__all__ + quaternion.__all__ + vector.__all__
