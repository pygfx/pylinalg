"""
pylinalg

Linear algebra utilities for Python.
"""

# flake8: noqa

__version__ = "0.3.2"

version_info = tuple(map(int, __version__.split(".")))


from .func import *
from .obj import *
