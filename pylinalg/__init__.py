"""
pylinalg

Linear algebra utilities for Python.
"""

# flake8: noqa
from importlib.metadata import version  


__version__ = version('pylinalg') 
version_info = tuple(map(int, __version__.split(".")))


from .func import *
from .obj import *
