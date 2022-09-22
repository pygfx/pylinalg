"""
Functional API

These functions have rather verbose names, but it makes things
explicit. Each function accepts either singletons or arrays of
"things", and uses Numpy's broadcasting to just make it work. Each
function also accepts an out argument.

This API is for internal use and for power-users that want to
vectorize operations on large sets of things.
"""

from .matrix import *
from .quaternion import *
from .vector import *
