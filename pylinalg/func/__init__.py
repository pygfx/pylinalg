"""
Functional API

These functions have rather verbose names, but it makes things
explicit. Each function accepts either singletons or arrays of
"things", and uses Numpy's broadcasting to just make it work. Each
function also accepts an out argument.

This API is for internal use and for power-users that want to
vectorize operations on large sets of things.

Requirements for all functions in this subpackage are as follows:

* Support out parameters for all return values, to enable in-place operations
* Inherit dtype from input parameters, or support an optional keyword dtype
  in creation routines
"""

from .matrix import *
from .quaternion import *
from .vector import *
