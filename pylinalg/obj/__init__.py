"""
Object API

In this API each "thing" is represented as one object.
These objects are array-like and iterable to make them easy to
convert to native Python/Numpy objects. The objects support mul
and add where applicable, and have methods specific to the type of
object.

This API should make any linalg work much easier and safer, partly
because semantics matters here: a point is not the same as a vector.
"""

from .base import *
from .point import *
from .vector import *
