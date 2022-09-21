import numpy as np

from .base import LinalgBase
from .vector import Vector
from ..func import vector_add_vector


class Point(LinalgBase):
    """A representation of a location in 3D Euclidean space."""

    _n = 3

    def __init__(self, x, y, z):
        self._val = np.array([x, y, z], dtype="f8")

    def __add__(self, vector):
        if isinstance(vector, LinalgBase):
            if not isinstance(vector, Vector):
                raise TypeError("Can only add a Vector to a Point")
            arr = vector._val
        else:
            arr = np.asanyarray(vector)
        assert arr.shape == (3,)
        new = Point(0, 0, 0)
        vector_add_vector(self._val, arr, out=new._val)
        return new
