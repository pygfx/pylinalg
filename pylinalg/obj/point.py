import numpy as np

from .base import LinalgBase
from .vector import Vector
from ..func import vector_add_vector


class Point(LinalgBase):
    """A representation of a location in 3D Euclidean space."""

    def __init__(self, x=0, y=0, z=0, /, *, dtype="f8"):
        self._val = np.array([x, y, z], dtype=dtype)

    @property
    def x(self):
        return self._val[0]

    @x.setter
    def x(self, val):
        self._val[0] = val

    @property
    def y(self):
        return self._val[1]

    @y.setter
    def y(self, val):
        self._val[1] = val

    @property
    def z(self):
        return self._val[2]

    @z.setter
    def z(self, val):
        self._val[2] = val

    def set(self, x, y, z):
        self._val[:] = x, y, z

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
