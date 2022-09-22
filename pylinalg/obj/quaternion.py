import numpy as np

from .base import LinalgBase


class Quaternion(LinalgBase):
    """A representation of a rotation in 3D Euclidean space."""

    def __init__(self, x=0, y=0, z=0, w=0, /, *, dtype="f8"):
        self._val = np.array([x, y, z, w], dtype=dtype)

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

    @property
    def w(self):
        return self._val[3]

    @w.setter
    def w(self, val):
        self._val[3] = val
