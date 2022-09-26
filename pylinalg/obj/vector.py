import numpy as np

from .base import LinalgBase
from ..func import vector_add_vector, vector_mul_scalar, vector_mul_vector


class Vector(LinalgBase):
    """A representation of a direction and length in 3D Euclidean space."""

    def __init__(self, x=0, y=0, z=0, /, *, dtype="f8"):
        self._val = np.array([x, y, z], dtype=dtype)

    def copy(self):
        return self.__class__(*self._val.copy(), dtype=self._val.dtype)

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

    def __add__(self, vector):
        if isinstance(vector, LinalgBase):
            if not isinstance(vector, Vector):
                raise TypeError("Can only add a Vector to a Vector")
            arr = vector._val
        else:
            arr = np.asanyarray(vector)
        assert arr.shape == (4,)
        new = Vector(0, 0, 0)
        vector_add_vector(self._val, arr, out=new._val)
        return new

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new = Vector(0, 0, 0)
            vector_mul_scalar(self._val, other, out=new._val)
            return new
        else:
            if isinstance(other, LinalgBase):
                # if not isinstance(other, Scalor):
                #     raise TypeError(
                #         "Can only multiply a Vector with a scalar or Scalor."
                #     )
                arr = other._val
            else:
                arr = np.asanyarray(other)
            assert arr.shape == (3,)
            new = Vector(0, 0, 0)
            vector_mul_vector(self._val, arr, out=new._val)
            return new

    def __rmul__(self, other):
        return Vector.__mul__(self, other)
