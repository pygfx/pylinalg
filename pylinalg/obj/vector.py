import numpy as np

from .base import LinalgBase
from ..func import vector_add_vector, vector_mul_scalar, vector_mul_vector


class Vector(LinalgBase):
    """A representation of a direction and length in 3D Euclidean space."""

    _n = 3

    def __init__(self, dx, dy, dz):
        self._val = np.array([dx, dy, dz], "f8")

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
