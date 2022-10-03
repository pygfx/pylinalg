import numpy as np

from .base import LinalgBase
from ..func import (
    matrix_to_quaternion,
    quaternion_from_unit_vectors,
    quaternion_multiply_quaternion,
    quaternion_norm,
    quaternion_to_matrix,
)


class Quaternion(LinalgBase):
    """A representation of a rotation in 3D Euclidean space."""

    def __init__(self, x=0, y=0, z=0, w=1, /, *, dtype="f8"):
        self._val = np.array([x, y, z, w], dtype=dtype)

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

    @property
    def w(self):
        return self._val[3]

    @w.setter
    def w(self, val):
        self._val[3] = val

    def to_matrix(self):
        from .matrix import Matrix

        m = Matrix(dtype=self.dtype)
        quaternion_to_matrix(self._val, out=m._val)
        return m

    @classmethod
    def from_matrix(cls, matrix, dtype=None):
        return cls(*matrix_to_quaternion(matrix), dtype=dtype)

    def ifrom_matrix(self, matrix):
        matrix_to_quaternion(matrix, out=self._val)

        return self

    @classmethod
    def from_unit_vectors(cls, a, b, dtype=None):
        return cls(*quaternion_from_unit_vectors(a, b, dtype=dtype), dtype=dtype)

    def ifrom_unit_vectors(self, a, b):
        quaternion_from_unit_vectors(a, b, out=self._val)

        return self

    def multiply(self, quaternion):
        return Quaternion(*quaternion_multiply_quaternion(self, quaternion))

    def imultiply(self, quaternion):
        return quaternion_multiply_quaternion(self, quaternion, out=self)

    def premultiply(self, quaternion):
        return Quaternion(*quaternion_multiply_quaternion(quaternion, self))

    def ipremultiply(self, quaternion):
        return quaternion_multiply_quaternion(quaternion, self, out=self)

    def __mul__(self, quaternion):
        return self.multiply(quaternion)

    def __imul__(self, quaternion):
        return self.imultiply(quaternion)

    def norm(self):
        return quaternion_norm(self._val)

    def normalize(self):
        return Quaternion(*(self._val / quaternion_norm(self._val)))

    def inormalize(self):
        self._val /= quaternion_norm(self._val)
        return self
