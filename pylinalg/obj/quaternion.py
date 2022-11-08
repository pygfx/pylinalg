from functools import partial

import numpy as np

from ..func import (
    matrix_to_quaternion,
    quaternion_add_quaternion,
    quaternion_from_axis_angle,
    quaternion_from_unit_vectors,
    quaternion_inverse,
    quaternion_multiply_quaternion,
    quaternion_norm,
    quaternion_subtract_quaternion,
    quaternion_to_matrix,
)
from .base import LinalgBase


class Quaternion(LinalgBase):
    """A representation of a rotation in 3D Euclidean space."""

    _initializer = partial(np.array, [0, 0, 0, 1])

    @property
    def x(self):
        return self.val[0]

    @x.setter
    def x(self, val):
        self.val[0] = val

    @property
    def y(self):
        return self.val[1]

    @y.setter
    def y(self, val):
        self.val[1] = val

    @property
    def z(self):
        return self.val[2]

    @z.setter
    def z(self, val):
        self.val[2] = val

    @property
    def w(self):
        return self.val[3]

    @w.setter
    def w(self, val):
        self.val[3] = val

    def to_matrix(self):
        from .matrix import Matrix

        m = Matrix(dtype=self.dtype)
        quaternion_to_matrix(self.val, out=m.val)
        return m

    @classmethod
    def from_matrix(cls, matrix, dtype=None):
        return cls(*matrix_to_quaternion(matrix), dtype=dtype)

    def ifrom_matrix(self, matrix):
        matrix_to_quaternion(matrix, out=self.val)

        return self

    @classmethod
    def from_unit_vectors(cls, a, b, dtype=None):
        return cls(*quaternion_from_unit_vectors(a, b, dtype=dtype), dtype=dtype)

    def ifrom_unit_vectors(self, a, b):
        quaternion_from_unit_vectors(a, b, out=self.val)

        return self

    @classmethod
    def from_axis_angle(cls, axis, angle, dtype=None):
        return cls(*quaternion_from_axis_angle(axis, angle, dtype=dtype), dtype=dtype)

    def ifrom_axis_angle(self, axis, angle):
        quaternion_from_axis_angle(axis, angle, out=self.val)

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

    def add(self, quaternion):
        return Quaternion(*quaternion_add_quaternion(self.val, quaternion))

    def iadd(self, quaternion):
        quaternion_add_quaternion(self, quaternion, out=self.val)
        return self

    def __add__(self, quaternion):
        return self.add(quaternion)

    def __iadd__(self, quaternion):
        return self.iadd(quaternion)

    def subtract(self, quaternion):
        return Quaternion(*quaternion_subtract_quaternion(self.val, quaternion))

    def isubtract(self, quaternion):
        quaternion_subtract_quaternion(self, quaternion, out=self.val)
        return self

    def __sub__(self, quaternion):
        return self.subtract(quaternion)

    def __isub__(self, quaternion):
        return self.isubtract(quaternion)

    def norm(self):
        return quaternion_norm(self.val)

    def normalize(self):
        return Quaternion(*(self.val / quaternion_norm(self.val)))

    def inormalize(self):
        self.val /= quaternion_norm(self.val)
        return self

    def inverse(self):
        return Quaternion(*quaternion_inverse(self.val))

    def iinverse(self):
        quaternion_inverse(self.val, out=self.val)
        return self
