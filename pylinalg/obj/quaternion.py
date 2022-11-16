from functools import partial

import numpy as np

from ..func import quaternion
from ..func.matrix import matrix_to_quaternion
from .base import LinalgBase


class Quaternion(LinalgBase):
    """A representation of a rotation in 3D Euclidean space."""

    _initializer = partial(np.array, [0.0, 0.0, 0.0, 1.0])

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

    @classmethod
    def from_matrix(cls, matrix):
        return matrix_to_quaternion(matrix, out=cls())

    @classmethod
    def from_unit_vectors(cls, a, b):
        return cls(quaternion.quaternion_make_from_unit_vectors(a, b))

    @classmethod
    def from_axis_angle(cls, axis, angle):
        return cls(quaternion.quaternion_make_from_axis_angle(axis, angle))

    def __mul__(self, other):
        return Quaternion(quaternion.quaternion_multiply(self, other))

    def __imul__(self, other):
        return quaternion.quaternion_multiply(self, other, out=self)

    def norm(self):
        return np.linalg.norm(self.val)

    def normalize(self):
        return Quaternion(self.val / self.norm())

    def inverse(self):
        return Quaternion(quaternion.quaternion_inverse(self.val))
