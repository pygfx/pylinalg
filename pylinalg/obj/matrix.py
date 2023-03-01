from functools import partial

import numpy as np

from ..func import matrix
from ..func.quaternion import quaternion_to_matrix
from .base import LinalgBase


class Matrix(LinalgBase):
    """A representation of a transformation matrix in 4D
    homogeneous (to 3D Euclidean) space."""

    _initializer = partial(np.identity, 4)

    @classmethod
    def transform(cls, translation, rotation, scaling):
        return cls(matrix.matrix_make_transform(translation, rotation, scaling))

    @classmethod
    def translation(cls, vector):
        return cls(matrix.matrix_make_translation(vector))

    @classmethod
    def scaling(cls, factors):
        return cls(matrix.matrix_make_scaling(factors))

    @classmethod
    def rotation_from_axis_angle(cls, axis, angle):
        return cls(matrix.matrix_make_rotation_from_axis_angle(axis, angle))

    @classmethod
    def rotation_from_euler_angles(cls, angles, order="xyz"):
        return cls(matrix.matrix_make_rotation_from_euler_angles(angles, order=order))

    @classmethod
    def perspective(cls, left, right, top, bottom, near, far):
        return cls(matrix.matrix_make_perspective(left, right, top, bottom, near, far))

    @classmethod
    def orthographic(cls, left, right, top, bottom, near, far):
        return cls(matrix.matrix_make_orthographic(left, right, top, bottom, near, far))

    @classmethod
    def combine(cls, matrices):
        return cls(matrix.matrix_combine(matrices))

    def decompose(self):
        from .quaternion import Quaternion
        from .vector import Vector

        return matrix.matrix_decompose(self, out=(Vector(), Quaternion(), Vector()))

    def inverse(self):
        return Matrix(np.linalg.inv(self))

    @classmethod
    def from_quaternion(cls, quaternion):
        return quaternion_to_matrix(quaternion, out=cls())
