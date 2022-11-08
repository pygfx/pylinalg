from functools import partial

import numpy as np

from ..func import (
    matrix_compose,
    matrix_decompose,
    matrix_inverse,
    matrix_make_orthographic,
    matrix_make_perspective,
)
from .base import LinalgBase


class Matrix(LinalgBase):
    """A representation of a transformation matrix in 4D
    homogeneous (to 3D Euclidean) space."""

    _initializer = partial(np.identity, 4)

    def icompose(self, translation, rotation, scaling):
        matrix_compose(translation, rotation, scaling, out=self)
        return self

    @classmethod
    def make_perspective(cls, left, right, top, bottom, near, far, dtype=None):
        return cls(
            matrix_make_perspective(left, right, top, bottom, near, far, dtype=dtype)
        )

    def imake_perspective(self, left, right, top, bottom, near, far):
        matrix_make_perspective(left, right, top, bottom, near, far, out=self.val)

        return self

    @classmethod
    def make_orthographic(cls, left, right, top, bottom, near, far, dtype=None):
        return cls(
            matrix_make_orthographic(left, right, top, bottom, near, far, dtype=dtype)
        )

    def imake_orthographic(self, left, right, top, bottom, near, far):
        matrix_make_orthographic(left, right, top, bottom, near, far, out=self.val)

        return self

    def decompose(self, translation=None, rotation=None, scaling=None):
        from .quaternion import Quaternion
        from .vector import Vector

        if translation is None:
            translation = Vector()
        if rotation is None:
            rotation = Quaternion()
        if scaling is None:
            scaling = Vector()

        matrix_decompose(self, translation, rotation, scaling)
        return translation, rotation, scaling

    def inverse(self):
        return Matrix(matrix_inverse(self.val))

    def iinverse(self):
        self.val[:] = matrix_inverse(self.val)
        return self

    def multiply(self, matrix):
        return self @ matrix

    def imultiply(self, matrix):
        self @= matrix

    def premultiply(self, matrix):
        return matrix @ self

    def ipremultiply(self, matrix):
        self.val[:] = matrix @ self.val

    def __imatmul__(self, matrix):
        self.val[:] = self.val @ matrix
        return self

    def __matmul__(self, matrix):
        return Matrix(self.val @ matrix)
