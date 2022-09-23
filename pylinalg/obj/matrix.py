import numpy as np

from .base import LinalgBase
from ..func import (
    matrix_compose,
    matrix_decompose,
    matrix_inverse,
)


class Matrix(LinalgBase):
    """A representation of a transformation matrix in 4D
    homogeneous (to 3D Euclidean) space."""

    def __init__(self, matrix=None, /, *, dtype=None):
        if matrix is not None:
            if dtype is None:
                self._val = np.asarray(matrix)
            else:
                self._val = np.asarray(matrix, dtype=dtype)
        else:
            if dtype is None:
                dtype = "f8"
            self._val = np.identity(4, dtype=dtype)

    def icompose(self, translation, rotation, scaling):
        matrix_compose(translation, rotation, scaling, out=self)
        return self

    def decompose(self, translation=None, rotation=None, scaling=None):
        from .vector import Vector
        from .quaternion import Quaternion

        if translation is None:
            translation = Vector()
        if rotation is None:
            rotation = Quaternion()
        if scaling is None:
            scaling = Vector()

        matrix_decompose(self, translation, rotation, scaling)
        return translation, rotation, scaling

    def inverse(self):
        return Matrix(matrix_inverse(self._val))

    def iinverse(self):
        self._val[:] = matrix_inverse(self._val)
        return self

    def multiply(self, matrix):
        return self @ matrix

    def imultiply(self, matrix):
        self @= matrix

    def premultiply(self, matrix):
        return matrix @ self

    def ipremultiply(self, matrix):
        self._val[:] = matrix @ self._val

    def __imatmul__(self, matrix):
        self._val @= matrix

    def __matmul__(self, matrix):
        return self._val @ matrix
