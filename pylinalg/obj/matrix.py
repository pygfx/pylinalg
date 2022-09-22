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

    def compose(self, translation, rotation, scaling):
        matrix_compose(translation._val, rotation._val, scaling._val, out=self._val)
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

        matrix_decompose(self._val, translation._val, rotation._val, scaling._val)
        return translation, rotation, scaling

    def inverse(self):
        return Matrix(matrix_inverse(self._val))
