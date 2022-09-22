import numpy as np

from .base import LinalgBase


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
