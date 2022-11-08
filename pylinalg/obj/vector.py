from functools import partial

import numpy as np

from .base import LinalgBase


class Vector(LinalgBase):
    """A representation of a direction and length in 3D Euclidean space."""

    _initializer = partial(np.zeros, 3)

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
