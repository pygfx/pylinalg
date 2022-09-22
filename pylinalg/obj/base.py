"""
The LinalgBase base class makes our objects more performant and compatible
with numpy.

Related docs:

* https://docs.python.org/3/reference/datamodel.html#slots
* https://numpy.org/doc/stable/user/basics.interoperability.html
"""
import numpy as np


class LinalgBase:
    __slots__ = ["_val"]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._val}>"

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        return self._val[index]

    def __iter__(self):
        return self._val.__iter__()

    @property
    def __array_interface__(self):
        return self._val.__array_interface__

    def __eq__(self, other):
        if isinstance(other, LinalgBase):
            return isinstance(other, self.__class__) and np.array_equal(
                self._val, other._val
            )
        else:
            return np.array_equal(self._val, other)
