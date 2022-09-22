import numpy as np
import pytest

import pylinalg as pla


def test_matrix_init():
    m = pla.Matrix()
    assert m == np.identity(4, dtype="f8")
    assert m.dtype == "f8"

    m = pla.Matrix(dtype="i2")
    assert m == np.identity(4, dtype="i2")
    assert m.dtype == "i2"

    m = pla.Matrix(np.zeros((4, 4)))
    assert m == np.zeros((4, 4))
    assert m.dtype == "f8"

    m = pla.Matrix(np.zeros((4, 4), dtype="i2"))
    assert m == np.zeros((4, 4), dtype="i2")
    assert m.dtype == "i2"

    with pytest.raises(TypeError):
        m = pla.Matrix(matrix=np.zeros((4, 4)))

    with pytest.raises(TypeError):
        m = pla.Matrix(np.zeros((4, 4)), "i2")


def test_matrix_set():
    m = pla.Matrix()
    assert m == np.identity(4)
    assert m[0, 0] == 1

    val = m._val
    m[:] = np.zeros((4, 4))
    assert m[0, 0] == 0
    assert m._val is val
