import numpy as np
import numpy.testing as npt
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


def test_matrix_copy():
    data = np.arange(16, dtype="f4").reshape(4, 4)
    m = pla.Matrix(data)
    m2 = m.copy()
    assert m == m2
    assert m.val is not m2.val
    assert m2.val.flags.owndata
    assert m.dtype == m2.dtype


def test_matrix_set():
    m = pla.Matrix()
    assert m == np.identity(4)
    assert m[0, 0] == 1

    val = m.val
    m[:] = np.zeros((4, 4))
    assert m[0, 0] == 0
    assert m.val is val


def test_matrix_compose():
    m = pla.Matrix.transform(
        pla.Vector([2, 2, 2]),
        # quaternion corresponding to 90 degree rotation about z-axis
        pla.Quaternion([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        pla.Vector([1, 2, 1]),
    )
    npt.assert_array_almost_equal(
        m,
        [
            [0, -2, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ],
    )


def test_matrix_decompose():
    m = pla.Matrix(
        [
            [0, -2, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ]
    )
    translation, rotation, scaling = m.decompose()
    assert translation == pla.Vector([2, 2, 2])
    # quaternion corresponding to 90 degree rotation about z-axis
    npt.assert_array_almost_equal(
        rotation, pla.Quaternion([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    )
    assert isinstance(rotation, pla.Quaternion)
    assert scaling == pla.Vector([1, 2, 1])


def test_matrix_inverse():
    matrix = pla.Matrix()
    matrix[0, 1] = 5
    result = matrix.inverse()
    assert result is not matrix
    assert result == np.linalg.inv(matrix)


def test_matrix_multiply():
    a = np.arange(16).reshape((4, 4))
    b = a.copy() + 1
    ab = a @ b

    ma = pla.Matrix(a)
    mb = pla.Matrix(b)

    mamb = ma @ mb
    assert isinstance(mamb, pla.Matrix)

    assert mamb == ab

    ma2 = ma.copy()
    ma2 = ma2 @ mb

    assert ma2 == ab

    ma2 = ma.copy()
    ma2 = ma2 @ mb

    assert ma2 == ab


def test_matrix_perspective():
    a = pla.Matrix.perspective(-1, 1, -1, 1, 1, 100)
    npt.assert_array_almost_equal(
        a,
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -101 / 99, -200 / 99],
            [0, 0, -1, 0],
        ],
    )


def test_matrix_orthographic():
    a = pla.Matrix.orthographic(-1, 1, -1, 1, 1, 100)
    npt.assert_array_almost_equal(
        a,
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -2 / 99, -101 / 99],
            [0, 0, 0, 1],
        ],
    )
