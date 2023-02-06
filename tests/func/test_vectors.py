import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given

import pylinalg as pla

from .. import conftest as ct


def test_vector_normalize():
    vectors = np.array(
        [
            [2, 0, 0],
            [1, 1, 1],
            [-1, -1, -1],
            [1, 0, 0],
        ],
        dtype="f8",
    )
    npt.assert_array_almost_equal(
        pla.vector_normalize(vectors),
        [
            [1, 0, 0],
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [-1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)],
            [1, 0, 0],
        ],
    )


@pytest.mark.parametrize(
    "vectors,value,expected",
    [
        ([1, 1, 1], 1, [1, 1, 1, 1]),
        ([1, 1, 1], 0, [1, 1, 1, 0]),
        ([[1, 1, 1]], 1, [[1, 1, 1, 1]]),
        ([[1, 1, 1], [2, 2, 2]], 1, [[1, 1, 1, 1], [2, 2, 2, 1]]),
        ([[1, 1, 1], [2, 2, 2]], 0, [[1, 1, 1, 0], [2, 2, 2, 0]]),
    ],
)
def test_vector_make_homogeneous(vectors, value, expected):
    vectors = np.asarray(vectors)
    expected = np.asarray(expected)
    result = pla.vector_make_homogeneous(vectors, w=value)
    npt.assert_array_equal(result, expected)


def test_vector_apply_translation():
    vectors = np.array([[1, 0, 0]])
    expected = np.array([[0, 2, 2]])
    matrix = pla.matrix_make_translation([-1, 2, 2])
    result = pla.vector_apply_matrix(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


def test_vector_apply_matrix_out():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    out = np.empty_like(vectors, dtype="i4")
    matrix = pla.matrix_make_translation([-1, 2, 2])
    result = pla.vector_apply_matrix(vectors, matrix, out=out)

    assert result is out


@given(ct.test_vector)
def test_vector_euclidean_to_spherical(vector):
    # result = pla.vector_euclidean_to_spherical(vector)

    raise NotImplementedError("Waiting for upstream PR.")


def test_vector_apply_matrix_out_performant():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    out = np.empty_like(vectors, dtype="f8")
    matrix = pla.matrix_make_translation([-1, 2, 2])
    result = pla.vector_apply_matrix(vectors, matrix, out=out)

    assert result is out


def test_vector_apply_matrix_dtype():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    matrix = pla.matrix_make_translation([-1, 2, 2])
    result = pla.vector_apply_matrix(vectors, matrix, dtype="i2")

    assert result.dtype == "i2"


def test_vector_apply_matrix_out_dtype():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    matrix = pla.matrix_make_translation([-1, 2, 2])
    out = np.empty_like(vectors, dtype="i4")
    result = pla.vector_apply_matrix(vectors, matrix, out=out, dtype="i2")

    assert result is out
    assert result.dtype == "i4"


def test_vector_apply_rotation_about_z_matrix():
    """Test that a positive pi/2 rotation about the z-axis results
    in counter clockwise rotation, in accordance with the unit circle."""
    vectors = np.array(
        [1, 0, 0],
    )
    expected = np.array(
        [0, 1, 0],
    )
    matrix = pla.matrix_make_rotation_from_euler_angles([0, 0, np.pi / 2])
    result = pla.vector_apply_matrix(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


def test_vector_apply_rotation_ordered():
    """Test that a positive pi/2 rotation about the z-axis and then the y-axis
    results in a different output then in standard rotation ordering."""
    vectors = np.array(
        [1, 0, 0],
    )
    expected = np.array(
        [0, 1, 0],
    )
    matrix = pla.matrix_make_rotation_from_euler_angles(
        [0, -np.pi / 2, np.pi / 2], order="zyx"
    )
    result = pla.vector_apply_matrix(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )
