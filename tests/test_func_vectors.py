import numpy as np
import numpy.testing as npt
import pytest

import pylinalg as pla


@pytest.mark.parametrize(
    "v1,v2,out,expected",
    [
        (np.array([1, 1, 1]), np.array([2, 2, 2]), None, np.array([3, 3, 3])),
        (
            np.array([1, 1, 1]),
            np.array([2, 2, 2]),
            np.array([0, 0, 0]),
            np.array([3, 3, 3]),
        ),
        (
            np.array([[1, 1, 1], [2, 2, 2]]),
            np.array([2, 2, 2]),
            None,
            np.array([[3, 3, 3], [4, 4, 4]]),
        ),
        (
            np.array([[1, 1, 1], [2, 2, 2]]),
            np.array([[2, 2, 2], [3, 3, 3]]),
            None,
            np.array([[3, 3, 3], [5, 5, 5]]),
        ),
    ],
)
def test_vector_add_vector(v1, v2, out, expected):
    result = pla.vector_add_vector(v1, v2, out=out)
    npt.assert_array_equal(
        result,
        expected,
    )
    if out is not None:
        assert result is out


@pytest.mark.parametrize(
    "v,s,out,expected",
    [
        (np.array([1, 1, 1]), 2, None, np.array([3, 3, 3])),
        (
            np.array([1, 1, 1]),
            2,
            np.array([0, 0, 0]),
            np.array([3, 3, 3]),
        ),
        (
            np.array([[1, 1, 1], [2, 2, 2]]),
            2,
            None,
            np.array([[3, 3, 3], [4, 4, 4]]),
        ),
    ],
)
def test_vector_add_scalar(v, s, out, expected):
    result = pla.vector_add_scalar(v, s, out=out)
    npt.assert_array_equal(
        result,
        expected,
    )
    if out is not None:
        assert result is out


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
