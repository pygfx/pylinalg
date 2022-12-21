import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given
from hypothesis.strategies import none

import pylinalg as pla

from ..conftest import test_vector, test_dtype, legal_numbers


@given(legal_numbers | test_vector, none() | test_dtype)
def test_matrix_make_translation(position, dtype):
    result = pla.matrix_make_translation(position, dtype=dtype)

    expected = np.eye(4, dtype=dtype)
    expected[:3, 3] = np.asarray(position)

    npt.assert_array_almost_equal(result, expected)


@given(legal_numbers | test_vector, none() | test_dtype)
def test_matrix_make_scaling(scale, dtype):
    result = pla.matrix_make_scaling(scale, dtype=dtype)

    if isinstance(scale, np.ndarray):
        scaling = scale
    else:
        scaling = np.repeat(scale, 3)

    expected = np.diag((*scaling, 1))
    expected = expected.astype(dtype)

    npt.assert_array_almost_equal(result, expected)
    assert result.dtype == dtype


def test_matrix_make_rotation_from_euler_angles():
    """Test that a positive pi/2 rotation about the z-axis results
    in counter clockwise rotation, in accordance with the unit circle."""
    result = pla.matrix_make_rotation_from_euler_angles([0, 0, np.pi / 2])
    npt.assert_array_almost_equal(
        result,
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )


def test_matrix_make_rotation_from_euler_angles_ordered():
    """Test that an ordered sequence of rotations about the z and then y axis results
    in the correct result."""
    result = pla.matrix_make_rotation_from_euler_angles(
        [0, np.pi / 2, np.pi / 2], order="zyx"
    )
    npt.assert_array_almost_equal(
        result,
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
    )


def test_matrix_make_rotation_from_euler_angles_dtype():
    result = pla.matrix_make_rotation_from_euler_angles([0, 0, np.pi / 2], dtype="i2")
    npt.assert_array_almost_equal(
        result,
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )
    assert result.dtype == "i2"


def test_matrix_make_rotation_from_axis_angle_direction():
    """Test that a positive pi/2 rotation about the z-axis results
    in counter clockwise rotation, in accordance with the unit circle."""
    result = pla.matrix_make_rotation_from_axis_angle([0, 0, 1], np.pi / 2)
    npt.assert_array_almost_equal(
        result,
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )


def test_matrix_make_rotation_from_axis_angle_xy():
    """Test that a negative pi rotation about the diagonal of the x-y plane
    flips the x and y coordinates, and negates the z coordinate."""
    axis = np.array([1, 1, 0], dtype="f8")
    axis /= np.linalg.norm(axis)
    result = pla.matrix_make_rotation_from_axis_angle(axis, -np.pi)
    npt.assert_array_almost_equal(
        result,
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
    )


@pytest.mark.parametrize(
    "angles,expected,dtype",
    [
        # case a
        ([0, 0, np.pi / 2], [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2], "f8"),
        # case a, two ordered rotations
        ([0, -np.pi / 2, np.pi / 2], [0.5, -0.5, 0.5, 0.5], "f8"),
        # non-default dtype
        ([0, -np.pi / 2, np.pi / 2], [0.5, -0.5, 0.5, 0.5], "f4"),
        # case b (contrived example for code coverage)
        (
            [0, np.pi * 0.51, np.pi * 0.51],
            [0.515705, -0.499753, -0.499753, -0.484295],
            "f8",
        ),
        # case c (contrived example for code coverage)
        (
            [np.pi * 1.2, np.pi * 1.8, np.pi],
            [-0.095492, 0.904508, -0.293893, -0.293893],
            "f8",
        ),
        # case d (contrived example for code coverage)
        (
            [np.pi * 0.45, np.pi * 1.8, np.pi],
            [0.234978, 0.617662, 0.723189, -0.20069],
            "f8",
        ),
    ],
)
def test_matrix_to_quaternion(angles, expected, dtype):
    matrix = pla.matrix_make_rotation_from_euler_angles(angles, dtype=dtype)
    quaternion = pla.matrix_to_quaternion(matrix, dtype=dtype)
    npt.assert_array_almost_equal(
        quaternion,
        expected,
    )
    assert matrix.dtype == dtype
    assert quaternion.dtype == dtype


def test_matrix_combine():
    """Test that the matrices are combined in the expected order."""
    # non-uniform scaling such that the test would fail if rotation/scaling are
    # applied in the incorrect order
    scaling = pla.matrix_make_scaling([1, 2, 1])
    rotation = pla.matrix_make_rotation_from_euler_angles([0, np.pi / 4, np.pi / 2])
    translation = pla.matrix_make_translation(2)
    # apply the standard SRT ordering
    result = pla.matrix_combine([translation, rotation, scaling])
    # therefore translation should be unaffected in the combined matrix
    npt.assert_array_almost_equal(
        result,
        [
            [0, -2, 0, 2],
            [np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 2],
            [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 2],
            [0, 0, 0, 1],
        ],
    )

    with pytest.raises(TypeError):
        pla.matrix_combine()
    with pytest.raises(ValueError):
        pla.matrix_combine([])
    with pytest.raises(ValueError):
        pla.matrix_combine([translation])

    result = pla.matrix_combine([translation, translation], dtype="f4")
    npt.assert_array_almost_equal(
        result,
        [
            [1, 0, 0, 4],
            [0, 1, 0, 4],
            [0, 0, 1, 4],
            [0, 0, 0, 1],
        ],
    )
    assert result.dtype == "f4"

    temp = np.identity(4, dtype="i4")
    result = pla.matrix_combine([translation, translation], out=temp, dtype="f4")
    assert result is temp
    npt.assert_array_almost_equal(
        result,
        [
            [1, 0, 0, 4],
            [0, 1, 0, 4],
            [0, 0, 1, 4],
            [0, 0, 0, 1],
        ],
    )
    assert result.dtype == "i4"


def test_matrix_make_transform():
    """Test that the matrices are composed correctly in SRT order."""
    # non-uniform scaling such that the test would fail if rotation/scaling are
    # applied in the incorrect order
    scaling = [1, 2, 1]
    # quaternion corresponding to 90 degree rotation about z-axis
    rotation = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
    translation = [2, 2, 2]
    # compose the transform
    result = pla.matrix_make_transform(translation, rotation, scaling)
    npt.assert_array_almost_equal(
        result,
        [
            [0, -2, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ],
    )


def test_matrix_decompose():
    """Test that the matrices are decomposed correctly."""
    matrix = [
        [0, -2, 0, 2],
        [1, 0, 0, 2],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ]
    translation, rotation, scaling = pla.matrix_decompose(matrix)
    npt.assert_array_equal(translation, [2, 2, 2])
    npt.assert_array_equal(scaling, [1, 2, 1])
    npt.assert_array_almost_equal(rotation, [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])


def test_matrix_make_perspective():
    a = pla.matrix_make_perspective(-1, 1, -1, 1, 1, 100)
    npt.assert_array_almost_equal(
        a,
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -101 / 99, -200 / 99],
            [0, 0, -1, 0],
        ],
    )


def test_matrix_make_orthographic():
    a = pla.matrix_make_orthographic(-1, 1, -1, 1, 1, 100)
    npt.assert_array_almost_equal(
        a,
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -2 / 99, -101 / 99],
            [0, 0, 0, 1],
        ],
    )
