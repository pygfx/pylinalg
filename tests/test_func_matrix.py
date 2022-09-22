import numpy as np
import numpy.testing as npt

import pylinalg as pla


def test_matrix_make_translation():
    """Test that the translation offsets ends up in the right slots
    in the matrix (not transposed)."""
    result = pla.matrix_make_translation([1, 2, 3])
    npt.assert_array_equal(
        result,
        [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
    )


def test_matrix_make_translation_dtype():
    result = pla.matrix_make_translation([1, 2, 3], dtype="i2")
    npt.assert_array_equal(
        result,
        [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
    )
    assert result.dtype == "i2"


def test_matrix_make_translation_uniform():
    result = pla.matrix_make_translation(2)
    npt.assert_array_equal(
        result,
        [
            [1, 0, 0, 2],
            [0, 1, 0, 2],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ],
    )


def test_matrix_make_scaling():
    result = pla.matrix_make_scaling([2, 3, 4])
    npt.assert_array_equal(
        result,
        [
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 1],
        ],
    )


def test_matrix_make_scaling_dtype():
    result = pla.matrix_make_scaling([2, 3, 4], dtype="i2")
    npt.assert_array_equal(
        result,
        [
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 1],
        ],
    )
    assert result.dtype == "i2"


def test_matrix_make_scaling_uniform():
    result = pla.matrix_make_scaling(2)
    npt.assert_array_equal(
        result,
        [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ],
    )


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


def test_matrix_combine():
    """Test that the matrices are combined in the expected order."""
    scaling = pla.matrix_make_scaling(2)
    rotation = pla.matrix_make_rotation_from_euler_angles([0, 0, np.pi / 2])
    translation = pla.matrix_make_translation(2)
    # apply the standard SRT ordering
    result = pla.matrix_combine([translation, rotation, scaling])
    # therefore translation should be unaffected in the combined matrix
    npt.assert_array_almost_equal(
        result,
        [
            [0, -2, 0, 2],
            [2, 0, 0, 2],
            [0, 0, 2, 2],
            [0, 0, 0, 1],
        ],
    )
