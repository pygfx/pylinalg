import numpy as np
import numpy.testing as npt
import pytest

import pylinalg as pla


@pytest.mark.parametrize(
    "expected,quaternion,dtype",
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
def test_quaternion_to_matrix(expected, quaternion, dtype):
    quaternion = np.asarray(quaternion, dtype=dtype)
    matrix = pla.quaternion_to_matrix(quaternion)

    expected_matrix = pla.matrix_make_rotation_from_euler_angles(expected, dtype=dtype)
    npt.assert_array_almost_equal(
        matrix,
        expected_matrix,
        decimal=5,
    )
    assert matrix.dtype == dtype


def test_quaternion_add_quaternion():
    # quaternion corresponding to 90 degree rotation about z-axis
    a = np.array([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    b = np.array([0, 0, 0, 1])
    c = pla.quaternion_add_quaternion(a, b)
    npt.assert_array_equal(a + b, c)


def test_quaternion_subtract_quaternion():
    # quaternion corresponding to 90 degree rotation about z-axis
    a = np.array([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    b = np.array([0, 0, 0, 1])
    c = pla.quaternion_subtract_quaternion(a, b)
    npt.assert_array_equal(a - b, c)


def test_quaternion_multiply_quaternion():
    # quaternion corresponding to 90 degree rotation about z-axis
    a = np.array([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    b = np.array([0, 0, 0, 1])
    c = pla.quaternion_multiply_quaternion(a, b)
    # multiplying by the identity quaternion
    npt.assert_array_equal(c, a)

    d = pla.quaternion_multiply_quaternion(a, a)
    # should be 180 degree rotation about z-axis
    npt.assert_array_almost_equal(d, [0, 0, 1, 0])


def test_quaternion_norm():
    a = np.array([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    b = np.array([0, 0, 0, 1])
    c = np.array([0, 0, 1, 0])
    assert pla.quaternion_norm(a) == 1
    assert pla.quaternion_norm(b) == 1
    assert pla.quaternion_norm(c) == 1
