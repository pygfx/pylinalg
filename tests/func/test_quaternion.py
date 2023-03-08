import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given
from hypothesis.strategies import text

import pylinalg as la

from .. import conftest as ct


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
    matrix = la.quaternion_to_matrix(quaternion, dtype=dtype)

    expected_matrix = la.matrix_make_rotation_from_euler_angles(expected, dtype=dtype)
    npt.assert_array_almost_equal(
        matrix,
        expected_matrix,
        decimal=5,
    )
    assert matrix.dtype == dtype


def test_quaternion_multiply_quaternion():
    # quaternion corresponding to 90 degree rotation about z-axis
    a = np.array([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    b = np.array([0, 0, 0, 1])
    c = la.quaternion_multiply(a, b)
    # multiplying by the identity quaternion
    npt.assert_array_equal(c, a)

    d = la.quaternion_multiply(a, a)
    # should be 180 degree rotation about z-axis
    npt.assert_array_almost_equal(d, [0, 0, 1, 0])


def test_quaternion_norm():
    a = np.array([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    b = np.array([0, 0, 0, 1])
    c = np.array([0, 0, 1, 0])
    assert np.linalg.norm(a) == 1
    assert np.linalg.norm(b) == 1
    assert np.linalg.norm(c) == 1


def test_quaternion_norm_vectorized():
    a = np.array([[0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]])
    npt.assert_array_equal(np.linalg.norm(a, axis=-1), [1])


def test_quaternion_from_unit_vectors():
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    q = la.quaternion_make_from_unit_vectors(a, b)
    npt.assert_almost_equal(q, [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])


def test_quaternion_inverse():
    a = np.array([0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])
    ai = la.quaternion_inverse(a)

    npt.assert_array_equal(a[:3], -ai[:3])
    npt.assert_array_equal(a[3], ai[3])

    # broadcasting over multiple quaternions
    b = np.array(
        [
            [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    bi = la.quaternion_inverse(b)

    npt.assert_array_equal(b[..., :3], -bi[..., :3])
    npt.assert_array_equal(b[..., 3], bi[..., 3])


def test_quaternion_from_axis_angle():
    axis = np.array([1, 0, 0], dtype="f4")
    angle = np.pi / 2
    q = la.quaternion_make_from_axis_angle(axis, angle)

    npt.assert_array_almost_equal(q, [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])


@given(ct.test_unit_vector, ct.test_quaternion)
def test_quaternion_vs_matrix_rotate(vector, quaternion):
    matrix = la.quaternion_to_matrix(quaternion)
    hom_vector = np.ones(4, dtype=vector.dtype)
    hom_vector[:3] = vector

    expected = (matrix @ hom_vector)[:3]
    actual = la.quaternion_rotate(vector, quaternion)

    npt.assert_array_almost_equal(actual, expected)


@given(ct.test_unit_vector, ct.test_quaternion)
def test_quaternion_rotate_inversion(vector, quaternion):
    inverse = la.quaternion_inverse(quaternion)

    tmp = la.quaternion_rotate(vector, quaternion)
    result = la.quaternion_rotate(tmp, inverse)
    npt.assert_array_almost_equal(result, vector)

    tmp = la.quaternion_rotate(vector, inverse)
    result = la.quaternion_rotate(tmp, quaternion)
    npt.assert_array_almost_equal(result, vector)


@given(ct.test_angles_rad, text("xyz", min_size=1, max_size=3))
def test_quaternion_make_from_euler_angles(angles, order):
    angles = angles[: len(order)]
    result = la.quaternion_make_from_euler_angles(angles, order=order)
    actual = la.quaternion_to_matrix(result)

    expected = la.matrix_make_rotation_from_euler_angles(angles, order=order)
    assert np.allclose(actual, expected)
