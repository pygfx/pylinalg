import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import assume, given
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


@given(ct.test_unit_vector, ct.test_unit_vector, ct.legal_positive_number)
def test_quaternion_from_unit_vectors(
    source_direction, target_direction, source_length
):
    assume(abs(source_length) > 1e-8)

    # Note: the length of the cross product of two large vectors can overflow
    # and become Inf. to avoid this, we only scale source.
    source = source_length * source_direction
    target = target_direction

    rotation = la.quaternion_make_from_unit_vectors(source, target)
    actual = la.vector_apply_quaternion(source_direction, rotation)

    assert np.allclose(actual, target_direction)


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


@given(ct.legal_positive_number)
def test_quaternion_from_axis_angle(length):
    assume(abs(length) > 1e-10)

    axis = np.array([1, 0, 0], dtype="f4")
    angle = np.pi / 2
    q = la.quaternion_make_from_axis_angle(length * axis, angle)

    npt.assert_array_almost_equal(q, [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])


@given(ct.test_unit_vector, ct.legal_angle, ct.legal_positive_number)
def test_quaternion_from_axis_angle_roundtrip(true_axis, true_angle, axis_scaling):
    assume(abs(axis_scaling) > 1e-6)

    assume(abs(true_angle) > 1e-6)
    assume(abs(true_angle) < 2 * np.pi - 1e-6)

    quaternion = la.quaternion_make_from_axis_angle(
        axis_scaling * true_axis, true_angle
    )
    axis, angle = la.axis_angle_from_quaternion(quaternion)

    assert np.allclose(angle, true_angle)

    # Note: We loose the scaling of the axis, but can (roughly) reconstruct the
    # direction
    actual_dot = np.dot(axis, true_axis)
    assert np.allclose(actual_dot, 1)


@given(ct.test_angles_rad, text("xyz", min_size=1, max_size=3))
def test_quaternion_make_from_euler_angles(angles, order):
    angles = np.squeeze(angles[: len(order)])
    result = la.quaternion_make_from_euler_angles(angles, order=order)
    actual = la.quaternion_to_matrix(result)

    expected = la.matrix_make_rotation_from_euler_angles(angles, order=order)
    assert np.allclose(actual, expected)
