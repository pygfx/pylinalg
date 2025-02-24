import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis.strategies import none

import pylinalg as la

from . import conftest as ct


def test_vec_normalize():
    vectors = [
        [2, 0, 0],
        [1, 1, 1],
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
    ]
    expected = [
        [1, 0, 0],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
        [-1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)],
        [1, 0, 0],
    ]

    # Test individuals
    for vec, e_vec in zip(vectors, expected):
        npt.assert_array_almost_equal(
            la.vec_normalize(vec),
            e_vec,
        )

    # Test the batch
    npt.assert_array_almost_equal(
        la.vec_normalize(vectors),
        expected,
    )


##
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
def test_vec_homogeneous(vectors, value, expected):
    vectors = np.asarray(vectors)
    expected = np.asarray(expected)
    result = la.vec_homogeneous(vectors, w=value)
    npt.assert_array_equal(result, expected)


@given(ct.test_vector)
def test_mat_decompose_translation(translation):
    matrix = la.mat_from_translation(translation)
    result = la.mat_decompose_translation(matrix)

    npt.assert_equal(result, translation)


def test_vector_apply_translation():
    vectors = np.array([[1, 0, 0]])
    expected = np.array([[0, 2, 2]])
    matrix = la.mat_from_translation([-1, 2, 2])
    result = la.vec_transform(vectors, matrix, projection=False)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


@given(ct.test_vector)
def test_vec_spherical_safe(vector):
    result = la.vec_spherical_safe(vector)

    assert np.all((0 <= result[..., 1]) & (result[..., 1] < np.pi))
    assert np.all((0 <= result[..., 2]) & (result[..., 2] < 2 * np.pi))


def test_vec_transform_out():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    out = np.empty_like(vectors, dtype="i4")
    matrix = la.mat_from_translation([-1, 2, 2])
    result = la.vec_transform(vectors, matrix, out=out)

    assert result is out


def test_vec_transform_projection_flag():
    vectors = np.array(
        [
            [1, 0, 0],
            [1, 2, 3],
            [1, 1, 1],
            [0, 0, 0],
            [7, 8, -9],
        ],
        dtype="f8",
    )
    translation = np.array([-1, 2, 2], dtype="f8")
    expected = vectors + translation[None, :]

    matrix = la.mat_from_translation(translation)

    for projection in [True, False]:
        for batch in [True, False]:
            if batch:
                vectors_in = vectors
                expected_out = expected
            else:
                vectors_in = vectors[0]
                expected_out = expected[0]
            result = la.vec_transform(vectors_in, matrix, projection=projection)
            npt.assert_array_equal(result, expected_out)


def test_vec_transform_ndim():
    vectors_2d = np.array(
        [
            [1, 0, 0],
            [1, 2, 3],
            [1, 1, 1],
            [1, 1, -1],
            [0, 0, 0],
            [7, 8, -9],
        ],
        dtype="f8",
    )
    translation = np.array([-1, 2, 2], dtype="f8")

    vectors_3d = vectors_2d.reshape((3, 2, 3))
    vectors_4d = vectors_2d.reshape((6, 1, 1, 3))

    expected_3d = vectors_3d + translation[None, None, :]
    expected_4d = vectors_4d + translation[None, None, None, :]

    matrix = la.mat_from_translation(translation)

    for projection in [True, False]:
        result = la.vec_transform(vectors_3d, matrix, projection=projection)
        npt.assert_array_equal(result, expected_3d)

        result = la.vec_transform(vectors_4d, matrix, projection=projection)
        npt.assert_array_equal(result, expected_4d)


@given(ct.test_spherical, none())
@example((1, 0, np.pi / 2), (0, 0, 1))
@example((1, np.pi / 2, np.pi / 2), (1, 0, 0))
@example((1, 0, 0), (0, 1, 0))
def test_vec_euclidean_to_spherical(expected, vector):
    if vector is None:
        assume(abs(expected[0]) > 1e-10)
        vector = la.vec_spherical_to_euclidean(expected)
    else:
        expected = np.asarray(expected)
        vector = np.asarray(vector)

    actual = la.vec_euclidean_to_spherical(vector)

    assert np.allclose(actual, expected, rtol=1e-10)


def test_vec_transform_out_performant():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    out = np.empty_like(vectors, dtype="f8")
    matrix = la.mat_from_translation([-1, 2, 2])
    result = la.vec_transform(vectors, matrix, out=out)

    assert result is out


def test_vec_transform_dtype():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    matrix = la.mat_from_translation([-1, 2, 2])
    result = la.vec_transform(vectors, matrix, dtype="i2")

    assert result.dtype == "i2"


@given(ct.test_vector, ct.test_vector)
def test_vec_dist(vector_a, vector_b):
    expected = np.linalg.norm(vector_a - vector_b)
    result = la.vec_dist(vector_a, vector_b)

    assert np.allclose(result, expected, rtol=1e-10)


def test_vec_dist_exceptions():
    tmp = np.array(0)
    with pytest.raises(IndexError):
        la.vec_dist((0, 0, 0), (0, 1, 0), out=tmp)


def test_vec_angle():
    cases = [
        ((0, 0, 1), (0, 1, 0), 90),
        ((0, 0, 1), (0, 2, 0), 90),
        ((1, 0, 0), (0, 3, 0), 90),
        ((2, 0, 0), (3, 0, 0), 0),
        ((2, 2, 0), (3, 3, 0), 0),
        ((2, 2, 2), (3, 3, 3), 0),
        ((1, 0, 0), (-2, 0, 0), 180),
        ((0, 1, 2), (0, -3, -6), 180),
        ((1, 0, 0), (2, 2, 0), 45),
        ((1, 0, 1), (0, 0, 80), 45),
    ]
    for v1, v2, deg in cases:
        rad = deg * np.pi / 180
        result = la.vec_angle(v1, v2)
        assert np.abs(result - rad) < 0.0001

    v1 = np.array([case[0] for case in cases])
    v2 = np.array([case[1] for case in cases])
    expected = np.array([case[2] for case in cases]) * np.pi / 180
    result = la.vec_angle(v1, v2)

    assert np.allclose(result, expected, rtol=1e-8)


def test_vec_transform_out_dtype():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    matrix = la.mat_from_translation([-1, 2, 2])
    out = np.empty_like(vectors, dtype="i4")
    result = la.vec_transform(vectors, matrix, out=out, dtype="i2")

    assert result is out
    assert result.dtype == "i4"


@given(ct.test_spherical)
def test_vec_spherical_to_euclidean(spherical):
    # accuracy of trigonometric ops close to 0, 90, 180, 270, 360 deg dependes a
    # lot on the underlying hardware. Let's avoid it here.
    angles = spherical[..., [1, 2]]
    assume(np.all(np.abs(angles - 0) > 1e-100))
    assume(np.all(np.abs(angles - np.pi / 2) > 1e-100))
    assume(np.all(np.abs(angles - np.pi) > 1e-100))
    assume(np.all(np.abs(angles - 2 * np.pi) > 1e-100))

    # same for really short vectors (can produce 0)
    assume(np.all(np.abs(spherical[0] - 0) > 1e-200))

    # we can't do a simple round trip test. Instead we ensure that we are
    # rotating in the right direction and that the radius/length match
    result = la.vec_spherical_to_euclidean(spherical)

    # ensure azimuth rotates CCW
    expected_sign = np.where(spherical[1] < np.pi / 2, 1, -1)
    actual_sign = np.prod(np.sign(result[..., [0, 2]]))
    assert np.all(expected_sign == actual_sign)

    # ensure inclination is measured from positive y
    expected_sign = np.where(spherical[2] < np.pi / 2, 1, -1)
    expected_sign = np.where(spherical[2] > 3 / 2 * np.pi, 1, expected_sign)
    actual_sign = np.sign(result[..., 1])
    assert np.all(expected_sign == actual_sign)

    # ensure length is what we expect
    length = np.linalg.norm(result)
    assert np.allclose(length, spherical[0], rtol=1e-16, atol=np.inf)


def test_vec_spherical_to_euclidean_refs():
    # ensure that the reference axes get transformed as expected
    result = la.vec_spherical_to_euclidean((1, 0, 0))
    assert np.allclose(result, (0, 1, 0))

    result = la.vec_spherical_to_euclidean((1, 0, np.pi / 2))
    assert np.allclose(result, (0, 0, 1))


def test_vector_apply_rotation_about_z_matrix():
    """Test that a positive pi/2 rotation about the z-axis results
    in counter clockwise rotation, in accordance with the unit circle."""
    vectors = np.array(
        [1, 0, 0],
    )
    expected = np.array(
        [0, 1, 0],
    )
    matrix = la.mat_from_euler([0, 0, np.pi / 2])
    result = la.vec_transform(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


@settings(suppress_health_check=(HealthCheck.filter_too_much,))
@given(ct.test_vector, ct.test_projection)
def test_vec_unproject(expected, projection_matrix):
    expected_2d = la.vec_transform(expected, projection_matrix)

    depth = expected_2d[..., 2]
    vector = expected_2d[..., [0, 1]]

    actual = la.vec_unproject(vector, projection_matrix, depth=depth)

    # only test stable results
    assume(not np.any(np.isnan(actual) | np.isinf(actual)))
    assert np.allclose(actual, expected, rtol=1e-16, atol=np.inf)


def test_unproject_explicitly():
    # see https://github.com/pygfx/pylinalg/pull/60#discussion_r1159522602
    # and the following comments

    # cube at origin with side length 10
    cube_corners = np.array(
        [
            [-5, -5, -5],
            [5, -5, -5],
            [5, 5, -5],
            [-5, 5, -5],
            [-5, -5, 5],
            [5, -5, 5],
            [5, 5, 5],
            [-5, 5, 5],
        ]
    )
    cube_world_matrix = np.eye(4)

    # camera 10 units away from cube origin
    camera_pos = (0, 0, 10)
    cam_world_matrix = la.mat_from_translation(camera_pos)
    view_matrix = np.linalg.inv(cam_world_matrix)

    # Scenario 1: near=4, far=16
    projection_matrix = la.mat_orthographic(-10, 10, 10, -10, 4, 16, depth_range=(0, 1))
    cube_local_to_cam_ndc = projection_matrix @ view_matrix @ cube_world_matrix
    corners_ndc = la.vec_transform(cube_corners, cube_local_to_cam_ndc)
    corner_in_view = np.all(
        ((-1, -1, 0) < corners_ndc) & (corners_ndc < (1, 1, 1)), axis=-1
    )
    assert np.sum(corner_in_view) == 8

    # Scenario 2: near=6, far=14
    projection_matrix = la.mat_orthographic(-10, 10, 10, -10, 6, 14, depth_range=(0, 1))
    cube_local_to_cam_ndc = projection_matrix @ view_matrix @ cube_world_matrix
    corners_ndc = la.vec_transform(cube_corners, cube_local_to_cam_ndc)
    corner_in_view = np.all(
        ((-1, -1, 0) < corners_ndc) & (corners_ndc < (1, 1, 1)), axis=-1
    )
    assert np.sum(corner_in_view) == 0


def test_vec_unproject_exceptions():
    vector = np.ones(2)
    matrix = np.eye(4)
    matrix[1, 1] = 0

    with pytest.raises(ValueError):
        la.vec_unproject(vector, matrix)


def test_vec_unproject_is_inverse():
    a = la.mat_perspective(-1, 1, -1, 1, 1, 100)
    a_inv = la.mat_inverse(a)
    vecs = np.array([[1, 2], [4, 5], [7, 8]])

    expected = la.vec_unproject(vecs, a)
    actual = la.vec_unproject(vecs, a_inv, matrix_is_inv=True)
    npt.assert_array_equal(expected, actual)


def test_vector_apply_rotation_ordered():
    """Test that a positive pi/2 rotation about the z-axis and then the y-axis
    results in a different output then in standard rotation ordering."""
    vectors = np.array(
        [1, 0, 0],
    )
    expected = np.array(
        [0, 1, 0],
    )
    matrix = la.mat_from_euler([0, np.pi / 2, np.pi / 2], order="zyx")
    result = la.vec_transform(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


@given(ct.test_vector, ct.test_quaternion)
def test_vec_transform_quat(vector, quaternion):
    scale = np.linalg.norm(vector)
    if scale > 1e100:
        vector = vector / scale * 1e100

    actual = la.vec_transform_quat(vector, quaternion)

    # reference implementation
    matrix = la.mat_from_quat(quaternion)
    vector = la.vec_homogeneous(vector)
    expected = (matrix @ vector)[..., :-1]

    # assert relative proximity only
    assert np.allclose(actual, expected, rtol=1e-10, atol=np.inf)


@given(ct.test_quaternion)
def test_matrix_vs_quaternion_apply(quaternion):
    basis = np.eye(3)
    matrix = la.mat_from_quat(quaternion)

    expected = la.vec_transform(basis, matrix)
    actual = la.vec_transform_quat(basis, quaternion)

    assert np.allclose(actual, expected)


@given(ct.test_vector, ct.test_quaternion)
def test_vec_transform_quat_identity(vector, quaternion):
    scale = np.linalg.norm(vector)
    if scale > 1e100:
        vector = vector / scale * 1e100

    inv_quaternion = la.quat_inv(quaternion)
    tmp = la.vec_transform_quat(vector, quaternion)
    actual = la.vec_transform_quat(tmp, inv_quaternion)

    # assert relative proximity only
    assert np.allclose(actual, vector, rtol=1e-10, atol=np.inf)


def test_vec_transform__perspective():
    # Test for OpenGL, wgpu, and arbitrary depth ranges
    depth_ranges = (-1, 1), (0, 1), (-2, 9)

    for depth_range in depth_ranges:
        m = la.mat_perspective(-1, 1, -1, 1, 1, 17, depth_range=depth_range)

        # Check the depth range
        vec2 = la.vec_transform((0, 0, -1), m)
        assert vec2[2] == depth_range[0]
        vec2 = la.vec_transform((0, 0, -17), m)
        assert vec2[2] == depth_range[1]
        # vec2 = la.vec_transform((0, 0, -9), m) -> skip: halfway is not 0.5 ndc

        cases = [
            [(1, 0, -2), 0.5],
            [(1, 0, -4), 0.25],
            [(0, 0, -4), 0.0],
            [(-1, 0, -4), -0.25],
            [(-1, 0, -2), -0.5],
        ]

        # Check cases one by one
        for vec1, expected in cases:
            vec2 = la.vec_transform(vec1, m)
            assert vec2[0] == expected

        # Check cases batched
        vectors1 = np.vstack([v for v, _ in cases])
        vectors2 = la.vec_transform(vectors1, m)
        assert vectors2[0][0] == cases[0][1]
        assert vectors2[1][0] == cases[1][1]
        assert vectors2[2][0] == cases[2][1]

        # Check cases batched, via out
        vectors2 = la.vec_transform(vectors1, m, out=vectors2)
        assert vectors2[0][0] == cases[0][1]
        assert vectors2[1][0] == cases[1][1]
        assert vectors2[2][0] == cases[2][1]


def test_vec_transform_orthographic():
    # Test for OpenGL, wgpu, and arbitrary depth ranges
    depth_ranges = (-1, 1), (0, 1), (-2, 9)

    for depth_range in depth_ranges:
        m = la.mat_orthographic(-1, 1, -1, 1, 1, 17, depth_range=depth_range)

        # Check the depth range
        vec2 = la.vec_transform((0, 0, -1), m)
        assert vec2[2] == depth_range[0]
        vec2 = la.vec_transform((0, 0, -17), m)
        assert vec2[2] == depth_range[1]
        vec2 = la.vec_transform((0, 0, -9), m)
        assert vec2[2] == (depth_range[0] + depth_range[1]) / 2

        # This point would be at the edge of NDC
        vec2 = la.vec_transform((1, 0, -2), m)
        assert vec2[0] == 1

        # This point would be at the egde of NDC
        vec2 = la.vec_transform((1, 0, -4), m)
        assert vec2[0] == 1


@given(ct.test_angles_rad)
def test_quat_to_euler(angles):
    """
    Test that we can recover a rotation in euler angles from a given quaternion.

    This test applies the recovered rotation to a vector.
    """
    order = "xyz"
    quaternion = la.quat_from_euler(angles, order=order)
    matrix = la.mat_from_euler(angles, order=order)

    angles_reconstructed = la.quat_to_euler(quaternion, order=order)
    matrix_reconstructed = la.mat_from_euler(angles_reconstructed)

    expected = la.vec_transform([1, 2, 3], matrix)
    actual = la.vec_transform([1, 2, 3], matrix_reconstructed)

    assert np.allclose(actual, expected)


@given(ct.test_angles_rad)
def test_quat_to_euler_roundtrip(angles):
    """
    Test that we can recover a rotation in euler angles from a given quaternion.

    This test creates another quaternion with the recovered angles and
    test for equality.
    """
    order = "xyz"
    quaternion = la.quat_from_euler(angles, order=order)

    angles_reconstructed = la.quat_to_euler(quaternion, order=order)
    quaternion_reconstructed = la.quat_from_euler(angles_reconstructed, order=order)

    assert np.allclose(quaternion, quaternion_reconstructed) or np.allclose(
        quaternion, -quaternion_reconstructed
    )


def test_quat_from_euler_upper_case_order():
    order = "XYZ"
    angles = np.array([np.pi / 2, np.pi / 180, 0])
    quat = la.quat_from_euler(angles, order=order)
    actual = la.quat_to_euler(quat, order=order)

    npt.assert_allclose(actual, angles)


def test_quat_from_euler_lower_case_order():
    order = "xyz"
    angles = np.array([np.pi / 2, np.pi / 180, 0])
    quat = la.quat_from_euler(angles, order=order)
    actual = la.quat_to_euler(quat, order=order)

    npt.assert_allclose(actual, angles)


def test_quat_euler_vs_scipy():
    """Compare our implementation with scipy's."""
    from scipy.spatial.transform import Rotation as R  # noqa: N817

    cases = [
        ("xyz", [np.pi / 2, np.pi / 180, 0]),
        ("XYZ", [np.pi / 2, np.pi / 180, 0]),
        ("zxy", [np.pi, np.pi / 180, -np.pi / 180]),
        ("ZXY", [np.pi, np.pi / 180, -np.pi / 180]),
    ]

    for order, angles in cases:
        npt.assert_allclose(
            la.quat_from_euler(angles, order=order),
            R.from_euler(order, angles).as_quat(),
        )

    cases = [(order, la.quat_from_euler(euler, order=order)) for order, euler in cases]

    for order, quat in cases:
        npt.assert_allclose(
            la.quat_to_euler(quat, order=order),
            R.from_quat(quat).as_euler(order),
        )


def test_quat_to_euler_broadcasting():
    """
    Test that quat_to_euler supports broadcasting.
    """
    quaternions = la.quat_from_axis_angle(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        [
            np.pi,
            np.pi * 2,
            np.pi / 2,
            np.pi * 1.5,
        ],
    )

    expected = np.array(
        [
            [np.pi, 0, 0],
            [0, 0, 0],
            [0, 0, np.pi / 2],
            [0, 0, -np.pi / 2],
        ]
    )
    actual = la.quat_to_euler(quaternions)

    npt.assert_array_almost_equal(actual, expected)
