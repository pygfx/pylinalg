import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import assume, example, given
from hypothesis.strategies import none

import pylinalg as la

from .. import conftest as ct


def test_vector_normalize():
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
            la.vector_normalize(vec),
            e_vec,
        )

    # Test the batch
    npt.assert_array_almost_equal(
        la.vector_normalize(vectors),
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
def test_vector_make_homogeneous(vectors, value, expected):
    vectors = np.asarray(vectors)
    expected = np.asarray(expected)
    result = la.vector_make_homogeneous(vectors, w=value)
    npt.assert_array_equal(result, expected)


@given(ct.test_vector)
def test_vector_from_matrix_position(translation):
    matrix = la.matrix_make_translation(translation)
    result = la.vector_from_matrix_position(matrix)

    npt.assert_equal(result, translation)


def test_vector_apply_translation():
    vectors = np.array([[1, 0, 0]])
    expected = np.array([[0, 2, 2]])
    matrix = la.matrix_make_translation([-1, 2, 2])
    result = la.vector_apply_matrix(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


@given(ct.test_vector)
def test_vector_make_spherical_safe(vector):
    result = la.vector_make_spherical_safe(vector)

    assert np.all((0 <= result[..., 1]) & (result[..., 1] < np.pi))
    assert np.all((0 <= result[..., 2]) & (result[..., 2] < 2 * np.pi))


def test_vector_apply_matrix_out():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    out = np.empty_like(vectors, dtype="i4")
    matrix = la.matrix_make_translation([-1, 2, 2])
    result = la.vector_apply_matrix(vectors, matrix, out=out)

    assert result is out


@given(ct.test_spherical, none())
@example((1, 0, np.pi / 2), (0, 0, 1))
@example((1, np.pi / 2, np.pi / 2), (1, 0, 0))
@example((1, 0, 0), (0, 1, 0))
def test_vector_euclidean_to_spherical(expected, vector):
    if vector is None:
        assume(abs(expected[0]) > 1e-10)
        vector = la.vector_spherical_to_euclidean(expected)
    else:
        expected = np.asarray(expected)
        vector = np.asarray(vector)

    actual = la.vector_euclidean_to_spherical(vector)

    assert np.allclose(actual, expected, rtol=1e-10)


def test_vector_apply_matrix_out_performant():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    out = np.empty_like(vectors, dtype="f8")
    matrix = la.matrix_make_translation([-1, 2, 2])
    result = la.vector_apply_matrix(vectors, matrix, out=out)

    assert result is out


def test_vector_apply_matrix_dtype():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    matrix = la.matrix_make_translation([-1, 2, 2])
    result = la.vector_apply_matrix(vectors, matrix, dtype="i2")

    assert result.dtype == "i2"


@given(ct.test_vector, ct.test_vector)
def test_vector_distance_between(vector_a, vector_b):
    expected = np.linalg.norm(vector_a - vector_b)
    result = la.vector_distance_between(vector_a, vector_b)

    assert np.allclose(result, expected, rtol=1e-10)


def test_vector_distance_between_exceptions():
    tmp = np.array(0)
    with pytest.raises(IndexError):
        la.vector_distance_between((0, 0, 0), (0, 1, 0), out=tmp)


def test_vector_angle_between():
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
        result = la.vector_angle_between(v1, v2)
        assert np.abs(result - rad) < 0.0001

    v1 = np.array([case[0] for case in cases])
    v2 = np.array([case[1] for case in cases])
    expected = np.array([case[2] for case in cases]) * np.pi / 180
    result = la.vector_angle_between(v1, v2)

    assert np.allclose(result, expected, rtol=1e-8)


def test_vector_apply_matrix_out_dtype():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    matrix = la.matrix_make_translation([-1, 2, 2])
    out = np.empty_like(vectors, dtype="i4")
    result = la.vector_apply_matrix(vectors, matrix, out=out, dtype="i2")

    assert result is out
    assert result.dtype == "i4"


@given(ct.test_spherical)
def test_vector_spherical_to_euclidean(spherical):
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
    result = la.vector_spherical_to_euclidean(spherical)

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


def test_vector_spherical_to_euclidean_refs():
    # ensure that the reference axes get transformed as expected
    result = la.vector_spherical_to_euclidean((1, 0, 0))
    assert np.allclose(result, (0, 1, 0))

    result = la.vector_spherical_to_euclidean((1, 0, np.pi / 2))
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
    matrix = la.matrix_make_rotation_from_euler_angles([0, 0, np.pi / 2])
    result = la.vector_apply_matrix(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


@given(ct.test_vector, ct.test_projection)
@example(
    ((250, 250, 0), (-250, -250, 0)),
    la.matrix_make_orthographic(250, -250, 250, -250, -100, 100, depth_range=(0, 1)),
)
def test_vector_unproject(expected, projection_matrix):
    expected_2d = la.vector_apply_matrix(expected, projection_matrix)

    depth = expected_2d[..., 2]
    vector = expected_2d[..., [0, 1]]

    actual = la.vector_unproject(vector, projection_matrix, depth=depth)

    # only test stable results
    assume(not np.any(np.isnan(actual) | np.isinf(actual)))
    assert np.allclose(actual, expected, rtol=1e-16, atol=np.inf)


def test_vector_unproject_exceptions():
    vector = np.ones(2)
    matrix = np.eye(4)
    matrix[1, 1] = 0

    with pytest.raises(ValueError):
        la.vector_unproject(vector, matrix)


def test_vector_apply_rotation_ordered():
    """Test that a positive pi/2 rotation about the z-axis and then the y-axis
    results in a different output then in standard rotation ordering."""
    vectors = np.array(
        [1, 0, 0],
    )
    expected = np.array(
        [0, 1, 0],
    )
    matrix = la.matrix_make_rotation_from_euler_angles(
        [0, np.pi / 2, np.pi / 2], order="zyx"
    )
    result = la.vector_apply_matrix(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


@given(ct.test_vector, ct.test_quaternion)
def test_vector_apply_quaternion(vector, quaternion):
    scale = np.linalg.norm(vector)
    if scale > 1e100:
        vector = vector / scale * 1e100

    actual = la.vector_apply_quaternion(vector, quaternion)

    # reference implementation
    matrix = la.quaternion_to_matrix(quaternion)
    vector = la.vector_make_homogeneous(vector)
    expected = (matrix @ vector)[..., :-1]

    # assert relative proximity only
    assert np.allclose(actual, expected, rtol=1e-10, atol=np.inf)


@given(ct.test_vector, ct.test_quaternion)
def test_vector_apply_quaternion_identity(vector, quaternion):
    scale = np.linalg.norm(vector)
    if scale > 1e100:
        vector = vector / scale * 1e100

    inv_quaternion = la.quaternion_inverse(quaternion)
    tmp = la.vector_apply_quaternion(vector, quaternion)
    actual = la.vector_apply_quaternion(tmp, inv_quaternion)

    # assert relative proximity only
    assert np.allclose(actual, vector, rtol=1e-10, atol=np.inf)


def test_vector_apply_matrix__perspective():
    # Test for OpenGL, wgpu, and arbitrary depth ranges
    depth_ranges = (-1, 1), (0, 1), (-2, 9)

    for depth_range in depth_ranges:
        m = la.matrix_make_perspective(-1, 1, -1, 1, 1, 17, depth_range=depth_range)

        # Check the depth range
        vec2 = la.vector_apply_matrix((0, 0, -1), m)
        assert vec2[2] == depth_range[0]
        vec2 = la.vector_apply_matrix((0, 0, -17), m)
        assert vec2[2] == depth_range[1]
        # vec2 = la.vector_apply_matrix((0, 0, -9), m) -> skip: halfway is not 0.5 ndc

        cases = [
            [(1, 0, -2), 0.5],
            [(1, 0, -4), 0.25],
            [(0, 0, -4), 0.0],
            [(-1, 0, -4), -0.25],
            [(-1, 0, -2), -0.5],
        ]

        # Check cases one by one
        for vec1, expected in cases:
            vec2 = la.vector_apply_matrix(vec1, m)
            assert vec2[0] == expected

        # Check cases batched
        vectors1 = np.row_stack([v for v, _ in cases])
        vectors2 = la.vector_apply_matrix(vectors1, m)
        assert vectors2[0][0] == cases[0][1]
        assert vectors2[1][0] == cases[1][1]
        assert vectors2[2][0] == cases[2][1]

        # Check cases batched, via out
        vectors2 = la.vector_apply_matrix(vectors1, m, out=vectors2)
        assert vectors2[0][0] == cases[0][1]
        assert vectors2[1][0] == cases[1][1]
        assert vectors2[2][0] == cases[2][1]


def test_vector_apply_matrix__orthographic():
    # Test for OpenGL, wgpu, and arbitrary depth ranges
    depth_ranges = (-1, 1), (0, 1), (-2, 9)

    for depth_range in depth_ranges:
        m = la.matrix_make_orthographic(-1, 1, -1, 1, 1, 17, depth_range=depth_range)

        # Check the depth range
        vec2 = la.vector_apply_matrix((0, 0, -1), m)
        assert vec2[2] == depth_range[0]
        vec2 = la.vector_apply_matrix((0, 0, -17), m)
        assert vec2[2] == depth_range[1]
        vec2 = la.vector_apply_matrix((0, 0, -9), m)
        assert vec2[2] == (depth_range[0] + depth_range[1]) / 2

        # This point would be at the edge of NDC
        vec2 = la.vector_apply_matrix((1, 0, -2), m)
        assert vec2[0] == 1

        # This point would be at the egde of NDC
        vec2 = la.vector_apply_matrix((1, 0, -4), m)
        assert vec2[0] == 1
