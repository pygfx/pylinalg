import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import assume, example, given
from hypothesis.strategies import none

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


@given(ct.test_vector)
def test_vector_from_matrix_position(translation):
    matrix = pla.matrix_make_translation(translation)
    result = pla.vector_from_matrix_position(matrix)

    npt.assert_equal(result, translation)


def test_vector_apply_translation():
    vectors = np.array([[1, 0, 0]])
    expected = np.array([[0, 2, 2]])
    matrix = pla.matrix_make_translation([-1, 2, 2])
    result = pla.vector_apply_matrix(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


@given(ct.test_vector)
def test_vector_make_spherical_safe(vector):
    result = pla.vector_make_spherical_safe(vector)

    assert np.all((0 <= result[..., 1]) & (result[..., 1] < np.pi))
    assert np.all((0 <= result[..., 2]) & (result[..., 2] < 2 * np.pi))


def test_vector_apply_matrix_out():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    out = np.empty_like(vectors, dtype="i4")
    matrix = pla.matrix_make_translation([-1, 2, 2])
    result = pla.vector_apply_matrix(vectors, matrix, out=out)

    assert result is out


@given(ct.test_spherical, none())
@example((1, 0, np.pi / 2), (0, 0, 1))
@example((1, np.pi / 2, np.pi / 2), (1, 0, 0))
@example((1, 0, 0), (0, 1, 0))
def test_vector_euclidean_to_spherical(expected, vector):
    if vector is None:
        assume(abs(expected[0]) > 1e-10)
        vector = pla.vector_spherical_to_euclidean(expected)
    else:
        expected = np.asarray(expected)
        vector = np.asarray(vector)

    actual = pla.vector_euclidean_to_spherical(vector)

    assert np.allclose(actual, expected, rtol=1e-10)


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


@given(ct.test_vector, ct.test_vector)
def test_vector_distance_between(vector_a, vector_b):
    expected = np.linalg.norm(vector_a - vector_b)
    result = pla.vector_distance_between(vector_a, vector_b)

    assert np.allclose(result, expected, rtol=1e-10)


def test_vector_distance_between_exceptions():
    tmp = np.array(0)
    with pytest.raises(IndexError):
        pla.vector_distance_between((0, 0, 0), (0, 1, 0), out=tmp)


def test_vector_apply_matrix_out_dtype():
    vectors = np.array([[1, 0, 0]], dtype="f4")
    matrix = pla.matrix_make_translation([-1, 2, 2])
    out = np.empty_like(vectors, dtype="i4")
    result = pla.vector_apply_matrix(vectors, matrix, out=out, dtype="i2")

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
    result = pla.vector_spherical_to_euclidean(spherical)

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
    result = pla.vector_spherical_to_euclidean((1, 0, 0))
    assert np.allclose(result, (0, 1, 0))

    result = pla.vector_spherical_to_euclidean((1, 0, np.pi / 2))
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
    matrix = pla.matrix_make_rotation_from_euler_angles([0, 0, np.pi / 2])
    result = pla.vector_apply_matrix(vectors, matrix)

    npt.assert_array_almost_equal(
        result,
        expected,
    )


@given(ct.test_vector, ct.test_projection)
def test_vector_unproject(expected, projection_matrix):
    expected_hom = pla.vector_make_homogeneous(expected)
    expected_2d = projection_matrix @ expected_hom

    depth = expected_2d[..., 0]
    vector = expected_2d[..., [1, 2]]

    actual = pla.vector_unproject(vector, projection_matrix, depth=depth)

    # only test stable results
    assume(not np.any(np.isnan(actual) | np.isinf(actual)))
    assert np.allclose(actual, expected, rtol=1e-16, atol=np.inf)


def test_vector_unproject_exceptions():
    vector = np.ones(2)
    matrix = np.eye(4)
    matrix[1, 1] = 0

    with pytest.raises(ValueError):
        pla.vector_unproject(vector, matrix)


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


@given(ct.test_vector, ct.test_quaternion)
def test_vector_apply_quaternion_rotation(vector, quaternion):
    scale = np.linalg.norm(vector)
    if scale > 1e100:
        vector = vector / scale * 1e100

    actual = pla.vector_apply_quaternion_rotation(vector, quaternion)

    # reference implementation
    matrix = pla.quaternion_to_matrix(quaternion)
    vector = pla.vector_make_homogeneous(vector)
    expected = (matrix @ vector)[..., :-1]

    # assert relative proximity only
    assert np.allclose(actual, expected, rtol=1e-10, atol=np.inf)


@given(ct.test_vector, ct.test_quaternion)
def test_vector_apply_quaternion_rotation_identity(vector, quaternion):
    scale = np.linalg.norm(vector)
    if scale > 1e100:
        vector = vector / scale * 1e100

    inv_quaternion = pla.quaternion_inverse(quaternion)
    tmp = pla.vector_apply_quaternion_rotation(vector, quaternion)
    actual = pla.vector_apply_quaternion_rotation(tmp, inv_quaternion)

    # assert relative proximity only
    assert np.allclose(actual, vector, rtol=1e-10, atol=np.inf)
