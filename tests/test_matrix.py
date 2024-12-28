import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import assume, example, given

import pylinalg as la

from . import conftest as ct


@given(ct.legal_numbers | ct.test_vector, st.none() | ct.test_dtype)
def test_mat_from_translation(position, dtype):
    result = la.mat_from_translation(position, dtype=dtype)

    expected = np.eye(4, dtype=dtype)
    expected[:3, 3] = np.asarray(position)

    npt.assert_array_almost_equal(result, expected)


@given(ct.legal_numbers | ct.test_scaling, st.none() | ct.test_dtype)
def test_mat_from_scale(scale, dtype):
    result = la.mat_from_scale(scale, dtype=dtype)

    scaling = np.ones(4, dtype=dtype)
    scaling[:3] = np.asarray(scale, dtype=dtype)

    expected = np.identity(4, dtype=dtype)
    np.fill_diagonal(expected, scaling)

    npt.assert_array_almost_equal(result, expected)
    assert result.dtype == dtype


@given(ct.test_angles_rad, st.permutations("xyz"), ct.test_dtype)
@example((np.pi, -np.pi / 2, 0), "zyx", "f8")
@example((0, np.pi / 2, 0), "xyz", "f8")
def test_mat_from_euler(angles, order, dtype):
    result = la.mat_from_euler(angles, order="".join(order), dtype=dtype)

    expected = np.eye(4, dtype=dtype)
    for axis, angle in zip(order, angles):
        matrix = np.eye(4, dtype=dtype)
        matrix[:3, :3] = ct.rotation_matrix(axis, angle)
        expected = matrix @ expected

    npt.assert_array_almost_equal(result, expected)
    assert result.dtype == dtype


def test_mat_from_axis_angle_direction():
    """Test that a positive pi/2 rotation about the z-axis results
    in counter clockwise rotation, in accordance with the unit circle."""
    result = la.mat_from_axis_angle([0, 0, 1], np.pi / 2)
    npt.assert_array_almost_equal(
        result,
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )


def test_mat_from_axis_angle_xy():
    """Test that a negative pi rotation about the diagonal of the x-y plane
    flips the x and y coordinates, and negates the z coordinate."""
    axis = np.array([1, 1, 0], dtype="f8")
    axis /= np.linalg.norm(axis)
    result = la.mat_from_axis_angle(axis, -np.pi)
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
def test_quat_from_mat(angles, expected, dtype):
    matrix = la.mat_from_euler(angles, dtype=dtype)
    quaternion = la.quat_from_mat(matrix, dtype=dtype)
    npt.assert_array_almost_equal(
        quaternion,
        expected,
    )
    assert matrix.dtype == dtype
    assert quaternion.dtype == dtype


def test_mat_combine():
    """Test that the matrices are combined in the expected order."""
    # non-uniform scaling such that the test would fail if rotation/scaling are
    # applied in the incorrect order
    scaling = la.mat_from_scale([1, 2, 1])
    rotation = la.mat_from_euler([0, np.pi / 4, np.pi / 2])
    translation = la.mat_from_translation(2)
    # apply the standard SRT ordering
    result = la.mat_combine([translation, rotation, scaling])
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
        la.mat_combine()

    result = la.mat_combine([translation, translation], dtype="f4")
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


def test_mat_compose():
    """Test that the matrices are composed correctly in SRT order."""
    # non-uniform scaling such that the test would fail if rotation/scaling are
    # applied in the incorrect order
    scaling = [1, 2, 1]
    # quaternion corresponding to 90 degree rotation about z-axis
    rotation = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
    translation = [2, 2, 2]
    # compose the transform
    result = la.mat_compose(translation, rotation, scaling)
    npt.assert_array_almost_equal(
        result,
        [
            [0, -2, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 1, 2],
            [0, 0, 0, 1],
        ],
    )


def test_mat_decompose():
    """Test that the matrices are decomposed correctly."""
    matrix = [
        [0, -2, 0, 2],
        [1, 0, 0, 2],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ]
    translation, rotation, scaling = la.mat_decompose(matrix)
    npt.assert_array_equal(translation, [2, 2, 2])
    npt.assert_array_equal(scaling, [1, 2, 1])
    npt.assert_array_almost_equal(rotation, [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])


def test_mat_decompose_scaling_0():
    """Test that the matrices are decomposed correctly when scaling is 0."""

    scaling = [0, 0, 2]
    rotation = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
    translation = [2, 2, 2]

    matrix = la.mat_compose(translation, rotation, scaling)
    translation_, rotation_, scaling_ = la.mat_decompose(matrix)

    npt.assert_array_almost_equal(translation_, translation)
    npt.assert_array_almost_equal(scaling_, scaling)
    # rotation is not uniquely defined when scaling is 0, but it should not be NaN
    assert not np.isnan(rotation_).any()


@pytest.mark.parametrize(
    "signs",
    [
        # enumerate all combinations of signs
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [1, 1, -1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
    ],
    ids=str,
)
def test_mat_compose_roundtrip(signs):
    """Test that transform components survive a matrix
    compose -> decompose roundtrip."""
    scaling = np.array([1, 2, 3]) * signs
    rotation = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
    translation = [-100, -6, 5]
    matrix = la.mat_compose(translation, rotation, scaling)

    # decompose cannot reconstruct original scaling
    # so this is expected to fail
    translation2, rotation2, scaling2 = la.mat_decompose(matrix)
    npt.assert_array_equal(translation, translation2)
    if signs in ([1, 1, 1], [-1, 1, 1]):
        # if there are no flips, or if the flip happens to be the first axis
        # then we can correctly reconstruct the scaling without
        # prior knowledge
        npt.assert_array_almost_equal(scaling, scaling2)
        npt.assert_array_almost_equal(rotation, rotation2)
    else:
        with pytest.raises(AssertionError):
            npt.assert_array_almost_equal(scaling, scaling2)
        with pytest.raises(AssertionError):
            npt.assert_array_almost_equal(rotation, rotation2)

    # now inform decompose of the original scaling
    translation3, rotation3, scaling3 = la.mat_decompose(
        matrix, scaling_signs=np.sign(scaling)
    )
    npt.assert_array_equal(translation, translation3)
    npt.assert_array_almost_equal(scaling, scaling3)
    npt.assert_array_almost_equal(rotation, rotation3)


def test_mat_compose_validation():
    """Test that decompose validates consistency of scaling signs."""
    signs = [-1, -1, -1]
    scaling = np.array([1, 2, -3]) * signs
    rotation = [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2]
    translation = [-100, -6, 5]
    matrix = la.mat_compose(translation, rotation, scaling)

    with pytest.raises(ValueError):
        la.mat_decompose(matrix, scaling_signs=signs)


def naive_mat_compose(translation, rotation, scaling, /, *, out=None, dtype=None):
    return la.mat_combine(
        [
            la.mat_from_translation(translation),
            la.mat_from_quat(rotation),
            la.mat_from_scale(scaling),
        ],
        out=out,
        dtype=dtype,
    )


def test_mat_compose_naive():
    """Compare the direct composition with the naive composition."""
    npt.assert_equal(
        la.mat_compose([1, 2, 3], [np.pi, np.pi / 4, 0, 1], [1, -2, 9]),
        naive_mat_compose([1, 2, 3], [np.pi, np.pi / 4, 0, 1], [1, -2, 9]),
    )


def test_mat_compose_scalar_scaling():
    """Check that a scaler scaling argument is supported in mat_compose."""
    npt.assert_equal(
        la.mat_compose([1, 2, 3], [np.pi, np.pi / 4, 0, 1], [1.25, 1.25, 1.25]),
        la.mat_compose([1, 2, 3], [np.pi, np.pi / 4, 0, 1], 1.25),
    )

    npt.assert_equal(
        la.mat_compose([1, 2, 3], [np.pi, np.pi / 4, 0, 1], [1.25, 1.25, 1.25]),
        la.mat_compose([1, 2, 3], [np.pi, np.pi / 4, 0, 1], [1.25]),
    )


def test_mat_perspective():
    a = la.mat_perspective(-1, 1, -1, 1, 1, 100)
    npt.assert_array_almost_equal(
        a,
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -101 / 99, -200 / 99],
            [0, 0, -1, 0],
        ],
    )


def test_mat_orthographic():
    a = la.mat_orthographic(-1, 1, -1, 1, 1, 100)
    npt.assert_array_almost_equal(
        a,
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -2 / 99, -101 / 99],
            [0, 0, 0, 1],
        ],
    )


@given(ct.test_unit_vector, ct.test_unit_vector, ct.test_unit_vector)
def test_mat_look_at(eye, target, up_reference):
    # Note: to run this test, we need to choose 2 independent vectors (eye,
    # target) and one arbitrary vector. Scale doesn't matter, so doing this on
    # the unit-sphere will (almost) always succeed.
    independence_matrix = np.stack((eye, target), axis=0)
    assume(np.linalg.matrix_rank(independence_matrix) == 2)
    assume(np.linalg.norm(up_reference - (target - eye)) > 1e-10)

    rotation = la.mat_look_at(eye, target, up_reference)

    inverse_rotation = np.eye(4)
    inverse_rotation[:3, :3] = rotation[:3, :3].T

    # ensure matrix is inverted by its transpose
    identity = rotation @ inverse_rotation
    assert np.allclose(identity, np.eye(4), rtol=1e-10)

    # ensure z_new is along target - eye
    target_pointer = target - eye
    target_pointer = target_pointer / np.linalg.norm(target_pointer)
    target_pointer = la.vec_homogeneous(target_pointer)
    result = rotation @ (0, 0, 1, 1)
    assert np.allclose(result, target_pointer, rtol=1e-16)

    # ensure y_new, z_new, and up_reference roughly align
    # (map up_reference from target to source space and check if it's in the YZ-plane)
    new_reference = rotation.T @ la.vec_homogeneous(up_reference)
    assert np.abs(new_reference[0]) < 1e-10


def test_mat_euler_vs_scipy():
    """Compare our implementation with scipy's."""
    from scipy.spatial.transform import Rotation as R  # noqa: N817

    cases = [
        ("XYZ", [np.pi / 2, np.pi / 180, 0]),
        ("xyz", [np.pi / 2, np.pi / 180, 0]),
        ("ZXY", [np.pi, np.pi / 180, -np.pi / 180]),
        ("zxy", [np.pi, np.pi / 180, -np.pi / 180]),
        ("ZYX", [0, np.pi / 2, np.pi / 2]),
        ("zyx", [0, np.pi / 2, np.pi / 2]),
    ]

    for order, angles in cases:
        scipy_mat = np.identity(4)
        scipy_mat[:3, :3] = R.from_euler(order, angles).as_matrix()

        npt.assert_allclose(
            la.mat_from_euler(angles, order=order),
            scipy_mat,
            atol=1e-15,
        )
