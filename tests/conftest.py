from math import cos, sin

import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import arrays, from_dtype

import pylinalg as pla

# upper bound on approximation error
EPS = 1e-6


@st.composite
def generate_quaternion(
    draw,
    elements=st.floats(
        min_value=0, max_value=360, allow_infinity=False, allow_nan=False
    ),
    snap_precision=0,
):
    """
    Generate a valid quaternion

    This function generates a quaternion from three angles. It first generates a
    point on the unit-sphere (in polar coordinates) that represents the rotation
    vector. It then generates an angle to rotate by, and finally constructs a
    valid quaternion from this axis-angle representation.

    Parameters
    ----------
    draw : Any
        Mandatory input from Hypothesis to track when elements are drawn from
        strategies to allow test-case simplification on failure.
    elements : strategy
        A strategy that creates valid elements. Defaults to any degree in [0, 360].
    snap_precision : int
        The precision to which to round ("snap") angles to.

    Returns
    -------
    quaternion : ndarray, [4]
        The generated quaternion.

    """
    theta, phi, angle = draw(elements), draw(elements), draw(elements)
    theta, phi, angle = (
        round(theta, snap_precision),
        round(phi, snap_precision),
        round(angle, snap_precision),
    )
    hp.assume(theta <= 180)

    theta, phi, angle = (
        2 * np.pi * theta / 360,
        2 * np.pi * phi / 360,
        2 * np.pi * angle / 360,
    )

    # spherical to euclidean (r = 1)
    x = cos(theta) * sin(phi)
    y = sin(theta) * sin(phi)
    z = cos(phi)

    # axis-angle to quaternion
    qx = x * sin(angle / 2)
    qy = y * sin(angle / 2)
    qz = z * sin(angle / 2)
    qw = cos(angle / 2)

    quaternion = np.array((qx, qy, qz, qw))

    # reject samples that are not precise
    hp.assume(np.linalg.norm(quaternion) - 1 < EPS)

    return quaternion


@st.composite
def dtype_string(draw):
    letter = draw(st.sampled_from("?iuf"))

    if letter == "?":
        code = letter
    elif letter == "f":
        code = letter + draw(st.sampled_from("48"))
    else:
        code = letter + draw(st.sampled_from("1248"))

    return code


def rotation_matrix(axis, angle):
    """Rotation by angle around the given cardinal axis.

    Parameters
    ----------
    axis : str
        One of "x", "y", or "z".
    angle : float
        The angle to rotate by (in rad).

    """

    matrix = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    matrix = np.insert(matrix, axis_idx, 0, axis=0)
    matrix = np.insert(matrix, axis_idx, 0, axis=1)
    matrix[axis_idx, axis_idx] = 1

    return matrix


@st.composite
def unit_vector(
    draw,
    elements=st.floats(
        allow_infinity=False, allow_nan=False, min_value=0, max_value=2 * np.pi
    ),
):
    """
    Generate a unit vector using a point on the unit-sphere
    (essentially spherical coordinates to euclidean coordinates)
    """
    theta, phi = draw(elements), draw(elements)

    # spherical to euclidean (r = 1)
    x = cos(theta) * sin(phi)
    y = sin(theta) * sin(phi)
    z = cos(phi)

    return np.array((x, y, z))


@st.composite
def perspecitve_matrix(
    draw, elements=st.floats(allow_infinity=False, allow_nan=False, min_value=1e-16)
):
    top, bottom = draw(elements), draw(elements)
    hp.assume(top != bottom)

    left, right = draw(elements), draw(elements)
    hp.assume(left != right)

    near, far = draw(elements), draw(elements)
    hp.assume(near != far)
    hp.assume(0 < near)
    hp.assume(near < far)

    matrix = pla.matrix_make_perspective(left, right, top, bottom, near, far)
    hp.assume(not (np.any(np.isinf(matrix) | np.isnan(matrix))))

    try:
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # only stable/invertible matrices
        hp.assume(False)

    return matrix


@st.composite
def orthographic_matrix(
    draw, elements=st.floats(allow_infinity=False, allow_nan=False)
):
    top, bottom = draw(elements), draw(elements)
    hp.assume(top != bottom)

    left, right = draw(elements), draw(elements)
    hp.assume(left != right)

    near, far = draw(elements), draw(elements)
    hp.assume(near != far)
    hp.assume(0 < near)
    hp.assume(near < far)

    matrix = pla.matrix_make_orthographic(left, right, top, bottom, near, far)
    hp.assume(not (np.any(np.isinf(matrix) | np.isnan(matrix))))

    try:
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # only stable/invertible matrices
        hp.assume(False)

    return matrix


def nonzero_scale(scale):
    return np.where(np.abs(scale) < EPS, 1, scale)


# Hypthesis testing strategies
legal_numbers = from_dtype(np.dtype(float), allow_infinity=False, allow_nan=False)
legal_angle = from_dtype(
    np.dtype(float),
    allow_infinity=False,
    allow_nan=False,
    min_value=0,
    max_value=2 * np.pi,
)
test_vector = arrays(float, (3,), elements=legal_numbers)
test_quaternion = generate_quaternion()
test_matrix_affine = arrays(float, (4, 4), elements=legal_numbers)
test_scaling = arrays(float, (3,), elements=legal_numbers).map(nonzero_scale)
test_dtype = dtype_string()
test_angles_rad = arrays(float, (3,), elements=legal_angle)
test_unit_vector = unit_vector()
test_projection = perspecitve_matrix() | orthographic_matrix()
