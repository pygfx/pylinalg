from math import cos, sin

import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import arrays, from_dtype

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
    valid_letters = "?iuf"

    letter_idx = draw(st.integers(min_value=0, max_value=len(valid_letters) - 1))
    letter = valid_letters[letter_idx]

    if letter == "?":
        code = letter
    elif letter == "f":
        valid_lengths = "48"
        n_bytes = draw(st.integers(min_value=0, max_value=len(valid_lengths) - 1))
        code = letter + valid_lengths[n_bytes]
    else:
        valid_lengths = "1248"
        n_bytes = draw(st.integers(min_value=0, max_value=len(valid_lengths) - 1))
        code = letter + valid_lengths[n_bytes]

    return code


def nonzero_scale(scale):
    return np.where(np.abs(scale) < EPS, 1, scale)


# Hypthesis testing strategies
legal_numbers = from_dtype(np.dtype(float), allow_infinity=False, allow_nan=False)
test_vector = arrays(float, (3,), elements=legal_numbers)
test_quaternion = generate_quaternion()
test_matrix_affine = arrays(float, (4, 4), elements=legal_numbers)
test_scaling = arrays(float, (3,), elements=legal_numbers).map(nonzero_scale)
test_dtype = dtype_string()
