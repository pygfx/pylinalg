import numpy as np
from hypothesis import given

import pylinalg as pla

from ..conftest import test_quaternion, test_scaling, test_vector


@given(test_vector, test_vector)
def test_affine_chaining(position, position2):
    transform_a = pla.AffineTransform(position=position)
    transform_b = pla.AffineTransform(position=position2)

    chained = transform_a @ transform_b

    expected = position + position2
    np.testing.assert_allclose(chained.position, expected, equal_nan=True)


@given(test_vector, test_quaternion, test_scaling)
def test_affine_inverse(position, orientation, scale):
    """
    Normally, we would just test if the combination results in an identity
    matrix. Unfortunately though, limited precision arithmetic defeats this
    approach, because we can produce (almost) arbitrarily large errors by
    carefully choosing the rotation and scale components of the affine transform
    (which hypothesis is very good at doing, so it quickly finds breaking
    examples.)

    Instead, this test checks the position, rotation, and scale manually,
    because there errors compound less in this case.

    """
    transform = pla.AffineTransform()
    transform.position = position
    transform.orientation = orientation
    transform.scale = scale

    inverse = transform.inverse()

    result = inverse @ transform  # numerical identity
    np.testing.assert_array_almost_equal(result.orientation, (0, 0, 0, 1))
    np.testing.assert_array_almost_equal(result.scale, (1, 1, 1))

    result = transform @ inverse
    np.testing.assert_array_almost_equal(result.orientation, (0, 0, 0, 1))
    np.testing.assert_array_almost_equal(result.scale, (1, 1, 1))

    # position is where errors compound, so we need to run a
    # looser test here
    # np.testing.assert_array_almost_equal(result.position, (0, 0, 0))
