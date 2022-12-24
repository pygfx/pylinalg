import numpy as np
from hypothesis import given

import pylinalg as pla

from ..conftest import test_quaternion, test_scaling, test_vector, test_unit_vector


@given(test_vector, test_vector)
def test_affine_chaining(position, position2):
    transform_a = pla.AffineTransform(position=position)
    transform_b = pla.AffineTransform(position=position2)

    chained = transform_a @ transform_b

    expected = position + position2
    np.testing.assert_allclose(chained.position, expected, equal_nan=True)


@given(test_unit_vector, test_quaternion, test_scaling)
def test_affine_inverse(position, orientation, scale):
    """
    Checks if an affine transform times its inverse is the identity.

    Note that this only uses unit vectors for position, because allowing
    vectors with other scales leads to numerical inaccuracies that cause this
    test to fail.

    """
    transform = pla.AffineTransform()
    transform.position = position
    transform.orientation = orientation
    transform.scale = scale

    inverse = transform.inverse()

    expected = np.identity(4)
    actual = (transform @ inverse).as_matrix()
    np.testing.assert_almost_equal(actual, expected)

    expected = np.identity(4)
    actual = (inverse @ transform).as_matrix()
    np.testing.assert_almost_equal(actual, expected)
