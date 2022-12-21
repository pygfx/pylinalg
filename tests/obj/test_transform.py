import numpy as np
from hypothesis import given

import pylinalg as pla

from ..conftest import test_quaternion, test_scaling, test_vector, EPS


@given(test_vector, test_vector)
def test_affine_chaining(position, position2):
    transform_a = pla.AffineTransform(position=position)
    transform_b = pla.AffineTransform(position=position2)

    chained = transform_a @ transform_b

    expected = position + position2
    np.testing.assert_allclose(chained.position, expected, equal_nan=True)


@given(test_vector, test_quaternion, test_scaling)
def test_inverse(position, orientation, scale):
    transform = pla.AffineTransform()
    transform.position = position
    transform.orientation = orientation
    transform.scale = scale

    inverse = transform.inverse()

    expected = np.eye(4)

    result = (inverse @ transform).as_matrix()
    np.testing.assert_allclose(result, expected, atol=EPS)

    result = (transform @ inverse).as_matrix()
    np.testing.assert_allclose(result, expected, atol=EPS)
