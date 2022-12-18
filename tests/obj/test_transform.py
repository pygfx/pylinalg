import numpy as np
from hypothesis import given

import pylinalg as pla
from ..conftest import test_vector, test_quaternion, test_scaling


@given(test_vector, test_vector)
def test_affine_chaining(position, position2):
    transform_a = pla.AffineTransform(position=position)
    transform_b = pla.AffineTransform(position=position2)

    chained = transform_a @ transform_b

    expected = position + position2
    np.testing.assert_allclose(chained.position, expected, equal_nan=True)


@given(test_vector, test_quaternion, test_scaling)
def test_inverse(position, orientation_raw, scale):

    if np.any(orientation_raw != 0):
        orientation = orientation_raw / orientation_raw[-1]
        orientation = orientation_raw / np.linalg.norm(orientation_raw)
    else:
        orientation = np.array((0, 0, 0, 1))

    transform = pla.AffineTransform()
    transform.position = position
    transform.orientation = orientation
    transform.scale = scale

    inverse = transform.inverse()

    expected = np.eye(4)

    result = (inverse @ transform).as_matrix()
    np.testing.assert_allclose(result, expected)

    result = (transform @ inverse).as_matrix()
    np.testing.assert_allclose(result, expected)
