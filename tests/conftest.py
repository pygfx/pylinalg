import numpy as np
from hypothesis.extra.numpy import arrays, from_dtype


def normalize_quaternion(quaternion):
    if np.linalg.norm(quaternion < 1e-16):
        return np.array((0, 0, 0, 1))

    return quaternion / np.linalg.norm(quaternion)


def nonzero_scale(scale):
    if np.all(scale == 0):
        return np.array((1, 1, 1))

    return scale


# Hypthesis testing strategies
legal_numbers = from_dtype(np.dtype(float), allow_infinity=False, allow_nan=False)
test_vector = arrays(float, (3,), elements=legal_numbers)
test_quaternion = arrays(float, (4,), elements=legal_numbers).map(normalize_quaternion)
test_matrix_affine = arrays(float, (4, 4), elements=legal_numbers)
test_scaling = arrays(float, (3,), elements=legal_numbers).map(nonzero_scale)
