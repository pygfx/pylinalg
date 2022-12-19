import numpy as np
from hypothesis.extra.numpy import arrays, from_dtype
import hypothesis.strategies as st
import hypothesis as hp


@st.composite
def generate_quaternion(draw, elements=st.floats()):
    x, y, z, w = draw(elements), draw(elements), draw(elements), draw(elements)

    quaternion = np.array((x, y, z, w))
    quaternion /= np.linalg.norm(quaternion)

    # reject samples that produce inf or nan (due to insufficient machine
    # precision)
    hp.assume(np.linalg.norm(quaternion) == 1)

    return quaternion


def nonzero_scale(scale):
    return np.where(np.abs(scale) < 1e-6, 1, scale)


# Hypthesis testing strategies
legal_numbers = from_dtype(np.dtype(float), allow_infinity=False, allow_nan=False)
test_vector = arrays(float, (3,), elements=legal_numbers)
test_quaternion = generate_quaternion(elements=legal_numbers)
test_matrix_affine = arrays(float, (4, 4), elements=legal_numbers)
test_scaling = arrays(float, (3,), elements=legal_numbers).map(nonzero_scale)
