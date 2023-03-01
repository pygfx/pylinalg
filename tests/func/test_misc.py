import numpy as np
from hypothesis import assume, given
from hypothesis.strategies import floats

import pylinalg as la

from .. import conftest as ct


@given(
    ct.test_unit_vector,
    ct.test_unit_vector,
    floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
def test_aabb_to_sphere(point, offset, scale):
    assume(abs(scale) > 1e-16)
    assume(np.linalg.norm(point - (0, 0, 1)) > 1e-16)
    assume(np.linalg.norm(point - (0, 1, 0)) > 1e-16)
    assume(np.linalg.norm(point - (1, 0, 0)) > 1e-16)

    candidate = np.stack(
        (
            scale * point + offset,
            -scale * point + offset,
        )
    )

    aabb = np.empty_like(candidate)
    aabb[0, :] = np.min(candidate, axis=0)
    aabb[1, :] = np.max(candidate, axis=0)

    sphere = la.aabb_to_sphere(aabb)

    assert np.allclose(sphere[:3], offset, atol=1e-10)
    assert np.allclose(sphere[-1], abs(scale), rtol=1e-10)


@given(ct.test_unit_vector, ct.test_vector, ct.test_scaling)
def test_aabb_transform(point, translation, scale):
    candidate = np.stack((point, -point))
    aabb = np.empty_like(candidate)
    aabb[0, :] = np.min(candidate, axis=0)
    aabb[1, :] = np.max(candidate, axis=0)

    translation_matrix = la.matrix_make_translation(translation)
    result = la.aabb_transform(aabb, translation_matrix)
    assert np.allclose(result, aabb + translation, atol=1e-10)

    scale_matrix = la.matrix_make_scaling(scale)
    result = la.aabb_transform(aabb, scale_matrix)
    assert np.allclose(result, np.sort(aabb * scale, axis=0), atol=1e-10)
