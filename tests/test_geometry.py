import numpy as np
import numpy.testing as npt

import pylinalg as la


def test_vertex_normals():
    # a simple quad in the XY plane
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]
    )
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ]
    )

    expected = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    actual = la.vertex_normals(vertices, indices)
    npt.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32

    actual = la.vertex_normals(vertices, indices, dtype=np.float64)
    npt.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64

    block = np.zeros_like(vertices, dtype=np.float64)
    la.vertex_normals(vertices, indices, out=block)
    npt.assert_array_equal(block, expected)
    assert block.dtype == np.float64
