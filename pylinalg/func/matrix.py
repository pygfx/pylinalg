from functools import partial, reduce

import numpy as np


matrix_combine = partial(reduce, np.dot)


def matrix_make_translation(vector, dtype="f8"):
    """Make a matrix given a translation vector, or a
    single offset to apply to all axes."""
    vector = np.asarray(vector, dtype=dtype)

    vector = np.atleast_1d(vector)
    if vector.ndim != 1:
        raise NotImplementedError()

    if vector.size == 1:
        vector = np.full((3,), vector[0], dtype=dtype)
    elif vector.shape != (3,):
        raise NotImplementedError()

    matrix = np.identity(4, dtype=dtype)
    matrix[:-1, -1] = vector
    return matrix


def matrix_make_scaling(factors, dtype="f8"):
    """Make a matrix given scaling factors per axis, or a
    single uniform scaling factor."""
    factors = np.asarray(factors, dtype=dtype)

    factors = np.atleast_1d(factors)
    if factors.ndim != 1:
        raise NotImplementedError()

    if factors.size == 1:
        factors = np.full((3,), factors[0], dtype=dtype)
    elif factors.shape != (3,):
        raise NotImplementedError()

    matrix = np.identity(4, dtype=dtype)
    matrix[0, 0] = factors[0]
    matrix[1, 1] = factors[1]
    matrix[2, 2] = factors[2]
    return matrix


def matrix_make_rotation_from_euler_angles(angles, order="xyz", dtype="f8"):
    """Make a matrix given euler angles per axis."""
    angles = np.asarray(angles)
    matrix = {
        "x": np.identity(4, dtype=dtype),
        "y": np.identity(4, dtype=dtype),
        "z": np.identity(4, dtype=dtype),
    }

    matrix["x"][1, 1] = np.cos(angles[0])
    matrix["x"][1, 2] = -np.sin(angles[0])
    matrix["x"][2, 1] = np.sin(angles[0])
    matrix["x"][2, 2] = np.cos(angles[0])

    matrix["y"][0, 0] = np.cos(angles[1])
    matrix["y"][0, 2] = np.sin(angles[1])
    matrix["y"][2, 0] = -np.sin(angles[1])
    matrix["y"][2, 2] = np.cos(angles[1])

    matrix["z"][0, 0] = np.cos(angles[2])
    matrix["z"][0, 1] = -np.sin(angles[2])
    matrix["z"][1, 0] = np.sin(angles[2])
    matrix["z"][1, 1] = np.cos(angles[2])

    return matrix_combine([matrix[i] for i in reversed(order.lower())])


def matrix_to_quaternion(matrix, out=None):
    m = matrix[:3, :3]
    t = np.trace(m)

    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
        w = 0.25 / s

    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
        w = (m[2, 1] - m[1, 2]) / s

    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
        w = (m[0, 2] - m[2, 0]) / s

    else:
        s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
        w = (m[1, 0] - m[0, 1]) / s

    if out is None:
        out = np.empty((4,), dtype=matrix.dtype)
    out[:] = x, y, z, w
    return out


matrix_inverse = np.linalg.inv


def matrix_compose(translation, rotation, scaling, out=None):
    from .quaternion import quaternion_to_matrix

    translation = np.asarray(translation)
    rotation = np.asarray(rotation)
    scaling = np.asarray(scaling)

    if out is None:
        out = np.empty((4, 4), dtype=translation.dtype)
    out[:] = matrix_combine(
        [
            matrix_make_translation(translation),
            quaternion_to_matrix(rotation),
            matrix_make_scaling(scaling),
        ]
    )
    return out


def matrix_decompose(matrix, translation=None, rotation=None, scaling=None):
    matrix = np.asarray(matrix)

    if translation is None:
        translation = np.empty((3,), dtype=matrix.dtype)
    translation[:] = matrix[:-1, -1]

    if scaling is None:
        scaling = np.empty((3,), dtype=matrix.dtype)
    scaling[:] = np.linalg.norm(matrix[:-1, :-1], axis=0)
    if np.linalg.det(matrix) < 0:
        scaling[0] *= -1

    normal_rotation_matrix = matrix[:-1, :-1] * (1 / scaling)[None, :]
    rotation = matrix_to_quaternion(normal_rotation_matrix)

    return translation, rotation, scaling
