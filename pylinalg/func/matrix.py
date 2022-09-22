from functools import reduce

import numpy as np


def matrix_combine(matrices):
    return reduce(np.dot, matrices)


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


def matrix_make_rotation_from_euler_angles(angles, dtype="f8"):
    """Make a matrix given euler angles per axis."""
    angles = np.asarray(angles)

    matrix_x = np.identity(4, dtype=dtype)  # x axis rotation
    matrix_x[1, 1] = np.cos(angles[0])
    matrix_x[1, 2] = -np.sin(angles[0])
    matrix_x[2, 1] = np.sin(angles[0])
    matrix_x[2, 2] = np.cos(angles[0])

    matrix_y = np.identity(4, dtype=dtype)  # y axis rotation
    matrix_y[0, 0] = np.cos(angles[1])
    matrix_y[0, 2] = np.sin(angles[1])
    matrix_y[2, 0] = -np.sin(angles[1])
    matrix_y[2, 2] = np.cos(angles[1])

    matrix_z = np.identity(4, dtype=dtype)  # z axis rotation
    matrix_z[0, 0] = np.cos(angles[2])
    matrix_z[0, 1] = -np.sin(angles[2])
    matrix_z[1, 0] = np.sin(angles[2])
    matrix_z[1, 1] = np.cos(angles[2])

    return matrix_combine([matrix_z, matrix_y, matrix_x])
