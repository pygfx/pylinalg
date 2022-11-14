from functools import partial, reduce

import numpy as np


matrix_combine = partial(reduce, np.dot)
matrix_combine.__doc__ = """
Combine a list of affine matrices by multiplying them.

Note that by matrix multiplication rules, the output matrix will applied the
given transformations in reverse order. For example, passing a scaling,
rotation and translation matrix (in that order), will lead to a combined
transformation matrix that applies translation first, then rotation and finally
scaling.

Parameters
----------
matrices : list of ndarray, [4, 4]
    List of affine matrices to combine.

Returns
-------
ndarray, [4, 4]
    Combined transformation matrix.
"""


def matrix_make_translation(vector, /, *, out=None, dtype=None):
    """
    Make a translationmatrix given a translation vector.

    Parameters
    ----------
    vector : number or ndarray, [3]
        translation vector
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [4, 4]
        Translation matrix.
    """
    vector = np.asarray(vector)

    matrix = np.identity(4, dtype=dtype)
    matrix[:-1, -1] = vector

    if out is not None:
        out[:] = matrix
        return out
    
    return matrix


def matrix_make_scaling(factors, /, *, out=None, dtype=None):
    """
    Make a scaling matrix given scaling factors per axis, or a
    single uniform scaling factor.

    Parameters
    ----------
    factor : number or ndarray, [3]
        scaling factor(s)
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [4, 4]
        Scaling matrix.
    """
    factors = np.asarray(factors)

    matrix = np.identity(4, dtype=dtype)
    matrix[np.diag_indices(3)] = factors

    if out is not None:
        out[:] = matrix
        return out

    return matrix


def matrix_make_rotation_from_euler_angles(angles, /, *, order="xyz", out=None, dtype=None):
    """
    Make a matrix given euler angles (in radians) per axis.

    Parameters
    ----------
    angles : ndarray, [3]
        The euler angles.
    order : string, optional
        The order in which the rotations should be applied. Default
        is "xyz".
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [4, 4]
        Rotation matrix.
    """
    angles = np.asarray(angles)

    matrix_x = np.identity(4)
    matrix_y = np.identity(4)
    matrix_z = np.identity(4)

    matrix_x[1, 1] = np.cos(angles[0])
    matrix_x[1, 2] = -np.sin(angles[0])
    matrix_x[2, 1] = np.sin(angles[0])
    matrix_x[2, 2] = np.cos(angles[0])

    matrix_y[0, 0] = np.cos(angles[1])
    matrix_y[0, 2] = np.sin(angles[1])
    matrix_y[2, 0] = -np.sin(angles[1])
    matrix_y[2, 2] = np.cos(angles[1])

    matrix_z[0, 0] = np.cos(angles[2])
    matrix_z[0, 1] = -np.sin(angles[2])
    matrix_z[1, 0] = np.sin(angles[2])
    matrix_z[1, 1] = np.cos(angles[2])

    lookup = {
        "x": matrix_x,
        "y": matrix_y,
        "z": matrix_z,
    }
    matrix = matrix_combine([lookup[i] for i in reversed(order.lower())])

    if out is not None:
        out[:] = matrix
        return out

    if dtype is not None:
        matrix = matrix.astype(dtype, copy=False)

    return matrix


def matrix_make_rotation_from_axis_angle(axis, angle, /, *, out=None, dtype=None):
    """
    Make a rotation matrix given a rotation axis and an angle (in radians).

    Parameters
    ----------
    axis : ndarray, [3]
        The rotation axis.
    angle : number
        The angle (in radians) to rotate about the axis.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [4, 4]
        Rotation matrix.
    """
    axis = np.asarray(axis)

    matrix = np.identity(4)
    rotation = np.cos(angle) * matrix[:3, :3]
    # the second component here is the "cross product matrix" of axis
    rotation += np.sin(angle) * np.cross(axis, matrix[:3, :3] * -1)
    rotation += (1 - np.cos(angle)) * (np.outer(axis, axis))
    matrix[:3, :3] = rotation

    if out is not None:
        out[:] = matrix
        return out

    if dtype is not None:
        matrix = matrix.astype(dtype, copy=False)

    return matrix


def matrix_to_quaternion(matrix, out=None, dtype=None):
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
        out = np.empty((4,), dtype=dtype)
    out[:] = np.array([x, y, z, w])
    return out


def matrix_compose(translation, rotation, scaling, out=None, dtype=None):
    from .quaternion import quaternion_to_matrix

    translation = np.asarray(translation)
    rotation = np.asarray(rotation)
    scaling = np.asarray(scaling)

    if out is None:
        out = np.empty((4, 4), dtype=dtype)
    out[:] = matrix_combine(
        [
            matrix_make_translation(translation),
            quaternion_to_matrix(rotation),
            matrix_make_scaling(scaling),
        ]
    )
    return out


def matrix_decompose(matrix, translation=None, rotation=None, scaling=None, dtype=None):
    matrix = np.asarray(matrix)

    if translation is None:
        translation = np.empty((3,), dtype=dtype)
    translation[:] = matrix[:-1, -1]

    if scaling is None:
        scaling = np.empty((3,), dtype=dtype)
    scaling[:] = np.linalg.norm(matrix[:-1, :-1], axis=0)
    if np.linalg.det(matrix) < 0:
        scaling[0] *= -1

    normal_rotation_matrix = matrix[:-1, :-1] * (1 / scaling)[None, :]
    if rotation is None:
        rotation = np.empty((4,), dtype=dtype)
    matrix_to_quaternion(normal_rotation_matrix, out=rotation)

    return translation, rotation, scaling


def matrix_make_perspective(left, right, top, bottom, near, far, out=None, dtype=None):
    if out is None:
        out = np.empty((4, 4), dtype=dtype)

    x = 2 * near / (right - left)
    y = 2 * near / (top - bottom)

    a = (right + left) / (right - left)
    b = (top + bottom) / (top - bottom)
    c = -(far + near) / (far - near)
    d = -2 * far * near / (far - near)

    out[:] = 0
    out[0, 0] = x
    out[0, 2] = a
    out[1, 1] = y
    out[1, 2] = b
    out[2, 2] = c
    out[2, 3] = d
    out[3, 2] = -1

    return out


def matrix_make_orthographic(left, right, top, bottom, near, far, out=None, dtype=None):
    if out is None:
        out = np.empty((4, 4), dtype=dtype)

    w = 1.0 / (right - left)
    h = 1.0 / (top - bottom)
    p = 1.0 / (far - near)

    x = (right + left) * w
    y = (top + bottom) * h
    z = (far + near) * p

    out[:] = 0
    out[0, 0] = 2 * w
    out[0, 3] = -x
    out[1, 1] = 2 * h
    out[1, 3] = -y
    out[2, 2] = -2 * p
    out[2, 3] = -z
    out[3, 3] = 1

    return out
