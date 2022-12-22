from math import cos, sin

import numpy as np


def matrix_combine(matrices, /, *, out=None, dtype=None):
    """
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
        Combined transformation matrix.
    """
    if out is None:
        out = np.empty((4, 4), dtype=dtype)
    out[:] = np.identity(4, dtype=dtype)
    for matrix in matrices:
        try:
            np.dot(out, matrix, out=out)
        except ValueError:
            out[:] = np.dot(out, matrix)
    return out


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


def matrix_make_rotation_from_euler_angles(
    angles, /, *, order="xyz", out=None, dtype=None
):
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


    Notes
    -----
    The current implementation only supports euler-angles that are permutations
    of "xyz". I.e., other formats like "yzy" are not supported.

    """
    order = order.lower()

    matrices = []
    for angle, axis in zip(angles, order):
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]

        matrix = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        matrix = np.insert(matrix, axis_idx, 0, axis=0)
        matrix = np.insert(matrix, axis_idx, 0, axis=1)
        matrix[axis_idx, axis_idx] = 1

        affine_matrix = np.identity(4, dtype=dtype)
        affine_matrix[:3, :3] = matrix

        matrices.append(affine_matrix)

    # note: combining in the loop would save time and memory usage
    return matrix_combine([x for x in reversed(matrices)], out=out, dtype=dtype)


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

    if out is None:
        out = np.identity(4, dtype=dtype)
    else:
        out[:] = np.identity(4)

    eye = out[:3, :3]
    rotation = np.cos(angle) * eye
    # the second component here is the "cross product matrix" of axis
    rotation += np.sin(angle) * np.cross(axis, eye * -1)
    rotation += (1 - np.cos(angle)) * (np.outer(axis, axis))
    out[:3, :3] = rotation

    return out


def matrix_to_quaternion(matrix, /, *, out=None, dtype=None):
    """
    Make a quaternion given a rotation matrix.

    Parameters
    ----------
    matrix : ndarray, [3]
        The rotation matrix.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [4]
        Quaternion.
    """
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


def matrix_make_transform(translation, rotation, scaling, /, *, out=None, dtype=None):
    """
    Compose a transformation matrix given a translation vector, a
    quaternion and a scaling vector.

    Parameters
    ----------
    translation : number or ndarray, [3]
        translation vector
    rotation : ndarray, [4]
        quaternion
    scaling : number or ndarray, [3]
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
        Transformation matrix
    """
    from .quaternion import quaternion_to_matrix

    return matrix_combine(
        [
            matrix_make_translation(translation),
            quaternion_to_matrix(rotation),
            matrix_make_scaling(scaling),
        ],
        out=out,
        dtype=dtype,
    )


def matrix_decompose(matrix, /, *, dtype=None, out=None):
    """
    Decompose a transformation matrix into a translation vector, a
    quaternion and a scaling vector.

    Parameters
    ----------
    matrix : ndarray, [4, 4]
        transformation matrix
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    translation : ndarray, [3]
        translation vector
    rotation : ndarray, [4]
        quaternion
    scaling : ndarray, [3]
        scaling factor(s)
    """
    matrix = np.asarray(matrix)

    if out is not None:
        translation = out[0]
    else:
        translation = np.empty((3,), dtype=dtype)
    translation[:] = matrix[:-1, -1]

    if out is not None:
        scaling = out[2]
    else:
        scaling = np.empty((3,), dtype=dtype)
    scaling[:] = np.linalg.norm(matrix[:-1, :-1], axis=0)
    if np.linalg.det(matrix) < 0:
        scaling[0] *= -1

    rotation = out[1] if out is not None else None
    rotation_matrix = matrix[:-1, :-1] * (1 / scaling)[None, :]
    rotation = matrix_to_quaternion(rotation_matrix, out=rotation, dtype=dtype)

    return translation, rotation, scaling


def matrix_make_perspective(
    left, right, top, bottom, near, far, /, *, out=None, dtype=None
):
    """
    Create a perspective projection matrix.

    Parameters
    ----------
    left : number
        distance between the left frustum plane and the origin
    right : number
        distance between the right frustum plane and the origin
    top : number
        distance between the top frustum plane and the origin
    bottom : number
        distance between the bottom frustum plane and the origin
    near : number
        distance between the near frustum plane and the origin
    far : number
        distance between the far frustum plane and the origin
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    matrix : ndarray, [4, 4]
        perspective projection matrix
    """
    if out is None:
        out = np.zeros((4, 4), dtype=dtype)
    else:
        out[:] = 0.0

    x = 2 * near / (right - left)
    y = 2 * near / (top - bottom)

    a = (right + left) / (right - left)
    b = (top + bottom) / (top - bottom)
    c = -(far + near) / (far - near)
    d = -2 * far * near / (far - near)

    out[0, 0] = x
    out[0, 2] = a
    out[1, 1] = y
    out[1, 2] = b
    out[2, 2] = c
    out[2, 3] = d
    out[3, 2] = -1

    return out


def matrix_make_orthographic(
    left, right, top, bottom, near, far, /, *, out=None, dtype=None
):
    """
    Create an orthographic projection matrix.

    Parameters
    ----------
    left : number
        distance between the left frustum plane and the origin
    right : number
        distance between the right frustum plane and the origin
    top : number
        distance between the top frustum plane and the origin
    bottom : number
        distance between the bottom frustum plane and the origin
    near : number
        distance between the near frustum plane and the origin
    far : number
        distance between the far frustum plane and the origin
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    matrix : ndarray, [4, 4]
        orthographic projection matrix
    """
    if out is None:
        out = np.zeros((4, 4), dtype=dtype)
    else:
        out[:] = 0.0

    w = 1.0 / (right - left)
    h = 1.0 / (top - bottom)
    p = 1.0 / (far - near)

    x = (right + left) * w
    y = (top + bottom) * h
    z = (far + near) * p

    out[0, 0] = 2 * w
    out[0, 3] = -x
    out[1, 1] = 2 * h
    out[1, 3] = -y
    out[2, 2] = -2 * p
    out[2, 3] = -z
    out[3, 3] = 1

    return out


def matrix_make_look_at(eye, target, up, /, *, out=None, dtype=None):
    """
    Rotation that aligns two vectors.

    Computes a rotation matrix that rotates the input frame's z-axis (forward)
    to point in direction ``target - eye`` and the input frame's y-axis (up) to
    point in direction ``up``.

    Parameters
    ----------
    eye : ndarray, [3]
        A vector indicating the direction that should be aligned.
    target : ndarray, [3]
        A vector indicating the direction to align on.
    up : ndarray, [3]
        The direction of the camera's up axis.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.


    Returns
    -------
    rotation_matrix : ndarray, [4, 4]
        A matrix describing the rotation.

    """

    raise NotImplementedError()
