"""Note that we assume unit quaternions for faster implementations"""

import numpy as np


def quaternion_to_matrix(quaternion, /, *, out=None, dtype=None):
    """
    Make a rotation matrix given a quaternion.

    Parameters
    ----------
    quaternion : ndarray, [4]
        Quaternion.
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
        rotation matrix.
    """
    quaternion = np.asarray(quaternion)
    x, y, z, w = quaternion

    if out is None:
        out = np.identity(4, dtype=dtype)
    else:
        out[:] = np.identity(4)

    # credit to: http://www.songho.ca/opengl/gl_quaternion.htm
    # fmt: off
    out[:3, :3] = np.array([
        [1 - 2*y**2 - 2*z**2,       2*x*y - 2*w*z,       2*x*z + 2*w*y],  # noqa: E201, E501
        [      2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2,       2*y*w - 2*w*x],  # noqa: E201, E501
        [      2*x*w - 2*w*y,       2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2],  # noqa: E201, E501
    ]).T
    # fmt: on

    return out


def quaternion_multiply(a, b, /, *, out=None, dtype=None):
    """
    Multiply two quaternions

    Parameters
    ----------
    a : ndarray, [4]
        Left-hand quaternion
    b : ndarray, [4]
        Right-hand quaternion
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
    if out is None:
        out = np.empty(4, dtype=dtype)

    xyz = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    w = a[3] * b[3] - a[:3].dot(b[:3])

    out[:3] = xyz
    out[3] = w

    return out


def quaternion_make_from_unit_vectors(a, b, /, *, out=None, dtype=None):
    """
    Create a quaternion representing the rotation from one unit vectors
    to another

    Parameters
    ----------
    a : ndarray, [3]
        First unit vector
    b : ndarray, [3]
        Second unit vector
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
    if out is None:
        out = np.empty(4, dtype=dtype)

    w = 1 + np.dot(a, b)
    xyz = np.cross(a, b)

    out[:3] = xyz
    out[3] = w

    out /= np.linalg.norm(out)

    return out


def quaternion_inverse(quaternion, /, *, out=None, dtype=None):
    """
    Inverse of a given quaternion

    Parameters
    ----------
    a : ndarray, [3]
        First unit vector
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
    quaternion = np.asarray(quaternion)

    if out is None:
        out = np.empty_like(quaternion, dtype=dtype)

    out[:] = quaternion
    out[..., :3] *= -1

    return out


def quaternion_make_from_axis_angle(axis, angle, /, *, out=None, dtype=None):
    """
    Create a quaternion representing the rotation of an given angle
    about a given unit vector

    Parameters
    ----------
    axis : ndarray, [3]
        Unit vector
    angle : number
        The angle (in radians) to rotate about axis
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
    if out is None:
        out = np.empty(4, dtype=dtype)

    angle_half = angle / 2
    out[:3] = np.asarray(axis) * np.sin(angle_half)
    out[3] = np.cos(angle_half)

    return out
