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

    x2 = x * 2
    y2 = y * 2
    z2 = z * 2
    xx = x * x2
    xy = x * y2
    xz = x * z2
    yy = y * y2
    yz = y * z2
    zz = z * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2

    if out is None:
        out = np.identity(4, dtype=dtype)
    else:
        out[:] = np.identity(4)

    out[0, 0] = 1 - (yy + zz)
    out[1, 0] = xy + wz
    out[2, 0] = xz - wy
    out[0, 1] = xy - wz
    out[1, 1] = 1 - (xx + zz)
    out[2, 1] = yz + wx
    out[0, 2] = xz + wy
    out[1, 2] = yz - wx
    out[2, 2] = 1 - (xx + yy)

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
    a = np.asarray(a)
    b = np.asarray(b)

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


def quaternion_rotate(vector, quaternion, /, *, out=None, dtype=None):
    """
    Rotate a vector using a quaternion.

    Parameters
    ----------
    vector : ndarray, [3]
        The vector to rotate in local space.
    quaternion : ndarray, [4]
        The quaternion to rotate by in ``(x, y, z, w)`` format.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    rotated_vector : ndarray, [3]
        The input vector rotated by the given quaternion.

    Notes
    -----
    For improved accuracy consider normalizing the vector before applying the
    rotation and then re-apply the original scale afterwards.

    """

    vector = np.asarray(vector, dtype=dtype)
    quaternion = np.asarray(quaternion, dtype=dtype)

    scalar = quaternion[..., -1]
    q_vector = quaternion[..., :3]

    # the required linalg products
    q_v = np.tensordot(q_vector, vector, axes=(-1, -1))
    q_q = np.tensordot(q_vector, q_vector, axes=(-1, -1))
    qxv = np.cross(q_vector, vector, axis=-1)

    return (2 * q_v * q_vector) + (scalar**2 - q_q) * vector + 2 * scalar * qxv


def quaternion_make_from_euler_angles(angles, /, *, order="XYZ", out=None, dtype=None):
    """
    Create a quaternion from euler angles.

    Parameters
    ----------
    angles : ndarray, [3]
        A set of XYZ euler angles.
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
    quaternion : ndarray, [4]
        The rotation expressed as a quaternion.

    """

    angles = np.asarray(angles, dtype=float)
    batch_shape = angles.shape[:-1]

    if out is None:
        out = np.empty((*batch_shape, 4), dtype=dtype)

    # work out the sequence in which to apply the rotations
    is_extrinsic = [x.isupper() for x in order]
    order = [{"X": 0, "Y": 1, "Z": 2}[x.upper()] for x in order]

    # convert each euler matrix into a quaternion
    quaternions = np.zeros((len(order), *batch_shape, 4), dtype=float)
    quaternions[:, ..., -1] = np.cos(angles / 2)
    quaternions[np.arange(len(order)), ..., order] = np.sin(angles / 2)

    # multiple euler-angle quaternions respecting
    out[:] = quaternions[0]
    for idx in range(1, len(quaternions)):
        if is_extrinsic[idx]:
            quaternion_multiply(out, quaternions[idx], out=out)
        else:
            quaternion_multiply(quaternions[idx], out, out=out)

    return out
