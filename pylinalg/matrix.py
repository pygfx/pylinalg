import numpy as np
from numpy.lib.stride_tricks import as_strided


def mat_combine(matrices, /, *, out=None, dtype=None):
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

    matrices = [np.asarray(matrix) for matrix in matrices]
    result_shape = np.broadcast_shapes(*[matrix.shape for matrix in matrices])

    if out is None:
        out = np.empty(result_shape, dtype=dtype)

    out[:] = matrices[0]
    for matrix in matrices[1:]:
        np.matmul(out, matrix, out=out)

    return out


def mat_from_translation(vector, /, *, out=None, dtype=None):
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
    result_shape = (*vector.shape[:-1], 4, 4)

    if out is None:
        out = np.empty(result_shape, dtype=dtype)

    # view into the diagonal of the result
    n_matrices = np.prod(result_shape[:-2], dtype=int)
    itemsize = out.itemsize
    diagonal = as_strided(
        out, shape=(n_matrices, 4), strides=(16 * itemsize, 5 * itemsize)
    )

    out[:] = 0
    diagonal[:] = 1
    out[..., :-1, -1] = vector

    return out


def mat_from_scale(factors, /, *, out=None, dtype=None):
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
    factors = np.asarray(factors, dtype=dtype)

    matrix = np.identity(4, dtype=dtype)
    matrix[np.diag_indices(3)] = factors

    if out is not None:
        out[:] = matrix
        return out

    return matrix


def mat_from_euler(angles, /, *, order="xyz", out=None, dtype=None):
    """
    Make a matrix given euler angles (in radians) per axis.

    Parameters
    ----------
    angles : ndarray, [3]
        The euler angles.
    order : string, optional
        The rotation order as a string. Can include 'X', 'Y', 'Z' for intrinsic
        rotation (uppercase) or 'x', 'y', 'z' for extrinsic rotation (lowercase).
        Default is "xyz".
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
    If you are familiar with TreeJS note that this function uses ``order`` to
    denote both the order in which rotations are applied *and* the order in
    which angles are provided in ``angles``. I.e.,
    ``mat_from_euler([np.pi, np.pi, 0], order="zyx")``
    will first rotate 180° ccw (counter-clockwise) around the z-axis, then 180°
    ccw around the y-axis, and finally 0° around the x axis.

    """
    fill_out_first = out is not None
    angles = np.atleast_1d(np.asarray(angles))
    extrinsic = order.islower()
    order = order.upper()
    axis_lookup = {"X": 0, "Y": 1, "Z": 2}
    if extrinsic:
        angles = reversed(angles)
        order = reversed(order)
    for angle, axis in zip(angles, order):
        axis_idx = axis_lookup[axis]
        affine_matrix = np.identity(4, dtype=dtype)

        if axis_idx == 0:
            affine_matrix[1, 1] = np.cos(angle)
            affine_matrix[1, 2] = -np.sin(angle)
            affine_matrix[2, 1] = np.sin(angle)
            affine_matrix[2, 2] = np.cos(angle)
        elif axis_idx == 1:
            affine_matrix[0, 0] = np.cos(angle)
            affine_matrix[0, 2] = np.sin(angle)
            affine_matrix[2, 0] = -np.sin(angle)
            affine_matrix[2, 2] = np.cos(angle)
        elif axis_idx == 2:
            affine_matrix[0, 0] = np.cos(angle)
            affine_matrix[0, 1] = -np.sin(angle)
            affine_matrix[1, 0] = np.sin(angle)
            affine_matrix[1, 1] = np.cos(angle)

        if fill_out_first:
            out[:] = affine_matrix
            fill_out_first = False
        elif out is None:
            out = affine_matrix
        else:
            out @= affine_matrix
    return out


def mat_from_axis_angle(axis, angle, /, *, out=None, dtype=None):
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


def quat_from_mat(matrix, /, *, out=None, dtype=None):
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


def mat_compose(translation, rotation, scaling, /, *, out=None, dtype=None):
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
    if out is None:
        out = np.empty((4, 4), dtype=dtype)

    x, y, z, w = rotation
    x2 = x + x
    y2 = y + y
    z2 = z + z
    xx = x * x2
    xy = x * y2
    xz = x * z2
    yy = y * y2
    yz = y * z2
    zz = z * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2

    scaling = np.asarray(scaling)
    if scaling.size == 1:
        scaling = np.broadcast_to(scaling, (3,))
    sx, sy, sz = scaling

    out[0, 0] = (1 - (yy + zz)) * sx
    out[1, 0] = (xy + wz) * sx
    out[2, 0] = (xz - wy) * sx
    out[3, 0:3] = 0

    out[0, 1] = (xy - wz) * sy
    out[1, 1] = (1 - (xx + zz)) * sy
    out[2, 1] = (yz + wx) * sy

    out[0, 2] = (xz + wy) * sz
    out[1, 2] = (yz - wx) * sz
    out[2, 2] = (1 - (xx + yy)) * sz

    out[0:3, 3] = translation
    out[3, 3] = 1

    return out


def mat_decompose(matrix, /, *, scaling_signs=None, dtype=None, out=None):
    """
    Decompose a transformation matrix into a translation vector, a
    quaternion and a scaling vector.

    Parameters
    ----------
    matrix : ndarray, [4, 4]
        transformation matrix
    scaling_signs : ndarray, [3], optional
        scaling factor signs. If you wish to preserve the original scaling
        factors through a compose-decompose roundtrip, you should
        provide the original scaling factors here, or alternatively just
        the signs.
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
        scaling factors
    """
    matrix = np.asarray(matrix)

    if out is not None:
        translation = out[0]
    else:
        translation = np.empty((3,), dtype=dtype)
    translation[:] = matrix[:-1, -1]

    flip = 1 if np.linalg.det(matrix) >= 0 else -1
    if scaling_signs is not None:
        # if the user provides the scaling signs, always use them
        scaling_signs = np.sign(scaling_signs)
        if np.prod(scaling_signs) != flip:
            raise ValueError(
                "Number of negative signs is inconsistent with the determinant"
            )
    else:
        # if not, detect if a flip is needed to reconstruct the transform
        # and apply it to the first axis arbitrarily
        scaling_signs = np.array([flip, 1, 1])

    if out is not None:
        scaling = out[2]
    else:
        scaling = np.empty((3,), dtype=dtype)
    scaling[:] = np.linalg.norm(matrix[:-1, :-1], axis=0)
    scaling *= scaling_signs

    rotation = out[1] if out is not None else None

    rotation_matrix = matrix[:-1, :-1].copy().astype(float)
    mask = scaling != 0
    rotation_matrix[:, mask] /= scaling[mask][None, :]
    rotation_matrix[:, ~mask] = 0.0
    rotation = quat_from_mat(rotation_matrix, out=rotation, dtype=dtype)

    return translation, rotation, scaling


def mat_perspective(
    left, right, top, bottom, near, far, /, *, depth_range=(-1, 1), out=None, dtype=None
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
    depth_range : Tuple[float, float]
        The interval along the z-axis in NDC that shall correspond to the region
        inside the viewing frustum.
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

    near_d = near * depth_range[0]
    far_d = far * depth_range[1]
    depth_diff = depth_range[1] - depth_range[0]

    a = (right + left) / (right - left)
    b = (top + bottom) / (top - bottom)
    c = -(far_d - near_d) / (far - near)
    d = -(far * near * depth_diff) / (far - near)

    out[0, 0] = x
    out[0, 2] = a
    out[1, 1] = y
    out[1, 2] = b
    out[2, 2] = c
    out[2, 3] = d
    out[3, 2] = -1

    return out


def mat_orthographic(
    left, right, top, bottom, near, far, /, *, depth_range=(-1, 1), out=None, dtype=None
):
    """Create an orthographic projection matrix.

    The result projects points from local space into NDC (normalized device
    coordinates). Elements inside the viewing frustum defind by left, right,
    top, bottom, near, far, are projected into the unit cube centered at the
    origin (default) or a cuboid (custom `depth_range`). The frustum is centered
    around the local frame's origin.

    Parameters
    ----------
    left : ndarray, [1]
        Distance between the left frustum plane and the origin
    right : ndarray, [1]
        Distance between the right frustum plane and the origin
    top : ndarray, [1]
        Distance between the top frustum plane and the origin
    bottom : ndarray, [1]
        Distance between the bottom frustum plane and the origin
    near : ndarray, [1]
        Distance between the near frustum plane and the origin
    far : ndarray, [1]
        Distance between the far frustum plane and the origin
    depth_range : ndarray, [2]
        The interval along the z-axis in NDC that shall correspond to the region
        inside the viewing frustum.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a
        shape that the inputs broadcast to. If not provided or None, a
        freshly-allocated array is returned. A tuple must have length equal to
        the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    matrix : ndarray, [4, 4]
        orthographic projection matrix

    Notes
    -----
    The parameters to this function are given in a left-handed frame that is
    obtained by mirroring source's Z-axis at the origin. In other words, if the
    returned matrix represents a camera's projection matrix then this function's
    parameters are given in a frame that is like the camera's local frame except
    that it's Z-axis is inverted. This means that positive values for `near` and
    `far` refer to a negative Z values in camera local.

    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    top = np.asarray(top, dtype=float)
    bottom = np.asarray(bottom, dtype=float)
    far = np.asarray(far, dtype=float)
    near = np.asarray(near, dtype=float)
    depth_range = np.asarray(depth_range, dtype=float)

    if out is None:
        batch_shape = np.broadcast_shapes(
            left.shape[:-1],
            right.shape[:-1],
            top.shape[:-1],
            bottom.shape[:-1],
            far.shape[:-1],
            near.shape[:-1],
            depth_range.shape[:-1],
        )
        out = np.zeros((*batch_shape, 4, 4), dtype=dtype)
    else:
        out[:] = 0

    # desired cuboid dimensions
    out[..., 0, 0] = 2
    out[..., 1, 1] = 2
    out[..., 2, 2] = -np.diff(depth_range, axis=-1)
    out[..., 3, 3] = 1

    # translation to cuboid origin
    out[..., 0, 3] = -(right + left)
    out[..., 1, 3] = -(top + bottom)
    out[..., 2, 3] = far * depth_range[..., 0] - near * depth_range[..., 1]

    # frustum-based scaling
    out[..., 0, :] /= right - left
    out[..., 1, :] /= top - bottom
    out[..., 2, :] /= far - near

    return out


def mat_look_at(eye, target, up_reference, /, *, out=None, dtype=None):
    """
    Rotation that aligns two vectors.

    Given an entity at position `eye` looking at position `target`, this
    function computes a rotation matrix that makes the local frame "look at" the
    same direction, i.e., the matrix will rotate the local frame's z-axes
    (forward) to point in direction ``target - eye``.

    This rotation matrix is not unique (yet), as the above doesn't specify the
    desired rotation around the new z-axis. The rotation around this axis is
    controlled by ``up_reference``, which indicates the direction of the y-axis
    (up) of a reference frame of choice expressed in local coordinates. The
    rotation around the new z-axis will then align `up_reference`, the new
    y-axis, and the new z-axis in the same plane.

    In many cases, a natural choice for ``up_reference`` is the world frame's
    y-axis, i.e., ``up_reference`` would be the world's y-axis expressed in
    local coordinates. This can be thought of as "gravity pulling on the
    rotation" (opposite direction of world frame's up) and will create a result
    with a level attitude.


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
        A homogeneous matrix describing the rotation.

    Notes
    -----
    If the new z-axis (``target - eye``) aligns with the chosen ``up_reference``
    then we can't compute the angle of rotation around the new z-axis. In this
    case, we will default to a rotation of 0, which may result in surprising
    behavior for some use-cases. It is the user's responsibility to ensure that
    these two directions don't align.

    """

    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up_reference = np.asarray(up_reference, dtype=float)

    new_z = target - eye
    up_reference = up_reference / np.linalg.norm(up_reference, axis=-1)

    result_shape = np.broadcast_shapes(eye.shape, target.shape, up_reference.shape)
    if out is None:
        out = np.zeros((*result_shape[:-1], 4, 4), dtype=dtype)
    else:
        out[:] = 0

    # Note: The below is equivalent to np.fill_diagonal(out, 1, axes=(-2, -1)),
    # i.e., treat the last two axes as a matrix and fill its diagonal with 1.
    # Currently numpy doesn't support axes on fill_diagonal, so we do it
    # ourselves to support input batches and mimic the `np.linalg` API.
    n_matrices = np.prod(result_shape[:-1], dtype=int)
    itemsize = out.itemsize
    view = as_strided(out, shape=(n_matrices, 4), strides=(16 * itemsize, 5 * itemsize))
    view[:] = 1

    out[..., :-1, 2] = new_z / np.linalg.norm(new_z, axis=-1)
    out[..., :-1, 0] = np.cross(
        up_reference, out[..., :-1, 2], axisa=-1, axisb=-1, axisc=-1
    )
    out[..., :-1, 1] = np.cross(
        out[..., :-1, 2], out[..., :-1, 0], axisa=-1, axisb=-1, axisc=-1
    )
    out /= np.linalg.norm(out, axis=-2)[..., None, :]

    return out


__all__ = [
    name for name in globals() if name.startswith(("vec_", "mat_", "quat_", "aabb_"))
]
