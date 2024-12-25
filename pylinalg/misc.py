import numpy as np


def aabb_to_sphere(aabb, /, *, out=None, dtype=None):
    """A sphere that envelops an Axis-Aligned Bounding Box.

    Parameters
    ----------
    aabb : ndarray, [2, 3]
        The axis-aligned bounding box.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    sphere : ndarray, [4]
        A sphere (x, y, z, radius).

    """

    aabb = np.asarray(aabb, dtype=float)

    if out is None:
        out = np.empty((*aabb.shape[:-2], 4), dtype=dtype)

    out[..., :3] = np.sum(aabb, axis=-2) / 2
    out[..., 3] = np.linalg.norm(np.diff(aabb, axis=-2), axis=-1) / 2

    return out


def aabb_transform(aabb, matrix, /, *, out=None, dtype=None):
    """Apply an affine transformation to an axis-aligned bounding box.

    Parameters
    ----------
    aabb : ndarray, [2, 3]
        The axis-aligned bounding box.
    homogeneous_matrix : [4, 4]
        The homogeneous transformation to apply.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    aabb : ndarray, [2, 3]
        The transformed axis-aligned bounding box.

    Notes
    -----
    This function preserves the alignment axes of the bounding box. This means
    the returned bounding box has the same alignment axes as the input bounding
    box, but contains the transformed object. In other words, the box will grow
    or shrink depending on how the contained object is transformed, but its
    alignment axis stay the same.

    """

    aabb = np.asarray(aabb, dtype=float)
    matrix = np.asarray(matrix, dtype=float).transpose((-1, -2))

    if out is None:
        out = np.empty_like(aabb, dtype=dtype)

    corners = np.full(
        aabb.shape[:-2] + (8, 4),
        # Fill value of 1 is used for homogeneous coordinates.
        fill_value=1.0,
        dtype=float,
    )
    # x
    corners[..., 0::2, 0] = aabb[..., 0, 0]
    corners[..., 1::2, 0] = aabb[..., 1, 0]

    # y
    corners[..., 0::4, 1] = aabb[..., 0, 1]
    corners[..., 1::4, 1] = aabb[..., 0, 1]
    corners[..., 2::4, 1] = aabb[..., 1, 1]
    corners[..., 3::4, 1] = aabb[..., 1, 1]

    # z
    corners[..., 0:4, 2] = aabb[..., 0, 2]
    corners[..., 4:8, 2] = aabb[..., 1, 2]

    corners = corners @ matrix
    out[..., 0, :] = np.min(corners[..., :-1], axis=-2)
    out[..., 1, :] = np.max(corners[..., :-1], axis=-2)

    return out


def quat_to_axis_angle(quaternion, /, *, out=None, dtype=None):
    """Convert a quaternion to axis-angle representation.

    Parameters
    ----------
    quaternion : ndarray, [4]
        A quaternion describing the rotation.
    out : Tuple[ndarray, ...], optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    axis : ndarray, [3]
        The axis around which the quaternion rotates in euclidean coordinates.
    angle : ndarray, [1]
        The angle (in rad) by which the quaternion rotates.

    Notes
    -----
    To use `out` with a single quaternion you need to provide a ndarray of shape
    ``(1,)`` for angle.

    """

    quaternion = np.asarray(quaternion)

    if out is None:
        quaternion = quaternion.astype(dtype)
        out = (
            quaternion[..., :3] / np.sqrt(1 - quaternion[..., 3] ** 2),
            2 * np.arccos(quaternion[..., 3]),
        )
    else:
        out[0][:] = quaternion[..., :3] / np.sqrt(1 - quaternion[..., 3] ** 2)
        out[1][:] = 2 * np.arccos(quaternion[..., 3])

    return out


__all__ = [
    name for name in globals() if name.startswith(("vec_", "mat_", "quat_", "aabb_"))
]
