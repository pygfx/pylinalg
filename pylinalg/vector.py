import numpy as np


def vec_normalize(vectors, /, *, out=None, dtype=None):
    """
    Normalize an array of vectors.

    Parameters
    ----------
    vectors : array_like, [..., 3]
        array of vectors
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [..., 3]
        array of normalized vectors.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if out is None:
        out = np.empty_like(vectors, dtype=dtype)

    lengths_shape = vectors.shape[:-1] + (1,)
    lengths = np.linalg.norm(vectors, axis=-1).reshape(lengths_shape)
    return np.divide(vectors, lengths, out=out)


def vec_homogeneous(vectors, /, *, w=1, out=None, dtype=None):
    """
    Append homogeneous coordinates to vectors.

    Parameters
    ----------
    vectors : array_like, [..., 3]
        array of vectors
    w : number, optional, default is 1
        the value for the homogeneous dimensionality.
        this affects the result of translation transforms. use 0 (vectors)
        if the translation component should not be applied, 1 (positions)
        otherwise.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [..., 4]
        The list of vectors with appended homogeneous value.
    """
    vectors = np.asarray(vectors)
    shape = list(vectors.shape)
    shape[-1] += 1
    if out is None:
        out = np.empty_like(vectors, shape=shape, dtype=dtype)
    out[..., -1] = w
    out[..., :-1] = vectors
    return out


def vec_transform(vectors, matrix, /, *, w=1, out=None, dtype=None):
    """
    Apply a transformation matrix to a vector.

    Parameters
    ----------
    vectors : ndarray, [3]
        Array of vectors
    matrix : ndarray, [4, 4]
        Transformation matrix
    w : ndarray, [1], optional
        The value of the scale component of the homogeneous coordinate. This
        affects the result of translation transforms. use 0 (vectors) if the
        translation component should not be applied, 1 (positions) otherwise.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a
        shape that the inputs broadcast to. If not provided or None, a
        freshly-allocated array is returned. A tuple must have length equal to
        the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [3]
        transformed vectors
    """

    vectors = np.asarray(vectors, dtype=float)
    matrix = np.asarray(matrix, dtype=float)

    if out is None:
        out_shape = np.broadcast_shapes(vectors.shape[:-1], matrix.shape[:-2])
        out = np.empty((*out_shape, 3), dtype=dtype)

    vectors = vec_homogeneous(vectors, w=w)
    result = matrix @ vectors[..., None]
    result /= result[..., -1, :][..., None, :]
    out[:] = result[..., :-1, 0]

    return out


def vec_unproject(vector, matrix, /, *, depth=0, out=None, dtype=None):
    """
    Un-project a vector from 2D space to 3D space.

    Find a ``vectorB`` in 3D euclidean space such that the projection
    ``matrix @ vectorB`` yields the provided vector (in 2D euclidean
    space). Since the solution to the above is a 1D subspace of 3D space (a
    line), ``depth`` is used to select a single vector within.

    Parameters
    ----------
    vector : ndarray, [2]
        The vector to be un-projected.
    matrix: ndarray, [4, 4]
        The camera's intrinsic matrix.
    depth : number, optional
        The distance of the unprojected vector from the camera.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    projected_vector : ndarray, [3]
        The unprojected vector in 3D space

    Notes
    -----
    The source frame of this operation is the XY-plane of the camera's NDC frame
    and the target frame is the camera's local frame.
    """

    vector = np.asarray(vector, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    depth = np.asarray(depth, dtype=float)

    result_shape = np.broadcast_shapes(
        vector.shape[:-1], matrix.shape[:-2], depth.shape
    )

    if out is None:
        out = np.empty((*result_shape, 3), dtype=dtype)

    try:
        inverse_projection = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("The provided matrix is not invertible.")

    vector_hom = np.empty((*result_shape, 4), dtype=dtype)
    vector_hom[..., 2] = depth
    vector_hom[..., [0, 1]] = vector
    vector_hom[..., 3] = 1

    out_hom = vector_hom @ inverse_projection.T
    scale = out_hom[..., -1][..., None]
    out[:] = (out_hom / scale)[..., :-1]

    return out


def vec_transform_quat(vector, quaternion, /, *, out=None, dtype=None):
    """Rotate a vector using a quaternion.

    Parameters
    ----------
    vector : ndarray, [3]
        The vector to be rotated.
    quaternion : ndarray, [4]
        The quaternion to apply (in xyzw format).
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
        The rotated vector.

    """

    vector = np.asarray(vector, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float)

    if out is None:
        out = np.zeros_like(vector, dtype=dtype)

    # based on https://gamedev.stackexchange.com/a/50545
    # (more readable than my attempt at doing the same)

    quat_vector = quaternion[..., :-1]
    quat_scalar = quaternion[..., -1]

    out += 2 * np.sum(quat_vector * vector, axis=-1, keepdims=True) * quat_vector
    out += (
        quat_scalar**2 - np.sum(quat_vector * quat_vector, axis=-1, keepdims=True)
    ) * vector
    out += 2 * quat_scalar * np.cross(quat_vector, vector)

    return out


def vec_spherical_to_euclidian(spherical, /, *, out=None, dtype=None):
    """Convert spherical -> euclidian coordinates.

    Parameters
    ----------
    spherical : ndarray, [3]
        A vector in spherical coordinates (r, phi, theta). Phi and theta are
        measured in radians.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    euclidean : ndarray, [3]
        A vector in euclidian coordinates.

    Notes
    -----
    This implementation follows pygfx's coordinate conventions. This means that
    the positive y-axis is the zenith reference and the positive z-axis is the
    azimuth reference. Angles are measured counter-clockwise.

    """

    spherical = np.asarray(spherical, dtype=float)

    if out is None:
        out = np.empty_like(spherical, dtype=dtype)

    r, theta, phi = np.split(spherical, 3, axis=-1)
    out[..., 0] = r * np.sin(phi) * np.sin(theta)
    out[..., 1] = r * np.cos(phi)
    out[..., 2] = r * np.sin(phi) * np.cos(theta)

    return out


def vec_dist(vector_a, vector_b, /, *, out=None, dtype=None):
    """The distance between two vectors

    Parameters
    ----------
    vector_a : ndarray, [3]
        The first vector.
    vector_b : ndarray, [3]
        The second vector.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    distance : ndarray
        The distance between both vectors.

    """

    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    shape = vector_a.shape[:-1]
    if out is None:
        out = np.linalg.norm(vector_a - vector_b, axis=-1).astype(dtype)
    elif len(shape) >= 0:
        out[:] = np.linalg.norm(vector_a - vector_b, axis=-1)
    else:
        raise ValueError("Can't use `out` with scalar output.")

    return out


def vec_angle(vector_a, vector_b, /, *, out=None, dtype=None):
    """The angle between two vectors

    Parameters
    ----------
    vector_a : ndarray, [3]
        The first vector.
    vector_b : ndarray, [3]
        The second vector.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    angle : float
        The angle between both vectors.

    """

    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    shape = vector_a.shape[:-1]

    # Cannot broadcast np.dot(), so just write it out
    dot_prod = sum(
        [
            vector_a[..., 0] * vector_b[..., 0],
            vector_a[..., 1] * vector_b[..., 1],
            vector_a[..., 2] * vector_b[..., 2],
        ]
    )
    the_cos = (
        dot_prod / np.linalg.norm(vector_a, axis=-1) / np.linalg.norm(vector_b, axis=-1)
    )

    if out is None:
        out = np.arccos(the_cos).astype(dtype)
    elif len(shape) >= 0:
        out[:] = np.arccos(the_cos)
    else:
        raise ValueError("Can't use `out` with scalar output.")

    return out


def mat_decompose_translation(homogeneous_matrix, /, *, out=None, dtype=None):
    """Position component of a homogeneous matrix.

    Parameters
    ----------
    homogeneous_matrix : ndarray, [4, 4]
        The matrix of which the position/translation component will be
        extracted.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    position : ndarray, [3]
        The position/translation component.

    """

    homogeneous_matrix = np.asarray(homogeneous_matrix, dtype=float)

    if out is None:
        out = np.empty((*homogeneous_matrix.shape[:-2], 3), dtype=dtype)

    out[:] = homogeneous_matrix[..., :-1, -1]

    return out


def vec_euclidian_to_spherical(euclidean, /, *, out=None, dtype=None):
    """Convert euclidean -> spherical coordinates

    Parameters
    ----------
    euclidean : ndarray, [3]
        A vector in euclidean coordinates.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    spherical : ndarray, [3]
        A vector in spherical coordinates (r, phi, theta).

    """

    euclidean = np.asarray(euclidean, dtype=float)

    if out is None:
        out = np.zeros_like(euclidean, dtype=dtype)
    else:
        out[:] = 0

    out[..., 0] = np.sqrt(np.sum(euclidean**2, axis=-1))

    # flags to handle all cases
    needs_flip = np.sign(euclidean[..., 0]) < 0
    len_xz = np.sum(euclidean[..., [0, 2]] ** 2, axis=-1)
    xz_nonzero = ~np.all(len_xz == 0, axis=-1)
    r_nonzero = ~np.all(out[..., [0]] == 0, axis=-1)

    # chooses phi = 0 if vector runs along y-axis
    out[..., 1] = np.divide(euclidean[..., 2], np.sqrt(len_xz), where=xz_nonzero)
    out[..., 1] = np.arccos(out[..., 1], where=xz_nonzero)
    out[..., 1] = np.where(needs_flip, np.abs(out[..., 1] - np.pi), out[..., 1])

    # chooses theta = 0 at the origin (0, 0, 0)
    out[..., 2] = np.divide(euclidean[..., 1], out[..., 0], where=r_nonzero)
    out[..., 2] = np.arccos(out[..., 2], where=r_nonzero)
    out[..., 2] = np.where(needs_flip, 2 * np.pi - out[..., 2], out[..., 2])

    return out


def vec_spherical_safe(vector, /, *, out=None, dtype=None):
    """Normalize sperhical coordinates.

    Normalizes a vector of spherical coordinates to restrict phi to [0, pi) and
    theta to [0, 2pi).

    Parameters
    ----------
    vector : ndarray, [3]
        A vector in spherical coordinates.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    normalized_vector : ndarray, [3]
        A vector in spherical coordinates with restricted angle values.

    """

    vector = np.asarray(vector, dtype=float)

    if out is None:
        out = np.zeros_like(vector, dtype=dtype)

    is_flipped = vector[..., 1] % (2 * np.pi) >= np.pi
    out[..., 2] = np.where(is_flipped, -vector[..., 2], vector[..., 2])

    out[..., 0] = vector[..., 0]
    out[..., 1] = vector[..., 1] % np.pi
    out[..., 2] = vector[..., 1] % (2 * np.pi)

    out[..., 1] = np.where(out[..., 1] == np.pi, 0, out[..., 1])
    out[..., 2] = np.where(out[..., 2] == 2 * np.pi, 0, out[..., 2])

    return out


def quat_to_euler(quaternion, /, *, out=None, dtype=None):
    """Convert quaternions to XYZ Euler angles.

    Parameters
    ----------
    quaternion : ndarray, [4]
        The quaternion to convert (in xyzw format).
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : ndarray, [3]
        The XYZ Euler angles.
    """
    quaternion = np.asarray(quaternion, dtype=float)

    if out is None:
        out = np.empty((*quaternion.shape[:-1], 3), dtype=dtype)

    ysqr = quaternion[..., 1] ** 2

    t0 = 2 * (
        quaternion[..., 3] * quaternion[..., 0]
        + quaternion[..., 1] * quaternion[..., 2]
    )
    t1 = 1 - 2 * (quaternion[..., 0] ** 2 + ysqr)
    out[..., 0] = np.arctan2(t0, t1)

    t2 = 2 * (
        quaternion[..., 3] * quaternion[..., 1]
        - quaternion[..., 2] * quaternion[..., 0]
    )
    t2 = np.clip(t2, a_min=-1, a_max=1)
    out[..., 1] = np.arcsin(t2)

    t3 = 2 * (
        quaternion[..., 3] * quaternion[..., 2]
        + quaternion[..., 0] * quaternion[..., 1]
    )
    t4 = 1 - 2 * (ysqr + quaternion[..., 2] ** 2)
    out[..., 2] = np.arctan2(t3, t4)

    return out


__all__ = [
    name for name in globals() if name.startswith(("vec_", "mat_", "quat_", "aabb_"))
]
