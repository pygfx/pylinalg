import numpy as np


def vector_normalize(vectors, /, *, out=None, dtype=None):
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
    vectors = np.asarray(vectors)
    if out is None:
        out = np.empty_like(vectors, dtype=dtype)
    return np.divide(vectors, np.linalg.norm(vectors, axis=-1)[:, None], out=out)


def vector_make_homogeneous(vectors, /, *, w=1, out=None, dtype=None):
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


def vector_apply_matrix(vectors, matrix, /, *, w=1, out=None, dtype=None):
    """
    Transform vectors by a transformation matrix.

    Parameters
    ----------
    vectors : ndarray, [..., 3]
        Array of vectors
    matrix : ndarray, [4, 4]
        Transformation matrix
    w : number, optional, default 1
        The value for the homogeneous dimensionality.
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
    ndarray, [..., 3]
        transformed vectors
    """
    vectors = vector_make_homogeneous(vectors, w=w)
    # usually when applying a transformation matrix to a vector
    # the vector is a column, so if you were to have an array of vectors
    # it would have shape (ndim, nvectors).
    # however, we instead have the convention (nvectors, ndim) where
    # vectors are rows.
    # therefore it is necessary to transpose the transformation matrix
    # additionally we slice off the last row of the matrix, since we are not interested
    # in the resulting w coordinate
    transform = matrix[:-1, :].T
    if out is not None:
        try:
            # if `out` is exactly compatible, that is the most performant
            return np.dot(vectors, transform, out=out)
        except ValueError:
            # otherwise we need a temporary array and cast
            out[:] = np.dot(vectors, transform)
            return out
    # otherwise just return whatever dot computes
    out = np.dot(vectors, transform)
    # cast if requested
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def project_camera(vector, projection_matrix) -> np.ndarray:
    """Project a 3D vector into 2D space.

    Parameters
    ----------
    vector : ArrayLike
        The vector to be projected.
    projection_matrix: ArrayLike
        The camera's intrinsic matrix.

    Returns
    -------
    projected_vector : ndarray
        The projection of the vector.

    """

    raise NotImplementedError()


def unproject_camera(vector, projection_matrix, /, *, depth=0) -> np.ndarray:
    """Un-project a vector from 2D space to 3D space.

    Parameters
    ----------
    vector : ArrayLike
        The vector to be un-projected.
    projection_matrix: ArrayLike
        The camera's intrinsic matrix.
    depth : float
        The distance of the unprojected vector from the camera.

    Returns
    -------
    projected_vector : ndarray
        The unprojected vector in 3D space
    """

    raise NotImplementedError()


def apply_quaternion(vector, quaternion) -> np.ndarray:
    """Rotate a vector using a quaternion.

    Parameters
    ----------
    vector : ArrayLike
        The vector to be rotated.
    quaternion : ArrayLike
        The quaternion to apply (in xyzw format).

    Returns
    -------
    rotated_vector : np.ndarray
        The rotated vector.

    """

    raise NotImplementedError()


def from_spherical(spherical) -> np.ndarray:
    """Convert Spherical --> Euclidian coordinates

    Parameters
    ----------
    spherical : ArrayLike
        A vector in spherical coordinates (r, phi, theta).

    Returns
    -------
    euclidean_coordinates : ArrayLike
        A vector in euclidian coordinates.

    """

    raise NotImplementedError()


def distance_to(vectorA, vectorB, /) -> np.ndarray:
    """The distance between two vectors

    Parameters
    ----------
    vectorA : ArrayLike
        The first vector.
    vectorB : ArrayLike
        The second vector.

    Returns
    -------
    distance : np.ndarray
        The distance between both vectors.

    """

    raise NotImplementedError()


def from_matrix_position(homogeneous_matrix) -> np.ndarray:
    """Return the position of the matrix (??)

    @Korijn: I can't work out what this function does on a high-level. It
    clearly extracts the bottom row of a homogeneous matrix, but why do we want
    that?

    """

    raise NotImplementedError()
