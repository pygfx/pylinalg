import numpy as np


vector_add_vector = np.add
vector_sub_vector = np.subtract
vector_mul_vector = np.multiply
vector_div_vector = np.divide
vector_add_scalar = np.add
vector_sub_scalar = np.subtract
vector_mul_scalar = np.multiply
vector_div_scalar = np.divide


def vector_make_homogeneous(vectors, value=0):
    """
    Append homogeneous coordinates to v.

    Parameters
    ----------
    vectors : ndarray, [..., ndim]
        array of vectors
    value : number, optional, default is 0
        the value for the additional dimensionality. use 0 for vectors, 1 for vectors.

    Returns
    -------
    ndarray, [..., ndim + 1]
        The list of vectors with appended homogeneous value.
    """
    vectors = np.asarray(vectors)
    shape = list(vectors.shape)
    shape[-1] += 1
    out = np.empty_like(vectors, shape=shape)
    out[..., -1] = value
    out[..., :-1] = vectors
    return out


def vector_apply_matrix(vectors, matrix):
    """
    Transform vectors by a transformation matrix.

    Parameters
    ----------
    vectors : ndarray, [..., ndim]
        array of vectors
    transform : ndarray, [ndim + 1, ndim + 1]
        transformation matrix

    Returns
    -------
    ndarray, [..., ndim]
        transformed vectors
    """
    vectors = vector_make_homogeneous(vectors, value=0)
    return np.dot(vectors, matrix)[..., :-1]
