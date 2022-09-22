import numpy as np


vector_add_vector = np.add
vector_sub_vector = np.subtract
vector_mul_vector = np.multiply
vector_div_vector = np.divide
vector_add_scalar = np.add
vector_sub_scalar = np.subtract
vector_mul_scalar = np.multiply
vector_div_scalar = np.divide


def vector_make_homogeneous(vectors, w=1):
    """
    Append homogeneous coordinates to vectors.

    Parameters
    ----------
    vectors : ndarray, [..., ndim]
        array of vectors
    w : number, optional, default is 1
        the value for the homogeneous dimensionality.
        this affects the result of translation transforms. use 0 (vectors)
        if the translation component should not be applied, 1 (positions)
        otherwise.

    Returns
    -------
    ndarray, [..., ndim + 1]
        The list of vectors with appended homogeneous value.
    """
    vectors = np.asarray(vectors)
    shape = list(vectors.shape)
    shape[-1] += 1
    out = np.empty_like(vectors, shape=shape)
    out[..., -1] = w
    out[..., :-1] = vectors
    return out


def vector_apply_matrix(vectors, matrix, w=1):
    """
    Transform vectors by a transformation matrix.

    Parameters
    ----------
    vectors : ndarray, [..., ndim]
        array of vectors
    transform : ndarray, [ndim + 1, ndim + 1]
        transformation matrix
    w : number, optional, default is 1
        the value for the homogeneous dimensionality.
        this affects the result of translation transforms. use 0 (vectors)
        if the translation component should not be applied, 1 (positions)
        otherwise.

    Returns
    -------
    ndarray, [..., ndim]
        transformed vectors
    """
    vectors = vector_make_homogeneous(vectors, w=w)
    # the transpose is necessary due to the shape of the vectors matrix
    return np.dot(vectors, matrix.T)[..., :-1]
