import numpy as np


def from_euclidian(vector) -> np.ndarray:
    """Convert Euclidean -> Spherical Coordinates

    Parameters
    ----------
    vector : ArrayLike
        A vector in euclidean coordinates.

    Returns
    -------
    spherical : np.ndarray
        A vector in spherical coordinates (r, phi, theta).

    """

    raise NotImplementedError()


def make_safe(vector) -> np.ndarray:
    """Normalize sperhical coordinates.

    Normalizes a vector of spherical coordinates to restrict phi to (eps, pi-eps) and
    theta to (0, 2pi)

    Parameters
    ----------
    vector : ArrayLike
        A vector in spherical coordinates.

    Returns
    -------
    normalized_vector : np.ndarray
        A vector in spherical coordinates with restricted angle values.

    """

    raise NotImplementedError()
