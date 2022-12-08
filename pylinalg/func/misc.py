import numpy as np


def aabb_to_sphere(aabb) -> np.ndarray:
    """A sphere that envelops an Axis-Aligned Bounding Box.

    Parameters
    ----------
    aabb : ArrayLike
        The axis-aligned bounding box.

    Returns
    -------
    sphere : np.ndarray
        A sphere (x, y, z, radius).

    """

    raise NotImplementedError()


def transform_aabb(aabb, homogenious_matrix) -> np.ndarray:
    """Apply an affine transformation to an axis-aligned bounding box.

    Parameters
    ----------
    aabb : ArrayLike
        The axis-aligned bounding box.
    homogeneous_matrix : ArrayLike
        The homogeneous transformation to apply.

    Returns
    -------
    aabb : ArrayLike
        The transformed axis-aligned bounding box.

    """
    raise NotImplementedError()
