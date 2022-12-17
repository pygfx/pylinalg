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

    raise NotImplementedError()


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
    raise NotImplementedError()
