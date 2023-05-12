import numpy as np


def vertex_normals(vertices, indices, /, *, out=None, dtype=None):
    """Efficiently compute vertex normals for a triangulated surface.

    Parameters
    ----------
    vertices : ndarray, [n, 3]
        The vertices for which normals should be computed.
    indices : ndarray, [n, 3]
        The triangle index array.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    out : ndarray, [n, 3]
        The vertex normals.
    """
    # This code was taken from Vispy
    # ensure highest precision for our summation/vectorization "trick"
    rr = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(indices).reshape((-1, 3))

    # first, compute triangle normals
    r1 = rr[tris[:, 0], :]
    r2 = rr[tris[:, 1], :]
    r3 = rr[tris[:, 2], :]
    tri_nn = np.cross((r2 - r1), (r3 - r1))

    # Triangle normals and areas
    size = np.sqrt(np.sum(tri_nn * tri_nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    tri_nn /= size[:, np.newaxis]

    npts = len(rr)

    # the following code replaces this, but is faster (vectorized):
    #
    # for p, verts in enumerate(tris):
    #     nn[verts, :] += tri_nn[p, :]
    #
    nn = np.zeros((npts, 3))
    for verts in tris.T:  # note this only loops 3x (number of verts per tri)
        for idx in range(3):  # x, y, z
            nn[:, idx] += np.bincount(
                verts.astype(np.int32), tri_nn[:, idx], minlength=npts
            )
    size = np.sqrt(np.sum(nn * nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    nn /= size[:, np.newaxis]

    if out is not None:
        out[:] = nn
        return out

    if dtype is None:
        dtype = np.float32
    return nn.astype(dtype)


__all__ = [name for name in globals() if name.startswith(("vertex_"))]
