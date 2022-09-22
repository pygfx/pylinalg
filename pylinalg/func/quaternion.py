import numpy as np


def quaternion_to_matrix(quaternion, out=None):
    x, y, z, w = quaternion
    x2 = x * 2
    y2 = y * 2
    z2 = z * 2
    xx = x * x2
    xy = x * y2
    xz = x * z2
    yy = y * y2
    yz = y * z2
    zz = z * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2

    if out is None:
        out = np.identity(4, dtype=quaternion.dtype)

    out[0, 0] = 1 - (yy + zz)
    out[0, 1] = xy + wz
    out[0, 2] = xz - wy
    out[1, 0] = xy - wz
    out[1, 1] = 1 - (xx + zz)
    out[1, 2] = yz + wx
    out[2, 0] = xz + wy
    out[2, 1] = yz - wx
    out[2, 2] = 1 - (xx + yy)

    return out
