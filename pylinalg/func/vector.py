import numpy as np


def vector_add_vector(v1, v2, out=None):
    if out is None:
        out = np.empty_like(v1 if v1.size > v2.size else v2)
    out[:] = v1 + v2
    return out


def vector_sub_vector(v1, v2, out=None):
    if out is None:
        out = np.empty_like(v1 if v1.size > v2.size else v2)
    out[:] = v1 - v2
    return out


def vector_mul_vector(v1, v2, out=None):
    if out is None:
        out = np.empty_like(v1 if v1.size > v2.size else v2)
    out[:] = v1 * v2
    return out


def vector_div_vector(v1, v2, out=None):
    if out is None:
        out = np.empty_like(v1 if v1.size > v2.size else v2)
    out[:] = v1 / v2
    return out


def vector_add_scalar(v, s, out=None):
    if out is None:
        out = np.empty_like(v)
    out[:] = v + s
    return out


def vector_sub_scalar(v, s, out=None):
    if out is None:
        out = np.empty_like(v)
    out[:] = v - s
    return out


def vector_mul_scalar(v, s, out=None):
    if out is None:
        out = np.empty_like(v)
    out[:] = v * s
    return out


def vector_div_scalar(v, s, out=None):
    if out is None:
        out = np.empty_like(v)
    out[:] = v / s
    return out
