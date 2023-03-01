import numpy as np

import pylinalg as pla


def test_linalgbase_eq():
    m = pla.LinalgBase([2, 2, 2, 2])
    v = pla.LinalgBase([2, 2, 2])

    t = m + 5
    assert t is not m
    assert isinstance(t, pla.LinalgBase)

    d = m
    m += 5
    assert d is m
    assert isinstance(m, pla.LinalgBase)

    # array_like comparison
    # type is ignored
    # good value
    assert v == [2, 2, 2]
    assert v == np.array([2, 2, 2])
    # wrong value
    assert v != [4, 5, 7]
    assert v != np.array([4, 5, 7])
    # wrong value, different shape
    assert v != [4, 5, 7, 8, 9]
    assert v != np.array([4, 5, 7, 8, 9])
    assert v != [[4, 5, 7, 8, 9]]
    assert v != np.array([[4, 5, 7, 8, 9]])

    # linalgbase type comparisons
    # type matters

    # good type, good value
    assert v == pla.LinalgBase([2, 2, 2])
    # good type, wrong value
    assert v != pla.LinalgBase([4, 5, 7])
    # wrong type
    assert m != v
