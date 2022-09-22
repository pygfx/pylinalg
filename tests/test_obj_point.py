import pytest

import pylinalg as pla


def test_point_init():
    p = pla.Point()
    assert p == [0, 0, 0]
    assert p.dtype == "f8"
    assert p.y == 0

    p = pla.Point(1, 2, 3, dtype="f4")
    assert p == [1, 2, 3]
    assert p.dtype == "f4"
    assert p.y == 2

    with pytest.raises(TypeError):
        p = pla.Point(x=1, y=2, z=3)

    with pytest.raises(TypeError):
        p = pla.Point(1, 2, 3, "f4")


def test_point_set():
    p = pla.Point()
    assert p.y == 0

    val = p._val
    p.set(0, 1, 0)
    assert p.y == 1
    assert p._val[1] == 1
    assert p._val is val

    p.y = 2
    assert p.y == 2
    assert p._val[1] == 2
    assert p._val is val
