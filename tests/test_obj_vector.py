import pytest

import pylinalg as pla


def test_vector_init():
    v = pla.Vector()
    assert v == [0, 0, 0]
    assert v.dtype == "f8"

    v = pla.Vector(1, 2, 3, dtype="f4")
    assert v == [1, 2, 3]
    assert v.dtype == "f4"

    with pytest.raises(TypeError):
        v = pla.Vector(x=1, y=2, z=3)

    with pytest.raises(TypeError):
        v = pla.Vector(1, 2, 3, "f4")


def test_vector_set():
    v = pla.Vector()
    assert v.y == 0

    val = v._val
    v.set(0, 1, 0)
    assert v.y == 1
    assert v._val[1] == 1
    assert v._val is val

    v.y = 2
    assert v.y == 2
    assert v._val[1] == 2
    assert v._val is val
