import pytest

import pylinalg as pla


def test_quaternion_init():
    q = pla.Quaternion()
    assert q == [0, 0, 0, 1]
    assert q.dtype == "f8"

    q = pla.Quaternion(1, 2, 3, 4, dtype="f4")
    assert q == [1, 2, 3, 4]
    assert q.dtype == "f4"

    with pytest.raises(TypeError):
        q = pla.Quaternion(x=1, y=2, z=3, w=4)

    with pytest.raises(TypeError):
        q = pla.Quaternion(1, 2, 3, 4, "f4")


def test_quaternion_set():
    q = pla.Quaternion()
    assert q.w == 1

    val = q._val
    q[:] = 0, 1, 0, 4
    assert q.w == 4
    assert q[3] == 4
    assert q._val[3] == 4
    assert q._val is val

    q.w = 2
    assert q.w == 2
    assert q[3] == 2
    assert q._val[3] == 2
    assert q._val is val
