import numpy as np
import numpy.testing as npt
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


def test_quaternion_copy():
    data = np.arange(4, dtype="f4")
    q = pla.Quaternion(*data)
    q2 = q.copy()
    assert q == q2
    assert q._val is not q2._val
    assert q2._val.flags.owndata
    assert q.dtype == q2.dtype


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


def test_quaternion_multiply():
    qu = pla.Quaternion()
    q180z = pla.Quaternion(0, 0, 1, 0)
    q90z = pla.Quaternion(0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2)
    q90y = pla.Quaternion(0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2)

    assert qu * q180z == q180z
    assert qu * q180z is not q180z

    q2 = pla.Quaternion()
    q2_val = q2._val
    q2 *= q180z
    assert q2 == q180z
    assert q2_val is q2._val

    npt.assert_array_almost_equal(q90z * q90z, q180z)
    npt.assert_array_almost_equal(
        q90z * q90y * q90z, [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0]
    )
    npt.assert_array_almost_equal(
        q90z * q90z * q90y, [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0]
    )
