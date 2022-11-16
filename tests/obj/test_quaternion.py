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
    assert isinstance(q2, pla.Quaternion)


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


def test_quaternion_norm():
    a = pla.Quaternion(0, 0, 1, 1)
    assert a.norm() == np.sqrt(2)

    b = a.normalize()
    npt.assert_almost_equal(b.norm(), 1)
    assert a is not b
    npt.assert_almost_equal(b, [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])

    a.inormalize()
    npt.assert_almost_equal(a.norm(), 1)
    npt.assert_almost_equal(a, [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])


def test_quaternion_from_unit_vectors():
    a = pla.Vector(1, 0, 0)
    b = pla.Vector(0, 1, 0)
    q = pla.Quaternion()
    q2 = q.ifrom_unit_vectors(a, b)
    assert q is q2
    npt.assert_almost_equal(q, [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])

    q3 = pla.Quaternion.from_unit_vectors(a, b)
    npt.assert_almost_equal(q3, [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2])


def test_quaternion_add():
    a = pla.Quaternion()
    b = pla.Quaternion(0, 0, 1, 1)
    npt.assert_almost_equal(a + b, [0, 0, 1, 2])
    npt.assert_almost_equal(a.add(b), [0, 0, 1, 2])

    c = pla.Quaternion()
    c += b
    npt.assert_almost_equal(c, [0, 0, 1, 2])
    c = pla.Quaternion()
    c.iadd(b)
    npt.assert_almost_equal(c, [0, 0, 1, 2])


def test_quaternion_subtract():
    a = pla.Quaternion()
    b = pla.Quaternion(0, 0, 1, 1)
    npt.assert_almost_equal(a - b, [0, 0, -1, 0])
    npt.assert_almost_equal(a.subtract(b), [0, 0, -1, 0])

    c = pla.Quaternion()
    c -= b
    npt.assert_almost_equal(c, [0, 0, -1, 0])
    c = pla.Quaternion()
    c.isubtract(b)
    npt.assert_almost_equal(c, [0, 0, -1, 0])


def test_quaternion_inverse():
    q = pla.Quaternion(0, 0, 1, 1)
    npt.assert_almost_equal(q.inverse(), [0, 0, -1, 1])

    q.iinverse()
    npt.assert_almost_equal(q, [0, 0, -1, 1])


def test_quaternion_from_axis_angle():
    q = pla.Quaternion.from_axis_angle([1, 0, 0], np.pi / 2)
    npt.assert_array_almost_equal(q, [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])

    q = pla.Quaternion()
    q.ifrom_axis_angle([1, 0, 0], np.pi / 2)
    npt.assert_array_almost_equal(q, [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
