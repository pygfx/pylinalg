from pylinalg import Vector4


x, y, z, w = 1, 2, 3, 4


# INSTANCING
def test_instancing():
    a = Vector4()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0
    assert a.w == 0

    a = Vector4(x, y, z, w)
    assert a.x == x
    assert a.y == y
    assert a.z == z
    assert a.w == w

    assert repr(a)


# PUBLIC STUFF
def test_set():
    a = Vector4()
    assert a.x == 0
    assert a.y == 0
    assert a.z == 0
    assert a.w == 0

    a.set(x, y, z, w)
    assert a.x == x
    assert a.y == y
    assert a.z == z
    assert a.w == w
