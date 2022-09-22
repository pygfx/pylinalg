import pylinalg as pla


def test_linalgbase_eq():
    p = pla.Point(2, 3, 4)
    v = pla.Vector(2, 2, 2)

    # array_like comparison
    # type is ignored
    assert p + v == [4, 5, 6]

    # linalgbase type comparisons
    # type matters

    # point + vector => point
    # good type, good value
    assert p + v == pla.Point(4, 5, 6)
    # good type, wrong value
    assert p + v != pla.Point(4, 5, 7)
    # wrong type, good value
    assert p + v != pla.Vector(4, 5, 6)
    # wrong type, wrong value
    assert p + v != pla.Vector(4, 5, 7)
