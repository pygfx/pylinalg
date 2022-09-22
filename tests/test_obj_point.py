import numpy.testing as npt

import pylinalg as pla


def test_point_add_vector():
    p = pla.Point(2, 3, 4)
    v = pla.Vector(2, 2, 2)
    assert p + v == [4, 5, 6]
