import pylinalg


def test_clamp():
    assert pylinalg.clamp(0.5, 0, 1) == 0.5, "Value already within limits"
    assert pylinalg.clamp(0, 0, 1) == 0, "Value equal to one limit"
    assert pylinalg.clamp(-0.1, 0, 1) == 0, "Value too low"
    assert pylinalg.clamp(1.1, 0, 1) == 1, "Value too high"
