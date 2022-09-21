from importlib.metadata import version

import pylinalg


def test_version_matches_metadata():
    assert pylinalg.__version__ == version("pylinalg")
