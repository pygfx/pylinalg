from importlib.metadata import version

import pylinalg as pla


def test_version_matches_metadata():
    assert pla.__version__ == version("pylinalg")
