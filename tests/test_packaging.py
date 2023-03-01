from importlib.metadata import version

import pylinalg as la


def test_version_matches_metadata():
    assert la.__version__ == version("pylinalg")
