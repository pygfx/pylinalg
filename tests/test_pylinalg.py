import inspect

import pylinalg as la


def test_api():
    api = dir(la)

    def popattr(key):
        val = getattr(la, key)
        api.remove(key)
        return val

    __version__ = popattr("__version__")
    version_info = popattr("version_info")

    legal_prefixes = ("vec_", "mat_", "quat_", "aabb_")
    for key in api:
        if key.startswith("__") and key.endswith("__"):
            continue
        if inspect.ismodule(getattr(la, key)):
            continue
        assert key.startswith(legal_prefixes)
        assert callable(getattr(la, key))
