import importlib
import inspect

import packaging.version

import pylinalg as la


def test_api():
    api = dir(la)

    def popattr(key):
        val = getattr(la, key)
        api.remove(key)
        return val

    # assert that we expose a valid version
    __version__ = popattr("__version__")
    packaging.version.parse(__version__)

    # we don't want a runtime dependency on `packaging`
    # so parsing of version_info is a little dumb in
    # that it always expects integer components
    # so we can just test for that
    version_info = popattr("version_info")
    assert isinstance(version_info, tuple)
    assert all(isinstance(x, int) for x in version_info)

    # assert that all the remaining elements of the
    # public api are either builtins, submodules/packages,
    # or callables with legal prefixes
    legal_prefixes = ("vec_", "mat_", "quat_", "aabb_")
    for key in api:
        if key.startswith("__") and key.endswith("__"):
            # builtins are OK
            continue
        if inspect.ismodule(getattr(la, key)):
            # submodule/package
            try:
                importlib.import_module(f".{key}", "pylinalg")
            except ImportError:
                # should be `del`eted
                raise AssertionError(
                    f"API includes module '{key}' which is not a submodule/package"
                )
            # actual pylinalg submodules/packages are OK
            continue
        # otherwise it should be a callable
        # with a legal prefix
        assert key.startswith(legal_prefixes)
        assert callable(getattr(la, key))
