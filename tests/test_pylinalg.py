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

    # assert that we expose __all__ for tools like sphinx
    __all__ = popattr("__all__")

    # assert that all the remaining elements of the
    # public api are either builtins, submodules/packages,
    # or callables with legal prefixes
    legal_prefixes = ("vec_", "mat_", "quat_", "aabb_")
    callables = []
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
                ) from None
            # actual pylinalg submodules/packages are OK
            continue
        # otherwise it should be a callable
        # with a legal prefix
        assert key.startswith(legal_prefixes)
        assert callable(getattr(la, key))
        callables.append(key)

        # confirm the signature of each callable
        sig = inspect.signature(getattr(la, key))

        assert (
            sig.return_annotation is not inspect.Signature.empty
        ), f"Missing return annotation on {key}"
        if sig.return_annotation is bool:
            key_parts = key.split("_")
            assert key_parts[1] in ("is", "has")
        else:
            has_out, has_dtype = False, False
            for param in sig.parameters.values():
                # all arguments are either positional-only, or keyword-only
                assert param.kind in (param.POSITIONAL_ONLY, param.KEYWORD_ONLY)
                # every function has out & dtype keyword-only arguments
                if param.name == "dtype":
                    assert param.KEYWORD_ONLY
                    has_dtype = True
                elif param.name == "out":
                    assert param.KEYWORD_ONLY
                    has_out = True
            assert has_out, f"Function {key} does not have 'out' keyword-only argument"
            assert (
                has_dtype
            ), f"Function {key} does not have 'dtype' keyword-only argument"

    # assert that all callables are available in __all__
    assert set(__all__) == set(callables)
