"""
The LinalgBase base class makes our objects more performant and compatible
with numpy.

Related docs:

* https://docs.python.org/3/reference/datamodel.html#slots
* https://numpy.org/doc/stable/user/basics.interoperability.html
"""
import numpy as np


class LinalgBase:
    __slots__ = ["val"]

    def __init__(self, val=None, /, *, dtype=None):
        if val is not None:
            self.val = np.asarray(val, dtype=dtype)
        else:
            self.val = self._initializer(dtype=dtype)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.val}>"

    def __eq__(self, other):
        if isinstance(other, LinalgBase) and not isinstance(other, self.__class__):
            return False
        return np.array_equal(self.val, other)


def create_proxy(name, is_callable, retval_wrap_cache={}):
    if is_callable:

        def proxy(self, *args, **kwargs):
            retval = getattr(self.val, name)(*args, **kwargs)
            # we try to intelligently determine if the return value should
            # be wrapped in a class instance
            # and cache the result of that evaluation
            if name not in retval_wrap_cache:
                if retval is None:
                    retval_wrap_cache[name] = False
                elif retval is self.val:
                    retval_wrap_cache[name] = "self"
                elif retval.shape == self.val.shape:
                    retval_wrap_cache[name] = True
                else:
                    retval_wrap_cache[name] = False
            wrap_strat = retval_wrap_cache[name]
            if not wrap_strat:
                return retval
            elif wrap_strat == "self":
                return self
            else:
                return self.__class__(retval)

    else:

        def proxy(self):
            return getattr(self.val, name)

        proxy = property(proxy)
    return proxy


def wrap_ndarray():
    # dummy ndarray so we can evaluate `callable`
    # at wrapping time
    val = np.empty(1)

    # loop over attributes
    for name in dir(val):
        # skip if either is true:
        # - name not in whitelist
        # - we already defined a custom wrapper
        if name not in NDARRAY_WHITELIST or hasattr(LinalgBase, name):
            continue

        # evaluate if the attribute is callable
        is_callable = callable(getattr(val, name))

        # create proxy via a function so `name`
        # is stored in a closure
        proxy = create_proxy(name, is_callable)

        # set the wrapping attribute
        setattr(LinalgBase, name, proxy)


# whitelist generated with numpy v1.23.4
# python -c "import numpy as np; import pprint; pprint.pprint(set(dir(np.empty(1))))"
NDARRAY_WHITELIST = {
    # 'T',
    "__abs__",
    "__add__",
    "__and__",
    # '__array__',
    # '__array_finalize__',
    # '__array_function__',
    "__array_interface__",
    # '__array_prepare__',
    # '__array_priority__',
    # '__array_struct__',
    # '__array_ufunc__',
    # '__array_wrap__',
    "__bool__",
    # '__class__',
    # '__class_getitem__',
    "__complex__",
    "__contains__",
    "__copy__",
    "__deepcopy__",
    # '__delattr__',
    "__delitem__",
    "__dir__",
    "__divmod__",
    # '__dlpack__',
    # '__dlpack_device__',
    # '__doc__',
    "__eq__",
    "__float__",
    "__floordiv__",
    # '__format__',
    "__ge__",
    # '__getattribute__',
    "__getitem__",
    "__gt__",
    "__hash__",
    "__iadd__",
    "__iand__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    # '__index__',
    # '__init__',
    # '__init_subclass__',
    # '__int__',
    # '__invert__',
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__iter__",
    "__itruediv__",
    "__ixor__",
    "__le__",
    "__len__",
    "__lshift__",
    "__lt__",
    "__matmul__",
    "__mod__",
    "__mul__",
    "__ne__",
    # '__neg__',
    # '__new__',
    "__or__",
    # '__pos__',
    "__pow__",
    "__radd__",
    # '__rand__',
    "__rdivmod__",
    # '__reduce__',
    # '__reduce_ex__',
    "__repr__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmatmul__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
    # '__setattr__',
    "__setitem__",
    # '__setstate__',
    # '__sizeof__',
    "__str__",
    "__sub__",
    # '__subclasshook__',
    "__truediv__",
    "__xor__",
    "all",
    "any",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "astype",
    "base",
    "byteswap",
    "choose",
    "clip",
    "compress",
    "conj",
    "conjugate",
    "copy",
    "ctypes",
    "cumprod",
    "cumsum",
    "data",
    "diagonal",
    "dot",
    "dtype",
    "dump",
    "dumps",
    "fill",
    "flags",
    "flat",
    "flatten",
    "getfield",
    "imag",
    "item",
    "itemset",
    "itemsize",
    "max",
    "mean",
    "min",
    "nbytes",
    "ndim",
    "newbyteorder",
    "nonzero",
    "partition",
    "prod",
    "ptp",
    "put",
    "ravel",
    "real",
    "repeat",
    "reshape",
    "resize",
    "round",
    "searchsorted",
    "setfield",
    "setflags",
    "shape",
    "size",
    "sort",
    "squeeze",
    "std",
    "strides",
    "sum",
    "swapaxes",
    "take",
    "tobytes",
    "tofile",
    "tolist",
    "tostring",
    "trace",
    "transpose",
    "var",
    "view",
}


wrap_ndarray()
