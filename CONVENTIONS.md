# General conventions

## Docstrings

Docstrings shall be written in NumpyDoc format.

## Type annotations

TBD: I'm not very familiar with the latest and greatest type annotation
options for libraries that make heavy use of numpy arrays like this one.

## Style

Linting shall be performed with flake8, flake8-isort, flake8-black and pep8-naming.

Max line length: 88.

Import order shall be google-style.

Automated formatting shall be performed with black and isort.

## Testing

All functions need to be covered by unit tests.


# Functional API conventions

This API is for internal use and for power-users that want to
vectorize operations on large sets of things.

Performance is prioritized over extensive input validation.

## Function naming

The functional API has rather verbose names, but it makes things
explicit.

## Function signatures

We strive to align closely with the conventions of [numpy](https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs)
in order to be least-surprising for pylinalg users.

All functions shall adhere as much possible to the following signature:

`func(x1, x2, ..., /, *, out=None, dtype=None, **kwargs)`

* All input arguments are positional-only.
* There can be as many input arguments as needed for the function signature to make sense.
* Input arguments are always [`array_like`](https://numpy.org/doc/stable/glossary.html?highlight=array_like#term-array_like):
    * "Any scalar or sequence that can be interpreted as an ndarray. In
      addition to ndarrays and scalars this category includes lists
      (possibly nested and with different element types) and tuples.
      Any argument accepted by `np.array` is `array_like`."
    * TBD: require a call to `np.asarray` or `np.asanyarray` on inputs?
* All non-input arguments are keyword-only.
* If `out` is `None` (the default), a new return array is created. Otherwise,
  the result of the function is written to `out`.
    * If there are multiple outputs, `out` is expected to be a tuple
      with matching number of elements.
* If `dtype` is `None` (the default), the dtype of the return array
  will be determined automatically by numpy based on the operations
  performed on the inputs. Otherwise, the dtype of the output array(s) is
  overridden with the
  provided dtype with a call to `ndarray.astype`. This should ensure
  a matching precision of the calculation. The exact calculation
  dtypes chosen may depend on the function and the inputs may be cast
  to this dtype to perform the calculation.
* There may be additional keyword-only arguments as required by
  the specifics of individual functions.

Since the conventions align with those of numpy, in some cases, it is
possible to simply alias a numpy function to avoid all overhead and
implement the conventions. Optionally `functools.partial`
can be used to limit the available arguments, or a small wrapper function
can be created.


# Object oriented API conventions

TBD

Extensive input validation and ease of use is prioritized over performance.
