# Style conventions

## Docstrings

Docstrings shall be written in NumpyDoc format.

At least the following sections shall be provided:

* the short summary (one sentence at the top)
* the parameters section (if the function has input arguments)
* the returns section (if the function returns anything other than `None`)

## Type annotations

Until Numpy version 1.22+ becomes generally adopted this library will not provide type annotations.

## Linting

Linting shall be performed with flake8, flake8-isort, flake8-black and pep8-naming.

The default configurations for these linting tools shall be upheld, which includes
pep8 style and naming conventions.

Automated formatting shall be performed with black and isort.

Black is left at defaults, flake8 and isort are configured to adhere to black.

## Testing

All functions need to be covered by unit tests.


# Functional API conventions

This API is for internal use and for power-users that want to
vectorize operations on large sets of things.

Performance is prioritized over extensive input validation.

The source for this API resides in the `pylinalg.func` subpackage and is organized
by type.

## Function naming

The functional API has rather verbose names, but it makes things
explicit.

TBD: Provide examples and identify patterns.

## Function signatures

We strive to align closely with numpy conventions in order to be least-surprising
for users accustomed to numpy.

* Data arguments feeding into computation are positional-only.
* Optional arguments affecting the result of computation are keyword-only.
* The `dtype` and `out` arguments are available whenever possible, and they
  work as they do in numpy:
  * The `out` argument can be provided to write the results to an existing array,
    instead of a new array.
  * The `dtype` argument can be provided to override the data type of the result.
  * If there are multiple outputs, `out` is expected to be a tuple
    with matching number of elements.
  * If `out` and `dtype` are both provided, the `dtype` argument is ignored.

Here is an example of a function that complies with the conventions posed in
this document:

```python
def vector_apply_matrix(vectors, matrix, /, *, w=1, out=None, dtype=None):
    """
    Transform vectors by a transformation matrix.

    Parameters
    ----------
    vectors : ndarray, [..., ndim]
        Array of vectors
    matrix : ndarray, [ndim + 1, ndim + 1]
        Transformation matrix
    w : number, optional, default 1
        The value for the homogeneous dimensionality.
        this affects the result of translation transforms. use 0 (vectors)
        if the translation component should not be applied, 1 (positions)
        otherwise.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    ndarray, [..., ndim]
        transformed vectors
    """
    vectors = vector_make_homogeneous(vectors, w=w)
    # usually when applying a transformation matrix to a vector
    # the vector is a column, so if you were to have an array of vectors
    # it would have shape (ndim, nvectors).
    # however, we instead have the convention (nvectors, ndim) where
    # vectors are rows.
    # therefore it is necessary to transpose the transformation matrix
    # additionally we slice off the last row of the matrix, since we are not interested
    # in the resulting w coordinate
    transform = matrix[:-1, :].T
    if out is not None:
        try:
            # if `out` is exactly compatible, that is the most performant
            return np.dot(vectors, transform, out=out)
        except ValueError:
            # otherwise we need a temporary array and cast
            out[:] = np.dot(vectors, transform)
            return out
    # otherwise just return whatever dot computes
    out = np.dot(vectors, transform)
    # cast if requested
    if dtype is not None:
        out = out.astype(dtype)
    return out
```

Since the conventions align with those of numpy, in some cases, it is
possible to simply alias a numpy function to avoid all overhead and
implement the conventions. Optionally `functools.partial`
can be used to limit the available arguments, or a small wrapper function
can be created.


# Object oriented API conventions

This API is for external use and for novice-users that want to
perform linear algebra operations on conceptually familiar primitives.

In this API each "thing" is represented as one object. 
This API should make any linalg work much easier and safer, partly
because semantics matters here: a point is not the same as a vector.

These objects are backed by an array-like structure and are thus easy to
convert to native Python/Numpy objects. The objects support native python
operators such as `__mul__` where applicable, and have methods
specific to the type of object.

Extensive input validation and ease of use is prioritized over performance.

The source for this API resides in the `pylinalg.obj` subpackage and is organized
by type.

## Imports cycles

Since the classes here will regularly need to instantiate other types,
circular import dependencies exist. To work around this, only the `LinalgBase`
type can be imported at module level, and other types will have to be imported
at runtime in methods.

## Function naming

* Names should be concise and short.
* For every method, there is an alternative in-place method, signified
  with the prefix `i`. This is not the prettiest option, but it is
  concise and short.

## Copying and mutability

* By default, methods return new objects, and do not mutate self. Such
  methods shall be referred to as "copying methods".
* In-place methods return `self` to enable function chaining.

## Function signatures

* Copying functions accept a `dtype` keyword argument, in-place methods do not.
