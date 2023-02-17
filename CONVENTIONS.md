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

# Coordinate frame conventions

As the purpose of pylinalg as a library primarily is to support linear algebra
operations in pygfx and applications based on pygfx, we adhere to a number of
standard coordinate frame conventions to align with expectations from pygfx.

Generally there are four transform steps when going from an object's local space,
all the way to screen space:

* Local space to world space: world matrix
* World space to view space: view matrix
* View space to NDC/clip space: projection matrix

In pygfx, every world object has its own model matrix, which
encapsulates scale, rotation and translation components. World objects
are additionally organized hierarchically using parent/child relationships.
The world matrix is then defined for every world object as follows:
* If the object has no parent, the world matrix is equal to the model matrix.
* Otherwise, the world matrix is equal to the multiplication of the model matrix
  with that of its parent.

The view and projection matrices are owned by camera objects. The view
matrix can be acquired by inverting the camera's world matrix. The projection matrix
is configured independently on the camera and depends on its
type (perspective, orthographic).

Pylinalg provides methods in its functional API to produce all of these matrices.
Since world matrices are of key interest to users and developers working with pygfx
there need to be some conventions in place regarding its axes.

* The positive Z axis is the forward direction.
* The positive Y axis is the up direction.
* The positive X axis is the right direction.

This means that if, for example, you add a car object to a scene, by default
it is assumed that the car is facing towards positive Z. Due to the above axis
conventions, you can then use `matrix_make_look_at` to easily create a new rotation
matrix that makes the car look in another direction.

For the view matrix:

* The positive Z axis is the viewing direction.
* The positive Y axis is the up direction.
* The positive X axis is the right direction.

For NDC/clip space:

* The positive Z axis is the viewing direction and ranges from [0, 1].
* The positive Y axis is the up direction and ranges from [-1, 1].
* The positive X axis is the right direction and ranges from [-1, 1].

# Memory layout conventions

Row-major can mean two things:

* Memory layout; are rows or columns contiguous in memory
* Are vectors columns or rows

Numpy is row-major in both cases, and so we adhere to the same
convention here in pylinalg.

# Functional API conventions

This API is for internal use and for power-users that want to
vectorize operations on large sets of things.

Performance is prioritized over extensive input validation.

The source for this API resides in the `pylinalg.func` subpackage and is organized
by type.

## Function naming

The functional API has rather verbose names, but it makes things
explicit.

Since all functions are exposed on the root `pylinalg` module object,
a simple naming scheme is put in place:

* Functions are organized by type. For example, functions that work on
  matrices, or create matrices, go into the `pylinalg/func/matrix.py` module
  and their function names are prefixed by `matrix_`.
* Creation routines, for example a function that creates a new rotation matrix
  based on an axis and an angle, are additionally prefixed with `make_`, e.g.
  `matrix_make_rotation_angle_axis` would be a candidate function name.
* Conversion routines are named simply, taking the example of a matrix to
  quaternion function: `matrix_to_quaternion`

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
    vectors : ndarray, [..., 3]
        Array of vectors
    matrix : ndarray, [4, 4]
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
    ndarray, [..., 3]
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
        out = out.astype(dtype, copy=False)
    return out
```

## Note on linear algebra operations already provided by numpy

Since the conventions align with those of numpy, in some cases, it just
does not make sense to add the function this library and incur all the overhead
of maintenance, documentation and testing. For example, a function to perform
vector addition would be exactly equal to the `np.add` function, and as such,
it is not necessary to add them to pylinalg.

# Object oriented API conventions

This API is for external use and for novice-users that want to
perform linear algebra operations on conceptually familiar primitives.

Extensive input validation and ease of use is prioritized over performance.

The source for this API resides in the `pylinalg.obj` subpackage.
