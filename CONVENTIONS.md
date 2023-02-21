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

To start with, we identify three important coordinate frames that we use as
references:

1. **world**: The world frame is the (global) inertial reference frame. All
    other coordinate frames are positioned relative to *world*; either
    explicitly or implicitly by being placed relative to a sequence of frames
    that were previously placed relative to *world*. This creates a graph of
    coordinate frames (called the scene graph), with *world* at its root and it
    allows finding a transformation matrix between pairs of coordinate frames.
2. **local**: The local frame is the coordinate frame in which an object's
    vertices are expressed. For example, when inspecting the position of a
    cube's corners, then the numerical values of the corners are given in the
    cube's *local frame*.
3. **parent**: The parent frame is the coordinate frame in which an object's
   pose (position + orientation) are expressed. *Parent* can either be the
   inertial reference frame (world) or it can be another object's local frame.
   An object's position is always relative to its parent; for example, if a
   lightbulb has a lamp's local frame as it's parent, then the lightbulb's
   position expressed in world coordinates will change whenever the position of
   lamp changes, i.e., if the lamp moves, so does the lightbulb.

Further, when talking about transformation matrices and coordinate transforms,
we use two additional frames to avoid confusion:

* **source**: The source frame is the coordinate frame in which
  to-be-transformed vectors are expressed, i.e., it is the reference frame of the
  input vectors.
* **target**: The target frame is the coordinate frame in which transformed
  vectors are expressed, i.e., it is the reference frame of the output vectors.

In addition to the above-mentioned reference frames, pylinalg uses a
standardized naming scheme for the axes of a reference frame. The convention
matches the convention chosen in pygfx:

* The positive X axis indicates the right direction.
* The positive Y axis indicates the up direction.
* The Z axis is interpreted differently depending on the type of object:
  * For cameras and lights, the _negative_ Z axis is the forward/viewing direction.
  * For all other objects, the positive Z axis is the forward/viewing direction.

This means that gravity is assumed to act along *world*'s negative y-axis.
Further, a space shuttle will launch forward in its *local* frame, meaning that
it will advance in the direction of the local frame's positive z-axis. At the
same time a launching space shuttle will move up in *world* coordinates, meaning
that it's *world* position will change along the positive y-axis.

To render an object we have to express its vertices in a camera's so-called NDC
coordinates, which stands for normalized device coordinates. Recall that
vertices are expressed in the object's *local* frame, which means we need to
work out the transformation from object *local* to NDC. We can do this by
following the chain of transformations in the scene graph from object *local*
via *world* to camera *local* and from camera *local* into NDC. Since this chain
is very important for rendering, it's parts have special names:

* **World Matrix**: The transformation from object's *local* into *world*.
* **View Matrix**: The transformation from *world* to camera's *local*. This is
  the inverse of the camera's world matrix.
* **Projection Matrix**: The transformation from camera's *local* into NDC.

The axes of NDC/clip space are defined as follows:

* The positive Z axis is the viewing direction and ranges from [0, 1].
* The positive Y axis is the up direction and ranges from [-1, 1].
* The positive X axis is the right direction and ranges from [-1, 1].

# Memory layout conventions

Row-major can mean two things:

* Memory layout; are rows or columns contiguous in memory
* Are vectors columns or rows

Pylinalg's kernels are written assuming a row-major (C-contigous) layout. If a
kernel supports batch processing of vetors, it assumes that the last dimension
contains the relevant vector data and that all other dimensions are batch
dimensions. As such, you can think of vectors being row vectors.

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
