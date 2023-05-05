# Style conventions

## Docstrings

Docstrings shall be written in NumpyDoc format.

At least the following sections shall be provided:

* the short summary (one sentence at the top)
* the parameters section (if the function has input arguments)
* the returns section (if the function returns anything other than `None`)

## Type annotations

Until Numpy version 1.22+ becomes generally adopted this library will not
provide type annotations.

## Linting

Linting shall be performed with flake8, flake8-isort, flake8-black and
pep8-naming.

The default configurations for these linting tools shall be upheld, which
includes pep8 style and naming conventions.

Automated formatting shall be performed with black and isort.

Black is left at defaults, flake8 and isort are configured to adhere to black.

## Testing

All functions need to be covered by unit tests.

# Coordinate frame conventions

The primary purpose of pylinalg as a library is to support linear algebra
operations in pygfx and applications based on pygfx. As such, pylinalg shares
its coordinate frame conventions with pygfx, which in turn follows [gITF's
conventions
](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#coordinate-system-and-units).

## Coordinate Systems

Unless stated otherwise, frames use 3-dimensional euclidean coordinates and a
right-handed coordinate frame. What differs depending on the type are the
semantics used to describe each axis. Distances are measured in `m` (meters) and
and angles in `rad` (radians) unless stated otherwise.

* **Object Coordinates** are represented using `(x, y, z)` vectors and are used
  to describe objects in a scene. They use the following semantics:
  * The *negative* X axis is called *right*.
  * The positive Y axis is called *up*.
  * The positive Z axis is called *forward*.
* **Camera Coordinates** are represented using `(x, y, z)` vectors and are used
  to describe cameras and lights. They use the following semantics:
  * The positive X axis is called *right*.
  * The positive Y axis is called *up*.
  * The *negative* Z axis is called *forward*.
* **Normalized Device Coordinates (NDC)** are left-handed and represented using
  `(x, y, z)` vectors. They are used to describe points to render/plot. Points
  inside the unit (half) box are rendered, others are not. NDC uses the
  following semantics:
  * The positive X axis is called *right* and the box extent is `[-1, 1]`.
  * The positive Y axis is called *up* and the box extent is `[-1, 1]`.
  * The positive Z axis is the viewing direction and the box extent is `[0, 1]`.
* **Spherical Coordinates** are represented using `(radius, theta, phi)` vectors
  and use the following semantics:
  * The *radius* measures the distance between a point and the origin and lies
    between `[0, inf)`.
  * The *phi* angle measures the counter-clockwise (CCW) rotation around the
    positive y-axis. It is measured from the positive Z-axis and lies between
    `[0, pi)`.
  * The *theta* angle measures the counter-clockwise (CCW) rotation around the
    negative X-axis. It is measured from the positive Y-axis and lies between
    `[0, 2*pi)`.
* **Homogeneous Coordinates** are represented using `(x, y, z, 1)` vectors and
  use the same semantics as their cartesian dual, i.e., if they represent an
  object, they use the semantics of object coordinates and when they represent a
  camera they use the semantics of camera coordinates.
* **Quaternion Coordinates** are represented using `(x, y, z, w)` vectors and
  have no explicit semantics.

## Named Coordinate Frames

We identify three important coordinate frames that we use as named references:

1. **world**: The world frame is the (global) inertial reference frame. All
    other coordinate frames are positioned relative to *world*; either
    explicitly, or implicitly via a sequence of frames that were previously
    placed relative to *world*. This creates a graph of coordinate frames
    (called the scene graph), with *world* at its root and it allows finding a
    transformation matrix between pairs of coordinate frames. *World* uses
    object coordinates.
2. **local**: The local frame is the coordinate frame in which an object's
    vertices are expressed. For example, when inspecting the position of a
    cube's corners their numerical values are given in the cube's *local frame*.
3. **parent**: The parent frame is the coordinate frame in which an object's
   pose (position + orientation) is expressed. *Parent* can either be the
   inertial reference frame (world) or it can be another object's local frame.
   An object's position is always relative to its parent; for example, if a
   lightbulb has a lamp's local frame as it's parent, then the lightbulb's
   position expressed in world coordinates will change whenever the position of
   lamp changes, i.e., if the lamp moves, so does the lightbulb.

Further, when talking about transformation matrices and coordinate transforms,
we use two additional frames to avoid confusion:

* **source**: The source frame is the coordinate frame in which
  to-be-transformed vectors are expressed, i.e., it is the reference frame of
  the input vectors. *Source* uses homogeneous coordinates.
* **target**: The target frame is the coordinate frame in which transformed
  vectors are expressed, i.e., it is the reference frame of the output vectors.
  *Target* uses homogeneous coordinates.

## Example using the named frames

To make this concrete, imagine a scene with a space shuttle that is about to
lift off. The *world* frame is a frame that is anchored to the surface of the
planet, the *local* frame is a frame attached to the space shuttle, and the
space shuttle's *parent* frame is the *local* frame of the rocket to which the
shuttle is attached to.

In the above example, gravity points along the negative y-axis in *world* (Y is
up) and along the positive z-axis in *local* (-Z is forward). During take-off,
the rocket will generate thrust and move in the direction of *parent*'s negative
Z (forward). From the perspective of *world*, however, the rocket launches in
the direction of the positive y-axis (up, as it should). The space shuttle,
being attached to the rocket, will be dragged along for the ride. It's position
relative to the rocket doesn't change; however, from the perspective of *world*
it, too, will accelerate along the positive y-axis.

## Rendering

To render an object we have to express its vertices in a camera's NDC. Recall
that vertices are expressed in the object's *local* frame, which means we need
to work out the transformation from object *local* to camera NDC. We can do this
by following the chain of transformations in the scene graph from object *local*
via *world* to camera *local* and from camera *local* into NDC. Since this chain
is very important for rendering, it's parts have special names:

* **World Matrix**: The transformation from object's *local* into *world*.
* **View Matrix**: The transformation from *world* to camera's *local*. This is
  the inverse of the camera's world matrix.
* **Projection Matrix**: The transformation from camera's *local* into NDC.

# Memory layout conventions

Row-major can mean two things:

* Memory layout; are rows or columns contiguous in memory
* Are vectors columns or rows

Pylinalg's kernels are written assuming a row-major (C-contigous) layout. If a
kernel supports batch processing of vetors, it assumes that the last dimension
contains the relevant vector data and that all other dimensions are batch/loop
dimensions. As such, you can think of vectors being row vectors.

# API conventions

This API is for power-users that want to
vectorize operations on large sets of things.

Performance is prioritized over extensive input validation.

The source for this API resides in the `pylinalg` package and is
organized by type.

## Function naming

Since all functions are exposed on the root `pylinalg` module object,
a simple naming scheme is put in place:

* Functions are organized by type. For example, functions that work on
  matrices, or create matrices, go into the `pylinalg/matrix.py` module
  and their function names are prefixed by `mat_`.

## Function signatures

We strive to align closely with numpy conventions in order to be
least-surprising for users accustomed to numpy.

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
def vec_transform(vectors, matrix, /, *, w=1, out=None, dtype=None):
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
    vectors = vec_homogeneous(vectors, w=w)
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

Since the conventions align with those of numpy, in some cases, it just does not
make sense to add the function this library and incur all the overhead of
maintenance, documentation and testing. For example, a function to perform
vector addition would be exactly equal to the `np.add` function, and as such, it
is not necessary to add them to pylinalg.
