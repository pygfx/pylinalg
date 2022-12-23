import numpy as np

from ..func import quaternion as qt


class Transform:
    """
    Abstract base class to define a common interface and allow future extension
    to non-linear transformations.
    """


class AffineTransform(Transform):
    """
    A affine transformation.

    This transform represents an affine mapping between two coordinate frames:
    parent and child. That is, a transformation that is linear in homogeneous
    coordinates and that consists of a translation, rotation, and scale (applied
    in that order).


    Parameters
    ----------
    position : ndarray, [3]
        The position of the child's frame in the parent's space. (The
        translation to apply to map vectors from child to parent.)
    orientation : ndarray, [4]
        The orientation of the parent's frame relative to the child's frame. (A
        quaternion (format: x, y, z, w) describing the rotation from child to
        parent.)
    scale : ndarray, [3]
        The scale of the parent expressed in child's units. (A axis-wise scaling
        to map units from child to parent.)

    """

    # some constants for syntax sugar
    FORWARD = np.array((0, 0, 1))
    BACKWARD = np.array((0, 0, -1))
    UP = np.array((0, 1, 0))
    DOWN = np.array((0, -1, 0))
    RIGHT = np.array((1, 0, 0))
    LEFT = np.array((-1, 0, 0))

    def __init__(self, /, *, position=None, orientation=None, scale=None) -> None:
        self._position = position
        self._orientation = orientation
        self._scale = scale

        if position is None:
            self._position = np.zeros(3)

        if orientation is None:
            self._orientation = np.array((0, 0, 0, 1))

        if scale is None:
            self._scale = np.ones(3)

    def as_matrix(self):
        """
        The affine transforms matrix

        The returned matrix is a transformation matrix that converts
        points in homogeneous coordinates according to the current parameters
        of this AffineTransform object. Typically, this maps from an object's
        child frame into an object's parent frame.
        """

        result = np.zeros((4, 4))
        result[-1, -1] = 1  # affine matrix :)

        # credit to: http://www.songho.ca/opengl/gl_quaternion.htm
        # @almarklein: Can we refactor quaternion_to_matrix to something like
        # this? It's easier to read than the current implementation.
        x, y, z, w = self._orientation
        # fmt: off
        result[:3, :3] = np.array([
            [1 - 2*y**2 - 2*z**2,       2*x*y - 2*w*z,       2*x*z + 2*w*y],  # noqa: E201, E501
            [      2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2,       2*y*w - 2*w*x],  # noqa: E201, E501
            [      2*x*w - 2*w*y,       2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2],  # noqa: E201, E501
        ]).T
        # fmt: on

        result[:, :3] = result[:, :3] * self._scale.reshape(1, 3)

        # we right-multiply transformations to allow broadcasting across
        # vectors. Hence we track the position in the bottom row
        result[-1, :3] = self._position

        return result

    def inverse(self):
        """
        Invert this affine transformation.


        Note
        ----
        The problem with this implementation is that this transform and its
        inverse don't remain in sync, i.e., if we store the returned transform
        somewhere and this transform updates later, the stored transform does no
        longer invert this transform. -- We could work around this by
        introducing `Transform.inverse_transform` and a
        `InverseTransform(Transform)` object if desired.
        """

        scale = 1 / self.scale
        orientation = qt.quaternion_inverse(self.orientation)
        position = -(
            scale * (self.position @ qt.quaternion_to_matrix(orientation)[:3, :3])
        )

        return AffineTransform(position=position, orientation=orientation, scale=scale)

    def __array__(self, dtype=None):
        return self.as_matrix().astype(dtype)

    def __matmul__(self, other):
        if not isinstance(other, AffineTransform):
            raise NotImplementedError()

        other_rotation = qt.quaternion_to_matrix(other.orientation)[:3, :3]
        position = other.position + other.scale * (self.position @ other_rotation)

        orientation = qt.quaternion_multiply(self._orientation, other._orientation)

        scale = self.scale * other.scale

        return AffineTransform(position=position, orientation=orientation, scale=scale)

    @property
    def position(self):
        return self._position

    @property
    def orientation(self):
        return self._orientation

    @property
    def scale(self):
        return self._scale

    @position.setter
    def position(self, value):
        self._position = np.asarray(value)

    @orientation.setter
    def orientation(self, value):
        self._orientation = np.asarray(value)

    @scale.setter
    def scale(self, value):
        self._scale = np.asarray(value)
