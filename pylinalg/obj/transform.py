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

        result = qt.quaternion_to_matrix(self._orientation)
        result[:, :3] *= self._scale.reshape(1, 3)
        result[:3, -1] = self._position

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
            scale * (qt.quaternion_to_matrix(orientation)[:3, :3] @ self._position)
        )

        return AffineTransform(position=position, orientation=orientation, scale=scale)

    def __array__(self, dtype=None):
        return self.as_matrix().astype(dtype)

    def __matmul__(self, other):
        if not isinstance(other, AffineTransform):
            raise NotImplementedError()

        rotation = qt.quaternion_to_matrix(self.orientation)[:3, :3]
        position = self.position + self.scale * (rotation @ other.position)

        orientation = qt.quaternion_multiply(other._orientation, self._orientation)

        scale = other.scale * self.scale

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
