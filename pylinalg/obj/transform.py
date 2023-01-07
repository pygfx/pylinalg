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
    source and target. That is, a transformation that is linear in homogeneous
    coordinates and that sequentially applies a translation, rotation, and
    scaling.


    Parameters
    ----------
    position : ndarray, [3]
        The position of target's frame in source's space. I.e., a vector
        describing the translation to apply.
    orientation : ndarray, [4]
        The orientation of the target's frame in source's space. I.e., a
        quaternion (format: x, y, z, w) describing the rotation to apply.
    scale : ndarray, [3]
        The scale of target's units expressed in parent's units. I.e., a vector
        describing the per-axis scaling to apply.

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
            self._orientation = np.array((0, 0, 0, 1), dtype=float)

        if scale is None:
            self._scale = np.ones(3)

    def as_matrix(self):
        """
        The affine transformation matrix.

        The returned matrix is a transformation matrix that takes points in
        source's frame and expresses them in target's frame. This works in
        homogeneous coordinates, i.e., both input and output vectors must be of
        the form ``(x, y, z, 1)``.

        """

        # AffineTransform describes the location/properties of the target frame
        # in the source frame. A matrix constructed from these parameters maps
        # points from the target's frame back into source's frame. We want to go
        # from source to target so we need to inverse before building the
        # matrix. 
        inverse = self.inverse()

        result = qt.quaternion_to_matrix(inverse._orientation)
        result[:, :3] *= inverse._scale.reshape(1, 3)
        result[:3, -1] = inverse._position

        return result

    def inverse(self):
        """
        Get the inverse affine transformation.

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

        # TODO: check if the order is correct here or if it should be swapped.
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
        self._position = np.asarray(value, dtype=float)

    @orientation.setter
    def orientation(self, value):
        self._orientation = np.asarray(value, dtype=float)

    @scale.setter
    def scale(self, value):
        self._scale = np.asarray(value, dtype=float)