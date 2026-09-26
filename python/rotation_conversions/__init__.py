"""NumPy rotation-matrix and quaternion conversion utilities."""

from .core import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

__all__ = [
    "normalize_quaternion",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
]
