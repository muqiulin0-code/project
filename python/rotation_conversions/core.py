"""Convert between 3D rotation matrices and ``[w, x, y, z]`` quaternions.

Conventions:
    * Rotation matrices multiply column vectors: ``v_rotated = R @ v``.
    * Quaternions use scalar-first order: ``q = [w, x, y, z]``.
    * Output quaternions are canonicalized to ``w >= 0`` because ``q`` and
      ``-q`` represent the same rotation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_vector(values: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity")
    return vector


def normalize_quaternion(quaternion: ArrayLike) -> FloatArray:
    """Return a unit quaternion in ``[w, x, y, z]`` order."""

    result = np.asarray(quaternion, dtype=np.float64)
    if result.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("quaternion contains NaN or infinity")
    norm = float(np.linalg.norm(result))
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    return result / norm


def quaternion_to_rotation_matrix(quaternion: ArrayLike) -> FloatArray:
    """Convert a unit quaternion ``[w, x, y, z]`` to a 3x3 rotation matrix."""

    w, x, y, z = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(
    matrix: ArrayLike,
    *,
    project_to_rotation: bool = True,
    tolerance: float = 1e-6,
) -> FloatArray:
    """Convert a 3x3 rotation matrix to ``[w, x, y, z]``.

    If ``project_to_rotation`` is true, a noisy matrix is projected onto the
    closest proper rotation using SVD. Set it to false to reject matrices that
    are not orthonormal with a positive determinant.
    """

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation matrix contains NaN or infinity")

    error = np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    determinant = float(np.linalg.det(rotation))
    is_rotation = error <= tolerance and abs(determinant - 1.0) <= tolerance
    if not is_rotation:
        if not project_to_rotation:
            raise ValueError(
                "matrix must be orthonormal with determinant +1 "
                f"(orthogonality error={error:.3g}, determinant={determinant:.3g})"
            )
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vt

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale

    quaternion = normalize_quaternion([w, x, y, z])
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def quaternion_multiply(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Compose quaternions so the result matches ``R_left @ R_right``."""

    lw, lx, ly, lz = normalize_quaternion(left)
    rw, rx, ry, rz = normalize_quaternion(right)
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )
    return normalize_quaternion(result)


def axis_angle_to_quaternion(axis: ArrayLike, angle_radians: float) -> FloatArray:
    """Create a unit quaternion from an axis and angle in radians."""

    unit_axis = _as_vector(axis, "axis")
    norm = float(np.linalg.norm(unit_axis))
    if norm < 1e-12:
        raise ValueError("axis must not be the zero vector")
    unit_axis = unit_axis / norm
    half_angle = 0.5 * float(angle_radians)
    return normalize_quaternion(
        np.concatenate(([np.cos(half_angle)], unit_axis * np.sin(half_angle)))
    )
