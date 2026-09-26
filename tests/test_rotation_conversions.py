import numpy as np
import pytest
from rotation_conversions import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def test_identity_rotation() -> None:
    matrix = quaternion_to_rotation_matrix([1, 0, 0, 0])
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation_matrix_to_quaternion(matrix), [1, 0, 0, 0], atol=1e-12)


def test_ninety_degrees_about_z_uses_scalar_first_order() -> None:
    quaternion = rotation_matrix_to_quaternion(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    expected = np.sqrt(0.5)
    np.testing.assert_allclose(quaternion, [expected, 0.0, 0.0, expected], atol=1e-12)


def test_random_round_trip() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        quaternion = normalize_quaternion(rng.normal(size=4))
        matrix = quaternion_to_rotation_matrix(quaternion)
        restored = rotation_matrix_to_quaternion(matrix)
        np.testing.assert_allclose(
            quaternion_to_rotation_matrix(restored), matrix, atol=1e-10, rtol=0
        )
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_composition_matches_matrix_multiplication() -> None:
    left = normalize_quaternion([0.4, -0.2, 0.7, 0.1])
    right = normalize_quaternion([0.2, 0.6, -0.3, 0.5])
    composed = quaternion_multiply(left, right)
    expected = quaternion_to_rotation_matrix(left) @ quaternion_to_rotation_matrix(right)
    np.testing.assert_allclose(quaternion_to_rotation_matrix(composed), expected, atol=1e-12)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        rotation_matrix_to_quaternion(np.eye(4))
    with pytest.raises(ValueError, match="zero"):
        normalize_quaternion([0, 0, 0, 0])
    with pytest.raises(ValueError, match="determinant"):
        rotation_matrix_to_quaternion([[1, 0, 0], [0, -1, 0], [0, 0, 1]], project_to_rotation=False)


def test_noisy_matrix_can_be_projected() -> None:
    matrix = np.eye(3)
    matrix[0, 1] = 0.01
    quaternion = rotation_matrix_to_quaternion(matrix, project_to_rotation=True)
    projected = quaternion_to_rotation_matrix(quaternion)
    np.testing.assert_allclose(projected.T @ projected, np.eye(3), atol=1e-12)
    assert np.linalg.det(projected) > 0.999999
    assert np.linalg.norm(projected - matrix) < 0.02
