"""CLI demonstration for NumPy rotation conversions."""

from __future__ import annotations

import argparse
import json
import math

from .core import (
    axis_angle_to_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--angle", type=float, default=90.0, help="Angle in degrees")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quaternion = axis_angle_to_quaternion(args.axis, math.radians(args.angle))
    matrix = quaternion_to_rotation_matrix(quaternion)
    round_trip = rotation_matrix_to_quaternion(matrix)
    result = {
        "convention": {
            "quaternion_order": "[w, x, y, z]",
            "matrix_action": "column_vector: v_rotated = R @ v",
        },
        "axis": args.axis,
        "angle_degrees": args.angle,
        "quaternion": quaternion.round(12).tolist(),
        "rotation_matrix": matrix.round(12).tolist(),
        "round_trip_quaternion": round_trip.round(12).tolist(),
        "round_trip_max_error": float(abs(round_trip - quaternion).max()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
