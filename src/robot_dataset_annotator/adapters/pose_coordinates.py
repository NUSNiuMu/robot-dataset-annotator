from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import read_json


def quaternion_rotation_matrices(
    quaternions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("pose quaternions must have shape Nx4")
    norms = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & (norms > 1e-12)
    normalized = np.zeros_like(values)
    normalized[valid] = values[valid] / norms[valid, None]
    x, y, z, w = normalized.T
    matrices = np.empty((len(values), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices, valid


def rotation_6d(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("rotation matrices must have shape Nx3x3")
    return np.concatenate((values[:, :, 0], values[:, :, 1]), axis=1)


def rotation_matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("rotation matrices must have shape Nx3x3")
    result = np.empty((len(values), 4), dtype=np.float64)
    for index, matrix in enumerate(values):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            result[index] = [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = (
                    np.sqrt(
                        1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
                    )
                    * 2.0
                )
                result[index] = [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            elif axis == 1:
                scale = (
                    np.sqrt(
                        1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
                    )
                    * 2.0
                )
                result[index] = [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            else:
                scale = (
                    np.sqrt(
                        1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
                    )
                    * 2.0
                )
                result[index] = [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def pose_matrices(
    positions: np.ndarray, quaternions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    translations = np.asarray(positions, dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("pose positions must have shape Nx3")
    rotations, rotation_valid = quaternion_rotation_matrices(quaternions)
    if len(translations) != len(rotations):
        raise ValueError("pose position and quaternion lengths differ")
    valid = np.isfinite(translations).all(axis=1) & rotation_valid
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(translations), axis=0)
    matrices[:, :3, :3] = rotations
    matrices[:, :3, 3] = translations
    return matrices, valid


def pose_state_from_matrices(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError("pose matrices must have shape Nx4x4")
    return np.concatenate((values[:, :3, 3], rotation_6d(values[:, :3, :3])), axis=1)


def validate_rigid_transform(values: Any, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{label} rotation determinant is not one")
    return matrix


def load_qr_transform(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = read_json(path)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("QR transform schema_version must be 1")
    global_from_qr = validate_rigid_transform(
        payload.get("global_from_qr"), "global_from_qr"
    )
    qr_from_global = np.linalg.inv(global_from_qr)
    stored_inverse = payload.get("qr_from_global")
    if stored_inverse is not None and not np.allclose(
        validate_rigid_transform(stored_inverse, "qr_from_global"),
        qr_from_global,
        atol=1e-6,
    ):
        raise ValueError("qr_from_global is not the inverse of global_from_qr")
    normalized = dict(payload)
    normalized["global_from_qr"] = global_from_qr.tolist()
    normalized["qr_from_global"] = qr_from_global.tolist()
    return normalized, qr_from_global
