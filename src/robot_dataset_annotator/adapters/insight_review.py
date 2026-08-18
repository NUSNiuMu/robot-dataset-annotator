from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _pose_by_role(payload: dict[str, Any], role: str) -> dict[str, Any]:
    poses = payload.get("poses")
    if not isinstance(poses, list):
        raise ValueError("review manifest must contain a poses array")
    matches = [pose for pose in poses if pose.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role!r} pose stream")
    return matches[0]


def _rotation_6d(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("pose quaternions must have shape Nx4")
    norms = np.linalg.norm(quaternions, axis=1)
    valid = np.isfinite(quaternions).all(axis=1) & (norms > 1e-12)
    normalized = np.zeros_like(quaternions, dtype=np.float64)
    normalized[valid] = quaternions[valid] / norms[valid, None]
    x, y, z, w = normalized.T
    matrices = np.empty((len(quaternions), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.concatenate((matrices[:, :, 0], matrices[:, :, 1]), axis=1), valid


def load_fused_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized dual-hand state from an Insight review manifest.

    Review manifests contain synchronized poses but no gripper widths. The
    missing width columns remain invalid so task plugins cannot mistake them
    for measurements.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("review manifest must be a JSON object")
    frame_count = int(payload.get("frame_count", 0))
    if frame_count < 1:
        raise ValueError("review manifest has no frames")
    state = np.full((frame_count, 20), np.nan, dtype=np.float64)
    state_valid = np.zeros((frame_count, 20), dtype=bool)
    for role, offset in (("left_hand", 0), ("right_hand", 10)):
        pose = _pose_by_role(payload, role)
        positions = np.asarray(pose.get("positions"), dtype=np.float64)
        quaternions = np.asarray(pose.get("quaternions_xyzw"), dtype=np.float64)
        stream_valid = np.asarray(pose.get("valid"), dtype=bool)
        if positions.shape != (frame_count, 3):
            raise ValueError(f"{role} positions must have shape ({frame_count}, 3)")
        if stream_valid.shape != (frame_count,):
            raise ValueError(f"{role} validity must have shape ({frame_count},)")
        rotation, rotation_valid = _rotation_6d(quaternions)
        position_valid = stream_valid & np.isfinite(positions).all(axis=1)
        orientation_valid = stream_valid & rotation_valid
        state[:, offset : offset + 3] = positions
        state[:, offset + 3 : offset + 9] = rotation
        state_valid[:, offset : offset + 3] = position_valid[:, None]
        state_valid[:, offset + 3 : offset + 9] = orientation_valid[:, None]
    return state, state_valid
