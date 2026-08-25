from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import read_json, write_json_atomic
from .pose_coordinates import pose_matrices, rotation_matrices_to_quaternions


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    first = left / np.linalg.norm(left)
    second = right / np.linalg.norm(right)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    scale = np.sin(angle)
    return (
        np.sin((1.0 - fraction) * angle) / scale * first
        + np.sin(fraction * angle) / scale * second
    )


def _robust_threshold(values: np.ndarray, absolute_floor: float) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return absolute_floor
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return max(absolute_floor, median + 12.0 * max(1.4826 * mad, 1e-6))


def _runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    result: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append((start, previous + 1))
            start = value
        previous = value
    result.append((start, previous + 1))
    return result


def _progressive_drift_clusters(
    jump_steps: list[int],
    matrices: np.ndarray,
    *,
    count: int,
    maximum_frames: int,
    minimum_translation: float,
) -> list[list[int]]:
    """Find short, directional runs of implausible coordinate-frame motion."""
    if not jump_steps:
        return []
    clusters: list[list[int]] = [[jump_steps[0]]]
    for step in jump_steps[1:]:
        if step - clusters[-1][-1] <= 3:
            clusters[-1].append(step)
        else:
            clusters.append([step])

    jump_set = set(jump_steps)
    result: list[list[int]] = []
    for cluster in clusters:
        start = cluster[0]
        end = cluster[-1] + 1
        if len(cluster) < 4 or end - start > maximum_frames:
            continue
        stable_end = min(count, end + 6)
        if stable_end - end < 6 or any(
            step in jump_set for step in range(end, stable_end)
        ):
            continue
        vectors = np.asarray(
            [
                matrices[step, :3, 3] - matrices[step - 1, :3, 3]
                for step in cluster
            ],
            dtype=np.float64,
        )
        path_length = float(np.linalg.norm(vectors, axis=1).sum())
        net_translation = float(np.linalg.norm(vectors.sum(axis=0)))
        directional_ratio = net_translation / max(path_length, 1e-12)
        if net_translation >= minimum_translation and directional_ratio >= 0.85:
            result.append(cluster)
    return result


def _settling_jump_clusters(
    jump_steps: list[int],
    translation_steps: np.ndarray,
    rotation_steps: np.ndarray,
    *,
    count: int,
    persistent_translation: float,
    persistent_rotation: float,
) -> list[list[int]]:
    """Find a short multi-step coordinate jump followed by stable tracking."""
    if not jump_steps:
        return []
    clusters: list[list[int]] = [[jump_steps[0]]]
    for step in jump_steps[1:]:
        if step - clusters[-1][-1] <= 3:
            clusters[-1].append(step)
        else:
            clusters.append([step])
    jump_set = set(jump_steps)
    result: list[list[int]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        end = cluster[-1] + 1
        stable_end = min(count, end + 6)
        if stable_end - end < 6 or any(
            step in jump_set for step in range(end, stable_end)
        ):
            continue
        if (
            max(float(translation_steps[step]) for step in cluster)
            >= persistent_translation
            or max(float(rotation_steps[step]) for step in cluster)
            >= persistent_rotation
        ):
            result.append(cluster)
    return result


def correct_pose_stream(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    valid: np.ndarray,
    *,
    maximum_spike_frames: int = 3,
    maximum_drift_transition_frames: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    translations = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64)
    stream_valid = np.asarray(valid, dtype=bool)
    matrices, matrix_valid = pose_matrices(translations, quaternions)
    pose_valid = stream_valid & matrix_valid
    count = len(matrices)
    translation_steps = np.full(count, np.nan, dtype=np.float64)
    rotation_steps = np.full(count, np.nan, dtype=np.float64)
    for index in range(1, count):
        if pose_valid[index - 1] and pose_valid[index]:
            translation_steps[index] = np.linalg.norm(
                matrices[index, :3, 3] - matrices[index - 1, :3, 3]
            )
            rotation_steps[index] = _rotation_angle(
                matrices[index - 1], matrices[index]
            )

    translation_threshold = _robust_threshold(translation_steps, 0.10)
    # Review streams run at 30 Hz while some pose topics update more slowly.
    # A duplicated sample can therefore concentrate a plausible hand rotation
    # into one review-frame step; 30 degrees avoids classifying that as drift.
    rotation_threshold = _robust_threshold(rotation_steps, np.deg2rad(30.0))
    jump_steps = [
        index
        for index in range(1, count)
        if pose_valid[index - 1]
        and pose_valid[index]
        and (
            translation_steps[index] > translation_threshold
            or rotation_steps[index] > rotation_threshold
        )
    ]
    jump_set = set(jump_steps)
    consumed: set[int] = set()
    spike_intervals: list[tuple[int, int]] = []
    events: list[dict[str, Any]] = []
    continuity_translation = max(0.05, translation_threshold / 2.0)
    continuity_rotation = max(np.deg2rad(10.0), rotation_threshold / 2.0)
    for start in jump_steps:
        if start in consumed:
            continue
        for return_step in range(start + 1, start + maximum_spike_frames + 2):
            if return_step >= count or return_step not in jump_set:
                continue
            if not (pose_valid[start - 1] and pose_valid[return_step]):
                continue
            bridge_translation = np.linalg.norm(
                matrices[return_step, :3, 3]
                - matrices[start - 1, :3, 3]
            )
            bridge_rotation = _rotation_angle(
                matrices[start - 1], matrices[return_step]
            )
            if (
                bridge_translation <= continuity_translation
                and bridge_rotation <= continuity_rotation
            ):
                spike_intervals.append((start, return_step))
                consumed.update(range(start, return_step + 1))
                events.append(
                    {
                        "type": "instantaneous_spike",
                        "frame_start": start,
                        "frame_end_exclusive": return_step,
                        "bridge_translation_m": float(bridge_translation),
                        "bridge_rotation_deg": float(np.rad2deg(bridge_rotation)),
                    }
                )
                break

    persistent_steps: list[int] = []
    unresolved_steps: list[int] = []
    persistent_translation = max(0.15, 1.5 * translation_threshold)
    persistent_rotation = max(np.deg2rad(30.0), 1.5 * rotation_threshold)
    progressive_clusters = _progressive_drift_clusters(
        [step for step in jump_steps if step not in consumed],
        matrices,
        count=count,
        maximum_frames=maximum_drift_transition_frames,
        minimum_translation=max(0.50, 3.0 * persistent_translation),
    )
    progressive_steps = {
        step for cluster in progressive_clusters for step in cluster
    }
    consumed.update(progressive_steps)
    for cluster in progressive_clusters:
        vectors = np.asarray(
            [
                matrices[step, :3, 3] - matrices[step - 1, :3, 3]
                for step in cluster
            ],
            dtype=np.float64,
        )
        path_length = float(np.linalg.norm(vectors, axis=1).sum())
        net_translation = float(np.linalg.norm(vectors.sum(axis=0)))
        events.append(
            {
                "type": "progressive_coordinate_drift",
                "frame_start": cluster[0],
                "frame_end_exclusive": cluster[-1] + 1,
                "jump_frames": cluster,
                "net_translation_m": net_translation,
                "directional_ratio": net_translation / max(path_length, 1e-12),
            }
        )
    settling_clusters = _settling_jump_clusters(
        [step for step in jump_steps if step not in consumed],
        translation_steps,
        rotation_steps,
        count=count,
        persistent_translation=persistent_translation,
        persistent_rotation=persistent_rotation,
    )
    settling_steps = {step for cluster in settling_clusters for step in cluster}
    consumed.update(settling_steps)
    for cluster in settling_clusters:
        events.append(
            {
                "type": "settling_coordinate_jump",
                "frame_start": cluster[0],
                "frame_end_exclusive": cluster[-1] + 1,
                "jump_frames": cluster,
                "maximum_translation_step_m": max(
                    float(translation_steps[step]) for step in cluster
                ),
                "maximum_rotation_step_deg": float(
                    np.rad2deg(max(rotation_steps[step] for step in cluster))
                ),
            }
        )
    remaining_steps = [step for step in jump_steps if step not in consumed]
    confirmed_persistent_instability = any(
        (
            translation_steps[index] > persistent_translation
            or rotation_steps[index] > persistent_rotation
        )
        and all(
            step not in jump_set
            for step in range(index + 1, min(count, index + 7))
        )
        for index in remaining_steps
    )
    for index in jump_steps:
        if index in consumed:
            continue
        window_end = min(count, index + 7)
        stable_after = all(
            step not in jump_set for step in range(index + 1, window_end)
        )
        very_large = (
            translation_steps[index] > persistent_translation
            or rotation_steps[index] > persistent_rotation
        )
        # Once this stream has already exhibited a confirmed coordinate-frame
        # jump, treat later stable jumps above the normal jump threshold as
        # part of the same instability chain. Trackers commonly recover in
        # several stages, leaving residual steps below the deliberately
        # conservative first-jump threshold.
        chained_instability = bool(
            persistent_steps
            or progressive_clusters
            or settling_clusters
            or confirmed_persistent_instability
        )
        if (
            stable_after
            and (very_large or chained_instability)
            and index >= 2
            and pose_valid[index - 2]
        ):
            persistent_steps.append(index)
        else:
            unresolved_steps.append(index)
            events.append(
                {
                    "type": "unresolved_jump",
                    "frame": index,
                    "translation_step_m": float(translation_steps[index]),
                    "rotation_step_deg": float(np.rad2deg(rotation_steps[index])),
                }
            )

    corrected_matrices = matrices.copy()
    correction = np.eye(4, dtype=np.float64)
    persistent_set = set(persistent_steps)
    for index in range(count):
        if (
            index in persistent_set
            or index in progressive_steps
            or index in settling_steps
        ):
            previous = corrected_matrices[index - 1]
            before_previous = corrected_matrices[index - 2]
            previous_motion = np.linalg.inv(before_previous) @ previous
            expected = previous @ previous_motion
            correction = expected @ np.linalg.inv(matrices[index])
            if index in persistent_set:
                events.append(
                    {
                        "type": "persistent_coordinate_jump",
                        "frame": index,
                        "translation_step_m": float(translation_steps[index]),
                        "rotation_step_deg": float(
                            np.rad2deg(rotation_steps[index])
                        ),
                        "correction_global_from_raw": correction.tolist(),
                    }
                )
        corrected_matrices[index] = correction @ matrices[index]

    corrected_positions = corrected_matrices[:, :3, 3].copy()
    corrected_quaternions = rotation_matrices_to_quaternions(
        corrected_matrices[:, :3, :3]
    )
    correction_mask = np.zeros(count, dtype=bool)
    for start, end in spike_intervals:
        correction_mask[start:end] = True
        for index in range(start, end):
            fraction = (index - start + 1) / (end - start + 1)
            corrected_positions[index] = (
                (1.0 - fraction) * corrected_positions[start - 1]
                + fraction * corrected_positions[end]
            )
            corrected_quaternions[index] = _slerp(
                corrected_quaternions[start - 1],
                corrected_quaternions[end],
                fraction,
            )
    for index in persistent_steps:
        correction_mask[index:] = True
    for cluster in progressive_clusters:
        correction_mask[cluster[0] :] = True
    for cluster in settling_clusters:
        correction_mask[cluster[0] :] = True
    corrected_positions[~pose_valid] = translations[~pose_valid]
    corrected_quaternions[~pose_valid] = quaternions[~pose_valid]
    audit = {
        "status": "NEEDS_REVIEW" if unresolved_steps else "PASS",
        "frames": count,
        "thresholds": {
            "translation_jump_m": float(translation_threshold),
            "rotation_jump_deg": float(np.rad2deg(rotation_threshold)),
            "persistent_translation_jump_m": float(persistent_translation),
            "persistent_rotation_jump_deg": float(
                np.rad2deg(persistent_rotation)
            ),
            "maximum_spike_frames": maximum_spike_frames,
            "maximum_drift_transition_frames": maximum_drift_transition_frames,
            "progressive_drift_minimum_translation_m": float(
                max(0.50, 3.0 * persistent_translation)
            ),
        },
        "events": sorted(
            events,
            key=lambda row: int(row.get("frame", row.get("frame_start", -1))),
        ),
        "corrected_frames": int(np.count_nonzero(correction_mask)),
        "correction_intervals": [
            list(row) for row in _runs(np.flatnonzero(correction_mask).tolist())
        ],
        "unresolved_jump_frames": unresolved_steps,
    }
    return corrected_positions, corrected_quaternions, correction_mask, audit


def correct_review_manifest_pose_drift(
    *,
    review_manifest_path: Path,
    output_manifest_path: Path,
    audit_path: Path,
    maximum_spike_frames: int = 3,
    maximum_drift_transition_frames: int = 45,
) -> dict[str, Any]:
    for output in (output_manifest_path, audit_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite pose drift output: {output}")
    manifest = read_json(review_manifest_path)
    corrected_manifest = dict(manifest)
    corrected_poses: list[dict[str, Any]] = []
    stream_audits: list[dict[str, Any]] = []
    overall_status = "PASS"
    for row in manifest.get("poses", []):
        corrected_positions, corrected_quaternions, mask, audit = correct_pose_stream(
            np.asarray(row["positions"], dtype=np.float64),
            np.asarray(row["quaternions_xyzw"], dtype=np.float64),
            np.asarray(row["valid"], dtype=bool),
            maximum_spike_frames=maximum_spike_frames,
            maximum_drift_transition_frames=maximum_drift_transition_frames,
        )
        corrected_row = dict(row)
        corrected_row["raw_positions"] = row["positions"]
        corrected_row["raw_quaternions_xyzw"] = row["quaternions_xyzw"]
        corrected_row["positions"] = corrected_positions.tolist()
        corrected_row["quaternions_xyzw"] = corrected_quaternions.tolist()
        corrected_row["pose_correction_mask"] = mask.tolist()
        corrected_poses.append(corrected_row)
        stream_audit = {"role": str(row.get("role", "")), **audit}
        stream_audits.append(stream_audit)
        if audit["status"] != "PASS":
            overall_status = "NEEDS_REVIEW"
    source_hash = _sha256(review_manifest_path)
    corrected_manifest["poses"] = corrected_poses
    corrected_manifest["pose_drift_correction"] = {
        "schema_version": 1,
        "status": overall_status,
        "source_manifest_sha256": source_hash,
        "audit_file": audit_path.name,
    }
    payload = {
        "schema_version": 1,
        "status": overall_status,
        "source_manifest": str(review_manifest_path),
        "source_manifest_sha256": source_hash,
        "corrected_manifest": str(output_manifest_path),
        "streams": stream_audits,
    }
    write_json_atomic(output_manifest_path, corrected_manifest)
    write_json_atomic(audit_path, payload)
    return payload
