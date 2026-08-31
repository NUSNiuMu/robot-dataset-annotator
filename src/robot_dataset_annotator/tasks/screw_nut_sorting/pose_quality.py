from __future__ import annotations

from typing import Any

import numpy as np


REQUIRED_ROLES = ("left_hand", "right_hand")


def _positive(config: dict[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _rotation_steps_deg(rotations: np.ndarray) -> np.ndarray:
    relative = np.einsum(
        "nij,njk->nik",
        np.transpose(rotations[:-1], (0, 2, 1)),
        rotations[1:],
    )
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def _pose_steps(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.linalg.norm(np.diff(matrices[:, :3, 3], axis=0), axis=1)
    rotation = _rotation_steps_deg(matrices[:, :3, :3])
    return translation, rotation


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _step_clusters(steps: list[int]) -> list[list[int]]:
    clusters: list[list[int]] = []
    for step in sorted(steps):
        if not clusters or step > clusters[-1][-1] + 1:
            clusters.append([step])
        else:
            clusters[-1].append(step)
    return clusters


def _comparison_arrays(
    role: str, comparison: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray(comparison["review_frames"], dtype=np.int64)
    native = np.asarray(comparison["native_matrices"], dtype=np.float64)
    global_pose = np.asarray(comparison["global_matrices"], dtype=np.float64)
    valid = np.asarray(comparison.get("valid", np.ones(len(frames))), dtype=bool)
    pair_skew_ms = np.asarray(
        comparison.get("pair_skew_ms", np.zeros(len(frames))), dtype=np.float64
    )
    expected_shape = (len(frames), 4, 4)
    if native.shape != expected_shape or global_pose.shape != expected_shape:
        raise ValueError(f"{role} pose comparisons must be Nx4x4 matrices")
    if valid.shape != (len(frames),) or pair_skew_ms.shape != (len(frames),):
        raise ValueError(f"{role} validity and skew must match review frames")
    if len(frames) < 2 or np.any(np.diff(frames) <= 0):
        raise ValueError(f"{role} review frames must be strictly increasing")
    finite = np.isfinite(native).all(axis=(1, 2)) & np.isfinite(global_pose).all(
        axis=(1, 2)
    )
    return frames, native, global_pose, valid & finite, pair_skew_ms


def _role_episode_audit(
    role: str,
    comparison: dict[str, Any],
    start: int,
    end: int,
    *,
    thresholds: dict[str, float | int],
) -> dict[str, Any]:
    frames, native, global_pose, valid, pair_skew_ms = _comparison_arrays(
        role, comparison
    )
    selected = (frames >= start) & (frames < end)
    selected_indices = np.flatnonzero(selected)
    if not len(selected_indices):
        return {
            "role": role,
            "status": "NEEDS_REVIEW",
            "training_usable": False,
            "correction_status": "MISSING_DATA",
            "reasons": ["episode has no synchronized VIO and Insight Global poses"],
            "events": [],
        }

    selected_valid = valid[selected_indices] & (
        pair_skew_ms[selected_indices] <= thresholds["maximum_pair_skew_ms"]
    )
    valid_fraction = float(np.mean(selected_valid))
    longest_invalid = _longest_false_run(selected_valid)
    reasons: list[str] = []
    if valid_fraction < thresholds["minimum_valid_fraction"]:
        reasons.append(f"paired pose validity is {valid_fraction:.1%}")
    if longest_invalid > thresholds["maximum_invalid_run_frames"]:
        reasons.append(
            f"paired pose gap reaches {longest_invalid} frames"
        )

    native_translation, native_rotation = _pose_steps(native)
    global_translation, global_rotation = _pose_steps(global_pose)
    alignment = np.einsum("nij,njk->nik", global_pose, np.linalg.inv(native))
    alignment_translation, alignment_rotation = _pose_steps(alignment)
    events: list[dict[str, Any]] = []
    event_by_step: dict[int, dict[str, Any]] = {}
    for step in range(1, len(frames)):
        previous_frame = int(frames[step - 1])
        frame = int(frames[step])
        if previous_frame < start or frame >= end:
            continue
        if frame != previous_frame + 1:
            events.append(
                {
                    "type": "review_frame_gap",
                    "frame_start": previous_frame,
                    "frame_end": frame,
                }
            )
            reasons.append(
                f"review pose samples skip frames {previous_frame} to {frame}"
            )
            continue
        pair_valid = bool(valid[step - 1] and valid[step])
        pair_valid &= bool(
            pair_skew_ms[step - 1] <= thresholds["maximum_pair_skew_ms"]
            and pair_skew_ms[step] <= thresholds["maximum_pair_skew_ms"]
        )
        if not pair_valid:
            continue
        native_jump = bool(
            native_translation[step - 1]
            >= thresholds["native_translation_jump_m"]
            or native_rotation[step - 1]
            >= thresholds["native_rotation_jump_deg"]
        )
        global_jump = bool(
            global_translation[step - 1]
            >= thresholds["global_translation_jump_m"]
            or global_rotation[step - 1]
            >= thresholds["global_rotation_jump_deg"]
        )
        alignment_update = bool(
            alignment_translation[step - 1]
            >= thresholds["alignment_translation_update_m"]
            or alignment_rotation[step - 1]
            >= thresholds["alignment_rotation_update_deg"]
        )
        if not (native_jump or global_jump or alignment_update):
            continue
        event = {
            "frame": frame,
            "native_jump": native_jump,
            "global_jump": global_jump,
            "alignment_update": alignment_update,
            "native_translation_step_m": float(native_translation[step - 1]),
            "global_translation_step_m": float(global_translation[step - 1]),
            "alignment_translation_step_m": float(
                alignment_translation[step - 1]
            ),
            "native_rotation_step_deg": float(native_rotation[step - 1]),
            "global_rotation_step_deg": float(global_rotation[step - 1]),
            "alignment_rotation_step_deg": float(alignment_rotation[step - 1]),
        }
        events.append(event)
        event_by_step[step] = event

    native_event_steps = [
        step for step, event in event_by_step.items() if event["native_jump"]
    ]
    alignment_event_steps = [
        step for step, event in event_by_step.items() if event["alignment_update"]
    ]
    native_clusters = _step_clusters(native_event_steps)
    alignment_clusters = _step_clusters(alignment_event_steps)
    corrected_same_frame = 0
    partially_corrected = 0
    uncorrected = 0
    matched_alignment_steps: set[int] = set()
    for cluster in native_clusters:
        cluster_events = [event_by_step[step] for step in cluster]
        cluster_start = cluster[0]
        cluster_end = cluster[-1]
        for event in cluster_events:
            event["native_jump_cluster"] = [
                int(frames[cluster_start]),
                int(frames[cluster_end]),
            ]
        if not any(event["global_jump"] for event in cluster_events):
            for event in cluster_events:
                event["correction"] = "CORRECTED_SAME_FRAME"
            corrected_same_frame += 1
            matched_alignment_steps.update(
                step for step in cluster if event_by_step[step]["alignment_update"]
            )
            continue
        later_cluster = next(
            (
                candidate
                for candidate in alignment_clusters
                if candidate[0] > cluster_end
            ),
            None,
        )
        if later_cluster is None:
            for event in cluster_events:
                event["correction"] = "UNCORRECTED"
            uncorrected += 1
        else:
            for event in cluster_events:
                event["correction"] = "ALIGNMENT_UPDATED_LATER"
                event["alignment_update_frame"] = int(frames[later_cluster[0]])
                event["correction_latency_frames"] = int(
                    frames[later_cluster[0]] - frames[cluster_end]
                )
            partially_corrected += 1
            matched_alignment_steps.update(later_cluster)

    native_event_step_set = set(native_event_steps)
    unresolved_global_steps = [
        step
        for step, event in event_by_step.items()
        if event["global_jump"]
        and step not in native_event_step_set
        and step not in matched_alignment_steps
    ]
    discontinuous_alignment_steps = [
        step
        for step, event in event_by_step.items()
        if event["alignment_update"]
        and event["global_jump"]
        and step not in matched_alignment_steps
    ]
    if uncorrected:
        reasons.append(f"{uncorrected} native VIO jump(s) remain uncorrected")
    if partially_corrected:
        reasons.append(
            f"{partially_corrected} native VIO jump(s) are only realigned later"
        )
    if unresolved_global_steps:
        reasons.append(
            f"{len(unresolved_global_steps)} unexplained Insight Global jump(s)"
        )
    if discontinuous_alignment_steps:
        reasons.append(
            f"{len(discontinuous_alignment_steps)} alignment refresh(es) "
            "are discontinuous"
        )

    training_usable = not reasons
    if training_usable and corrected_same_frame:
        status = "PASS_AFTER_CORRECTION"
        correction_status = "CORRECTED"
    elif training_usable:
        status = "PASS"
        correction_status = "NO_DRIFT"
    elif partially_corrected:
        status = "NEEDS_REVIEW"
        correction_status = "PARTIALLY_CORRECTED"
    elif uncorrected:
        status = "NEEDS_REVIEW"
        correction_status = "UNCORRECTED"
    else:
        status = "NEEDS_REVIEW"
        correction_status = "DISCONTINUOUS_GLOBAL_ALIGNMENT"
    return {
        "role": role,
        "status": status,
        "training_usable": training_usable,
        "correction_status": correction_status,
        "valid_fraction": valid_fraction,
        "maximum_pair_skew_ms": float(np.max(pair_skew_ms[selected_indices])),
        "native_jump_count": len(native_clusters),
        "corrected_same_frame_count": corrected_same_frame,
        "partially_corrected_count": partially_corrected,
        "uncorrected_count": uncorrected,
        "reasons": reasons,
        "events": events,
    }


def audit_episode_pose_quality(
    comparisons: dict[str, dict[str, Any]],
    episodes: list[dict[str, int]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Audit each sorting episode without carrying corrections across resets."""

    missing_roles = [role for role in REQUIRED_ROLES if role not in comparisons]
    if missing_roles:
        raise ValueError(f"missing pose comparisons for roles: {missing_roles}")
    thresholds: dict[str, float | int] = {
        "native_translation_jump_m": _positive(
            config, "native_translation_jump_m", 0.10
        ),
        "native_rotation_jump_deg": _positive(
            config, "native_rotation_jump_deg", 20.0
        ),
        "global_translation_jump_m": _positive(
            config, "global_translation_jump_m", 0.10
        ),
        "global_rotation_jump_deg": _positive(
            config, "global_rotation_jump_deg", 20.0
        ),
        "alignment_translation_update_m": _positive(
            config, "alignment_translation_update_m", 0.05
        ),
        "alignment_rotation_update_deg": _positive(
            config, "alignment_rotation_update_deg", 10.0
        ),
        "maximum_pair_skew_ms": _positive(
            config, "maximum_pair_skew_ms", 5.0
        ),
        "minimum_valid_fraction": float(
            config.get("minimum_valid_fraction", 0.99)
        ),
        "maximum_invalid_run_frames": int(
            config.get("maximum_invalid_run_frames", 2)
        ),
    }
    minimum_valid_fraction = thresholds["minimum_valid_fraction"]
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if thresholds["maximum_invalid_run_frames"] < 0:
        raise ValueError("maximum_invalid_run_frames must be non-negative")
    if not episodes:
        raise ValueError("episode pose-quality audit requires at least one episode")

    episode_results: list[dict[str, Any]] = []
    previous_end = -1
    for index, episode in enumerate(episodes):
        start = int(episode["episode_start_frame"])
        end = int(episode["episode_end_frame_exclusive"])
        if start < previous_end or end <= start:
            raise ValueError("episode ranges must be ordered and non-overlapping")
        role_results = [
            _role_episode_audit(
                role,
                comparisons[role],
                start,
                end,
                thresholds=thresholds,
            )
            for role in REQUIRED_ROLES
        ]
        usable = all(result["training_usable"] for result in role_results)
        after_correction = any(
            result["status"] == "PASS_AFTER_CORRECTION"
            for result in role_results
        )
        episode_results.append(
            {
                "episode_index": index,
                "episode_start_frame": start,
                "episode_end_frame_exclusive": end,
                "status": (
                    "PASS_AFTER_CORRECTION"
                    if usable and after_correction
                    else "PASS" if usable else "NEEDS_REVIEW"
                ),
                "training_usable": usable,
                "roles": {result["role"]: result for result in role_results},
            }
        )
        previous_end = end
    return {
        "schema_version": 1,
        "status": (
            "PASS"
            if all(result["training_usable"] for result in episode_results)
            else "NEEDS_REVIEW"
        ),
        "thresholds": thresholds,
        "subsequent_episodes_evaluated_independently": True,
        "episodes": episode_results,
        "usable_episode_indices": [
            result["episode_index"]
            for result in episode_results
            if result["training_usable"]
        ],
        "unusable_episode_indices": [
            result["episode_index"]
            for result in episode_results
            if not result["training_usable"]
        ],
    }
