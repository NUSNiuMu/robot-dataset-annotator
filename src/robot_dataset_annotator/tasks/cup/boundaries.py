from __future__ import annotations

from typing import Any

import numpy as np


BOUNDARY_METHOD = "carrier_grasp_drop_zone_release_and_motion_retreat_v3"


def _smooth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 5:
        return values.copy()
    padded = np.pad(values, (2, 2), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + 5]) for index in range(len(values))]
    )


def _sustained(
    mask: np.ndarray, start: int, window: int = 5, required: int = 3
) -> bool:
    return int(np.count_nonzero(mask[start : start + window])) >= required


def _repair(values: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(values)
    good = np.flatnonzero(valid)
    if len(good) < 5:
        return None
    result = values.copy()
    bad = np.flatnonzero(~valid)
    if len(bad):
        result[bad] = np.interp(bad, good, result[good])
    return _smooth(result)


def _release_event(
    width: np.ndarray,
    valid: np.ndarray,
    distance: np.ndarray,
    maximum_distance: float,
    transport_start: int,
    minimum_range: float,
) -> dict[str, Any] | None:
    repaired = _repair(width, valid)
    if repaired is None:
        return None
    valid = np.asarray(valid, dtype=bool) & np.isfinite(width)
    closed = float(np.percentile(repaired[valid], 5))
    opened = float(np.percentile(repaired[valid], 90))
    width_range = opened - closed
    if not np.isfinite(width_range) or width_range < minimum_range:
        return None
    openness = np.clip((repaired - closed) / width_range, 0.0, 1.5)
    closed_mask = (openness <= 0.30) & valid
    open_mask = (openness >= 0.70) & valid
    candidates = np.flatnonzero(
        closed_mask
        & (distance >= 0.90 * maximum_distance)
        & (np.arange(len(width)) >= transport_start)
    )
    for closed_frame in candidates[::-1]:
        end = next(
            (
                frame
                for frame in range(int(closed_frame) + 1, len(width))
                if open_mask[frame] and _sustained(open_mask, frame)
            ),
            None,
        )
        if end is None or end - int(closed_frame) > 40:
            continue
        start = int(closed_frame) + 1
        event_slice = slice(max(0, start - 3), min(len(width), end + 4))
        return {
            "start_frame": start,
            "end_frame": int(end),
            "width_range_mm": round(width_range * 1000.0, 3),
            "invalid_near_event": int(np.count_nonzero(~valid[event_slice])),
        }
    return None


def _acquisition_start(
    width: np.ndarray,
    valid: np.ndarray,
    release_start: int,
    minimum_range: float,
) -> int | None:
    repaired = _repair(width, valid)
    if repaired is None or release_start <= 0:
        return None
    valid = np.asarray(valid, dtype=bool) & np.isfinite(width)
    closed = float(np.percentile(repaired[valid], 5))
    opened = float(np.percentile(repaired[valid], 90))
    width_range = opened - closed
    if not np.isfinite(width_range) or width_range < minimum_range:
        return None
    openness = np.clip((repaired - closed) / width_range, 0.0, 1.5)
    low = (openness <= 0.30) & valid
    high = (openness >= 0.70) & valid
    sustained_high = high & np.asarray(
        [_sustained(high, frame) for frame in range(len(high))]
    )
    high_before = np.flatnonzero(
        sustained_high & (np.arange(len(high)) < release_start)
    )
    search_start = int(high_before[-1] + 1) if len(high_before) else 0
    for frame in range(search_start, release_start):
        if low[frame] and _sustained(low, frame):
            return frame
    return None


def _release_range(
    raw_start: int,
    raw_end: int,
    transport_start: int,
    episode_length: int,
    minimum_frames: int,
) -> tuple[int, int] | None:
    start, end = raw_start, raw_end
    while end - start < minimum_frames:
        can_pre = start > transport_start + minimum_frames
        can_post = end < episode_length - minimum_frames
        if can_pre:
            start -= 1
        if end - start >= minimum_frames:
            break
        if can_post:
            end += 1
        if not can_pre and not can_post:
            return None
    return start, end


def infer_fused_boundaries(
    states: np.ndarray,
    state_valid: np.ndarray,
    *,
    minimum_frames: int = 10,
    minimum_drop_excursion_m: float = 0.08,
    minimum_gripper_range_m: float = 0.01,
) -> dict[str, Any]:
    """Suggest four cup-task ranges from normalized dual-arm state.

    State layout is left xyz + rotation6d + gripper width followed by the same
    ten right-arm fields. An unresolved result is intentional: visual review is
    authoritative and automatic rejection must never discard source data.
    """

    values = np.asarray(states, dtype=np.float64)
    valid = np.asarray(state_valid, dtype=bool)
    length = len(values)
    if values.shape != (length, 20) or valid.shape != values.shape:
        raise ValueError("expected matching Nx20 state and validity arrays")
    if length < 4 * minimum_frames:
        return {"status": "unresolved", "reason": "episode_too_short"}
    pose_columns = [0, 1, 2, 10, 11, 12]
    pose_valid = np.all(valid[:, pose_columns], axis=1)
    if np.count_nonzero(pose_valid[:10]) < 3:
        return {"status": "unresolved", "reason": "invalid_pose_baseline"}
    midpoint = 0.5 * (values[:, :3] + values[:, 10:13])
    baseline = np.median(midpoint[:10][pose_valid[:10]], axis=0)
    distance = np.linalg.norm(midpoint - baseline, axis=1)
    if np.any(~pose_valid):
        good = np.flatnonzero(pose_valid)
        if len(good) < 5:
            return {"status": "unresolved", "reason": "insufficient_valid_pose"}
        distance[~pose_valid] = np.interp(
            np.flatnonzero(~pose_valid), good, distance[good]
        )
    distance = _smooth(distance)
    peak = int(np.argmax(distance))
    maximum_distance = float(distance[peak])
    if maximum_distance < minimum_drop_excursion_m or peak < minimum_frames:
        return {"status": "unresolved", "reason": "no_drop_zone_excursion"}
    far = np.flatnonzero(distance[: peak + 1] >= 0.80 * maximum_distance)
    if not len(far):
        return {"status": "unresolved", "reason": "no_far_zone"}
    near = np.flatnonzero(distance[: int(far[0])] <= 0.30 * maximum_distance)
    motion_start = (
        int(near[-1] + 1)
        if len(near)
        else max(1, int(far[0]) - minimum_frames)
    )

    events: dict[str, dict[str, Any]] = {}
    columns = {"left": 9, "right": 19}
    for side, column in columns.items():
        event = _release_event(
            values[:, column],
            valid[:, column],
            distance,
            maximum_distance,
            motion_start,
            minimum_gripper_range_m,
        )
        if event is not None:
            events[side] = event
    if not events:
        return {"status": "unresolved", "reason": "no_far_zone_release"}
    if any(event["invalid_near_event"] for event in events.values()):
        return {
            "status": "unresolved",
            "reason": "invalid_gripper_width_near_release",
            "side_events": events,
        }

    acquisitions: dict[str, int] = {}
    for side in list(events):
        start = _acquisition_start(
            values[:, columns[side]],
            valid[:, columns[side]],
            int(events[side]["start_frame"]),
            minimum_gripper_range_m,
        )
        if start is None:
            del events[side]
        else:
            acquisitions[side] = start
    if not events:
        return {"status": "unresolved", "reason": "no_acquisition_before_release"}
    transport_start = max(motion_start, min(acquisitions.values()))
    raw_start = min(int(event["start_frame"]) for event in events.values())
    returned = distance <= 0.90 * maximum_distance
    raw_end = next(
        (
            frame
            for frame in range(max(peak, raw_start + 1), length)
            if returned[frame] and _sustained(returned, frame)
        ),
        None,
    )
    if raw_end is None:
        return {"status": "unresolved", "reason": "no_retreat_after_release"}
    release = _release_range(
        raw_start, raw_end, transport_start, length, minimum_frames
    )
    if release is None:
        return {"status": "unresolved", "reason": "release_too_short"}
    release_start, retreat_start = release
    boundaries = [0, transport_start, release_start, retreat_start, length]
    lengths = np.diff(boundaries)
    if np.any(lengths < minimum_frames):
        return {
            "status": "unresolved",
            "reason": "semantic_segments_shorter_than_model_context",
            "raw_boundaries": boundaries,
        }
    return {
        "status": "candidate",
        "boundaries": boundaries,
        "segment_lengths": lengths.astype(int).tolist(),
        "release_event": [raw_start, raw_end],
        "carrier_style": (
            "bimanual_at_drop"
            if len(events) == 2
            else "single_carrier_or_handoff"
        ),
        "side_events": events,
        "max_excursion_m": round(maximum_distance, 6),
        "requires_video_confirmation": True,
        "boundary_method": BOUNDARY_METHOD,
    }
