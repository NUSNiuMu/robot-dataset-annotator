from __future__ import annotations

from typing import Any

import numpy as np


BOUNDARY_METHOD = "bimanual_home_return_cycle_v1"
POSITION_COLUMNS = (0, 1, 2, 10, 11, 12)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    width = max(1, min(int(window), len(values)))
    if width == 1:
        return values.copy()
    return np.convolve(values, np.ones(width, dtype=np.float64) / width, mode="same")


def _runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= minimum_length
    ]


def _longest_invalid_run(valid: np.ndarray) -> int:
    runs = _runs(~valid, 1)
    return max((end - start for start, end in runs), default=0)


def _repair_positions(
    states: np.ndarray,
    state_valid: np.ndarray,
    maximum_invalid_gap: int,
) -> tuple[np.ndarray | None, str | None]:
    pose_valid = np.all(state_valid[:, POSITION_COLUMNS], axis=1) & np.all(
        np.isfinite(states[:, POSITION_COLUMNS]), axis=1
    )
    good = np.flatnonzero(pose_valid)
    if len(good) < 5:
        return None, "insufficient_valid_hand_pose"
    if _longest_invalid_run(pose_valid) > maximum_invalid_gap:
        return None, "hand_pose_gap_too_long"
    positions = states[:, POSITION_COLUMNS].astype(np.float64, copy=True)
    missing = np.flatnonzero(~pose_valid)
    if len(missing):
        for column in range(positions.shape[1]):
            positions[missing, column] = np.interp(
                missing, good, positions[good, column]
            )
    return positions, None


def _phase_boundary(
    motion: np.ndarray,
    start: int,
    end: int,
    minimum_frames: int,
) -> tuple[int, int]:
    span = end - start
    search_start = max(start + minimum_frames, start + int(0.35 * span))
    search_end = min(end - minimum_frames, start + int(0.70 * span))
    if search_end <= search_start:
        return start + span // 2, 0
    section = motion[search_start:search_end]
    low_threshold = max(float(np.percentile(section, 25)), 1e-12)
    low_runs = _runs(section <= low_threshold, 1)
    if not low_runs:
        return search_start + int(np.argmin(section)), 0
    local_start, local_end = max(low_runs, key=lambda item: item[1] - item[0])
    boundary = search_start + local_end
    boundary = min(
        max(boundary, start + minimum_frames, start + int(0.52 * span)),
        end - minimum_frames,
    )
    return boundary, local_end - local_start


def infer_cycle_boundaries(
    states: np.ndarray,
    state_valid: np.ndarray,
    *,
    minimum_frames: int = 10,
    minimum_motion_m_per_frame: float = 0.0015,
    maximum_home_distance_m: float = 0.06,
    minimum_cycle_excursion_m: float = 0.08,
) -> dict[str, Any]:
    """Suggest repeated three-cup build-and-collapse cycles.

    State layout is left xyz + rotation6d + gripper width followed by the same
    ten right-arm fields. Object state is not inferred from hand pose, so every
    candidate and especially the build/collapse transition requires video
    confirmation.
    """

    values = np.asarray(states, dtype=np.float64)
    valid = np.asarray(state_valid, dtype=bool)
    length = len(values)
    if values.shape != (length, 20) or valid.shape != values.shape:
        raise ValueError("expected matching Nx20 state and validity arrays")
    if length < 2 * minimum_frames:
        return {"status": "unresolved", "reason": "source_segment_too_short"}
    positions, error = _repair_positions(
        values, valid, maximum_invalid_gap=minimum_frames
    )
    if positions is None:
        return {"status": "unresolved", "reason": error}

    baseline_frames = min(length, max(3 * minimum_frames, 15))
    baseline = np.median(positions[:baseline_frames], axis=0)
    excursion = np.linalg.norm(positions - baseline, axis=1)
    delta = np.diff(positions, axis=0, prepend=positions[:1])
    motion = _smooth(np.linalg.norm(delta, axis=1), max(3, minimum_frames))
    home = (motion <= minimum_motion_m_per_frame) & (
        excursion <= maximum_home_distance_m
    )
    dwells = _runs(home, minimum_frames)
    if not dwells or dwells[0][0] > minimum_frames:
        return {"status": "unresolved", "reason": "no_initial_nested_stack_dwell"}
    if dwells[-1][1] < length - minimum_frames:
        return {"status": "unresolved", "reason": "no_final_nested_stack_dwell"}

    episodes: list[dict[str, Any]] = []
    rejected_cycles: list[dict[str, Any]] = []
    for left, right in zip(dwells, dwells[1:]):
        start, end = left[1], right[0]
        if end - start < 2 * minimum_frames:
            continue
        maximum_excursion = float(np.max(excursion[start:end]))
        if maximum_excursion < minimum_cycle_excursion_m:
            rejected_cycles.append(
                {
                    "range": [start, end],
                    "reason": "insufficient_bimanual_excursion",
                    "max_excursion_m": round(maximum_excursion, 6),
                }
            )
            continue
        boundary, transition_dwell = _phase_boundary(
            motion, start, end, minimum_frames
        )
        boundaries = [start, boundary, end]
        episodes.append(
            {
                "episode_start_frame": start,
                "episode_end_frame_exclusive": end,
                "atomic_boundaries": boundaries,
                "segment_lengths": np.diff(boundaries).astype(int).tolist(),
                "max_excursion_m": round(maximum_excursion, 6),
                "transition_low_motion_frames": transition_dwell,
            }
        )
    if not episodes:
        return {
            "status": "unresolved",
            "reason": "no_complete_home_to_home_cycle",
            "rejected_cycles": rejected_cycles,
        }
    result: dict[str, Any] = {
        "status": "candidate",
        "episodes": episodes,
        "candidate_count": len(episodes),
        "rejected_cycles": rejected_cycles,
        "requires_video_confirmation": True,
        "boundary_method": BOUNDARY_METHOD,
    }
    if len(episodes) == 1:
        result["boundaries"] = episodes[0]["atomic_boundaries"]
    return result
