from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json, write_json_atomic
from .task_spec import TaskSpec


@dataclass(frozen=True)
class ValidationSummary:
    source_segments: int
    pass_segments: int
    fail_segments: int
    episodes: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "status": "PASS",
            "source_segments": self.source_segments,
            "pass_segments": self.pass_segments,
            "fail_segments": self.fail_segments,
            "episodes": self.episodes,
        }


def _manifest_segments(manifest: dict[str, Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    for row in manifest.get("source_segments", []):
        index = int(row["source_segment_index"])
        if index in result:
            raise ValueError(f"duplicate source segment in manifest: {index}")
        result[index] = int(row["frames"])
    if not result:
        raise ValueError("manifest has no source_segments")
    return result


def decision_template(manifest: dict[str, Any], task_id: str) -> dict[str, Any]:
    source = str(manifest.get("source_bag") or manifest.get("source") or "")
    reviews = [
        {
            "source": source,
            "source_segment_index": index,
            "visual_status": "NEEDS_REVIEW",
            "episodes": [],
            "reviewer": "",
            "failure_reason": "",
        }
        for index in sorted(_manifest_segments(manifest))
    ]
    return {"schema_version": 2, "task_id": task_id, "reviews": reviews}


def write_decision_template(
    manifest_path: Path, output_path: Path, task_id: str
) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite decisions: {output_path}")
    write_json_atomic(output_path, decision_template(read_json(manifest_path), task_id))


def _normalize_episodes(review: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = review.get("episodes")
    if isinstance(episodes, list):
        return episodes
    if review.get("episode_start_frame") is None:
        return []
    return [
        {
            "context_start_frame": review.get(
                "context_start_frame", review["episode_start_frame"]
            ),
            "episode_start_frame": review["episode_start_frame"],
            "episode_end_frame_exclusive": review["episode_end_frame_exclusive"],
            "atomic_boundaries": review.get("atomic_boundaries", []),
        }
    ]


def validate_decisions(
    manifest: dict[str, Any], decisions: dict[str, Any], task: TaskSpec
) -> ValidationSummary:
    segments = _manifest_segments(manifest)
    rows = decisions.get("reviews")
    if not isinstance(rows, list):
        raise ValueError("decisions must contain a reviews array")
    seen: set[int] = set()
    pass_count = fail_count = episode_count = 0
    for row in rows:
        index = int(row["source_segment_index"])
        if index not in segments:
            raise ValueError(f"decision references unknown source segment: {index}")
        if index in seen:
            raise ValueError(f"duplicate decision for source segment: {index}")
        seen.add(index)
        status = str(row.get("visual_status", "")).upper()
        reviewer = str(row.get("reviewer", "")).strip()
        episodes = _normalize_episodes(row)
        if status == "PASS":
            if not reviewer:
                raise ValueError(f"segment {index}: PASS requires reviewer")
            if not episodes:
                raise ValueError(f"segment {index}: PASS requires at least one episode")
            previous_end = -1
            for episode_index, episode in enumerate(episodes):
                context_start = int(
                    episode.get("context_start_frame", episode["episode_start_frame"])
                )
                start = int(episode["episode_start_frame"])
                end = int(episode["episode_end_frame_exclusive"])
                boundaries = [int(value) for value in episode["atomic_boundaries"]]
                if (
                    context_start < 0
                    or context_start > start
                    or end > segments[index]
                    or end <= start
                ):
                    raise ValueError(
                        f"segment {index} episode {episode_index}: invalid frame range"
                    )
                if context_start < previous_end:
                    raise ValueError(
                        f"segment {index}: episode ranges overlap or are unsorted"
                    )
                if context_start < start and task.context_action is None:
                    raise ValueError(
                        f"segment {index} episode {episode_index}: "
                        "context frames require a task context_action"
                    )
                if len(boundaries) != len(task.actions) + 1:
                    raise ValueError(
                        f"segment {index} episode {episode_index}: expected "
                        f"{len(task.actions) + 1} boundaries"
                    )
                if boundaries[0] != start or boundaries[-1] != end:
                    raise ValueError(
                        f"segment {index} episode {episode_index}: "
                        "boundaries must cover episode"
                    )
                lengths = [
                    right - left for left, right in zip(boundaries, boundaries[1:])
                ]
                if any(length < task.minimum_segment_frames for length in lengths):
                    raise ValueError(
                        f"segment {index} episode {episode_index}: "
                        "atomic action is shorter "
                        f"than {task.minimum_segment_frames} frames"
                    )
                previous_end = end
                episode_count += 1
            pass_count += 1
        elif status == "FAIL":
            if episodes:
                raise ValueError(f"segment {index}: FAIL cannot contain episodes")
            if not str(row.get("failure_reason", "")).strip():
                raise ValueError(f"segment {index}: FAIL requires failure_reason")
            if not reviewer:
                raise ValueError(f"segment {index}: FAIL requires reviewer")
            fail_count += 1
        else:
            raise ValueError(f"segment {index}: unresolved visual_status {status!r}")
    missing = sorted(set(segments) - seen)
    if missing:
        raise ValueError(f"missing decisions for source segments: {missing}")
    return ValidationSummary(len(segments), pass_count, fail_count, episode_count)


def validate_decision_files(
    manifest_path: Path, decisions_path: Path, task_path: Path
) -> ValidationSummary:
    return validate_decisions(
        read_json(manifest_path), read_json(decisions_path), TaskSpec.load(task_path)
    )
