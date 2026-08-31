from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import read_json, write_json_atomic
from ..core.plugins import load_episode_pose_quality_auditor
from ..core.task_spec import TaskSpec
from .pose_coordinates import pose_matrices, rotation_matrices_to_quaternions


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _matrix(message: Any) -> np.ndarray:
    position = message.pose.position
    orientation = message.pose.orientation
    matrices, valid = pose_matrices(
        np.asarray([[position.x, position.y, position.z]], dtype=np.float64),
        np.asarray(
            [[orientation.x, orientation.y, orientation.z, orientation.w]],
            dtype=np.float64,
        ),
    )
    if not valid[0]:
        raise ValueError("pose topic contains an invalid rigid transform")
    return matrices[0]


def _read_pose_topics(
    source: Path, topics: set[str]
) -> dict[str, list[tuple[int, int, np.ndarray]]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:  # pragma: no cover - optional Insight adapter
        raise RuntimeError(
            "ROS 2 Python with rosbag storage support is required"
        ) from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {row.name: row.type for row in reader.get_all_topics_and_types()}
    missing = sorted(topics - set(topic_types))
    if missing:
        raise ValueError(f"source bag is missing pose topics: {missing}")
    rows: dict[str, list[tuple[int, int, np.ndarray]]] = {
        topic: [] for topic in topics
    }
    while reader.has_next():
        topic, raw, bag_stamp_ns = reader.read_next()
        if topic not in rows:
            continue
        message = deserialize_message(raw, get_message(topic_types[topic]))
        rows[topic].append(
            (int(bag_stamp_ns), _stamp_ns(message), _matrix(message))
        )
    for topic, values in rows.items():
        if len(values) < 2:
            raise ValueError(f"pose topic has insufficient messages: {topic}")
        if any(right[0] < left[0] for left, right in zip(values, values[1:])):
            raise ValueError(f"pose topic timestamps are not monotonic: {topic}")
    return rows


def _nearest_index(stamps: np.ndarray, target: int) -> int:
    index = int(np.searchsorted(stamps, target))
    candidates = [
        candidate
        for candidate in (index - 1, index)
        if 0 <= candidate < len(stamps)
    ]
    if not candidates:
        raise ValueError("pose topic has no timestamp candidate")
    return min(candidates, key=lambda candidate: abs(int(stamps[candidate]) - target))


def _slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    start = left / np.linalg.norm(left)
    end = right / np.linalg.norm(right)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    if dot > 0.9995:
        result = start + fraction * (end - start)
        return result / np.linalg.norm(result)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    scale = np.sin(angle)
    return (
        np.sin((1.0 - fraction) * angle) / scale * start
        + np.sin(fraction * angle) / scale * end
    )


def _interpolate_matrices(
    stamps: np.ndarray, matrices: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    quaternions = rotation_matrices_to_quaternions(matrices[:, :3, :3])
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(targets), axis=0)
    for output_index, target in enumerate(targets):
        right = int(np.searchsorted(stamps, target))
        if right <= 0:
            left = right = 0
        elif right >= len(stamps):
            left = right = len(stamps) - 1
        else:
            left = right - 1
        if left == right or stamps[right] == stamps[left]:
            fraction = 0.0
        else:
            fraction = float((target - stamps[left]) / (stamps[right] - stamps[left]))
        result[output_index, :3, 3] = (
            (1.0 - fraction) * matrices[left, :3, 3]
            + fraction * matrices[right, :3, 3]
        )
        quaternion = _slerp(quaternions[left], quaternions[right], fraction)
        rotation, valid = pose_matrices(
            np.zeros((1, 3), dtype=np.float64), quaternion[None]
        )
        if not valid[0]:
            raise ValueError("interpolated pose has an invalid rotation")
        result[output_index, :3, :3] = rotation[0, :3, :3]
    return result


def _role_comparison(
    manifest: dict[str, Any],
    manifest_pose: dict[str, Any],
    native_rows: list[tuple[int, int, np.ndarray]],
    global_rows: list[tuple[int, int, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_count = int(manifest["frame_count"])
    fps = float(manifest["fps"])
    start_stamp_ns = int(manifest["start_stamp_ns"])
    native_header_stamps = np.asarray(
        [row[1] for row in native_rows], dtype=np.int64
    )
    global_bag_stamps = np.asarray(
        [row[0] for row in global_rows], dtype=np.int64
    )
    global_header_stamps = np.asarray(
        [row[1] for row in global_rows], dtype=np.int64
    )
    native_matrices = np.asarray([row[2] for row in native_rows])
    global_matrices = np.asarray([row[2] for row in global_rows])
    manifest_positions = np.asarray(manifest_pose["positions"], dtype=np.float64)
    if manifest_positions.shape != (frame_count, 3):
        raise ValueError("review manifest pose positions do not match frame_count")
    paired_native: list[np.ndarray] = []
    raw_pair_skew_ms: list[float] = []
    for global_index, global_header_stamp_value in enumerate(global_header_stamps):
        global_header_stamp = int(global_header_stamp_value)
        native_index = _nearest_index(native_header_stamps, global_header_stamp)
        paired_native.append(native_matrices[native_index])
        raw_pair_skew_ms.append(
            abs(int(native_header_stamps[native_index]) - global_header_stamp)
            / 1_000_000
        )
    targets = np.asarray(
        [
            start_stamp_ns + round(frame * 1_000_000_000 / fps)
            for frame in range(frame_count)
        ],
        dtype=np.int64,
    )
    selected_native_array = _interpolate_matrices(
        global_bag_stamps, np.asarray(paired_native), targets
    )
    source_global_array = _interpolate_matrices(
        global_bag_stamps, global_matrices, targets
    )
    manifest_quaternions = np.asarray(
        manifest_pose["quaternions_xyzw"], dtype=np.float64
    )
    selected_global_array, manifest_rotation_valid = pose_matrices(
        manifest_positions, manifest_quaternions
    )
    if not np.all(manifest_rotation_valid):
        raise ValueError("review manifest contains invalid pose rotations")
    manifest_difference = np.linalg.norm(
        source_global_array[:, :3, 3] - manifest_positions, axis=1
    )
    maximum_manifest_difference = float(np.max(manifest_difference))
    if maximum_manifest_difference > 0.001:
        raise ValueError(
            "source Insight Global poses do not reproduce the review manifest: "
            f"maximum position difference is {maximum_manifest_difference:.6f} m"
        )
    comparison = {
        "review_frames": np.arange(frame_count, dtype=np.int64),
        "native_matrices": selected_native_array,
        "global_matrices": selected_global_array,
        "valid": np.asarray(manifest_pose["valid"], dtype=bool),
        "pair_skew_ms": np.interp(
            targets,
            global_bag_stamps,
            np.asarray(raw_pair_skew_ms, dtype=np.float64),
        ),
    }
    synchronization = {
        "maximum_native_global_pair_skew_ms": float(max(raw_pair_skew_ms)),
        "maximum_global_sample_interval_ms": float(
            np.max(np.diff(global_bag_stamps)) / 1_000_000
        ),
        "maximum_manifest_global_position_difference_m": maximum_manifest_difference,
    }
    return comparison, synchronization


def _pass_episodes(
    decisions: dict[str, Any], frame_count: int
) -> list[dict[str, int]]:
    episodes: list[dict[str, int]] = []
    previous_end = -1
    for review in decisions.get("reviews", []):
        if str(review.get("visual_status", "")).upper() != "PASS":
            continue
        rows = review.get("episodes")
        if not isinstance(rows, list):
            if review.get("episode_start_frame") is None:
                continue
            rows = [review]
        for row in rows:
            start = int(row["episode_start_frame"])
            end = int(row["episode_end_frame_exclusive"])
            if start < previous_end or end <= start or end > frame_count:
                raise ValueError(
                    "PASS episode ranges must be ordered, non-overlapping, and "
                    "inside the review manifest"
                )
            episodes.append(
                {
                    "episode_start_frame": start,
                    "episode_end_frame_exclusive": end,
                }
            )
            previous_end = end
    return episodes


def audit_insight_episode_pose_quality(
    *,
    source: Path,
    review_manifest_path: Path,
    decisions_path: Path,
    task_path: Path,
    output: Path,
    left_native_topic: str | None = None,
    right_native_topic: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pose-quality audit: {output}")
    manifest = read_json(review_manifest_path)
    decisions = read_json(decisions_path)
    task = TaskSpec.load(task_path)
    if str(decisions.get("task_id", "")) != task.task_id:
        raise ValueError("decisions task_id does not match task spec")
    if task.episode_pose_quality is None:
        raise ValueError(f"task {task.task_id} has no episode pose-quality audit")
    episodes = _pass_episodes(decisions, int(manifest["frame_count"]))
    poses_by_role = {
        str(row.get("role", "")): row for row in manifest.get("poses", [])
    }
    native_overrides = {
        "left_hand": left_native_topic,
        "right_hand": right_native_topic,
    }
    topic_pairs: dict[str, tuple[str, str]] = {}
    for role in ("left_hand", "right_hand"):
        row = poses_by_role.get(role)
        if row is None:
            raise ValueError(f"review manifest is missing {role} pose")
        name = str(row.get("name", "")).strip()
        if not name and not native_overrides[role]:
            raise ValueError(f"cannot infer native VIO topic for {role}")
        native_topic = native_overrides[role] or f"/{name}/camera/vio_100hz"
        topic_pairs[role] = (str(native_topic), str(row["topic"]))
    rows_by_topic = _read_pose_topics(
        source, {topic for pair in topic_pairs.values() for topic in pair}
    )
    comparisons: dict[str, dict[str, Any]] = {}
    synchronization: dict[str, dict[str, Any]] = {}
    for role, (native_topic, global_topic) in topic_pairs.items():
        comparison, sync = _role_comparison(
            manifest,
            poses_by_role[role],
            rows_by_topic[native_topic],
            rows_by_topic[global_topic],
        )
        comparisons[role] = comparison
        synchronization[role] = sync
    auditor = load_episode_pose_quality_auditor(task)
    result = auditor(
        comparisons,
        episodes,
        config=task.episode_pose_quality.config,
    )
    payload = {
        **result,
        "task_id": task.task_id,
        "source": str(source),
        "review_manifest": str(review_manifest_path),
        "review_manifest_sha256": _sha256(review_manifest_path),
        "decisions": str(decisions_path),
        "decisions_sha256": _sha256(decisions_path),
        "task_spec": str(task_path),
        "task_spec_sha256": _sha256(task_path),
        "topics": {
            role: {"native_vio": pair[0], "insight_global": pair[1]}
            for role, pair in topic_pairs.items()
        },
        "synchronization": synchronization,
    }
    write_json_atomic(output, payload)
    return payload
