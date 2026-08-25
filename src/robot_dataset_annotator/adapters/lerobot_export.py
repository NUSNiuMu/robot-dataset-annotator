from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from ..core.decisions import validate_decisions
from ..core.io import read_json, write_json_atomic
from ..core.task_spec import TaskSpec
from .insight_frames import head_frame_calibration
from .pose_coordinates import pose_matrices, pose_state_from_matrices


HAND_STATE_NAMES = tuple(
    f"{hand}.{component}"
    for hand in ("left_hand", "right_hand")
    for component in (
        "x_m",
        "y_m",
        "z_m",
        "rotation_6d_0",
        "rotation_6d_1",
        "rotation_6d_2",
        "rotation_6d_3",
        "rotation_6d_4",
        "rotation_6d_5",
    )
)
HEAD_STATE_NAMES = tuple(
    f"head.{component}"
    for component in (
        "x_m",
        "y_m",
        "z_m",
        "rotation_6d_0",
        "rotation_6d_1",
        "rotation_6d_2",
        "rotation_6d_3",
        "rotation_6d_4",
        "rotation_6d_5",
    )
)


@dataclass(frozen=True)
class FramePlan:
    source_frame: int
    episode_index: int
    atomic_action_index: int
    left_hand_subtask_index: int
    right_hand_subtask_index: int
    task_progress: float
    left_hand_subtask_progress: float
    right_hand_subtask_progress: float
    task: str
    state: np.ndarray
    state_valid: np.ndarray
    head_pose: np.ndarray
    head_pose_valid: np.ndarray
    head_camera_pose_global: np.ndarray
    head_camera_pose_global_valid: np.ndarray
    action: np.ndarray
    action_valid: np.ndarray


@dataclass(frozen=True)
class CameraSpec:
    key: str
    topic: str
    width: int
    height: int


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_6d(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _pose_state(
    manifest: dict[str, Any],
    role: str,
    pose_from_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    poses = [pose for pose in manifest.get("poses", []) if pose.get("role") == role]
    if len(poses) != 1:
        raise ValueError(f"expected exactly one {role!r} pose stream")
    pose = poses[0]
    positions = np.asarray(pose.get("positions"), dtype=np.float64)
    quaternions = np.asarray(pose.get("quaternions_xyzw"), dtype=np.float64)
    stream_valid = np.asarray(pose.get("valid"), dtype=bool)
    frame_count = int(manifest["frame_count"])
    if positions.shape != (frame_count, 3):
        raise ValueError(f"{role} positions must have shape ({frame_count}, 3)")
    if quaternions.shape != (frame_count, 4):
        raise ValueError(f"{role} quaternions must have shape ({frame_count}, 4)")
    if stream_valid.shape != (frame_count,):
        raise ValueError(f"{role} validity must have shape ({frame_count},)")
    rotation, rotation_valid = _rotation_6d(quaternions)
    position_valid = stream_valid & np.isfinite(positions).all(axis=1)
    if pose_from_camera is None:
        state = np.concatenate((positions, rotation), axis=1).astype(np.float32)
    else:
        global_from_pose, matrix_valid = pose_matrices(positions, quaternions)
        global_from_camera = global_from_pose @ pose_from_camera
        state = pose_state_from_matrices(global_from_camera).astype(np.float32)
        position_valid &= matrix_valid
    valid = np.concatenate(
        (
            np.repeat(position_valid[:, None], 3, axis=1),
            np.repeat((stream_valid & rotation_valid)[:, None], 6, axis=1),
        ),
        axis=1,
    )
    state[~valid] = 0.0
    return state, valid


def build_frame_plan(
    review_manifest: dict[str, Any],
    decisions: dict[str, Any],
    task: TaskSpec,
    *,
    head_pose_from_camera: np.ndarray | None = None,
) -> list[FramePlan]:
    left, left_valid = _pose_state(review_manifest, "left_hand")
    right, right_valid = _pose_state(review_manifest, "right_hand")
    head, head_valid = _pose_state(review_manifest, "head")
    head_camera, head_camera_valid = _pose_state(
        review_manifest, "head", head_pose_from_camera
    )
    hands = np.concatenate((left, right), axis=1)
    hands_valid = np.concatenate((left_valid, right_valid), axis=1)

    reviews = decisions.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("decisions must contain a reviews array")
    episodes: list[dict[str, Any]] = []
    for review in reviews:
        if str(review.get("visual_status", "")).upper() != "PASS":
            continue
        rows = review.get("episodes")
        if isinstance(rows, list):
            episodes.extend(rows)
        elif review.get("episode_start_frame") is not None:
            episodes.append(
                {
                    "context_start_frame": review.get(
                        "context_start_frame", review["episode_start_frame"]
                    ),
                    "episode_start_frame": review["episode_start_frame"],
                    "episode_end_frame_exclusive": review[
                        "episode_end_frame_exclusive"
                    ],
                    "atomic_boundaries": review.get("atomic_boundaries", []),
                }
            )

    plans: list[FramePlan] = []
    for episode_index, episode in enumerate(episodes):
        boundaries = [int(value) for value in episode["atomic_boundaries"]]
        episode_start = int(episode.get("episode_start_frame", boundaries[0]))
        episode_end = int(
            episode.get("episode_end_frame_exclusive", boundaries[-1])
        )
        context_start = int(episode.get("context_start_frame", episode_start))
        episode_plans: list[FramePlan] = []

        ranges: list[tuple[int, Any, int, int]] = []
        if context_start < episode_start:
            if task.context_action is None:
                raise ValueError("context frames require a task context_action")
            ranges.append((-1, task.context_action, context_start, episode_start))
        ranges.extend(
            (
                action_index,
                action,
                boundaries[action_index],
                boundaries[action_index + 1],
            )
            for action_index, action in enumerate(task.actions)
        )
        task_denominator = max(1, episode_end - episode_start - 1)
        for action_index, action, range_start, range_end in ranges:
            subtask_denominator = max(1, range_end - range_start - 1)
            for source_frame in range(range_start, range_end):
                task_progress = (
                    0.0
                    if action_index == -1
                    else (source_frame - episode_start) / task_denominator
                )
                subtask_progress = (
                    0.0
                    if range_end - range_start == 1
                    else (source_frame - range_start) / subtask_denominator
                )
                episode_plans.append(
                    FramePlan(
                        source_frame=source_frame,
                        episode_index=episode_index,
                        atomic_action_index=action_index,
                        left_hand_subtask_index=task.subtask_index(
                            action_index, "left_hand"
                        ),
                        right_hand_subtask_index=task.subtask_index(
                            action_index, "right_hand"
                        ),
                        task_progress=float(task_progress),
                        left_hand_subtask_progress=float(subtask_progress),
                        right_hand_subtask_progress=float(subtask_progress),
                        task=action.instruction,
                        state=hands[source_frame],
                        state_valid=hands_valid[source_frame],
                        head_pose=head[source_frame],
                        head_pose_valid=head_valid[source_frame],
                        head_camera_pose_global=head_camera[source_frame],
                        head_camera_pose_global_valid=head_camera_valid[source_frame],
                        action=np.empty(0, dtype=np.float32),
                        action_valid=np.empty(0, dtype=bool),
                    )
                )
        for index, plan in enumerate(episode_plans):
            target = episode_plans[min(index + 1, len(episode_plans) - 1)]
            episode_plans[index] = FramePlan(
                source_frame=plan.source_frame,
                episode_index=plan.episode_index,
                atomic_action_index=plan.atomic_action_index,
                left_hand_subtask_index=plan.left_hand_subtask_index,
                right_hand_subtask_index=plan.right_hand_subtask_index,
                task_progress=plan.task_progress,
                left_hand_subtask_progress=plan.left_hand_subtask_progress,
                right_hand_subtask_progress=plan.right_hand_subtask_progress,
                task=plan.task,
                state=plan.state,
                state_valid=plan.state_valid,
                head_pose=plan.head_pose,
                head_pose_valid=plan.head_pose_valid,
                head_camera_pose_global=plan.head_camera_pose_global,
                head_camera_pose_global_valid=plan.head_camera_pose_global_valid,
                action=target.state.copy(),
                action_valid=target.state_valid.copy(),
            )
        plans.extend(episode_plans)
    if not plans:
        raise ValueError("decisions contain no accepted frames")
    return plans


def _camera_specs(manifest: dict[str, Any]) -> tuple[CameraSpec, ...]:
    specs: list[CameraSpec] = []
    for role in ("left_hand", "right_hand", "head"):
        rows = [
            row
            for row in manifest.get("source_cameras", [])
            if row.get("role") == role
        ]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one {role!r} source camera")
        row = rows[0]
        specs.append(
            CameraSpec(
                key=f"observation.images.{role}",
                topic=str(row["topic"]),
                width=int(row["source_width"]),
                height=int(row["source_height"]),
            )
        )
    return tuple(specs)


def _decode_ros_image(raw: bytes, message_type: str) -> np.ndarray:
    try:
        import cv2
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:  # pragma: no cover - optional ROS adapter
        raise RuntimeError(
            "ROS 2 Python and OpenCV are required for Insight export"
        ) from exc

    message = deserialize_message(raw, get_message(message_type))
    if message_type == "sensor_msgs/msg/CompressedImage":
        bgr = cv2.imdecode(
            np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if bgr is None:
            raise ValueError("OpenCV could not decode a compressed camera frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if message_type != "sensor_msgs/msg/Image":
        raise ValueError(f"unsupported camera message type: {message_type}")

    height, width, step = int(message.height), int(message.width), int(message.step)
    encoding = str(message.encoding).lower()
    data = np.frombuffer(message.data, dtype=np.uint8)
    if encoding == "nv12":
        rows = height * 3 // 2
        image = data.reshape(rows, step)[:, :width]
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_NV12)
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(
        encoding
    )
    if channels is None:
        raise ValueError(f"unsupported raw image encoding: {message.encoding}")
    row_width = width * channels
    image = data.reshape(height, step)[:, :row_width].reshape(height, width, channels)
    if encoding == "mono8":
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    if encoding == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image


def _stream_synchronized_images(
    source: Path,
    cameras: tuple[CameraSpec, ...],
    target_stamps_ns: list[int],
    max_skew_ns: int,
    emit: Callable[[int, str, np.ndarray, int], None],
) -> dict[str, dict[str, float | int]]:
    try:
        import rosbag2_py
    except ImportError as exc:  # pragma: no cover - optional ROS adapter
        raise RuntimeError("rosbag2_py with MCAP support is required") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {row.name: row.type for row in reader.get_all_topics_and_types()}
    camera_by_topic = {camera.topic: camera for camera in cameras}
    missing_topics = sorted(set(camera_by_topic) - set(topic_types))
    if missing_topics:
        raise ValueError(f"source bag is missing camera topics: {missing_topics}")

    pointers = {camera.topic: 0 for camera in cameras}
    previous: dict[str, tuple[int, bytes]] = {}
    selected_stamps: dict[str, list[int]] = {camera.key: [] for camera in cameras}

    def assign(camera: CameraSpec, stamp_ns: int, raw: bytes, stop_ns: int) -> None:
        pointer = pointers[camera.topic]
        chosen: list[int] = []
        while pointer < len(target_stamps_ns) and target_stamps_ns[pointer] <= stop_ns:
            skew = abs(target_stamps_ns[pointer] - stamp_ns)
            if skew > max_skew_ns:
                raise ValueError(
                    f"{camera.topic} target {pointer} exceeds synchronization "
                    "tolerance: "
                    f"{skew / 1_000_000:.3f} ms"
                )
            chosen.append(pointer)
            pointer += 1
        if chosen:
            image = _decode_ros_image(raw, topic_types[camera.topic])
            if image.shape != (camera.height, camera.width, 3):
                raise ValueError(
                    f"{camera.topic} decoded shape {image.shape} does not match "
                    f"{(camera.height, camera.width, 3)}"
                )
            for target_index in chosen:
                emit(target_index, camera.key, image, stamp_ns)
                selected_stamps[camera.key].append(stamp_ns)
        pointers[camera.topic] = pointer

    while reader.has_next():
        topic, raw, stamp_ns = reader.read_next()
        camera = camera_by_topic.get(topic)
        if camera is None:
            continue
        current = (int(stamp_ns), bytes(raw))
        prior = previous.get(topic)
        if prior is not None:
            midpoint = (prior[0] + current[0]) // 2
            assign(camera, prior[0], prior[1], midpoint)
        previous[topic] = current

    for camera in cameras:
        prior = previous.get(camera.topic)
        if prior is None:
            raise ValueError(f"source bag has no frames for {camera.topic}")
        assign(camera, prior[0], prior[1], target_stamps_ns[-1])
        if pointers[camera.topic] != len(target_stamps_ns):
            raise ValueError(f"source bag ended before all frames for {camera.topic}")

    result: dict[str, dict[str, float | int]] = {}
    for camera in cameras:
        stamps = selected_stamps[camera.key]
        skews = [
            abs(target - selected) / 1_000_000
            for target, selected in zip(target_stamps_ns, stamps)
        ]
        result[camera.key] = {
            "frames": len(stamps),
            "duplicate_frames": len(stamps) - len(set(stamps)),
            "max_skew_ms": round(max(skews), 3),
        }
    return result


def _features(cameras: tuple[CameraSpec, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (18,),
            "names": list(HAND_STATE_NAMES),
        },
        "observation.state_valid": {
            "dtype": "uint8",
            "shape": (18,),
            "names": list(HAND_STATE_NAMES),
        },
        "observation.head_pose": {
            "dtype": "float32",
            "shape": (9,),
            "names": list(HEAD_STATE_NAMES),
        },
        "observation.head_camera_pose_global": {
            "dtype": "float32",
            "shape": (9,),
            "names": list(HEAD_STATE_NAMES),
        },
        "observation.head_camera_pose_global_valid": {
            "dtype": "uint8",
            "shape": (9,),
            "names": list(HEAD_STATE_NAMES),
        },
        "observation.head_pose_valid": {
            "dtype": "uint8",
            "shape": (9,),
            "names": list(HEAD_STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (18,),
            "names": list(HAND_STATE_NAMES),
        },
        "action_is_valid": {
            "dtype": "uint8",
            "shape": (18,),
            "names": list(HAND_STATE_NAMES),
        },
        "annotation.source_frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["source_frame_index"],
        },
        "annotation.atomic_action_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["atomic_action_index"],
        },
        "annotation.left_hand_subtask_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["left_hand_subtask_index"],
        },
        "annotation.right_hand_subtask_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["right_hand_subtask_index"],
        },
        "annotation.task_progress": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["task_progress"],
        },
        "annotation.left_hand_subtask_progress": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["left_hand_subtask_progress"],
        },
        "annotation.right_hand_subtask_progress": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["right_hand_subtask_progress"],
        },
    }
    for camera in cameras:
        result[camera.key] = {
            "dtype": "video",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def export_insight_lerobot(
    *,
    source: Path,
    review_manifest_path: Path,
    annotation_manifest_path: Path,
    decisions_path: Path,
    task_path: Path,
    output: Path,
    repo_id: str,
    max_skew_ms: float | None = None,
    vcodec: str = "h264",
    head_pose_child_frame: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    review_manifest = read_json(review_manifest_path)
    annotation_manifest = read_json(annotation_manifest_path)
    decisions = read_json(decisions_path)
    task = TaskSpec.load(task_path)
    validation = validate_decisions(annotation_manifest, decisions, task)
    if len(annotation_manifest.get("source_segments", [])) != 1:
        raise ValueError("Insight LeRobot export currently requires one source segment")
    head_calibration = head_frame_calibration(
        source,
        review_manifest,
        head_pose_child_frame=head_pose_child_frame,
    )
    plans = build_frame_plan(
        review_manifest,
        decisions,
        task,
        head_pose_from_camera=head_calibration["tracking_from_camera"],
    )
    cameras = _camera_specs(review_manifest)
    fps = float(review_manifest["fps"])
    if fps <= 0:
        raise ValueError("review manifest fps must be positive")
    start_stamp_ns = int(review_manifest["start_stamp_ns"])
    target_stamps = [
        start_stamp_ns + round(plan.source_frame * 1_000_000_000 / fps)
        for plan in plans
    ]
    if max_skew_ms is not None and max_skew_ms <= 0:
        raise ValueError("max_skew_ms must be positive")
    tolerance_ns = round((max_skew_ms or (1000.0 / fps)) * 1_000_000)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - optional export dependency
        raise RuntimeError("install robot-dataset-annotator[lerobot]") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=round(fps),
            root=temporary,
            robot_type="human_hand_tracking",
            features=_features(cameras),
            use_videos=True,
            image_writer_threads=4,
            vcodec=vcodec,
        )
        images: dict[int, dict[str, np.ndarray]] = {}
        next_to_write = 0

        def flush_ready() -> None:
            nonlocal next_to_write
            while (
                next_to_write < len(plans)
                and len(images.get(next_to_write, {})) == len(cameras)
            ):
                plan = plans[next_to_write]
                frame: dict[str, Any] = {
                    "observation.state": plan.state,
                    "observation.state_valid": plan.state_valid.astype(np.uint8),
                    "observation.head_pose": plan.head_pose,
                    "observation.head_camera_pose_global": (
                        plan.head_camera_pose_global
                    ),
                    "observation.head_camera_pose_global_valid": (
                        plan.head_camera_pose_global_valid.astype(np.uint8)
                    ),
                    "observation.head_pose_valid": plan.head_pose_valid.astype(
                        np.uint8
                    ),
                    "action": plan.action,
                    "action_is_valid": plan.action_valid.astype(np.uint8),
                    "annotation.source_frame_index": np.asarray(
                        [plan.source_frame], dtype=np.int64
                    ),
                    "annotation.atomic_action_index": np.asarray(
                        [plan.atomic_action_index], dtype=np.int64
                    ),
                    "annotation.left_hand_subtask_index": np.asarray(
                        [plan.left_hand_subtask_index], dtype=np.int64
                    ),
                    "annotation.right_hand_subtask_index": np.asarray(
                        [plan.right_hand_subtask_index], dtype=np.int64
                    ),
                    "annotation.task_progress": np.asarray(
                        [plan.task_progress], dtype=np.float32
                    ),
                    "annotation.left_hand_subtask_progress": np.asarray(
                        [plan.left_hand_subtask_progress], dtype=np.float32
                    ),
                    "annotation.right_hand_subtask_progress": np.asarray(
                        [plan.right_hand_subtask_progress], dtype=np.float32
                    ),
                    "task": plan.task,
                    **images.pop(next_to_write),
                }
                dataset.add_frame(frame)
                is_episode_end = next_to_write + 1 == len(plans) or (
                    plans[next_to_write + 1].episode_index != plan.episode_index
                )
                if is_episode_end:
                    dataset.save_episode(parallel_encoding=True)
                next_to_write += 1

        def emit(index: int, key: str, image: np.ndarray, stamp_ns: int) -> None:
            images.setdefault(index, {})[key] = image
            flush_ready()

        synchronization = _stream_synchronized_images(
            source, cameras, target_stamps, tolerance_ns, emit
        )
        flush_ready()
        if next_to_write != len(plans):
            raise ValueError(
                f"only wrote {next_to_write} of {len(plans)} planned frames"
            )
        dataset.finalize()

        action_counts = {
            action.key: sum(plan.atomic_action_index == index for plan in plans)
            for index, action in enumerate(task.actions)
        }
        context_frames = sum(plan.atomic_action_index == -1 for plan in plans)
        export_manifest = {
            "schema_version": 1,
            "status": "PASS",
            "format": "LeRobotDataset-v3.0",
            "lerobot_version": version("lerobot"),
            "code_revision": current_code_revision(Path.cwd()),
            "repo_id": repo_id,
            "video_codec": vcodec,
            "task_id": task.task_id,
            "episodes": validation.episodes,
            "frames": len(plans),
            "fps": fps,
            "duration_s": len(plans) / fps,
            "atomic_action_frames": action_counts,
            "context_frames": context_frames,
            "subtask_semantics": task.subtask_catalog(),
            "camera_synchronization": synchronization,
            "action_semantics": (
                "Next-frame dual-hand 6D pose target in the source tracking frame; "
                "the final frame repeats the current target. Not robot-retargeted and "
                "contains no gripper command."
            ),
            "pose_semantics": {
                "observation.state": (
                    "Left- and right-hand global poses from the synchronized "
                    "review manifest."
                ),
                "observation.head_pose": (
                    "Head tracking-frame global pose from the synchronized review "
                    "manifest."
                ),
                "observation.head_camera_pose_global": (
                    "Head RGB-camera global pose after applying the recorded static "
                    "tracking-to-camera transform."
                ),
            },
            "head_camera_frames": {
                "global_frame": head_calibration["global_frame"],
                "head_pose_child_frame": head_calibration[
                    "head_pose_child_frame"
                ],
                "head_camera_frame": head_calibration["camera_frame"],
                "head_pose_from_camera": head_calibration[
                    "tracking_from_camera"
                ].tolist(),
            },
            "progress_semantics": {
                "annotation.task_progress": (
                    "Zero during preserved context; linear from 0 to 1 across the "
                    "reviewed manipulation interval."
                ),
                "annotation.left_hand_subtask_progress": (
                    "Linear from 0 to 1 inside the current left-hand subtask."
                ),
                "annotation.right_hand_subtask_progress": (
                    "Linear from 0 to 1 inside the current right-hand subtask."
                ),
            },
            "source": {
                "bag": source.name,
                "review_manifest_sha256": _sha256(review_manifest_path),
                "annotation_manifest_sha256": _sha256(annotation_manifest_path),
                "decisions_sha256": _sha256(decisions_path),
                "task_spec_sha256": _sha256(task_path),
            },
        }
        write_json_atomic(temporary / "rda" / "export_manifest.json", export_manifest)
        os.replace(temporary, output)
        return export_manifest
    except Exception:
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def current_code_revision(workspace: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None
