from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .pose_coordinates import pose_matrices


def camera_info_topic(image_topic: str) -> str:
    suffixes = (
        "/image_rect_raw/compressed",
        "/image_rect_raw",
        "/image_raw/compressed",
    )
    for suffix in suffixes:
        if image_topic.endswith(suffix):
            return image_topic[: -len(suffix)] + "/camera_info"
    raise ValueError("cannot infer camera_info topic from head image topic")


def read_ros_frame_calibration(
    source: Path,
    *,
    pose_topic: str,
    camera_info_topic_name: str,
    required_static_path: tuple[str, str] | None = None,
) -> dict[str, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:  # pragma: no cover - optional ROS adapter
        raise RuntimeError("ROS 2 Python with rosbag storage support is required") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {row.name: row.type for row in reader.get_all_topics_and_types()}
    required = {pose_topic, camera_info_topic_name, "/tf_static"}
    missing = sorted(required - set(types))
    if missing:
        raise ValueError(f"source bag is missing calibration topics: {missing}")
    result: dict[str, Any] = {"static_transforms": []}
    seen: set[str] = set()
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic not in required or (topic != "/tf_static" and topic in seen):
            continue
        message = deserialize_message(raw, get_message(types[topic]))
        if topic == pose_topic:
            result["global_frame"] = str(message.header.frame_id)
        elif topic == camera_info_topic_name:
            result.update(
                {
                    "camera_frame": str(message.header.frame_id),
                    "camera_matrix": np.asarray(
                        message.k, dtype=np.float64
                    ).reshape(3, 3),
                    "distortion": np.asarray(message.d, dtype=np.float64),
                    "distortion_model": str(message.distortion_model),
                }
            )
        else:
            for transform in message.transforms:
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                matrix, valid = pose_matrices(
                    np.asarray([[translation.x, translation.y, translation.z]]),
                    np.asarray([[rotation.x, rotation.y, rotation.z, rotation.w]]),
                )
                if not valid[0]:
                    raise ValueError("tf_static contains an invalid transform")
                result["static_transforms"].append(
                    (
                        str(transform.header.frame_id),
                        str(transform.child_frame_id),
                        matrix[0],
                    )
                )
        seen.add(topic)
        if seen != required:
            continue
        if required_static_path is None:
            break
        try:
            lookup_static_transform(
                result["static_transforms"],
                required_static_path[0],
                required_static_path[1],
            )
        except ValueError:
            # Static transforms may be split across multiple latched messages.
            continue
        break
    return result


def lookup_static_transform(
    rows: list[tuple[str, str, np.ndarray]], source_frame: str, target_frame: str
) -> np.ndarray:
    if source_frame == target_frame:
        return np.eye(4, dtype=np.float64)
    graph: dict[str, list[tuple[str, np.ndarray]]] = {}
    for parent, child, parent_from_child in rows:
        graph.setdefault(parent, []).append((child, parent_from_child))
        graph.setdefault(child, []).append((parent, np.linalg.inv(parent_from_child)))
    queue: deque[tuple[str, np.ndarray]] = deque(
        [(source_frame, np.eye(4, dtype=np.float64))]
    )
    seen = {source_frame}
    while queue:
        frame, source_from_frame = queue.popleft()
        for neighbor, frame_from_neighbor in graph.get(frame, []):
            if neighbor in seen:
                continue
            source_from_neighbor = source_from_frame @ frame_from_neighbor
            if neighbor == target_frame:
                return source_from_neighbor
            seen.add(neighbor)
            queue.append((neighbor, source_from_neighbor))
    raise ValueError(
        f"no tf_static path from {source_frame!r} to {target_frame!r}"
    )


def head_frame_calibration(
    source: Path,
    manifest: dict[str, Any],
    *,
    head_pose_child_frame: str | None = None,
    static_calibration_source: Path | None = None,
) -> dict[str, Any]:
    cameras = [
        row for row in manifest.get("source_cameras", []) if row.get("role") == "head"
    ]
    poses = [row for row in manifest.get("poses", []) if row.get("role") == "head"]
    if len(cameras) != 1 or len(poses) != 1:
        raise ValueError(
            "review manifest must contain one head camera and one head pose"
        )
    camera_row, pose_row = cameras[0], poses[0]
    image_topic = str(camera_row["topic"])
    pose_topic = str(pose_row["topic"])
    camera_name = str(camera_row.get("name", ""))
    pose_child = head_pose_child_frame or f"{camera_name}_camera_imu"
    calibration_source = static_calibration_source or source
    ros = read_ros_frame_calibration(
        source,
        pose_topic=pose_topic,
        camera_info_topic_name=camera_info_topic(image_topic),
        required_static_path=(
            (pose_child, f"{camera_name}_camera_rgb")
            if calibration_source == source
            else None
        ),
    )
    if calibration_source != source:
        reference = read_ros_frame_calibration(
            calibration_source,
            pose_topic=pose_topic,
            camera_info_topic_name=camera_info_topic(image_topic),
            required_static_path=(pose_child, f"{camera_name}_camera_rgb"),
        )
        if reference.get("camera_frame") != ros.get("camera_frame"):
            raise ValueError(
                "static calibration source uses a different head camera frame"
            )
        ros["static_transforms"] = reference["static_transforms"]
    ros.update(
        {
            "camera_row": camera_row,
            "pose_row": pose_row,
            "image_topic": image_topic,
            "pose_topic": pose_topic,
            "head_pose_child_frame": pose_child,
            "static_calibration_source": calibration_source.name,
            "static_calibration_borrowed": calibration_source != source,
            "tracking_from_camera": lookup_static_transform(
                ros["static_transforms"], pose_child, ros["camera_frame"]
            ),
        }
    )
    return ros
