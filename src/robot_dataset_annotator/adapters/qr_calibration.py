from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import read_json, write_json_atomic
from .insight_frames import head_frame_calibration
from .lerobot_export import CameraSpec, _stream_synchronized_images
from .pose_coordinates import pose_matrices, validate_rigid_transform


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transform_inlier_indices(matrices: list[np.ndarray]) -> list[int]:
    if not matrices:
        return []
    translations = np.asarray([matrix[:3, 3] for matrix in matrices])
    center = np.median(translations, axis=0)
    distances = np.linalg.norm(translations - center, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = max(0.02, median_distance + 3.0 * max(mad, 1e-6))
    translation_inliers = np.flatnonzero(distances <= threshold)
    if not len(translation_inliers):
        return []

    rotations = np.asarray([matrix[:3, :3] for matrix in matrices])
    pairwise_angles = np.empty((len(rotations), len(rotations)), dtype=np.float64)
    for left_index, left_rotation in enumerate(rotations):
        for right_index, right_rotation in enumerate(rotations):
            relative = left_rotation.T @ right_rotation
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            pairwise_angles[left_index, right_index] = np.arccos(cosine)
    rotation_medoid = translation_inliers[
        np.argmin(
            np.median(
                pairwise_angles[
                    np.ix_(translation_inliers, translation_inliers)
                ],
                axis=1,
            )
        )
    ]
    angle_distances = pairwise_angles[rotation_medoid]
    candidate_angles = angle_distances[translation_inliers]
    median_angle = float(np.median(candidate_angles))
    angle_mad = float(np.median(np.abs(candidate_angles - median_angle)))
    angle_threshold = max(
        np.deg2rad(5.0), median_angle + 3.0 * max(angle_mad, 1e-6)
    )
    return [
        int(index)
        for index in translation_inliers
        if angle_distances[index] <= angle_threshold
    ]


def _average_rigid_transforms(matrices: list[np.ndarray]) -> np.ndarray:
    inliers = _transform_inlier_indices(matrices)
    if not inliers:
        raise ValueError("all QR pose estimates were rejected as transform outliers")
    selected = [matrices[index] for index in inliers]
    rotation_mean = np.mean([matrix[:3, :3] for matrix in selected], axis=0)
    left, _, right = np.linalg.svd(rotation_mean)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.median([matrix[:3, 3] for matrix in selected], axis=0)
    return result


def _marker_corner_detector(
    cv2: Any,
    *,
    marker_type: str,
    aruco_dictionary: str,
    aruco_marker_id: int | None,
):
    if marker_type == "qr_code":
        detector = cv2.QRCodeDetector()

        def detect_qr(image: np.ndarray) -> np.ndarray | None:
            detected, points = detector.detect(image)
            if not detected or points is None:
                return None
            return np.asarray(points, dtype=np.float64).reshape(4, 2)

        return detect_qr
    if marker_type != "aruco":
        raise ValueError(f"unsupported marker_type: {marker_type}")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support is required for ArUco calibration")
    dictionary_id = getattr(cv2.aruco, aruco_dictionary, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {aruco_dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters")
        else cv2.aruco.DetectorParameters_create()
    )
    detector = (
        cv2.aruco.ArucoDetector(dictionary, parameters)
        if hasattr(cv2.aruco, "ArucoDetector")
        else None
    )

    def detect_aruco(image: np.ndarray) -> np.ndarray | None:
        if detector is None:  # pragma: no cover - OpenCV before 4.7
            corners, ids, _ = cv2.aruco.detectMarkers(
                image, dictionary, parameters=parameters
            )
        else:
            corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            return None
        flat_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if aruco_marker_id is None:
            matches = np.arange(len(flat_ids)) if len(flat_ids) == 1 else []
        else:
            matches = np.flatnonzero(flat_ids == aruco_marker_id)
        if len(matches) != 1:
            return None
        return np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)

    return detect_aruco


def estimate_qr_transform(
    *,
    source: Path,
    review_manifest_path: Path,
    output: Path,
    marker_size_m: float,
    frame_start: int = 0,
    frame_end_exclusive: int | None = None,
    head_pose_child_frame: str | None = None,
    minimum_detections: int = 3,
    maximum_reprojection_error_px: float = 3.0,
    marker_type: str = "qr_code",
    aruco_dictionary: str = "DICT_4X4_50",
    aruco_marker_id: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QR transform: {output}")
    if marker_size_m <= 0:
        raise ValueError("marker_size_m must be positive")
    if minimum_detections < 1:
        raise ValueError("minimum_detections must be positive")
    manifest = read_json(review_manifest_path)
    frame_count = int(manifest["frame_count"])
    end = frame_count if frame_end_exclusive is None else int(frame_end_exclusive)
    if frame_start < 0 or end > frame_count or end <= frame_start:
        raise ValueError("invalid QR calibration frame range")
    ros = head_frame_calibration(
        source, manifest, head_pose_child_frame=head_pose_child_frame
    )
    if ros["distortion_model"] not in {"", "plumb_bob", "rational_polynomial"}:
        raise ValueError(
            f"unsupported camera distortion model: {ros['distortion_model']}"
        )
    camera_row = ros["camera_row"]
    pose_row = ros["pose_row"]
    image_topic = ros["image_topic"]
    pose_topic = ros["pose_topic"]
    pose_child = ros["head_pose_child_frame"]
    tracking_from_camera = ros["tracking_from_camera"]

    positions = np.asarray(pose_row["positions"], dtype=np.float64)
    quaternions = np.asarray(pose_row["quaternions_xyzw"], dtype=np.float64)
    stream_valid = np.asarray(pose_row["valid"], dtype=bool)
    global_from_tracking, pose_valid = pose_matrices(positions, quaternions)
    pose_valid &= stream_valid

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional Insight adapter
        raise RuntimeError("OpenCV is required for QR calibration") from exc
    half = marker_size_m / 2.0
    object_points = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    detect_marker = _marker_corner_detector(
        cv2,
        marker_type=marker_type,
        aruco_dictionary=aruco_dictionary,
        aruco_marker_id=aruco_marker_id,
    )
    estimates: list[np.ndarray] = []
    accepted_frames: list[int] = []
    reprojection_errors: list[float] = []

    def emit(index: int, key: str, image: np.ndarray, stamp_ns: int) -> None:
        source_frame = frame_start + index
        if not pose_valid[source_frame]:
            return
        image_points = detect_marker(image)
        if image_points is None:
            return
        height, width = image.shape[:2]
        if np.any(image_points[:, 0] <= 1.0) or np.any(
            image_points[:, 0] >= width - 2.0
        ):
            return
        if np.any(image_points[:, 1] <= 1.0) or np.any(
            image_points[:, 1] >= height - 2.0
        ):
            return
        ok, rotation_vector, translation = cv2.solvePnP(
            object_points,
            image_points,
            ros["camera_matrix"],
            ros["distortion"],
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return
        projected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation,
            ros["camera_matrix"],
            ros["distortion"],
        )
        squared_errors = np.sum(
            (projected.reshape(4, 2) - image_points) ** 2, axis=1
        )
        error = float(np.sqrt(np.mean(squared_errors)))
        if error > maximum_reprojection_error_px:
            return
        camera_from_qr = np.eye(4, dtype=np.float64)
        camera_from_qr[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        camera_from_qr[:3, 3] = translation.reshape(3)
        estimates.append(
            global_from_tracking[source_frame]
            @ tracking_from_camera
            @ camera_from_qr
        )
        accepted_frames.append(source_frame)
        reprojection_errors.append(error)

    fps = float(manifest["fps"])
    start_stamp_ns = int(manifest["start_stamp_ns"])
    target_stamps = [
        start_stamp_ns + round(frame * 1_000_000_000 / fps)
        for frame in range(frame_start, end)
    ]
    camera = CameraSpec(
        key="head",
        topic=image_topic,
        width=int(camera_row["source_width"]),
        height=int(camera_row["source_height"]),
    )
    _stream_synchronized_images(
        source,
        (camera,),
        target_stamps,
        round(1_000_000_000 / fps),
        emit,
    )
    if len(estimates) < minimum_detections:
        raise ValueError(
            f"only {len(estimates)} valid marker detections; require "
            f"{minimum_detections}"
        )
    inlier_indices = _transform_inlier_indices(estimates)
    if len(inlier_indices) < minimum_detections:
        raise ValueError(
            f"only {len(inlier_indices)} marker transform inliers after rejecting "
            f"{len(estimates) - len(inlier_indices)} outliers; require "
            f"{minimum_detections}"
        )
    inlier_estimates = [estimates[index] for index in inlier_indices]
    inlier_errors = [reprojection_errors[index] for index in inlier_indices]
    global_from_qr = validate_rigid_transform(
        _average_rigid_transforms(inlier_estimates), "global_from_qr"
    )
    qr_from_global = np.linalg.inv(global_from_qr)
    translations = np.asarray([matrix[:3, 3] for matrix in inlier_estimates])
    payload = {
        "schema_version": 1,
        "marker_type": marker_type,
        "marker_size_m": marker_size_m,
        "coordinate_convention": (
            "Marker origin at its center; +X toward its right edge, +Y toward its "
            "top edge, and +Z outward from the printed face."
        ),
        "global_frame": ros["global_frame"],
        "head_pose_child_frame": pose_child,
        "head_camera_frame": ros["camera_frame"],
        "head_pose_from_camera": tracking_from_camera.tolist(),
        "global_from_qr": global_from_qr.tolist(),
        "qr_from_global": qr_from_global.tolist(),
        "calibration": {
            "frame_range": [frame_start, end],
            "accepted_source_frames": [
                accepted_frames[index] for index in inlier_indices
            ],
            "detections_before_transform_outlier_rejection": len(estimates),
            "transform_inliers": len(inlier_indices),
            "rejected_transform_outliers": len(estimates) - len(inlier_indices),
            "translation_std_m": np.std(translations, axis=0).tolist(),
            "mean_reprojection_error_px": float(np.mean(inlier_errors)),
            "maximum_reprojection_error_px": float(max(inlier_errors)),
        },
        "source": {
            "bag": source.name,
            "review_manifest_sha256": _sha256(review_manifest_path),
            "head_image_topic": image_topic,
            "head_pose_topic": pose_topic,
        },
    }
    if marker_type == "aruco":
        payload["aruco"] = {
            "dictionary": aruco_dictionary,
            "marker_id": aruco_marker_id,
        }
    write_json_atomic(output, payload)
    return payload
