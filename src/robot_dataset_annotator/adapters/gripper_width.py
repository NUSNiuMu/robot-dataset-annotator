from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import read_json


@dataclass(frozen=True)
class GripperCalibration:
    camera_name: str
    marker_dictionary: str
    left_marker_id: int
    right_marker_id: int
    reference_width_px: int
    closed_distance_px: float
    open_distance_px: float
    closed_width_m: float
    open_width_m: float

    def width_from_distance(
        self, distance_px: float, image_width_px: int
    ) -> tuple[float, str | None]:
        if image_width_px <= 0:
            raise ValueError("image width must be positive")
        scale = image_width_px / self.reference_width_px
        closed = self.closed_distance_px * scale
        opened = self.open_distance_px * scale
        normalized = (distance_px - closed) / (opened - closed)
        clipped = None
        if normalized < 0.0:
            clipped = "low"
        elif normalized > 1.0:
            clipped = "high"
        normalized = float(np.clip(normalized, 0.0, 1.0))
        width = self.closed_width_m + normalized * (
            self.open_width_m - self.closed_width_m
        )
        return float(width), clipped


@dataclass(frozen=True)
class GripperMeasurement:
    width_m: float
    valid: bool
    distance_px: float | None
    marker_counts: dict[int, int]
    clipped: str | None = None


def load_gripper_calibrations(
    path: Path,
) -> tuple[dict[str, GripperCalibration], dict[str, Any]]:
    payload = read_json(path)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("gripper calibration schema_version must be 1")
    marker_dictionary = str(payload.get("marker_dictionary", ""))
    if not marker_dictionary:
        raise ValueError("gripper calibration requires marker_dictionary")
    marker_ids = payload.get("marker_ids")
    if not isinstance(marker_ids, dict):
        raise ValueError("gripper calibration requires marker_ids")
    physical = payload.get("physical_width_m")
    if not isinstance(physical, dict):
        raise ValueError("gripper calibration requires physical_width_m")
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("gripper calibration requires cameras")

    calibrations: dict[str, GripperCalibration] = {}
    for camera_name, row in cameras.items():
        if not isinstance(row, dict):
            raise ValueError(f"invalid calibration for camera {camera_name!r}")
        calibration = GripperCalibration(
            camera_name=str(camera_name),
            marker_dictionary=marker_dictionary,
            left_marker_id=int(marker_ids["left_jaw"]),
            right_marker_id=int(marker_ids["right_jaw"]),
            reference_width_px=int(row["reference_width_px"]),
            closed_distance_px=float(row["closed_distance_px"]),
            open_distance_px=float(row["open_distance_px"]),
            closed_width_m=float(physical["closed"]),
            open_width_m=float(physical["open"]),
        )
        if calibration.reference_width_px <= 0:
            raise ValueError(f"{camera_name} reference width must be positive")
        if calibration.open_distance_px <= calibration.closed_distance_px:
            raise ValueError(f"{camera_name} open distance must exceed closed distance")
        if calibration.open_width_m <= calibration.closed_width_m:
            raise ValueError(f"{camera_name} open width must exceed closed width")
        calibrations[calibration.camera_name] = calibration
    return calibrations, payload


class GripperWidthDetector:
    def __init__(self, calibration: GripperCalibration):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional adapter dependency
            raise RuntimeError("OpenCV with the aruco module is required") from exc

        dictionary_id = getattr(cv2.aruco, calibration.marker_dictionary, None)
        if dictionary_id is None:
            raise ValueError(
                f"unknown ArUco dictionary: {calibration.marker_dictionary}"
            )
        self._cv2 = cv2
        self.calibration = calibration
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = (
            cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "DetectorParameters")
            else cv2.aruco.DetectorParameters_create()
        )
        self._detector = (
            cv2.aruco.ArucoDetector(dictionary, parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self._dictionary = dictionary
        self._parameters = parameters
        self._frames = 0
        self._valid_frames = 0
        self._missing_frames = 0
        self._ambiguous_frames = 0
        self._clipped_low = 0
        self._clipped_high = 0
        self._distances: list[float] = []

    def measure(self, image: np.ndarray) -> GripperMeasurement:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("gripper detector expects an HxWx3 RGB image")
        gray = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:  # pragma: no cover - OpenCV < 4.7
            corners, ids, _ = self._cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._parameters
            )

        centers: dict[int, list[np.ndarray]] = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
                centers.setdefault(int(marker_id), []).append(
                    np.asarray(marker_corners, dtype=np.float64)
                    .reshape(4, 2)
                    .mean(axis=0)
                )
        marker_ids = (
            self.calibration.left_marker_id,
            self.calibration.right_marker_id,
        )
        counts = {
            marker_id: len(centers.get(marker_id, [])) for marker_id in marker_ids
        }
        self._frames += 1
        if any(count == 0 for count in counts.values()):
            self._missing_frames += 1
            return GripperMeasurement(0.0, False, None, counts)
        if any(count != 1 for count in counts.values()):
            self._ambiguous_frames += 1
            return GripperMeasurement(0.0, False, None, counts)

        distance = float(
            np.linalg.norm(
                centers[self.calibration.left_marker_id][0]
                - centers[self.calibration.right_marker_id][0]
            )
        )
        width, clipped = self.calibration.width_from_distance(
            distance, int(image.shape[1])
        )
        self._valid_frames += 1
        self._distances.append(distance)
        if clipped == "low":
            self._clipped_low += 1
        elif clipped == "high":
            self._clipped_high += 1
        return GripperMeasurement(width, True, distance, counts, clipped)

    def audit(self) -> dict[str, Any]:
        distances = np.asarray(self._distances, dtype=np.float64)
        return {
            "camera_name": self.calibration.camera_name,
            "frames": self._frames,
            "valid_frames": self._valid_frames,
            "valid_fraction": (
                self._valid_frames / self._frames if self._frames else 0.0
            ),
            "missing_marker_frames": self._missing_frames,
            "ambiguous_marker_frames": self._ambiguous_frames,
            "clipped_low_frames": self._clipped_low,
            "clipped_high_frames": self._clipped_high,
            "raw_distance_px": {
                "min": float(distances.min()) if distances.size else None,
                "median": float(np.median(distances)) if distances.size else None,
                "max": float(distances.max()) if distances.size else None,
            },
            "calibration": {
                "reference_width_px": self.calibration.reference_width_px,
                "closed_distance_px": self.calibration.closed_distance_px,
                "open_distance_px": self.calibration.open_distance_px,
                "closed_width_m": self.calibration.closed_width_m,
                "open_width_m": self.calibration.open_width_m,
            },
        }
