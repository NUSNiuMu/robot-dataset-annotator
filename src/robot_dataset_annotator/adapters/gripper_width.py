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
    reference_height_px: int | None = None
    symmetric_marker_midpoint_px: tuple[float, float] | None = None

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
    source: str = "direct"


GRIPPER_WIDTH_SOURCE_CODES = {
    "invalid": 0,
    "direct": 1,
    "symmetric_inference": 2,
    "temporal_interpolation": 3,
}


class StreamingGripperInterpolator:
    """Resolve short marker dropouts without crossing episode boundaries."""

    def __init__(self, maximum_gap_frames: int = 3):
        if maximum_gap_frames < 0:
            raise ValueError("maximum gripper interpolation gap must be non-negative")
        self.maximum_gap_frames = maximum_gap_frames
        self._raw: list[GripperMeasurement] = []
        self._episode_indices: list[int] = []
        self._resolved: list[GripperMeasurement] = []
        self._interpolated_runs = 0
        self._interpolated_frames = 0

    def push(self, measurement: GripperMeasurement, episode_index: int) -> None:
        if self._episode_indices and episode_index < self._episode_indices[-1]:
            raise ValueError("gripper episode indices must be monotonic")
        self._raw.append(measurement)
        self._episode_indices.append(int(episode_index))
        self._drain(final=False)

    def finish(self) -> None:
        self._drain(final=True)
        if len(self._resolved) != len(self._raw):
            raise ValueError("gripper interpolation did not resolve every frame")

    def get(self, index: int) -> GripperMeasurement | None:
        return self._resolved[index] if index < len(self._resolved) else None

    def _append_interpolated_run(self, start: int, end: int) -> None:
        left = self._resolved[start - 1]
        right = self._raw[end]
        denominator = end - start + 1
        for index in range(start, end):
            fraction = (index - start + 1) / denominator
            width = left.width_m + fraction * (right.width_m - left.width_m)
            self._resolved.append(
                GripperMeasurement(
                    width_m=float(width),
                    valid=True,
                    distance_px=None,
                    marker_counts=dict(self._raw[index].marker_counts),
                    source="temporal_interpolation",
                )
            )
        self._interpolated_runs += 1
        self._interpolated_frames += end - start

    def _drain(self, *, final: bool) -> None:
        while len(self._resolved) < len(self._raw):
            start = len(self._resolved)
            measurement = self._raw[start]
            if measurement.valid:
                self._resolved.append(measurement)
                continue

            episode_index = self._episode_indices[start]
            end = start
            while (
                end < len(self._raw)
                and self._episode_indices[end] == episode_index
                and not self._raw[end].valid
            ):
                end += 1
            bounded_on_right = (
                end < len(self._raw)
                and self._episode_indices[end] == episode_index
                and self._raw[end].valid
            )
            bounded_on_left = (
                start > 0
                and self._episode_indices[start - 1] == episode_index
                and self._resolved[start - 1].valid
            )
            gap = end - start
            if bounded_on_left and bounded_on_right and gap <= self.maximum_gap_frames:
                self._append_interpolated_run(start, end)
                continue

            crossed_episode = (
                end < len(self._raw)
                and self._episode_indices[end] != episode_index
            )
            known_unusable = (
                bounded_on_right
                or crossed_episode
                or gap > self.maximum_gap_frames
                or final
            )
            if not known_unusable:
                break
            self._resolved.extend(self._raw[start:end])

    def audit(self) -> dict[str, int | float]:
        valid_frames = sum(row.valid for row in self._resolved)
        invalid_frames = len(self._resolved) - valid_frames
        return {
            "maximum_gap_frames": self.maximum_gap_frames,
            "interpolated_runs": self._interpolated_runs,
            "interpolated_frames": self._interpolated_frames,
            "resolved_frames": len(self._resolved),
            "resolved_valid_frames": valid_frames,
            "resolved_invalid_frames": invalid_frames,
            "resolved_valid_fraction": (
                valid_frames / len(self._resolved) if self._resolved else 0.0
            ),
        }


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
        inference = row.get("single_marker_inference")
        if inference is not None and not isinstance(inference, dict):
            raise ValueError(
                f"{camera_name} single_marker_inference must be an object"
            )
        reference_height = (
            int(row["reference_height_px"])
            if row.get("reference_height_px") is not None
            else None
        )
        symmetric_midpoint = None
        if inference is not None:
            if inference.get("method") != "symmetric_midpoint":
                raise ValueError(
                    f"{camera_name} single_marker_inference method must be "
                    "symmetric_midpoint"
                )
            midpoint = inference.get("marker_midpoint_px")
            if not isinstance(midpoint, list) or len(midpoint) != 2:
                raise ValueError(
                    f"{camera_name} marker_midpoint_px must contain x and y"
                )
            if not all(np.isfinite(float(value)) for value in midpoint):
                raise ValueError(
                    f"{camera_name} marker_midpoint_px must be finite"
                )
            if reference_height is None or reference_height <= 0:
                raise ValueError(
                    f"{camera_name} symmetric inference requires a positive "
                    "reference_height_px"
                )
            symmetric_midpoint = tuple(float(value) for value in midpoint)
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
            reference_height_px=reference_height,
            symmetric_marker_midpoint_px=symmetric_midpoint,
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
        self._direct_frames = 0
        self._symmetric_inferred_frames = 0
        self._inferred_from_marker = {
            calibration.left_marker_id: 0,
            calibration.right_marker_id: 0,
        }
        self._single_marker_frames = 0
        self._missing_frames = 0
        self._ambiguous_frames = 0
        self._symmetry_geometry_mismatch_frames = 0
        self._clipped_low = 0
        self._clipped_high = 0
        self._distances: list[float] = []
        self._direct_distances: list[float] = []
        self._inferred_distances: list[float] = []
        self._midpoint_errors: list[float] = []

    def _scaled_symmetric_midpoint(
        self, image: np.ndarray
    ) -> np.ndarray | None:
        midpoint = self.calibration.symmetric_marker_midpoint_px
        reference_height = self.calibration.reference_height_px
        if midpoint is None or reference_height is None:
            return None
        scale_x = image.shape[1] / self.calibration.reference_width_px
        scale_y = image.shape[0] / reference_height
        if not np.isclose(scale_x, scale_y, rtol=0.01, atol=1e-9):
            return None
        return np.asarray(
            [midpoint[0] * scale_x, midpoint[1] * scale_y], dtype=np.float64
        )

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
        if any(count > 1 for count in counts.values()):
            self._ambiguous_frames += 1
            return GripperMeasurement(0.0, False, None, counts, source="invalid")

        visible = [marker_id for marker_id in marker_ids if counts[marker_id] == 1]
        inferred = False
        if len(visible) == 2:
            distance = float(
                np.linalg.norm(
                    centers[self.calibration.left_marker_id][0]
                    - centers[self.calibration.right_marker_id][0]
                )
            )
            symmetric_midpoint = self._scaled_symmetric_midpoint(image)
            if symmetric_midpoint is not None:
                observed_midpoint = (
                    centers[self.calibration.left_marker_id][0]
                    + centers[self.calibration.right_marker_id][0]
                ) / 2.0
                self._midpoint_errors.append(
                    float(np.linalg.norm(observed_midpoint - symmetric_midpoint))
                )
        elif len(visible) == 1:
            self._single_marker_frames += 1
            symmetric_midpoint = self._scaled_symmetric_midpoint(image)
            if symmetric_midpoint is None:
                if self.calibration.symmetric_marker_midpoint_px is not None:
                    self._symmetry_geometry_mismatch_frames += 1
                self._missing_frames += 1
                return GripperMeasurement(0.0, False, None, counts, source="invalid")
            marker_id = visible[0]
            distance = 2.0 * float(
                np.linalg.norm(centers[marker_id][0] - symmetric_midpoint)
            )
            inferred = True
            self._inferred_from_marker[marker_id] += 1
        else:
            self._missing_frames += 1
            return GripperMeasurement(0.0, False, None, counts, source="invalid")
        width, clipped = self.calibration.width_from_distance(
            distance, int(image.shape[1])
        )
        self._valid_frames += 1
        self._distances.append(distance)
        if inferred:
            self._symmetric_inferred_frames += 1
            self._inferred_distances.append(distance)
        else:
            self._direct_frames += 1
            self._direct_distances.append(distance)
        if clipped == "low":
            self._clipped_low += 1
        elif clipped == "high":
            self._clipped_high += 1
        return GripperMeasurement(
            width,
            True,
            distance,
            counts,
            clipped,
            "symmetric_inference" if inferred else "direct",
        )

    def audit(self) -> dict[str, Any]:
        distances = np.asarray(self._distances, dtype=np.float64)
        direct_distances = np.asarray(self._direct_distances, dtype=np.float64)
        inferred_distances = np.asarray(self._inferred_distances, dtype=np.float64)
        midpoint_errors = np.asarray(self._midpoint_errors, dtype=np.float64)

        def distance_summary(values: np.ndarray) -> dict[str, float | None]:
            return {
                "min": float(values.min()) if values.size else None,
                "median": float(np.median(values)) if values.size else None,
                "max": float(values.max()) if values.size else None,
            }

        return {
            "camera_name": self.calibration.camera_name,
            "frames": self._frames,
            "valid_frames": self._valid_frames,
            "valid_fraction": (
                self._valid_frames / self._frames if self._frames else 0.0
            ),
            "direct_paired_marker_frames": self._direct_frames,
            "symmetric_inferred_frames": self._symmetric_inferred_frames,
            "single_marker_frames": self._single_marker_frames,
            "inferred_from_marker_frames": {
                str(marker_id): count
                for marker_id, count in self._inferred_from_marker.items()
            },
            "missing_marker_frames": self._missing_frames,
            "ambiguous_marker_frames": self._ambiguous_frames,
            "symmetry_geometry_mismatch_frames": (
                self._symmetry_geometry_mismatch_frames
            ),
            "clipped_low_frames": self._clipped_low,
            "clipped_high_frames": self._clipped_high,
            "raw_distance_px": distance_summary(distances),
            "direct_raw_distance_px": distance_summary(direct_distances),
            "inferred_raw_distance_px": distance_summary(inferred_distances),
            "paired_midpoint_error_px": distance_summary(midpoint_errors),
            "calibration": {
                "reference_width_px": self.calibration.reference_width_px,
                "reference_height_px": self.calibration.reference_height_px,
                "closed_distance_px": self.calibration.closed_distance_px,
                "open_distance_px": self.calibration.open_distance_px,
                "closed_width_m": self.calibration.closed_width_m,
                "open_width_m": self.calibration.open_width_m,
                "symmetric_marker_midpoint_px": (
                    list(self.calibration.symmetric_marker_midpoint_px)
                    if self.calibration.symmetric_marker_midpoint_px is not None
                    else None
                ),
            },
        }
