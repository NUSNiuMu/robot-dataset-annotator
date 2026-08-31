from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import os
from pathlib import Path
import shutil
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from ..core.decisions import ValidationSummary, validate_decisions
from ..core.episode_pose_quality import validate_episode_pose_audit_for_export
from ..core.io import read_json, write_json_atomic
from ..core.task_spec import TaskSpec
from .gripper_width import (
    GRIPPER_WIDTH_SOURCE_CODES,
    GripperMeasurement,
    GripperWidthDetector,
    StreamingGripperInterpolator,
    load_gripper_calibrations,
)
from .insight_frames import head_frame_calibration
from .lerobot_export import (
    CameraSpec,
    FramePlan,
    _camera_specs,
    _dataset_manifest,
    _features,
    _modality_metadata,
    _sha256,
    _stream_synchronized_images,
    _with_gripper,
    build_frame_plan,
    current_code_revision,
)


@dataclass(frozen=True)
class InsightExportRecording:
    source: Path
    review_manifest_path: Path
    annotation_manifest_path: Path
    decisions_path: Path
    episode_pose_audit_path: Path | None = None
    head_pose_child_frame: str | None = None
    head_static_calibration_source: Path | None = None


@dataclass(frozen=True)
class _PreparedRecording:
    request: InsightExportRecording
    review_manifest: dict[str, Any]
    validation: ValidationSummary
    plans: tuple[FramePlan, ...]
    cameras: tuple[CameraSpec, ...]
    head_calibration: dict[str, Any]
    fps: float


def _resolve_path(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"recording entry requires {field}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_insight_export_recordings(path: Path) -> list[InsightExportRecording]:
    payload = read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported LeRobot recordings manifest schema")
    rows = payload.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("recordings manifest must contain a non-empty recordings array")
    base = path.resolve().parent
    result: list[InsightExportRecording] = []
    seen: set[Path] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"recording entry {index} must be an object")
        source = _resolve_path(base, row.get("source"), "source")
        if source in seen:
            raise ValueError(f"duplicate recording source: {source}")
        seen.add(source)
        audit = row.get("episode_pose_audit")
        static_source = row.get("head_static_calibration_source")
        result.append(
            InsightExportRecording(
                source=source,
                review_manifest_path=_resolve_path(
                    base, row.get("review_manifest"), "review_manifest"
                ),
                annotation_manifest_path=_resolve_path(
                    base, row.get("annotation_manifest"), "annotation_manifest"
                ),
                decisions_path=_resolve_path(
                    base, row.get("decisions"), "decisions"
                ),
                episode_pose_audit_path=(
                    _resolve_path(base, audit, "episode_pose_audit")
                    if audit is not None
                    else None
                ),
                head_pose_child_frame=(
                    str(row["head_pose_child_frame"])
                    if row.get("head_pose_child_frame")
                    else None
                ),
                head_static_calibration_source=(
                    _resolve_path(
                        base,
                        static_source,
                        "head_static_calibration_source",
                    )
                    if static_source is not None
                    else None
                ),
            )
        )
    return result


def _prepare_recording(
    request: InsightExportRecording,
    task: TaskSpec,
    task_path: Path,
) -> _PreparedRecording:
    if not request.source.is_dir():
        raise FileNotFoundError(f"recording source does not exist: {request.source}")
    review_manifest = read_json(request.review_manifest_path)
    annotation_manifest = read_json(request.annotation_manifest_path)
    decisions = read_json(request.decisions_path)
    validation = validate_decisions(annotation_manifest, decisions, task)
    if len(annotation_manifest.get("source_segments", [])) != 1:
        raise ValueError(
            f"{request.source.name} must contain exactly one source segment"
        )
    if task.episode_pose_quality is not None:
        if request.episode_pose_audit_path is None:
            raise ValueError(
                f"{request.source.name} requires an episode pose-quality audit"
            )
        validate_episode_pose_audit_for_export(
            request.episode_pose_audit_path,
            review_manifest_path=request.review_manifest_path,
            decisions_path=request.decisions_path,
            task_path=task_path,
        )
    head_calibration = head_frame_calibration(
        request.source,
        review_manifest,
        head_pose_child_frame=request.head_pose_child_frame,
        static_calibration_source=request.head_static_calibration_source,
    )
    plans = tuple(
        build_frame_plan(
            review_manifest,
            decisions,
            task,
            head_pose_from_camera=head_calibration["tracking_from_camera"],
        )
    )
    cameras = _camera_specs(review_manifest)
    fps = float(review_manifest["fps"])
    if fps <= 0:
        raise ValueError(f"{request.source.name} review fps must be positive")
    return _PreparedRecording(
        request=request,
        review_manifest=review_manifest,
        validation=validation,
        plans=plans,
        cameras=cameras,
        head_calibration=head_calibration,
        fps=fps,
    )


def _check_recording_compatibility(
    prepared: Sequence[_PreparedRecording],
) -> tuple[tuple[CameraSpec, ...], float]:
    if not prepared:
        raise ValueError("at least one prepared recording is required")
    cameras = prepared[0].cameras
    fps = prepared[0].fps
    expected_shapes = {
        camera.role: (camera.width, camera.height) for camera in cameras
    }
    for row in prepared[1:]:
        if not np.isclose(row.fps, fps, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{row.request.source.name} fps {row.fps} does not match {fps}"
            )
        actual_shapes = {
            camera.role: (camera.width, camera.height) for camera in row.cameras
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"{row.request.source.name} camera geometry does not match the batch"
            )
    return cameras, fps


def _batch_features(
    cameras: tuple[CameraSpec, ...], *, include_gripper: bool
) -> dict[str, dict[str, Any]]:
    result = _features(cameras, include_gripper=include_gripper)
    result["annotation.source_recording_index"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["source_recording_index"],
    }
    if include_gripper:
        result["annotation.gripper_width_source"] = {
            "dtype": "int64",
            "shape": (2,),
            "names": ["left_hand", "right_hand"],
        }
        result["annotation.action_gripper_width_source"] = {
            "dtype": "int64",
            "shape": (2,),
            "names": ["left_hand", "right_hand"],
        }
    return result


def _batch_modality_metadata(
    cameras: tuple[CameraSpec, ...], *, include_gripper: bool
) -> dict[str, Any]:
    payload = _modality_metadata(cameras, include_gripper=include_gripper)
    payload["annotation"]["source_recording_index"] = {
        "original_key": "annotation.source_recording_index",
        "mapping": "rda/export_manifest.json:source_recordings",
    }
    if include_gripper:
        payload["annotation"]["gripper_width_source"] = {
            "original_key": "annotation.gripper_width_source",
            "codes": GRIPPER_WIDTH_SOURCE_CODES,
        }
        payload["annotation"]["action_gripper_width_source"] = {
            "original_key": "annotation.action_gripper_width_source",
            "codes": GRIPPER_WIDTH_SOURCE_CODES,
        }
    return payload


def _measurement_source_code(measurement: GripperMeasurement) -> int:
    source = measurement.source if measurement.valid else "invalid"
    try:
        return GRIPPER_WIDTH_SOURCE_CODES[source]
    except KeyError as exc:
        raise ValueError(f"unknown gripper measurement source: {source}") from exc


def _augment_gripper_audit(
    detector: GripperWidthDetector,
    resolver: StreamingGripperInterpolator,
) -> dict[str, Any]:
    result = detector.audit()
    interpolation = resolver.audit()
    result["temporal_interpolation"] = interpolation
    result["resolved_valid_frames"] = interpolation["resolved_valid_frames"]
    result["resolved_invalid_frames"] = interpolation["resolved_invalid_frames"]
    result["resolved_valid_fraction"] = interpolation["resolved_valid_fraction"]
    return result


def _aggregate_gripper_audits(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    count_fields = (
        "frames",
        "valid_frames",
        "direct_paired_marker_frames",
        "symmetric_inferred_frames",
        "single_marker_frames",
        "missing_marker_frames",
        "ambiguous_marker_frames",
        "symmetry_geometry_mismatch_frames",
        "clipped_low_frames",
        "clipped_high_frames",
        "resolved_valid_frames",
        "resolved_invalid_frames",
    )
    for role in ("left_hand", "right_hand"):
        role_rows = [row["roles"][role] for row in rows]
        aggregate = {
            field: sum(int(row.get(field, 0)) for row in role_rows)
            for field in count_fields
        }
        frames = aggregate["frames"]
        aggregate["valid_fraction"] = (
            aggregate["valid_frames"] / frames if frames else 0.0
        )
        aggregate["resolved_valid_fraction"] = (
            aggregate["resolved_valid_frames"] / frames if frames else 0.0
        )
        aggregate["temporal_interpolated_frames"] = sum(
            int(row["temporal_interpolation"]["interpolated_frames"])
            for row in role_rows
        )
        aggregate["temporal_interpolated_runs"] = sum(
            int(row["temporal_interpolation"]["interpolated_runs"])
            for row in role_rows
        )
        result[role] = aggregate
    return result


def _source_provenance(
    row: _PreparedRecording,
    *,
    source_recording_index: int,
    global_episode_start: int,
    synchronization: dict[str, Any],
    gripper_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    request = row.request
    episode_count = row.validation.episodes
    return {
        "source_recording_index": source_recording_index,
        "bag": request.source.name,
        "global_episode_start": global_episode_start,
        "global_episode_end_exclusive": global_episode_start + episode_count,
        "episodes": episode_count,
        "frames": len(row.plans),
        "review_manifest_sha256": _sha256(request.review_manifest_path),
        "annotation_manifest_sha256": _sha256(request.annotation_manifest_path),
        "decisions_sha256": _sha256(request.decisions_path),
        "episode_pose_audit_sha256": (
            _sha256(request.episode_pose_audit_path)
            if request.episode_pose_audit_path is not None
            else None
        ),
        "camera_synchronization": synchronization,
        "gripper_detection": gripper_audit,
        "head_camera_frames": {
            "global_frame": row.head_calibration["global_frame"],
            "head_pose_child_frame": row.head_calibration["head_pose_child_frame"],
            "head_camera_frame": row.head_calibration["camera_frame"],
            "head_pose_from_camera": row.head_calibration[
                "tracking_from_camera"
            ].tolist(),
            "static_calibration_source": row.head_calibration[
                "static_calibration_source"
            ],
            "static_calibration_borrowed": row.head_calibration[
                "static_calibration_borrowed"
            ],
        },
        "pose_drift_correction": row.review_manifest.get(
            "pose_drift_correction",
            {"schema_version": 1, "status": "NOT_RUN"},
        ),
    }


def export_insight_lerobot_batch(
    *,
    recordings: Sequence[InsightExportRecording],
    task_path: Path,
    output: Path,
    repo_id: str,
    max_skew_ms: float | None = None,
    vcodec: str = "h264",
    gripper_calibration_path: Path | None = None,
    maximum_gripper_interpolation_gap_frames: int = 3,
    streaming_video_encoding: bool = True,
    encoder_queue_maxsize: int = 256,
    encoder_threads: int = 2,
    recordings_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    if not recordings:
        raise ValueError("at least one recording is required")
    if max_skew_ms is not None and max_skew_ms <= 0:
        raise ValueError("max_skew_ms must be positive")
    if maximum_gripper_interpolation_gap_frames < 0:
        raise ValueError("maximum gripper interpolation gap must be non-negative")
    if encoder_queue_maxsize <= 0:
        raise ValueError("encoder_queue_maxsize must be positive")
    if encoder_threads <= 0:
        raise ValueError("encoder_threads must be positive")

    task = TaskSpec.load(task_path)
    prepared = tuple(
        _prepare_recording(request, task, task_path) for request in recordings
    )
    cameras, fps = _check_recording_compatibility(prepared)
    total_episodes = sum(row.validation.episodes for row in prepared)
    total_frames = sum(len(row.plans) for row in prepared)

    calibrations: dict[str, Any] = {}
    calibration_payload = None
    if gripper_calibration_path is not None:
        calibrations, calibration_payload = load_gripper_calibrations(
            gripper_calibration_path
        )
    include_gripper = bool(calibrations)
    if include_gripper:
        for row in prepared:
            for camera in row.cameras:
                if camera.role in {"left_hand", "right_hand"} and (
                    camera.name not in calibrations
                ):
                    raise ValueError(
                        f"no gripper calibration for source camera {camera.name!r}"
                    )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - optional export dependency
        raise RuntimeError("install robot-dataset-annotator[lerobot]") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    dataset = None
    source_rows: list[dict[str, Any]] = []
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=round(fps),
            root=temporary,
            robot_type="human_hand_tracking",
            features=_batch_features(cameras, include_gripper=include_gripper),
            use_videos=True,
            image_writer_threads=0 if streaming_video_encoding else 4,
            vcodec=vcodec,
            streaming_encoding=streaming_video_encoding,
            encoder_queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )
        global_episode_start = 0
        for source_recording_index, row in enumerate(prepared):
            plans = row.plans
            source_cameras = row.cameras
            camera_by_key = {camera.key: camera for camera in source_cameras}
            tolerance_ns = round(
                (max_skew_ms or (1000.0 / row.fps)) * 1_000_000
            )
            start_stamp_ns = int(row.review_manifest["start_stamp_ns"])
            target_stamps = [
                start_stamp_ns
                + round(plan.source_frame * 1_000_000_000 / row.fps)
                for plan in plans
            ]
            detectors = {
                camera.role: GripperWidthDetector(calibrations[camera.name])
                for camera in source_cameras
                if include_gripper and camera.role in {"left_hand", "right_hand"}
            }
            if include_gripper and set(detectors) != {"left_hand", "right_hand"}:
                raise ValueError("gripper calibration must cover both hand cameras")
            resolvers = {
                role: StreamingGripperInterpolator(
                    maximum_gripper_interpolation_gap_frames
                )
                for role in detectors
            }
            images: dict[int, dict[str, np.ndarray]] = {}
            raw_grippers: dict[int, dict[str, GripperMeasurement]] = {}
            next_to_measure = 0
            next_to_write = 0

            def flush_ready() -> None:
                nonlocal next_to_write
                while next_to_write < len(plans):
                    if len(images.get(next_to_write, {})) != len(source_cameras):
                        break
                    plan = plans[next_to_write]
                    target_index = next_to_write
                    if (
                        next_to_write + 1 < len(plans)
                        and plans[next_to_write + 1].episode_index
                        == plan.episode_index
                    ):
                        target_index += 1
                    state = plan.state
                    state_valid = plan.state_valid
                    action = plan.action
                    action_valid = plan.action_valid
                    state_sources = action_sources = None
                    if include_gripper:
                        current = {
                            role: resolver.get(next_to_write)
                            for role, resolver in resolvers.items()
                        }
                        target = {
                            role: resolver.get(target_index)
                            for role, resolver in resolvers.items()
                        }
                        if any(value is None for value in (*current.values(), *target.values())):
                            break
                        state, state_valid = _with_gripper(
                            plan.state, plan.state_valid, current
                        )
                        action, action_valid = _with_gripper(
                            plan.action, plan.action_valid, target
                        )
                        state_sources = np.asarray(
                            [
                                _measurement_source_code(current[role])
                                for role in ("left_hand", "right_hand")
                            ],
                            dtype=np.int64,
                        )
                        action_sources = np.asarray(
                            [
                                _measurement_source_code(target[role])
                                for role in ("left_hand", "right_hand")
                            ],
                            dtype=np.int64,
                        )
                    frame: dict[str, Any] = {
                        "observation.state": state,
                        "observation.state_valid": state_valid.astype(np.uint8),
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
                        "action": action,
                        "action_is_valid": action_valid.astype(np.uint8),
                        "annotation.source_recording_index": np.asarray(
                            [source_recording_index], dtype=np.int64
                        ),
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
                    if include_gripper:
                        frame["annotation.gripper_width_source"] = state_sources
                        frame["annotation.action_gripper_width_source"] = (
                            action_sources
                        )
                    dataset.add_frame(frame)
                    is_episode_end = next_to_write + 1 == len(plans) or (
                        plans[next_to_write + 1].episode_index != plan.episode_index
                    )
                    if is_episode_end:
                        dataset.save_episode(parallel_encoding=True)
                    raw_grippers.pop(next_to_write, None)
                    next_to_write += 1

            def resolve_completed_frames() -> None:
                nonlocal next_to_measure
                while next_to_measure < len(plans):
                    if len(images.get(next_to_measure, {})) != len(source_cameras):
                        break
                    if include_gripper and set(raw_grippers.get(next_to_measure, {})) != {
                        "left_hand",
                        "right_hand",
                    }:
                        break
                    if include_gripper:
                        for role, resolver in resolvers.items():
                            resolver.push(
                                raw_grippers[next_to_measure][role],
                                plans[next_to_measure].episode_index,
                            )
                    next_to_measure += 1
                flush_ready()

            def emit(index: int, key: str, image: np.ndarray, stamp_ns: int) -> None:
                images.setdefault(index, {})[key] = image
                camera = camera_by_key[key]
                detector = detectors.get(camera.role)
                if detector is not None:
                    raw_grippers.setdefault(index, {})[camera.role] = detector.measure(
                        image
                    )
                resolve_completed_frames()

            synchronization = _stream_synchronized_images(
                row.request.source,
                source_cameras,
                target_stamps,
                tolerance_ns,
                emit,
            )
            resolve_completed_frames()
            if next_to_measure != len(plans):
                raise ValueError(
                    f"{row.request.source.name} measured {next_to_measure} of "
                    f"{len(plans)} planned frames"
                )
            for resolver in resolvers.values():
                resolver.finish()
            flush_ready()
            if next_to_write != len(plans):
                raise ValueError(
                    f"{row.request.source.name} wrote {next_to_write} of "
                    f"{len(plans)} planned frames"
                )
            source_gripper_audit = (
                {
                    role: _augment_gripper_audit(detector, resolvers[role])
                    for role, detector in detectors.items()
                }
                if include_gripper
                else None
            )
            source_rows.append(
                _source_provenance(
                    row,
                    source_recording_index=source_recording_index,
                    global_episode_start=global_episode_start,
                    synchronization=synchronization,
                    gripper_audit=source_gripper_audit,
                )
            )
            global_episode_start += row.validation.episodes

        dataset.finalize()
        gripper_by_source = (
            [
                {
                    "source_recording_index": row["source_recording_index"],
                    "bag": row["bag"],
                    "roles": row["gripper_detection"],
                }
                for row in source_rows
            ]
            if include_gripper
            else []
        )
        gripper_audit = (
            _aggregate_gripper_audits(gripper_by_source)
            if include_gripper
            else None
        )
        write_json_atomic(
            temporary / "meta" / "modality.json",
            _batch_modality_metadata(cameras, include_gripper=include_gripper),
        )
        dataset_manifest = _dataset_manifest(
            repo_id=repo_id,
            task=task,
            episodes=total_episodes,
            frames=total_frames,
            fps=fps,
            include_gripper=include_gripper,
            gripper_audit=gripper_audit,
            calibration_payload=calibration_payload,
        )
        dataset_manifest["source_recordings"] = len(source_rows)
        dataset_manifest["source_recording_index_key"] = (
            "annotation.source_recording_index"
        )
        write_json_atomic(temporary / "meta" / "manifest.json", dataset_manifest)

        all_plans = [plan for row in prepared for plan in row.plans]
        action_counts = {
            action.key: sum(
                plan.atomic_action_index == index for plan in all_plans
            )
            for index, action in enumerate(task.actions)
        }
        export_manifest = {
            "schema_version": 2,
            "status": "PASS",
            "format": "LeRobotDataset-v3.0",
            "lerobot_version": version("lerobot"),
            "code_revision": current_code_revision(Path.cwd()),
            "repo_id": repo_id,
            "video_codec": vcodec,
            "video_encoding_mode": (
                "streaming" if streaming_video_encoding else "staged_png"
            ),
            "encoder_queue_maxsize": (
                encoder_queue_maxsize if streaming_video_encoding else None
            ),
            "encoder_threads": encoder_threads,
            "task_id": task.task_id,
            "episodes": total_episodes,
            "frames": total_frames,
            "fps": fps,
            "duration_s": total_frames / fps,
            "atomic_action_frames": action_counts,
            "context_frames": sum(
                plan.atomic_action_index == -1 for plan in all_plans
            ),
            "subtask_semantics": task.subtask_catalog(),
            "source_recordings": source_rows,
            "source_recording_index_semantics": (
                "Index into source_recordings; constant inside each episode."
            ),
            "state_dimension": 20 if include_gripper else 18,
            "gripper_semantics": (
                "physical_jaw_width_m" if include_gripper else None
            ),
            "gripper_width_source_codes": (
                GRIPPER_WIDTH_SOURCE_CODES if include_gripper else None
            ),
            "maximum_gripper_interpolation_gap_frames": (
                maximum_gripper_interpolation_gap_frames
                if include_gripper
                else None
            ),
            "gripper_detection": gripper_audit,
            "gripper_detection_by_source": gripper_by_source,
            "action_semantics": (
                "Next-frame dual-hand 6D pose"
                + (" and physical gripper-width" if include_gripper else "")
                + " target in each recording's source tracking frame; the final "
                "frame repeats the current target. Not robot-retargeted."
            ),
            "pose_semantics": {
                "observation.state": (
                    "Left- and right-hand global poses in the source recording's "
                    "tracking frame. Use annotation.source_recording_index for "
                    "recording provenance."
                ),
                "observation.head_pose": (
                    "Head tracking-frame global pose from the synchronized review "
                    "manifest."
                ),
                "observation.head_camera_pose_global": (
                    "Head RGB-camera global pose after applying that recording's "
                    "static tracking-to-camera transform."
                ),
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
            "task_spec_sha256": _sha256(task_path),
            "gripper_calibration_sha256": (
                _sha256(gripper_calibration_path)
                if gripper_calibration_path is not None
                else None
            ),
            "recordings_manifest_sha256": (
                _sha256(recordings_manifest_path)
                if recordings_manifest_path is not None
                else None
            ),
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
