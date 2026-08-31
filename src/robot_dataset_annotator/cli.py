from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path
import shlex

import numpy as np

from .core.audit import Stage, audit_batch
from .core.config import SessionConfig
from .core.decisions import validate_decision_files, write_decision_template
from .core.io import write_json_atomic
from .core.plugins import load_suggester
from .core.runner import execute, prepare_next
from .core.task_spec import TaskSpec


def _prompt_path(value: str | None, label: str) -> str:
    if value and value.strip():
        return str(Path(value).expanduser().resolve())
    answer = input(f"{label}: ").strip()
    if not answer:
        raise SystemExit(f"{label} is required")
    return str(Path(answer).expanduser().resolve())


def _configure(args: argparse.Namespace) -> int:
    workspace = _prompt_path(args.workspace, "Workspace path")
    payload = {
        "schema_version": 1,
        "workspace": workspace,
        "input_root": _prompt_path(args.input_root, "Input recordings root"),
        "review_root": _prompt_path(args.review_root, "Review artifacts root"),
        "dataset_root": _prompt_path(args.dataset_root, "Dataset output root"),
        "task_spec": _prompt_path(args.task_spec, "Task spec JSON"),
        "input_glob": args.input_glob,
        "start_item": args.start_item or None,
        "dataset_suffix": args.dataset_suffix,
        "review_manifest": args.review_manifest,
        "decisions_file": args.decisions_file,
        "checks": [],
        "commands": {},
    }
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force")
    write_json_atomic(output, payload)
    print(output)
    return 0


def _audit(args: argparse.Namespace) -> int:
    rows = audit_batch(SessionConfig.load(args.config.expanduser().resolve()))
    payload = {
        "status": (
            "PASS"
            if all(row.stage is Stage.COMPLETE for row in rows)
            else "INCOMPLETE"
        ),
        "items": [row.as_dict() for row in rows],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 2


def _init_decisions(args: argparse.Namespace) -> int:
    write_decision_template(args.manifest, args.output, args.task_id)
    print(args.output.resolve())
    return 0


def _validate_decisions(args: argparse.Namespace) -> int:
    summary = validate_decision_files(args.manifest, args.decisions, args.task_spec)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0


def _built_in_task(args: argparse.Namespace) -> int:
    source = resources.files("robot_dataset_annotator").joinpath(
        "task_specs", f"{args.task_id}.json"
    )
    if not source.is_file():
        raise SystemExit(f"unknown built-in task: {args.task_id}")
    args.output.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print(args.output.resolve())
    return 0


def _suggest(args: argparse.Namespace) -> int:
    task = TaskSpec.load(args.task_spec)
    if task.plugin is None:
        raise SystemExit(
            f"task {task.task_id} requires manual boundaries and has no suggester"
        )
    if args.format == "npz":
        with np.load(args.observations) as payload:
            observations = np.asarray(payload["state"], dtype=np.float64)
            valid = np.asarray(payload["state_valid"], dtype=bool)
    elif args.format == "insight-parquet":
        from .adapters.insight_lowdim import load_fused_state

        observations, valid = load_fused_state(args.observations)
    else:
        from .adapters.insight_review import load_fused_state

        observations, valid = load_fused_state(args.observations)
    result = load_suggester(task)(
        observations,
        valid,
        minimum_frames=task.minimum_segment_frames,
    )
    result["task_id"] = task.task_id
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "candidate" else 2


def _export_lerobot(args: argparse.Namespace) -> int:
    task_path = args.task_spec.expanduser().resolve()
    task = TaskSpec.load(task_path)
    if task.episode_pose_quality is not None and args.episode_pose_audit is None:
        raise SystemExit(
            f"task {task.task_id} requires --episode-pose-audit before export"
        )
    if args.episode_pose_audit is not None:
        from .core.episode_pose_quality import (
            validate_episode_pose_audit_for_export,
        )

        try:
            validate_episode_pose_audit_for_export(
                args.episode_pose_audit.expanduser().resolve(),
                review_manifest_path=args.review_manifest.expanduser().resolve(),
                decisions_path=args.decisions.expanduser().resolve(),
                task_path=task_path,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    from .adapters.lerobot_export import export_insight_lerobot

    result = export_insight_lerobot(
        source=args.source.expanduser().resolve(),
        review_manifest_path=args.review_manifest.expanduser().resolve(),
        annotation_manifest_path=args.annotation_manifest.expanduser().resolve(),
        decisions_path=args.decisions.expanduser().resolve(),
        task_path=task_path,
        output=args.output.expanduser().resolve(),
        repo_id=args.repo_id,
        max_skew_ms=args.max_skew_ms,
        vcodec=args.vcodec,
        head_pose_child_frame=args.head_pose_child_frame,
        head_static_calibration_source=(
            args.head_static_calibration_source.expanduser().resolve()
            if args.head_static_calibration_source
            else None
        ),
        gripper_calibration_path=(
            args.gripper_calibration.expanduser().resolve()
            if args.gripper_calibration
            else None
        ),
        streaming_video_encoding=args.video_encoding_mode == "streaming",
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        encoder_threads=args.encoder_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _export_lerobot_batch(args: argparse.Namespace) -> int:
    from .adapters.lerobot_batch_export import (
        export_insight_lerobot_batch,
        load_insight_export_recordings,
    )

    recordings_manifest = args.recordings_manifest.expanduser().resolve()
    result = export_insight_lerobot_batch(
        recordings=load_insight_export_recordings(recordings_manifest),
        recordings_manifest_path=recordings_manifest,
        task_path=args.task_spec.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        repo_id=args.repo_id,
        max_skew_ms=args.max_skew_ms,
        vcodec=args.vcodec,
        gripper_calibration_path=(
            args.gripper_calibration.expanduser().resolve()
            if args.gripper_calibration
            else None
        ),
        maximum_gripper_interpolation_gap_frames=(
            args.maximum_gripper_interpolation_gap_frames
        ),
        streaming_video_encoding=args.video_encoding_mode == "streaming",
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        encoder_threads=args.encoder_threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _calibrate_qr(args: argparse.Namespace) -> int:
    from .adapters.qr_calibration import estimate_qr_transform

    result = estimate_qr_transform(
        source=args.source.expanduser().resolve(),
        review_manifest_path=args.review_manifest.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        marker_size_m=args.marker_size_m,
        frame_start=args.frame_start,
        frame_end_exclusive=args.frame_end_exclusive,
        head_pose_child_frame=args.head_pose_child_frame,
        head_static_calibration_source=(
            args.head_static_calibration_source.expanduser().resolve()
            if args.head_static_calibration_source
            else None
        ),
        minimum_detections=args.minimum_detections,
        maximum_reprojection_error_px=args.maximum_reprojection_error_px,
        marker_type=args.marker_type,
        aruco_dictionary=args.aruco_dictionary,
        aruco_marker_id=args.aruco_marker_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _validate_lerobot(args: argparse.Namespace) -> int:
    from .adapters.lerobot_validation import validate_lerobot_dataset

    result = validate_lerobot_dataset(args.dataset.expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _correct_pose_drift(args: argparse.Namespace) -> int:
    from .adapters.pose_drift import correct_review_manifest_pose_drift

    result = correct_review_manifest_pose_drift(
        review_manifest_path=args.review_manifest.expanduser().resolve(),
        output_manifest_path=args.output_manifest.expanduser().resolve(),
        audit_path=args.audit.expanduser().resolve(),
        maximum_spike_frames=args.maximum_spike_frames,
        maximum_drift_transition_frames=args.maximum_drift_transition_frames,
        decisions_path=(
            args.decisions.expanduser().resolve() if args.decisions else None
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


def _audit_insight_episode_poses(args: argparse.Namespace) -> int:
    from .adapters.insight_pose_quality import audit_insight_episode_pose_quality

    result = audit_insight_episode_pose_quality(
        source=args.source.expanduser().resolve(),
        review_manifest_path=args.review_manifest.expanduser().resolve(),
        decisions_path=args.decisions.expanduser().resolve(),
        task_path=args.task_spec.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        left_native_topic=args.left_native_topic,
        right_native_topic=args.right_native_topic,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


def _visualize_trajectories(args: argparse.Namespace) -> int:
    from .adapters.trajectory_qc import visualize_trajectory_batch

    result = visualize_trajectory_batch(
        input_root=args.input_root,
        recordings_manifest=args.recordings_manifest,
        output=args.output,
        write_png=args.write_png,
        maximum_points=args.maximum_points,
        hand_step_warning_m=args.hand_step_warning_m,
        head_step_warning_m=args.head_step_warning_m,
        hand_head_distance_warning_m=args.hand_head_distance_warning_m,
        minimum_valid_fraction=args.minimum_valid_fraction,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _resume(args: argparse.Namespace) -> int:
    config = SessionConfig.load(args.config.expanduser().resolve())
    try:
        action = prepare_next(config, item_name=args.item)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if action is None:
        print(json.dumps({"status": "COMPLETE"}, ensure_ascii=False))
        return 0
    payload = {
        "item": action.row.item,
        "stage": action.row.stage.value,
        "action": action.row.next_action,
        "argv": list(action.argv),
        "command": shlex.join(action.argv),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return 0
    return execute(action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task-pluggable robot dataset annotation and batch audit"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    configure = commands.add_parser("configure", help="create a path-explicit session")
    configure.add_argument("--output", type=Path, required=True)
    configure.add_argument("--workspace")
    configure.add_argument("--input-root")
    configure.add_argument("--review-root")
    configure.add_argument("--dataset-root")
    configure.add_argument("--task-spec")
    configure.add_argument("--input-glob", default="*")
    configure.add_argument("--start-item")
    configure.add_argument("--dataset-suffix", default="")
    configure.add_argument("--review-manifest", default="manifest.json")
    configure.add_argument("--decisions-file", default="decisions.json")
    configure.add_argument("--force", action="store_true")
    configure.set_defaults(handler=_configure)

    audit = commands.add_parser("audit", help="derive batch state from artifacts")
    audit.add_argument("--config", type=Path, required=True)
    audit.set_defaults(handler=_audit)

    init = commands.add_parser(
        "init-decisions", help="create exhaustive review template"
    )
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--task-id", required=True)
    init.set_defaults(handler=_init_decisions)

    validate = commands.add_parser(
        "validate-decisions", help="validate review coverage"
    )
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--decisions", type=Path, required=True)
    validate.add_argument("--task-spec", type=Path, required=True)
    validate.set_defaults(handler=_validate_decisions)

    task = commands.add_parser("copy-task", help="copy a built-in task spec")
    task.add_argument("task_id")
    task.add_argument("--output", type=Path, required=True)
    task.set_defaults(handler=_built_in_task)

    suggest = commands.add_parser("suggest", help="run a task boundary plugin")
    suggest.add_argument("--task-spec", type=Path, required=True)
    suggest.add_argument("--observations", type=Path, required=True)
    suggest.add_argument(
        "--format",
        choices=("npz", "insight-parquet", "insight-review"),
        default="npz",
    )
    suggest.add_argument("--output", type=Path)
    suggest.set_defaults(handler=_suggest)

    export_lerobot = commands.add_parser(
        "export-lerobot", help="export reviewed Insight data as LeRobotDataset v3"
    )
    export_lerobot.add_argument("--source", type=Path, required=True)
    export_lerobot.add_argument("--review-manifest", type=Path, required=True)
    export_lerobot.add_argument("--annotation-manifest", type=Path, required=True)
    export_lerobot.add_argument("--decisions", type=Path, required=True)
    export_lerobot.add_argument("--task-spec", type=Path, required=True)
    export_lerobot.add_argument(
        "--episode-pose-audit",
        type=Path,
        help=(
            "required by tasks with episode pose-quality rules; must PASS and "
            "match the current manifest, decisions, and task spec"
        ),
    )
    export_lerobot.add_argument("--output", type=Path, required=True)
    export_lerobot.add_argument("--repo-id", required=True)
    export_lerobot.add_argument("--max-skew-ms", type=float)
    export_lerobot.add_argument(
        "--vcodec", choices=("h264", "hevc", "libsvtav1", "auto"), default="h264"
    )
    export_lerobot.add_argument(
        "--video-encoding-mode",
        choices=("streaming", "staged-png"),
        default="streaming",
        help=(
            "encode camera frames directly while reading the source (default), "
            "or stage PNG files before encoding"
        ),
    )
    export_lerobot.add_argument(
        "--encoder-queue-maxsize",
        type=int,
        default=256,
        help=(
            "maximum queued frames per streaming video encoder (default: 256); "
            "increase only when the encoder cannot keep up with source decoding"
        ),
    )
    export_lerobot.add_argument(
        "--encoder-threads",
        type=int,
        default=2,
        help=(
            "threads per camera encoder (default: 2); keep the combined camera "
            "encoders within the available CPU cores when batching exports"
        ),
    )
    export_lerobot.add_argument(
        "--head-pose-child-frame",
        help="tracking child frame represented by the head pose topic",
    )
    export_lerobot.add_argument(
        "--head-static-calibration-source",
        type=Path,
        help=(
            "recording whose tf_static supplies the head tracking-to-RGB transform "
            "when it is absent from the source recording"
        ),
    )
    export_lerobot.add_argument(
        "--gripper-calibration",
        type=Path,
        help=(
            "ArUco pixel-to-physical-width calibration; when supplied, export "
            "20D dual-hand pose and gripper state/action"
        ),
    )
    export_lerobot.set_defaults(handler=_export_lerobot)

    export_lerobot_batch = commands.add_parser(
        "export-lerobot-batch",
        help="export multiple reviewed Insight recordings as one LeRobotDataset v3",
    )
    export_lerobot_batch.add_argument(
        "--recordings-manifest", type=Path, required=True
    )
    export_lerobot_batch.add_argument("--task-spec", type=Path, required=True)
    export_lerobot_batch.add_argument("--output", type=Path, required=True)
    export_lerobot_batch.add_argument("--repo-id", required=True)
    export_lerobot_batch.add_argument("--max-skew-ms", type=float)
    export_lerobot_batch.add_argument(
        "--vcodec", choices=("h264", "hevc", "libsvtav1", "auto"), default="h264"
    )
    export_lerobot_batch.add_argument(
        "--video-encoding-mode",
        choices=("streaming", "staged-png"),
        default="streaming",
    )
    export_lerobot_batch.add_argument(
        "--encoder-queue-maxsize", type=int, default=256
    )
    export_lerobot_batch.add_argument("--encoder-threads", type=int, default=2)
    export_lerobot_batch.add_argument("--gripper-calibration", type=Path)
    export_lerobot_batch.add_argument(
        "--maximum-gripper-interpolation-gap-frames",
        type=int,
        default=3,
        help=(
            "interpolate only marker dropouts no longer than this many frames "
            "when bounded by valid measurements in the same episode"
        ),
    )
    export_lerobot_batch.set_defaults(handler=_export_lerobot_batch)

    calibrate_qr = commands.add_parser(
        "calibrate-qr", help="estimate the global QR pose from preserved head frames"
    )
    calibrate_qr.add_argument("--source", type=Path, required=True)
    calibrate_qr.add_argument("--review-manifest", type=Path, required=True)
    calibrate_qr.add_argument("--output", type=Path, required=True)
    calibrate_qr.add_argument("--marker-size-m", type=float, required=True)
    calibrate_qr.add_argument(
        "--marker-type", choices=("qr_code", "aruco"), default="qr_code"
    )
    calibrate_qr.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    calibrate_qr.add_argument("--aruco-marker-id", type=int)
    calibrate_qr.add_argument("--frame-start", type=int, default=0)
    calibrate_qr.add_argument("--frame-end-exclusive", type=int, required=True)
    calibrate_qr.add_argument("--head-pose-child-frame")
    calibrate_qr.add_argument(
        "--head-static-calibration-source",
        type=Path,
        help=(
            "recording whose tf_static supplies the head tracking-to-RGB transform "
            "when it is absent from the source recording"
        ),
    )
    calibrate_qr.add_argument("--minimum-detections", type=int, default=3)
    calibrate_qr.add_argument(
        "--maximum-reprojection-error-px", type=float, default=3.0
    )
    calibrate_qr.set_defaults(handler=_calibrate_qr)

    validate_lerobot = commands.add_parser(
        "validate-lerobot", help="validate a local LeRobotDataset v3 export"
    )
    validate_lerobot.add_argument("--dataset", type=Path, required=True)
    validate_lerobot.set_defaults(handler=_validate_lerobot)

    audit_insight_episode_poses = commands.add_parser(
        "audit-insight-episode-poses",
        help=(
            "compare native VIO with Insight Global and rate each reviewed episode"
        ),
    )
    audit_insight_episode_poses.add_argument("--source", type=Path, required=True)
    audit_insight_episode_poses.add_argument(
        "--review-manifest", type=Path, required=True
    )
    audit_insight_episode_poses.add_argument(
        "--decisions", type=Path, required=True
    )
    audit_insight_episode_poses.add_argument(
        "--task-spec", type=Path, required=True
    )
    audit_insight_episode_poses.add_argument("--output", type=Path, required=True)
    audit_insight_episode_poses.add_argument("--left-native-topic")
    audit_insight_episode_poses.add_argument("--right-native-topic")
    audit_insight_episode_poses.set_defaults(handler=_audit_insight_episode_poses)

    correct_pose_drift = commands.add_parser(
        "correct-pose-drift",
        help="repair high-confidence pose spikes and coordinate-frame jumps",
    )
    correct_pose_drift.add_argument("--review-manifest", type=Path, required=True)
    correct_pose_drift.add_argument("--output-manifest", type=Path, required=True)
    correct_pose_drift.add_argument("--audit", type=Path, required=True)
    correct_pose_drift.add_argument(
        "--decisions",
        type=Path,
        help=(
            "limit PASS/NEEDS_REVIEW gating to selected PASS episode ranges; "
            "out-of-scope unresolved jumps remain recorded"
        ),
    )
    correct_pose_drift.add_argument("--maximum-spike-frames", type=int, default=3)
    correct_pose_drift.add_argument(
        "--maximum-drift-transition-frames", type=int, default=45
    )
    correct_pose_drift.set_defaults(handler=_correct_pose_drift)

    visualize_trajectories = commands.add_parser(
        "visualize-trajectories",
        help="visualize raw and corrected 3D pose trajectories for a batch",
    )
    trajectory_source = visualize_trajectories.add_mutually_exclusive_group(
        required=True
    )
    trajectory_source.add_argument("--input-root", type=Path)
    trajectory_source.add_argument(
        "--recordings-manifest",
        type=Path,
        help="load the exact review artifacts referenced by a batch manifest",
    )
    visualize_trajectories.add_argument("--output", type=Path, required=True)
    visualize_trajectories.add_argument(
        "--write-png",
        action="store_true",
        help="also render per-take and overview PNG files (requires matplotlib)",
    )
    visualize_trajectories.add_argument("--maximum-points", type=int, default=800)
    visualize_trajectories.add_argument(
        "--hand-step-warning-m", type=float, default=0.12
    )
    visualize_trajectories.add_argument(
        "--head-step-warning-m", type=float, default=0.05
    )
    visualize_trajectories.add_argument(
        "--hand-head-distance-warning-m", type=float, default=1.25
    )
    visualize_trajectories.add_argument(
        "--minimum-valid-fraction", type=float, default=0.95
    )
    visualize_trajectories.set_defaults(handler=_visualize_trajectories)

    resume = commands.add_parser("resume", help="prepare or run the next transition")
    resume.add_argument("--config", type=Path, required=True)
    resume.add_argument("--item")
    resume.add_argument(
        "--execute",
        action="store_true",
        help="run the configured argv; omitted means a read-only preview",
    )
    resume.set_defaults(handler=_resume)
    return parser



def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
