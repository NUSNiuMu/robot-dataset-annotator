from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..core.io import read_json, write_json_atomic


ROLES = ("left_hand", "right_hand", "head")
ROLE_COLORS = {
    "left_hand": "#ff8c42",
    "right_hand": "#29b6f6",
    "head": "#d66efd",
}


def _select_review(take: Path) -> Path:
    candidates = [path.parent for path in take.glob("review*/decisions.json")]
    if len(candidates) != 1:
        raise ValueError(
            f"{take.name} must contain exactly one reviewed decisions file; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _select_pass_audit(review: Path) -> tuple[Path, dict[str, Any], Path]:
    candidates: list[tuple[Path, dict[str, Any], Path]] = []
    for audit_path in sorted(review.glob("pose_drift_audit*.json")):
        audit = read_json(audit_path)
        if audit.get("status") != "PASS":
            continue
        corrected_name = Path(str(audit.get("corrected_manifest", ""))).name
        corrected_path = review / corrected_name
        if corrected_name and corrected_path.is_file():
            candidates.append((audit_path, audit, corrected_path))
    if not candidates:
        raise ValueError(f"{review} has no PASS pose-drift audit with a manifest")
    canonical = [row for row in candidates if row[0].name == "pose_drift_audit.json"]
    return canonical[0] if canonical else candidates[-1]


def _transform_positions(positions: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [positions, np.ones((len(positions), 1), dtype=np.float64)], axis=1
    )
    return (transform @ homogeneous.T).T[:, :3]


def _episode_ranges(decisions: dict[str, Any]) -> list[tuple[int, int]]:
    review = decisions["reviews"][0]
    ranges: list[tuple[int, int]] = []
    for episode in review.get("episodes", []):
        start = int(episode.get("context_start_frame", episode["episode_start_frame"]))
        end = int(episode["episode_end_frame_exclusive"])
        ranges.append((start, end))
    return ranges


def _pair_mask(valid: np.ndarray, ranges: Iterable[tuple[int, int]]) -> np.ndarray:
    result = np.zeros(max(len(valid) - 1, 0), dtype=bool)
    for start, end in ranges:
        bounded_start = max(0, start)
        bounded_end = min(len(valid), end)
        if bounded_end - bounded_start > 1:
            result[bounded_start : bounded_end - 1] = True
    return result & valid[:-1] & valid[1:]


def _rotation_steps(quaternions: np.ndarray, pair_mask: np.ndarray) -> np.ndarray:
    if len(quaternions) < 2:
        return np.empty(0, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1)
    quaternion_valid = np.isfinite(norms) & (norms > 1e-12)
    normalized = np.zeros_like(quaternions)
    normalized[quaternion_valid] = (
        quaternions[quaternion_valid] / norms[quaternion_valid, None]
    )
    dots = np.abs(np.sum(normalized[:-1] * normalized[1:], axis=1))
    valid_pairs = pair_mask & quaternion_valid[:-1] & quaternion_valid[1:]
    return np.degrees(2.0 * np.arccos(np.clip(dots[valid_pairs], 0.0, 1.0)))


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else 0.0


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else 0.0


def _role_metrics(
    pose: dict[str, Any], ranges: list[tuple[int, int]], fps: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    corrected = np.asarray(pose["positions"], dtype=np.float64)
    raw = np.asarray(pose.get("raw_positions", pose["positions"]), dtype=np.float64)
    quaternions = np.asarray(pose["quaternions_xyzw"], dtype=np.float64)
    valid = np.asarray(pose["valid"], dtype=bool)
    correction_mask = np.asarray(
        pose.get("pose_correction_mask", [False] * len(valid)), dtype=bool
    )
    finite = np.all(np.isfinite(corrected), axis=1) & np.all(np.isfinite(raw), axis=1)
    valid = valid & finite
    full_pairs = valid[:-1] & valid[1:]
    training_pairs = _pair_mask(valid, ranges)
    corrected_steps = np.linalg.norm(np.diff(corrected, axis=0), axis=1)
    raw_steps = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    correction_offsets = np.linalg.norm(corrected - raw, axis=1)
    selected_frames = np.zeros(len(valid), dtype=bool)
    for start, end in ranges:
        selected_frames[max(start, 0) : min(end, len(valid))] = True
    selected_count = int(selected_frames.sum())
    selected_valid = selected_frames & valid
    selected_corrections = selected_valid & correction_mask
    selected_positions = corrected[selected_valid]
    if len(selected_positions):
        centroid = np.median(selected_positions, axis=0)
        lower = np.quantile(selected_positions, 0.01, axis=0)
        upper = np.quantile(selected_positions, 0.99, axis=0)
        centroid_value: list[float] | None = centroid.tolist()
        extent_value: list[float] | None = (upper - lower).tolist()
    else:
        centroid_value = None
        extent_value = None
    training_corrected_steps = corrected_steps[training_pairs]
    training_raw_steps = raw_steps[training_pairs]
    rotation_steps = _rotation_steps(quaternions, training_pairs)
    metrics = {
        "frames": len(valid),
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "training_frames": selected_count,
        "training_valid_fraction": (
            float(selected_valid.sum() / selected_count) if selected_count else 0.0
        ),
        "corrected_frames": int(correction_mask.sum()),
        "correction_fraction": (
            float(correction_mask.mean()) if len(correction_mask) else 0.0
        ),
        "training_corrected_frames": int(selected_corrections.sum()),
        "training_correction_fraction": (
            float(selected_corrections.sum() / selected_count)
            if selected_count
            else 0.0
        ),
        "maximum_correction_offset_m": _safe_max(correction_offsets[valid]),
        "training_maximum_correction_offset_m": _safe_max(
            correction_offsets[selected_valid]
        ),
        "raw_maximum_step_m": _safe_max(raw_steps[full_pairs]),
        "corrected_maximum_step_m": _safe_max(corrected_steps[full_pairs]),
        "training_raw_maximum_step_m": _safe_max(training_raw_steps),
        "training_corrected_maximum_step_m": _safe_max(training_corrected_steps),
        "training_corrected_step_p999_m": _safe_quantile(
            training_corrected_steps, 0.999
        ),
        "training_corrected_maximum_speed_mps": (
            _safe_max(training_corrected_steps) * fps
        ),
        "training_maximum_rotation_step_deg": _safe_max(rotation_steps),
        "training_centroid": centroid_value,
        "training_extent_p01_p99_m": extent_value,
    }
    arrays = {
        "raw": raw,
        "corrected": corrected,
        "quaternions_xyzw": quaternions,
        "valid": valid,
        "training_frames": selected_valid,
        "correction_mask": correction_mask,
        "corrected_steps": corrected_steps,
        "raw_steps": raw_steps,
        "training_pairs": training_pairs,
        "correction_offsets": correction_offsets,
    }
    return metrics, arrays


def _downsample(array: np.ndarray, maximum_points: int) -> list[list[float]]:
    if not len(array):
        return []
    indices = np.linspace(
        0, len(array) - 1, min(len(array), maximum_points), dtype=np.int64
    )
    return np.round(array[indices], 6).tolist()


def _audit_events_by_role(audit: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(stream["role"]): list(stream.get("events", []))
        for stream in audit.get("streams", [])
    }


def _qr_quality(qr: dict[str, Any] | None) -> dict[str, Any] | None:
    if qr is None:
        return None
    calibration = qr.get("calibration", {})
    translation_std = [
        float(value) for value in calibration.get("translation_std_m", [])
    ]
    return {
        "transform_inliers": int(calibration.get("transform_inliers", 0)),
        "detections": int(
            calibration.get("detections_before_transform_outlier_rejection", 0)
        ),
        "maximum_translation_std_m": max(translation_std, default=0.0),
        "mean_reprojection_error_px": float(
            calibration.get("mean_reprojection_error_px", 0.0)
        ),
        "maximum_reprojection_error_px": float(
            calibration.get("maximum_reprojection_error_px", 0.0)
        ),
    }


def _initial_rating(
    visual_status: str,
    role_metrics: dict[str, dict[str, Any]],
    qr_quality: dict[str, Any] | None,
    *,
    hand_step_warning_m: float,
    head_step_warning_m: float,
    minimum_valid_fraction: float,
) -> tuple[str, list[str]]:
    if visual_status != "PASS":
        return "NOT_TRAINING", ["visual review marked this recording FAIL"]
    reasons: list[str] = []
    for role, metrics in role_metrics.items():
        threshold = head_step_warning_m if role == "head" else hand_step_warning_m
        if metrics["training_corrected_maximum_step_m"] >= threshold:
            reasons.append(
                f"{role} training step reaches "
                f"{metrics['training_corrected_maximum_step_m']:.3f} m"
            )
        if metrics["training_valid_fraction"] < minimum_valid_fraction:
            reasons.append(
                f"{role} training validity is "
                f"{metrics['training_valid_fraction']:.1%}"
            )
    if qr_quality is None:
        reasons.append("QR global transform is missing")
    else:
        if qr_quality["maximum_translation_std_m"] >= 0.02:
            reasons.append(
                "QR translation standard deviation reaches "
                f"{qr_quality['maximum_translation_std_m']:.3f} m"
            )
        if qr_quality["maximum_reprojection_error_px"] >= 3.0:
            reasons.append(
                "QR reprojection error reaches "
                f"{qr_quality['maximum_reprojection_error_px']:.2f} px"
            )
    if reasons:
        return "REVIEW_REQUIRED", reasons
    corrected_frames = sum(
        row["training_corrected_frames"] for row in role_metrics.values()
    )
    return ("PASS_AFTER_CORRECTION" if corrected_frames else "PASS"), []


def _robust_centroid_scores(records: list[dict[str, Any]]) -> None:
    for role in ROLES:
        candidates = [
            row
            for row in records
            if row["visual_status"] == "PASS"
            and row["coordinate_frame"] == "qr"
            and row["roles"][role]["training_centroid"] is not None
        ]
        if len(candidates) < 5:
            continue
        centroids = np.asarray(
            [row["roles"][role]["training_centroid"] for row in candidates],
            dtype=np.float64,
        )
        center = np.median(centroids, axis=0)
        mad = np.median(np.abs(centroids - center), axis=0)
        scale = np.maximum(1.4826 * mad, 0.02)
        scores = np.linalg.norm((centroids - center) / scale, axis=1)
        for record, score in zip(candidates, scores, strict=True):
            record["roles"][role]["centroid_robust_z"] = float(score)
            if score >= 15.0:
                record["rating"] = "REVIEW_REQUIRED"
                record["rating_reasons"].append(
                    f"{role} QR-relative centroid robust score is {score:.1f}"
                )


def _record_for_paths(
    *,
    take_name: str,
    review: Path,
    decisions_path: Path,
    corrected_path: Path,
    audit_path: Path,
    audit: dict[str, Any],
    qr_path: Path | None,
    maximum_points: int,
    hand_step_warning_m: float,
    head_step_warning_m: float,
    hand_head_distance_warning_m: float,
    minimum_valid_fraction: float,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    decisions = read_json(decisions_path)
    visual_status = str(decisions["reviews"][0]["visual_status"])
    manifest = read_json(corrected_path)
    fps = float(manifest["fps"])
    ranges = _episode_ranges(decisions)
    qr = read_json(qr_path) if qr_path and qr_path.is_file() else None
    qr_from_global = (
        np.asarray(qr["qr_from_global"], dtype=np.float64) if qr else np.eye(4)
    )
    coordinate_frame = "qr" if qr else "global"
    role_metrics: dict[str, dict[str, Any]] = {}
    arrays_by_role: dict[str, dict[str, np.ndarray]] = {}
    tracks: dict[str, dict[str, Any]] = {}
    poses = {str(row["role"]): row for row in manifest.get("poses", [])}
    for role in ROLES:
        if role not in poses:
            raise ValueError(f"{corrected_path} is missing {role} pose")
        metrics, arrays = _role_metrics(poses[role], ranges, fps)
        arrays["raw_display"] = _transform_positions(arrays["raw"], qr_from_global)
        arrays["corrected_display"] = _transform_positions(
            arrays["corrected"], qr_from_global
        )
        # Render only reviewed context/training intervals. Otherwise a VIO-drift
        # tail remains visible after later episodes are removed from decisions.
        selected_valid = arrays["training_frames"]
        if metrics["training_centroid"] is not None:
            metrics["training_centroid"] = _transform_positions(
                np.asarray([metrics["training_centroid"]]), qr_from_global
            )[0].tolist()
        role_metrics[role] = metrics
        arrays_by_role[role] = arrays
        tracks[role] = {
            "color": ROLE_COLORS[role],
            "raw": _downsample(arrays["raw_display"][selected_valid], maximum_points),
            "corrected": _downsample(
                arrays["corrected_display"][selected_valid], maximum_points
            ),
            "corrections": _downsample(
                arrays["corrected_display"][selected_valid & arrays["correction_mask"]],
                min(maximum_points, 200),
            ),
        }
    selected_frames = np.zeros(int(manifest["frame_count"]), dtype=bool)
    for start, end in ranges:
        selected_frames[max(start, 0) : min(end, len(selected_frames))] = True
    for role in ("left_hand", "right_hand"):
        valid = (
            selected_frames
            & arrays_by_role[role]["valid"]
            & arrays_by_role["head"]["valid"]
        )
        distances = np.linalg.norm(
            arrays_by_role[role]["corrected"][valid]
            - arrays_by_role["head"]["corrected"][valid],
            axis=1,
        )
        role_metrics[role]["training_head_distance_p99_m"] = _safe_quantile(
            distances, 0.99
        )
        role_metrics[role]["training_head_distance_maximum_m"] = _safe_max(distances)
    role_metrics["head"]["training_head_distance_p99_m"] = 0.0
    role_metrics["head"]["training_head_distance_maximum_m"] = 0.0
    quality = _qr_quality(qr)
    rating, reasons = _initial_rating(
        visual_status,
        role_metrics,
        quality,
        hand_step_warning_m=hand_step_warning_m,
        head_step_warning_m=head_step_warning_m,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    record = {
        "record_id": take_name,
        "label": take_name,
        "take": take_name,
        "review_directory": str(review.resolve()),
        "decisions_file": str(decisions_path.resolve()),
        "pose_audit_file": str(audit_path.resolve()),
        "corrected_manifest": str(corrected_path.resolve()),
        "qr_transform_file": str(qr_path.resolve()) if qr else None,
        "visual_status": visual_status,
        "rating": rating,
        "rating_reasons": reasons,
        "coordinate_frame": coordinate_frame,
        "fps": fps,
        "frames": int(manifest["frame_count"]),
        "episodes": len(ranges),
        "pose_events": _audit_events_by_role(audit),
        "qr_quality": quality,
        "roles": role_metrics,
        "tracks": tracks,
    }
    return record, arrays_by_role


def _record_for_take(
    take: Path,
    *,
    maximum_points: int,
    hand_step_warning_m: float,
    head_step_warning_m: float,
    hand_head_distance_warning_m: float,
    minimum_valid_fraction: float,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    review = _select_review(take)
    decisions_path = review / "decisions.json"
    audit_path, audit, corrected_path = _select_pass_audit(review)
    qr_path = review / "qr_transform.json"
    return _record_for_paths(
        take_name=take.name,
        review=review,
        decisions_path=decisions_path,
        corrected_path=corrected_path,
        audit_path=audit_path,
        audit=audit,
        qr_path=qr_path if qr_path.is_file() else None,
        maximum_points=maximum_points,
        hand_step_warning_m=hand_step_warning_m,
        head_step_warning_m=head_step_warning_m,
        hand_head_distance_warning_m=hand_head_distance_warning_m,
        minimum_valid_fraction=minimum_valid_fraction,
    )


def _phase_semantics(
    definitions: list[dict[str, Any]], boundaries: list[int]
) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "key": definition["key"],
            "label": definition.get("label", definition["key"]),
            "start_frame": int(boundaries[index]),
            "end_frame_exclusive": int(boundaries[index + 1]),
        }
        for index, definition in enumerate(definitions)
    ]


def _episode_record(
    record: dict[str, Any],
    arrays_by_role: dict[str, dict[str, np.ndarray]],
    episode: dict[str, Any],
    *,
    local_episode_index: int,
    global_episode_index: int,
    task_spec: dict[str, Any],
    maximum_points: int,
    hand_step_warning_m: float,
    head_step_warning_m: float,
    minimum_valid_fraction: float,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    start = int(episode.get("context_start_frame", episode["episode_start_frame"]))
    end = int(episode["episode_end_frame_exclusive"])
    ranges = [(start, end)]
    role_metrics: dict[str, dict[str, Any]] = {}
    episode_arrays_by_role: dict[str, dict[str, np.ndarray]] = {}
    tracks: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        source = arrays_by_role[role]
        pose = {
            "positions": source["corrected"],
            "raw_positions": source["raw"],
            "quaternions_xyzw": source["quaternions_xyzw"],
            "valid": source["valid"],
            "pose_correction_mask": source["correction_mask"],
        }
        metrics, arrays = _role_metrics(pose, ranges, float(record["fps"]))
        arrays["raw_display"] = source["raw_display"]
        arrays["corrected_display"] = source["corrected_display"]
        selected_valid = arrays["training_frames"]
        selected_positions = arrays["corrected_display"][selected_valid]
        metrics["training_centroid"] = (
            np.median(selected_positions, axis=0).tolist()
            if len(selected_positions)
            else None
        )
        role_metrics[role] = metrics
        episode_arrays_by_role[role] = arrays
        tracks[role] = {
            "color": ROLE_COLORS[role],
            "raw": _downsample(arrays["raw_display"][selected_valid], maximum_points),
            "corrected": _downsample(
                arrays["corrected_display"][selected_valid], maximum_points
            ),
            "corrections": _downsample(
                arrays["corrected_display"][selected_valid & arrays["correction_mask"]],
                min(maximum_points, 200),
            ),
        }
    selected_frames = np.zeros(len(arrays_by_role["head"]["valid"]), dtype=bool)
    selected_frames[start:end] = True
    for role in ("left_hand", "right_hand"):
        valid = (
            selected_frames
            & arrays_by_role[role]["valid"]
            & arrays_by_role["head"]["valid"]
        )
        distances = np.linalg.norm(
            arrays_by_role[role]["corrected"][valid]
            - arrays_by_role["head"]["corrected"][valid],
            axis=1,
        )
        role_metrics[role]["training_head_distance_p99_m"] = _safe_quantile(
            distances, 0.99
        )
        role_metrics[role]["training_head_distance_maximum_m"] = _safe_max(distances)
    role_metrics["head"]["training_head_distance_p99_m"] = 0.0
    role_metrics["head"]["training_head_distance_maximum_m"] = 0.0
    rating, reasons = _initial_rating(
        record["visual_status"],
        role_metrics,
        record["qr_quality"],
        hand_step_warning_m=hand_step_warning_m,
        head_step_warning_m=head_step_warning_m,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    atomic_boundaries = [int(value) for value in episode["atomic_boundaries"]]
    hand_boundaries = episode["hand_subtask_boundaries"]
    episode_metadata = {
        "global_episode_index": global_episode_index,
        "local_episode_index": local_episode_index,
        "context_start_frame": start,
        "episode_start_frame": int(episode["episode_start_frame"]),
        "episode_end_frame_exclusive": end,
        "task_id": task_spec["task_id"],
        "atomic_actions": _phase_semantics(
            task_spec["atomic_actions"], atomic_boundaries
        ),
        "hand_subtasks": {
            hand: _phase_semantics(
                task_spec["hand_subtasks"][hand],
                [int(value) for value in hand_boundaries[hand]],
            )
            for hand in ("left_hand", "right_hand")
        },
    }
    label = (
        f"episode {global_episode_index:03d} | "
        f"{record['take'].split('_2026')[0]} local {local_episode_index}"
    )
    episode_record = {
        **record,
        "record_id": f"episode_{global_episode_index:03d}",
        "label": label,
        "rating": rating,
        "rating_reasons": reasons,
        "episodes": 1,
        "episode": episode_metadata,
        "roles": role_metrics,
        "tracks": tracks,
    }
    return episode_record, episode_arrays_by_role


def _html(
    records: list[dict[str, Any]], task_id: str, *, per_episode: bool = False
) -> str:
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    title = f"{task_id} 3D trajectory QC"
    selector_label = "Episode" if per_episode else "Take"
    record_header = "记录" if per_episode else "take"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #0b1018; color: #e7edf7; }}
header {{ padding: 18px 24px; border-bottom: 1px solid #263143; }}
main {{ display: grid; grid-template-columns: minmax(640px, 1.5fr) minmax(420px, 1fr); gap: 16px; padding: 16px; }}
.panel {{ background: #121a26; border: 1px solid #263143; border-radius: 12px; padding: 14px; }}
canvas {{ width: 100%; height: 680px; background: #080d14; border-radius: 8px; cursor: grab; }}
.controls {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }}
select, input {{ accent-color: #66d9ef; background: #0b1018; color: #e7edf7; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 7px; border-bottom: 1px solid #263143; text-align: left; }}
tbody tr {{ cursor: pointer; }} tbody tr:hover {{ background: #1b2839; }}
.PASS {{ color: #72e0a1; }} .PASS_AFTER_CORRECTION {{ color: #f5cf65; }}
.REVIEW_REQUIRED {{ color: #ff8b73; }} .NOT_TRAINING {{ color: #aab4c3; }}
pre {{ white-space: pre-wrap; max-height: 300px; overflow: auto; font-size: 12px; }}
.legend span {{ margin-right: 16px; }}
@media (max-width: 1100px) {{ main {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header><h2 style="margin:0">{title}</h2>
<div>拖动旋转，滚轮缩放。虚线为 raw，实线为 corrected；坐标优先使用二维码坐标系。</div></header>
<main>
<section class="panel">
<div class="controls">
<label>{selector_label} <select id="take"></select></label>
<label><input id="raw" type="checkbox" checked> raw</label>
<label><input id="corrected" type="checkbox" checked> corrected</label>
<label>进度 <input id="progress" type="range" min="1" max="100" value="100"></label>
<button id="reset">重置视角</button>
</div>
<canvas id="view" width="1200" height="760"></canvas>
<div class="legend"><span style="color:#ff8c42">左手</span><span style="color:#29b6f6">右手</span><span style="color:#d66efd">头部</span><span style="color:#ffe66d">修正区域</span></div>
<pre id="detail"></pre>
</section>
<section class="panel"><table><thead><tr><th>{record_header}</th><th>评级</th><th>episode</th><th>最大训练步长 L/R/H (m)</th></tr></thead><tbody id="rows"></tbody></table></section>
</main>
<script>
const records={payload};
const canvas=document.getElementById('view'), ctx=canvas.getContext('2d');
const select=document.getElementById('take'), rawBox=document.getElementById('raw'), correctedBox=document.getElementById('corrected');
const progress=document.getElementById('progress'), detail=document.getElementById('detail');
let yaw=-0.7,pitch=0.45,zoom=1.0,drag=false,lastX=0,lastY=0;
for (const [i,r] of records.entries()) {{ const o=document.createElement('option'); o.value=i; o.textContent=r.label||r.take; select.appendChild(o); }}
function rotate(p) {{ let [x,y,z]=p; const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch); let x1=cy*x-sy*y,y1=sy*x+cy*y; return [x1,cp*y1-sp*z,sp*y1+cp*z]; }}
function currentPoints(r) {{ const pts=[]; for (const role of Object.keys(r.tracks)) {{ if(rawBox.checked) pts.push(...r.tracks[role].raw); if(correctedBox.checked) pts.push(...r.tracks[role].corrected); }} return pts; }}
function drawLine(points,color,dashed,center,scale,fraction) {{ const count=Math.max(1,Math.floor(points.length*fraction)); if(count<2)return; ctx.beginPath();ctx.strokeStyle=color;ctx.globalAlpha=dashed?0.28:0.9;ctx.lineWidth=dashed?1.2:2.2;ctx.setLineDash(dashed?[7,5]:[]); for(let i=0;i<count;i++){{const q=rotate([points[i][0]-center[0],points[i][1]-center[1],points[i][2]-center[2]]);const x=canvas.width/2+q[0]*scale,y=canvas.height/2-q[1]*scale;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}}ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([]); }}
function drawAxis(center,scale) {{ const axes=[[[0,0,0],[.25,0,0],'#f66','X'],[[0,0,0],[0,.25,0],'#6e8','Y'],[[0,0,0],[0,0,.25],'#69f','Z']]; for(const [a,b,c,l] of axes){{const aa=rotate(a),bb=rotate(b);ctx.strokeStyle=c;ctx.beginPath();ctx.moveTo(canvas.width/2+aa[0]*scale,canvas.height/2-aa[1]*scale);ctx.lineTo(canvas.width/2+bb[0]*scale,canvas.height/2-bb[1]*scale);ctx.stroke();ctx.fillStyle=c;ctx.fillText(l,canvas.width/2+bb[0]*scale,canvas.height/2-bb[1]*scale);}} }}
function draw() {{ const r=records[+select.value||0], pts=currentPoints(r);ctx.clearRect(0,0,canvas.width,canvas.height);if(!pts.length)return;const lo=[0,1,2].map(k=>Math.min(...pts.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...pts.map(p=>p[k]))),center=lo.map((v,k)=>(v+hi[k])/2),span=Math.max(...hi.map((v,k)=>v-lo[k]),.1),scale=620/span*zoom,fraction=+progress.value/100;drawAxis(center,scale);for(const role of ['left_hand','right_hand','head']){{const t=r.tracks[role];if(rawBox.checked)drawLine(t.raw,t.color,true,center,scale,fraction);if(correctedBox.checked)drawLine(t.corrected,t.color,false,center,scale,fraction);}}ctx.fillStyle='#d9e3f0';ctx.font='16px sans-serif';ctx.fillText(`${{r.label||r.take}}  (${{r.coordinate_frame}} frame)`,18,28);detail.textContent=JSON.stringify({{episode:r.episode||null,rating:r.rating,reasons:r.rating_reasons,qr_quality:r.qr_quality,roles:r.roles}},null,2); }}
function selectRecord(i) {{ select.value=i; draw(); }}
const tbody=document.getElementById('rows');records.forEach((r,i)=>{{const tr=document.createElement('tr');const steps=['left_hand','right_hand','head'].map(k=>r.roles[k].training_corrected_maximum_step_m.toFixed(3)).join(' / ');const ep=r.episode?.global_episode_index??r.episodes;tr.innerHTML=`<td>${{r.label||r.take.match(/take_\\d+/)?.[0]||r.take}}</td><td class="${{r.rating}}">${{r.rating}}</td><td>${{ep}}</td><td>${{steps}}</td>`;tr.onclick=()=>selectRecord(i);tbody.appendChild(tr);}});
for(const el of [select,rawBox,correctedBox,progress])el.oninput=draw;document.getElementById('reset').onclick=()=>{{yaw=-.7;pitch=.45;zoom=1;draw();}};
canvas.onmousedown=e=>{{drag=true;lastX=e.clientX;lastY=e.clientY;canvas.style.cursor='grabbing';}};window.onmouseup=()=>{{drag=false;canvas.style.cursor='grab';}};window.onmousemove=e=>{{if(!drag)return;yaw+=(e.clientX-lastX)*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY;draw();}};canvas.onwheel=e=>{{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw();}};
draw();
</script></body></html>"""


def _equal_3d_axes(axis: Any, arrays: list[np.ndarray]) -> None:
    finite = [row[np.all(np.isfinite(row), axis=1)] for row in arrays if len(row)]
    if not finite:
        return
    points = np.concatenate(finite, axis=0)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 0.05)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _write_take_png(
    path: Path,
    record: dict[str, Any],
    arrays_by_role: dict[str, dict[str, np.ndarray]],
    *,
    hand_step_warning_m: float,
    head_step_warning_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    raw_axis = figure.add_subplot(2, 2, 1, projection="3d")
    corrected_axis = figure.add_subplot(2, 2, 2, projection="3d")
    step_axis = figure.add_subplot(2, 2, 3)
    correction_axis = figure.add_subplot(2, 2, 4)
    raw_arrays: list[np.ndarray] = []
    corrected_arrays: list[np.ndarray] = []
    fps = float(record["fps"])
    for role in ROLES:
        arrays = arrays_by_role[role]
        valid = arrays["training_frames"]
        raw = arrays["raw_display"][valid]
        corrected = arrays["corrected_display"][valid]
        raw_arrays.append(raw)
        corrected_arrays.append(corrected)
        color = ROLE_COLORS[role]
        raw_axis.plot(*raw.T, color=color, linewidth=0.8, alpha=0.65, label=role)
        corrected_axis.plot(
            *corrected.T, color=color, linewidth=1.0, alpha=0.85, label=role
        )
        correction_points = arrays["corrected_display"][
            valid & arrays["correction_mask"]
        ]
        if len(correction_points):
            stride = max(1, len(correction_points) // 150)
            corrected_axis.scatter(
                *correction_points[::stride].T,
                color="#ffe66d",
                s=4,
                alpha=0.5,
            )
        times = np.arange(1, len(arrays["corrected"])) / fps
        steps = arrays["corrected_steps"].copy()
        steps[~arrays["training_pairs"]] = np.nan
        step_axis.plot(times, steps, color=color, linewidth=0.8, label=role)
        correction_axis.plot(
            np.arange(len(arrays["correction_offsets"])) / fps,
            np.where(
                arrays["training_frames"], arrays["correction_offsets"], np.nan
            ),
            color=color,
            linewidth=0.8,
            label=role,
        )
    for axis, title, arrays in (
        (raw_axis, "Raw trajectories", raw_arrays),
        (corrected_axis, "Corrected trajectories", corrected_arrays),
    ):
        axis.set_title(f"{title} ({record['coordinate_frame']} frame)")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.legend(loc="best", fontsize=8)
        _equal_3d_axes(axis, arrays)
    step_axis.axhline(
        hand_step_warning_m, color="#ff8c42", linestyle="--", alpha=0.5
    )
    step_axis.axhline(
        head_step_warning_m, color="#d66efd", linestyle="--", alpha=0.5
    )
    step_axis.set_title("Corrected translation step inside training ranges")
    step_axis.set_xlabel("Time (s)")
    step_axis.set_ylabel("Step (m/frame)")
    step_axis.grid(alpha=0.2)
    step_axis.legend(fontsize=8)
    correction_axis.set_title("Raw-to-corrected offset inside reviewed ranges")
    correction_axis.set_xlabel("Time (s)")
    correction_axis.set_ylabel("Offset (m)")
    correction_axis.grid(alpha=0.2)
    correction_axis.legend(fontsize=8)
    reasons = "; ".join(record["rating_reasons"]) or "no automatic warning"
    title = f"{record['label']} — {record['rating']} — {reasons}"
    figure.suptitle(title, fontsize=12)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _write_overview(
    path: Path, records: list[dict[str, Any]], plots: Path, task_id: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(20, rows * 3.4))
    axes_array = np.asarray(axes).reshape(-1)
    status_colors = {
        "PASS": "#72e0a1",
        "PASS_AFTER_CORRECTION": "#f5cf65",
        "REVIEW_REQUIRED": "#ff8b73",
        "NOT_TRAINING": "#aab4c3",
    }
    for axis, record in zip(axes_array, records, strict=False):
        image = plt.imread(plots / f"{record['record_id']}.png")
        axis.imshow(image)
        axis.set_title(
            f"{record['label']}\n{record['rating']}",
            fontsize=8,
            color=status_colors[record["rating"]],
        )
        axis.axis("off")
    for axis in axes_array[len(records) :]:
        axis.axis("off")
    figure.suptitle(f"{task_id} trajectory QC overview", fontsize=18)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "take",
                "visual_status",
                "rating",
                "episodes",
                "coordinate_frame",
                "corrected_frames",
                "left_training_max_step_m",
                "right_training_max_step_m",
                "head_training_max_step_m",
                "left_training_head_distance_p99_m",
                "right_training_head_distance_p99_m",
                "rating_reasons",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["take"],
                    record["visual_status"],
                    record["rating"],
                    record["episodes"],
                    record["coordinate_frame"],
                    sum(row["corrected_frames"] for row in record["roles"].values()),
                    record["roles"]["left_hand"][
                        "training_corrected_maximum_step_m"
                    ],
                    record["roles"]["right_hand"][
                        "training_corrected_maximum_step_m"
                    ],
                    record["roles"]["head"][
                        "training_corrected_maximum_step_m"
                    ],
                    record["roles"]["left_hand"][
                        "training_head_distance_p99_m"
                    ],
                    record["roles"]["right_hand"][
                        "training_head_distance_p99_m"
                    ],
                    "; ".join(record["rating_reasons"]),
                ]
            )


def visualize_trajectory_batch(
    *,
    input_root: Path | None,
    recordings_manifest: Path | None = None,
    per_episode: bool = False,
    task_spec: Path | None = None,
    output: Path,
    write_png: bool = False,
    maximum_points: int = 800,
    hand_step_warning_m: float = 0.12,
    head_step_warning_m: float = 0.05,
    hand_head_distance_warning_m: float = 1.25,
    minimum_valid_fraction: float = 0.95,
) -> dict[str, Any]:
    if (input_root is None) == (recordings_manifest is None):
        raise ValueError("provide exactly one of input_root or recordings_manifest")
    if per_episode and (recordings_manifest is None or task_spec is None):
        raise ValueError(
            "per_episode requires recordings_manifest and task_spec"
        )
    input_root = input_root.expanduser().resolve() if input_root else None
    recordings_manifest = (
        recordings_manifest.expanduser().resolve() if recordings_manifest else None
    )
    task_spec = task_spec.expanduser().resolve() if task_spec else None
    output = output.expanduser().resolve()
    if input_root is not None and not input_root.is_dir():
        raise ValueError(f"input root does not exist: {input_root}")
    if recordings_manifest is not None and not recordings_manifest.is_file():
        raise ValueError(f"recordings manifest does not exist: {recordings_manifest}")
    if task_spec is not None and not task_spec.is_file():
        raise ValueError(f"task spec does not exist: {task_spec}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if maximum_points < 10:
        raise ValueError("maximum_points must be at least 10")
    temporary = output.parent / f".{output.name}.preparing.{uuid4().hex}"
    temporary.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    arrays_by_take: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    task_ids: set[str] = set()
    task_payload = read_json(task_spec) if task_spec else None
    global_episode_index = 0
    try:
        if input_root is not None:
            takes = sorted(
                path
                for path in input_root.iterdir()
                if path.is_dir() and list(path.glob("review*/decisions.json"))
            )
            if not takes:
                raise ValueError(f"no reviewed recordings found under {input_root}")
            for take in takes:
                record, arrays = _record_for_take(
                    take,
                    maximum_points=maximum_points,
                    hand_step_warning_m=hand_step_warning_m,
                    head_step_warning_m=head_step_warning_m,
                    hand_head_distance_warning_m=hand_head_distance_warning_m,
                    minimum_valid_fraction=minimum_valid_fraction,
                )
                decisions = read_json(Path(record["decisions_file"]))
                task_ids.add(str(decisions.get("task_id", "unknown-task")))
                records.append(record)
                arrays_by_take[take.name] = arrays
        else:
            batch = read_json(recordings_manifest)
            rows = batch.get("recordings", [])
            if not rows:
                raise ValueError(f"no recordings found in {recordings_manifest}")
            for row in rows:
                corrected_path = Path(str(row["review_manifest"])).resolve()
                decisions_path = Path(str(row["decisions"])).resolve()
                episode_audit_path = Path(str(row["episode_pose_audit"])).resolve()
                audit_path = episode_audit_path
                audit = read_json(episode_audit_path)
                correction_path = (
                    corrected_path.parent / "pose_drift_correction_audit.json"
                )
                if correction_path.is_file():
                    correction = read_json(correction_path)
                    referenced = Path(str(correction.get("corrected_manifest", "")))
                    if (
                        correction.get("status") == "PASS"
                        and referenced.resolve() == corrected_path
                    ):
                        audit_path = correction_path
                        audit = correction
                annotation_path = Path(str(row["annotation_manifest"])).resolve()
                qr_candidates = (
                    corrected_path.parent / "qr_transform.json",
                    annotation_path.parent / "qr_transform.json",
                )
                qr_path = next((path for path in qr_candidates if path.is_file()), None)
                decisions = read_json(decisions_path)
                task_ids.add(str(decisions.get("task_id", "unknown-task")))
                take_name = Path(str(row["source"])).name
                record, arrays = _record_for_paths(
                    take_name=take_name,
                    review=decisions_path.parent,
                    decisions_path=decisions_path,
                    corrected_path=corrected_path,
                    audit_path=audit_path,
                    audit=audit,
                    qr_path=qr_path,
                    maximum_points=maximum_points,
                    hand_step_warning_m=hand_step_warning_m,
                    head_step_warning_m=head_step_warning_m,
                    hand_head_distance_warning_m=hand_head_distance_warning_m,
                    minimum_valid_fraction=minimum_valid_fraction,
                )
                if per_episode:
                    episodes = decisions["reviews"][0]["episodes"]
                    for local_episode_index, episode in enumerate(episodes):
                        episode_row, episode_arrays = _episode_record(
                            record,
                            arrays,
                            episode,
                            local_episode_index=local_episode_index,
                            global_episode_index=(
                                global_episode_index + local_episode_index
                            ),
                            task_spec=task_payload,
                            maximum_points=maximum_points,
                            hand_step_warning_m=hand_step_warning_m,
                            head_step_warning_m=head_step_warning_m,
                            minimum_valid_fraction=minimum_valid_fraction,
                        )
                        records.append(episode_row)
                        arrays_by_take[episode_row["record_id"]] = episode_arrays
                    global_episode_index += len(episodes)
                else:
                    records.append(record)
                    arrays_by_take[record["record_id"]] = arrays
        if len(task_ids) != 1:
            raise ValueError(
                f"trajectory batch must contain one task_id: {sorted(task_ids)}"
            )
        task_id = next(iter(task_ids))
        if task_payload and task_payload.get("task_id") != task_id:
            raise ValueError(
                f"task spec {task_payload.get('task_id')} does not match {task_id}"
            )
        _robust_centroid_scores(records)
        report = {
            "schema_version": 1,
            "task_id": task_id,
            "granularity": "episode" if per_episode else "recording",
            "input_root": str(input_root) if input_root else None,
            "recordings_manifest": (
                str(recordings_manifest) if recordings_manifest else None
            ),
            "thresholds": {
                "hand_training_step_warning_m": hand_step_warning_m,
                "head_training_step_warning_m": head_step_warning_m,
                "hand_head_distance_p99_warning_m": hand_head_distance_warning_m,
                "minimum_training_valid_fraction": minimum_valid_fraction,
                "qr_translation_std_warning_m": 0.02,
                "qr_reprojection_warning_px": 3.0,
                "qr_centroid_robust_z_warning": 15.0,
            },
            "rating_counts": {
                rating: sum(row["rating"] == rating for row in records)
                for rating in (
                    "PASS",
                    "PASS_AFTER_CORRECTION",
                    "REVIEW_REQUIRED",
                    "NOT_TRAINING",
                )
            },
            "takes": records,
        }
        write_json_atomic(temporary / "report.json", report)
        (temporary / "index.html").write_text(
            _html(records, task_id, per_episode=per_episode), encoding="utf-8"
        )
        _write_summary_csv(temporary / "summary.csv", records)
        if write_png:
            plots = temporary / "plots"
            plots.mkdir()
            for record in records:
                _write_take_png(
                    plots / f"{record['record_id']}.png",
                    record,
                    arrays_by_take[record["record_id"]],
                    hand_step_warning_m=hand_step_warning_m,
                    head_step_warning_m=head_step_warning_m,
                )
            _write_overview(temporary / "overview.png", records, plots, task_id)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "PASS",
        "takes": len(records),
        "rating_counts": report["rating_counts"],
        "output": str(output),
        "interactive_html": str(output / "index.html"),
        "report": str(output / "report.json"),
        "overview_png": str(output / "overview.png") if write_png else None,
    }
