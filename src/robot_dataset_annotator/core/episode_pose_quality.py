from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from .io import read_json


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_episode_pose_audit_for_export(
    audit_path: Path,
    *,
    review_manifest_path: Path,
    decisions_path: Path,
    task_path: Path,
) -> dict[str, Any]:
    audit = read_json(audit_path)
    task = read_json(task_path)
    decisions = read_json(decisions_path)
    review_manifest = read_json(review_manifest_path)
    if audit.get("schema_version") != 1:
        raise ValueError("unsupported episode pose-quality audit schema")
    expected_hashes = {
        "review_manifest_sha256": _sha256(review_manifest_path),
        "decisions_sha256": _sha256(decisions_path),
        "task_spec_sha256": _sha256(task_path),
    }
    for key, expected in expected_hashes.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"episode pose-quality audit does not match current {key[:-7]}"
            )
    correction = review_manifest.get("pose_drift_correction")
    correction_evidence = audit.get("pose_drift_correction")
    if correction is not None:
        if not isinstance(correction, dict) or correction.get("status") != "PASS":
            raise ValueError("review manifest pose-drift correction is not PASS")
        if not isinstance(correction_evidence, dict) or correction_evidence.get(
            "status"
        ) != "PASS":
            raise ValueError("episode pose audit lacks PASS pose-drift evidence")
        if correction_evidence.get("source_manifest_sha256") != correction.get(
            "source_manifest_sha256"
        ):
            raise ValueError("episode pose audit pose-drift source hash is stale")
        correction_audit_path = Path(
            str(correction_evidence.get("audit", ""))
        ).expanduser()
        if not correction_audit_path.is_absolute():
            correction_audit_path = review_manifest_path.parent / correction_audit_path
        if not correction_audit_path.is_file() or _sha256(
            correction_audit_path
        ) != correction_evidence.get("audit_sha256"):
            raise ValueError("episode pose audit pose-drift evidence is stale")
        correction_audit = read_json(correction_audit_path)
        if correction_audit.get("status") != "PASS":
            raise ValueError("episode pose audit references a non-PASS drift audit")
    elif correction_evidence is not None:
        raise ValueError("episode pose audit has unexpected pose-drift evidence")
    if audit.get("task_id") != task.get("task_id"):
        raise ValueError("episode pose-quality audit task_id does not match task spec")
    if audit.get("subsequent_episodes_evaluated_independently") is not True:
        raise ValueError(
            "episode pose-quality audit lacks independent subsequent-episode evidence"
        )
    episodes = audit.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episode pose-quality audit has no episodes")
    expected_episode_count = 0
    for review in decisions.get("reviews", []):
        if str(review.get("visual_status", "")).upper() != "PASS":
            continue
        rows = review.get("episodes")
        expected_episode_count += len(rows) if isinstance(rows, list) else 1
    if len(episodes) != expected_episode_count:
        raise ValueError(
            "episode pose-quality audit episode count does not match decisions"
        )
    unusable: list[int] = []
    for index, row in enumerate(episodes):
        if not isinstance(row, dict) or row.get("episode_index") != index:
            raise ValueError("episode pose-quality audit indices are not contiguous")
        roles = row.get("roles")
        role_evidence_passes = isinstance(roles, dict) and all(
            isinstance(roles.get(role), dict)
            and roles[role].get("training_usable") is True
            for role in ("left_hand", "right_hand")
        )
        if row.get("training_usable") is not True or not role_evidence_passes:
            unusable.append(index)
    if audit.get("status") != "PASS" or unusable:
        raise ValueError(
            "episode pose-quality audit has unusable reviewed episodes: "
            f"{unusable}"
        )
    if audit.get("usable_episode_indices") != list(range(len(episodes))):
        raise ValueError("episode pose-quality audit usable indices are inconsistent")
    if audit.get("unusable_episode_indices") != []:
        raise ValueError("episode pose-quality audit unusable indices are inconsistent")
    return audit
