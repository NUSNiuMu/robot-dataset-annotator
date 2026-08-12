from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .config import CheckSpec, SessionConfig
from .decisions import validate_decision_files
from .io import read_json


class Stage(str, Enum):
    DISCOVERED = "DISCOVERED"
    REVIEW_READY = "REVIEW_READY"
    REVIEW_DECIDED = "REVIEW_DECIDED"
    EXPORTED = "EXPORTED"
    INTERNAL_VALID = "INTERNAL_VALID"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class AuditRow:
    item: str
    stage: Stage
    next_action: str
    error: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "item": self.item,
            "stage": self.stage.value,
            "next_action": self.next_action,
            "error": self.error,
        }


def _field(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def _check(root: Path, check: CheckSpec) -> tuple[bool, str]:
    path = root / check.path
    if not path.is_file():
        return False, f"missing {check.path}"
    try:
        actual = _field(read_json(path), check.field)
    except (OSError, ValueError, KeyError) as exc:
        return False, f"invalid {check.path}: {exc}"
    if actual != check.expected:
        return False, f"{check.path}:{check.field}={actual!r}"
    return True, ""


def discover_items(config: SessionConfig) -> list[Path]:
    if not config.input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {config.input_root}")
    items = sorted(
        path for path in config.input_root.glob(config.input_glob) if path.is_dir()
    )
    if config.start_item:
        items = [path for path in items if path.name >= config.start_item]
    return items


def audit_item(config: SessionConfig, item: Path) -> AuditRow:
    review = config.review_root / item.name
    manifest = review / config.review_manifest
    decisions = review / config.decisions_file
    dataset = config.dataset_root / f"{item.name}{config.dataset_suffix}"
    if not manifest.is_file():
        return AuditRow(item.name, Stage.DISCOVERED, "prepare_review")
    if not decisions.is_file():
        return AuditRow(item.name, Stage.REVIEW_READY, "review_all_segments")
    try:
        validate_decision_files(manifest, decisions, config.task_spec)
    except (OSError, ValueError, KeyError) as exc:
        return AuditRow(item.name, Stage.REVIEW_READY, "fix_decisions", str(exc))
    if not dataset.is_dir():
        return AuditRow(item.name, Stage.REVIEW_DECIDED, "export_dataset")
    if not config.checks:
        return AuditRow(item.name, Stage.EXPORTED, "configure_validation_checks")
    failed: list[str] = []
    passed_stages: set[str] = set()
    for check in config.checks:
        ok, detail = _check(dataset, check)
        if ok:
            passed_stages.add(check.stage)
        else:
            failed.append(detail)
    if failed:
        stage = Stage.INTERNAL_VALID if "internal" in passed_stages else Stage.EXPORTED
        return AuditRow(item.name, stage, "run_validation", "; ".join(failed))
    return AuditRow(item.name, Stage.COMPLETE, "skip")


def audit_batch(config: SessionConfig) -> list[AuditRow]:
    return [audit_item(config, item) for item in discover_items(config)]
