from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .audit import AuditRow, Stage, audit_batch
from .config import SessionConfig


@dataclass(frozen=True)
class PreparedAction:
    row: AuditRow
    argv: tuple[str, ...]


def _context(config: SessionConfig, item: str) -> dict[str, str]:
    review_dir = config.review_root / item
    dataset_dir = config.dataset_root / f"{item}{config.dataset_suffix}"
    return {
        "workspace": str(config.workspace),
        "input_root": str(config.input_root),
        "review_root": str(config.review_root),
        "dataset_root": str(config.dataset_root),
        "task_spec": str(config.task_spec),
        "item": item,
        "input": str(config.input_root / item),
        "review_dir": str(review_dir),
        "review_manifest": str(review_dir / config.review_manifest),
        "decisions": str(review_dir / config.decisions_file),
        "dataset_dir": str(dataset_dir),
    }


def _render(command: tuple[str, ...], context: dict[str, str]) -> tuple[str, ...]:
    try:
        return tuple(value.format_map(context) for value in command)
    except KeyError as exc:
        raise ValueError(f"unknown command placeholder: {exc.args[0]}") from exc


def prepare_next(
    config: SessionConfig, *, item_name: str | None = None
) -> PreparedAction | None:
    rows = audit_batch(config)
    if item_name is not None:
        rows = [row for row in rows if row.item == item_name]
        if not rows:
            raise ValueError(f"item is not in session scope: {item_name}")
    row = next(
        (candidate for candidate in rows if candidate.stage is not Stage.COMPLETE),
        None,
    )
    if row is None:
        return None
    commands = config.commands or {}
    command = commands.get(row.next_action)
    if command is None:
        raise ValueError(
            f"session has no command for action {row.next_action!r}; "
            "manual review actions may intentionally remain unconfigured"
        )
    return PreparedAction(row, _render(command, _context(config, row.item)))


def execute(action: PreparedAction, *, check: bool = False) -> int:
    result = subprocess.run(action.argv, check=False)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, action.argv)
    return int(result.returncode)
