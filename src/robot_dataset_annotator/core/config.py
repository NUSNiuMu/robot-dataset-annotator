from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json


@dataclass(frozen=True)
class CheckSpec:
    path: str
    field: str
    expected: Any
    stage: str


@dataclass(frozen=True)
class SessionConfig:
    workspace: Path
    input_root: Path
    review_root: Path
    dataset_root: Path
    task_spec: Path
    input_glob: str = "*"
    start_item: str | None = None
    dataset_suffix: str = ""
    review_manifest: str = "manifest.json"
    decisions_file: str = "decisions.json"
    checks: tuple[CheckSpec, ...] = ()
    commands: dict[str, tuple[str, ...]] | None = None

    @classmethod
    def load(cls, path: Path) -> "SessionConfig":
        payload = read_json(path)
        base = path.resolve().parent

        def resolve(value: object, name: str) -> Path:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing session path: {name}")
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return candidate.resolve()
            return (base / candidate).resolve()

        checks = tuple(
            CheckSpec(
                path=str(item["path"]),
                field=str(item["field"]),
                expected=item["expected"],
                stage=str(item["stage"]),
            )
            for item in payload.get("checks", [])
        )
        commands: dict[str, tuple[str, ...]] = {}
        for action, command in payload.get("commands", {}).items():
            if not isinstance(command, list) or not command:
                raise ValueError(f"command {action!r} must be a non-empty argv array")
            commands[str(action)] = tuple(str(value) for value in command)
        start_item = payload.get("start_item")
        return cls(
            workspace=resolve(payload.get("workspace"), "workspace"),
            input_root=resolve(payload.get("input_root"), "input_root"),
            review_root=resolve(payload.get("review_root"), "review_root"),
            dataset_root=resolve(payload.get("dataset_root"), "dataset_root"),
            task_spec=resolve(payload.get("task_spec"), "task_spec"),
            input_glob=str(payload.get("input_glob", "*")),
            start_item=str(start_item) if start_item else None,
            dataset_suffix=str(payload.get("dataset_suffix", "")),
            review_manifest=str(payload.get("review_manifest", "manifest.json")),
            decisions_file=str(payload.get("decisions_file", "decisions.json")),
            checks=checks,
            commands=commands,
        )
