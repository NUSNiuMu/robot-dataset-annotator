from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import read_json


@dataclass(frozen=True)
class AtomicAction:
    key: str
    label: str
    instruction: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    description: str
    plugin: str
    minimum_segment_frames: int
    actions: tuple[AtomicAction, ...]

    @classmethod
    def load(cls, path: Path) -> "TaskSpec":
        payload = read_json(path)
        actions = tuple(
            AtomicAction(
                key=str(item["key"]),
                label=str(item["label"]),
                instruction=str(item["instruction"]),
            )
            for item in payload.get("atomic_actions", [])
        )
        if not actions:
            raise ValueError("task spec must define at least one atomic action")
        minimum = int(payload.get("minimum_segment_frames", 1))
        if minimum < 1:
            raise ValueError("minimum_segment_frames must be positive")
        return cls(
            task_id=str(payload["task_id"]),
            description=str(payload["description"]),
            plugin=str(payload["plugin"]),
            minimum_segment_frames=minimum,
            actions=actions,
        )
