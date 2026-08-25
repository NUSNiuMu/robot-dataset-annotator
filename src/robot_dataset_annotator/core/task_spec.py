from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import read_json


@dataclass(frozen=True)
class SubtaskSemantic:
    key: str
    label: str
    instruction: str


@dataclass(frozen=True)
class AtomicAction:
    key: str
    label: str
    instruction: str
    left_hand: SubtaskSemantic | None = None
    right_hand: SubtaskSemantic | None = None

    def for_hand(self, hand: str) -> SubtaskSemantic:
        semantic = self.left_hand if hand == "left_hand" else self.right_hand
        if semantic is not None:
            return semantic
        return SubtaskSemantic(
            key=f"{hand}_{self.key}",
            label=f"{hand}_{self.label}",
            instruction=f"{hand.replace('_', ' ').title()}: {self.instruction}",
        )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    description: str
    plugin: str | None
    minimum_segment_frames: int
    actions: tuple[AtomicAction, ...]
    context_action: AtomicAction | None = None

    @staticmethod
    def _semantic(payload: dict[str, str]) -> SubtaskSemantic:
        return SubtaskSemantic(
            key=str(payload["key"]),
            label=str(payload["label"]),
            instruction=str(payload["instruction"]),
        )

    @classmethod
    def _action(cls, payload: dict[str, object]) -> AtomicAction:
        hand_subtasks = payload.get("hand_subtasks", {})
        if not isinstance(hand_subtasks, dict):
            raise ValueError("hand_subtasks must be an object")
        left = hand_subtasks.get("left_hand")
        right = hand_subtasks.get("right_hand")
        if left is not None and not isinstance(left, dict):
            raise ValueError("left_hand subtask must be an object")
        if right is not None and not isinstance(right, dict):
            raise ValueError("right_hand subtask must be an object")
        return AtomicAction(
            key=str(payload["key"]),
            label=str(payload["label"]),
            instruction=str(payload["instruction"]),
            left_hand=cls._semantic(left) if left is not None else None,
            right_hand=cls._semantic(right) if right is not None else None,
        )

    @classmethod
    def load(cls, path: Path) -> "TaskSpec":
        payload = read_json(path)
        actions = tuple(cls._action(item) for item in payload.get("atomic_actions", []))
        if not actions:
            raise ValueError("task spec must define at least one atomic action")
        minimum = int(payload.get("minimum_segment_frames", 1))
        if minimum < 1:
            raise ValueError("minimum_segment_frames must be positive")
        context_payload = payload.get("context_action")
        if context_payload is not None and not isinstance(context_payload, dict):
            raise ValueError("context_action must be an object")
        return cls(
            task_id=str(payload["task_id"]),
            description=str(payload["description"]),
            plugin=(str(payload["plugin"]) if payload.get("plugin") else None),
            minimum_segment_frames=minimum,
            actions=actions,
            context_action=(
                cls._action(context_payload) if context_payload is not None else None
            ),
        )

    def subtask_index(self, action_index: int, hand: str) -> int:
        if hand not in {"left_hand", "right_hand"}:
            raise ValueError(f"unknown hand: {hand}")
        if action_index < -1 or action_index >= len(self.actions):
            raise ValueError(f"unknown action index: {action_index}")
        if action_index == -1:
            if self.context_action is None:
                raise ValueError("task has no context action")
            phase = 0
        else:
            phase = action_index + (1 if self.context_action is not None else 0)
        return 2 * phase + (1 if hand == "right_hand" else 0)

    def subtask_catalog(self) -> list[dict[str, str | int]]:
        result: list[dict[str, str | int]] = []
        phases: list[tuple[int, AtomicAction]] = []
        if self.context_action is not None:
            phases.append((-1, self.context_action))
        phases.extend(enumerate(self.actions))
        for action_index, action in phases:
            for hand in ("left_hand", "right_hand"):
                semantic = action.for_hand(hand)
                result.append(
                    {
                        "index": self.subtask_index(action_index, hand),
                        "atomic_action_index": action_index,
                        "hand": hand,
                        "key": semantic.key,
                        "label": semantic.label,
                        "instruction": semantic.instruction,
                    }
                )
        return result
