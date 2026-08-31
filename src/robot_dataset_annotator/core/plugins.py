from __future__ import annotations

from importlib import import_module
from typing import cast

from robot_dataset_annotator.tasks.base import (
    BoundarySuggester,
    EpisodePoseQualityAuditor,
)

from .task_spec import TaskSpec


def load_suggester(task: TaskSpec) -> BoundarySuggester:
    if task.plugin is None:
        raise ValueError(
            f"task {task.task_id} requires manual boundaries and has no suggester"
        )
    module_name, separator, attribute = task.plugin.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"plugin must use module:attribute syntax: {task.plugin}")
    value = getattr(import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"task plugin is not callable: {task.plugin}")
    return cast(BoundarySuggester, value)


def load_episode_pose_quality_auditor(task: TaskSpec) -> EpisodePoseQualityAuditor:
    spec = task.episode_pose_quality
    if spec is None:
        raise ValueError(f"task {task.task_id} has no episode pose-quality audit")
    module_name, separator, attribute = spec.plugin.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "episode pose-quality plugin must use module:attribute syntax: "
            f"{spec.plugin}"
        )
    value = getattr(import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"episode pose-quality plugin is not callable: {spec.plugin}")
    return cast(EpisodePoseQualityAuditor, value)
