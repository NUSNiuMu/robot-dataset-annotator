from __future__ import annotations

from importlib import import_module
from typing import cast

from robot_dataset_annotator.tasks.base import BoundarySuggester

from .task_spec import TaskSpec


def load_suggester(task: TaskSpec) -> BoundarySuggester:
    module_name, separator, attribute = task.plugin.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"plugin must use module:attribute syntax: {task.plugin}")
    value = getattr(import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"task plugin is not callable: {task.plugin}")
    return cast(BoundarySuggester, value)
