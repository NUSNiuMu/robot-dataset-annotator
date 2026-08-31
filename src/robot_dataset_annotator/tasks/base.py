from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class BoundarySuggester(Protocol):
    def __call__(
        self,
        observations: np.ndarray,
        observation_valid: np.ndarray,
        *,
        minimum_frames: int,
    ) -> dict[str, Any]: ...


class EpisodePoseQualityAuditor(Protocol):
    def __call__(
        self,
        comparisons: dict[str, dict[str, Any]],
        episodes: list[dict[str, int]],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]: ...
