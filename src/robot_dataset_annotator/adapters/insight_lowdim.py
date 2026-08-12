from __future__ import annotations

from pathlib import Path

import numpy as np


def load_fused_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized Nx20 state and validity arrays from a review parquet.

    PyArrow remains an adapter dependency so importing the core package does not
    require ROS or the Insight capture repository.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("install robot-dataset-annotator[insight]") from exc
    table = pq.read_table(path)
    names = table.column_names
    state_name = next(
        (name for name in ("observation.state", "state") if name in names), None
    )
    valid_name = next(
        (name for name in ("observation.state_valid", "state_valid") if name in names),
        None,
    )
    if state_name is None or valid_name is None:
        raise ValueError("parquet must contain state and state_valid columns")
    state = np.asarray(table[state_name].to_pylist(), dtype=np.float64)
    valid = np.asarray(table[valid_name].to_pylist(), dtype=bool)
    if state.shape != valid.shape or state.ndim != 2 or state.shape[1] != 20:
        raise ValueError(
            f"expected matching Nx20 arrays, got {state.shape} and {valid.shape}"
        )
    return state, valid
