from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import read_json, write_json_atomic


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column(table: Any, name: str, dtype: Any) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"dataset table is missing {name}")
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype)


def _validate_low_dimensional_data(
    table: Any, episodes: list[dict[str, Any]], fps: float
) -> dict[str, Any]:
    row_count = len(table)
    index = _column(table, "index", np.int64)
    episode_index = _column(table, "episode_index", np.int64)
    frame_index = _column(table, "frame_index", np.int64)
    timestamps = _column(table, "timestamp", np.float64)
    source_frames = _column(table, "annotation.source_frame_index", np.int64)
    atomic_actions = _column(table, "annotation.atomic_action_index", np.int64)
    task_indices = _column(table, "task_index", np.int64)
    state = _column(table, "observation.state", np.float32)
    state_valid = _column(table, "observation.state_valid", np.uint8)
    head = _column(table, "observation.head_pose", np.float32)
    head_valid = _column(table, "observation.head_pose_valid", np.uint8)
    action = _column(table, "action", np.float32)
    action_valid = _column(table, "action_is_valid", np.uint8)

    if not np.array_equal(index, np.arange(row_count)):
        raise ValueError("global dataset index is not contiguous")
    if state.shape != (row_count, 18) or action.shape != (row_count, 18):
        raise ValueError("state and action must both have shape Nx18")
    if head.shape != (row_count, 9):
        raise ValueError("head pose must have shape Nx9")
    for values, valid, label in (
        (state, state_valid, "state"),
        (head, head_valid, "head pose"),
        (action, action_valid, "action"),
    ):
        if values.shape != valid.shape:
            raise ValueError(f"{label} validity shape does not match values")
        if not np.isin(valid, (0, 1)).all():
            raise ValueError(f"{label} validity is not binary")
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains non-finite values")
        if np.any(values[valid == 0] != 0):
            raise ValueError(f"{label} invalid values are not zero-filled")

    if sum(int(row["length"]) for row in episodes) != row_count:
        raise ValueError("episode lengths do not sum to the dataset row count")
    action_to_task: dict[int, int] = {}
    for expected_episode, metadata in enumerate(episodes):
        start = int(metadata["dataset_from_index"])
        end = int(metadata["dataset_to_index"])
        length = int(metadata["length"])
        if end - start != length:
            raise ValueError(f"episode {expected_episode} has inconsistent offsets")
        selection = slice(start, end)
        if not np.all(episode_index[selection] == expected_episode):
            raise ValueError(f"episode {expected_episode} index column mismatch")
        if not np.array_equal(frame_index[selection], np.arange(length)):
            raise ValueError(
                f"episode {expected_episode} frame index is not contiguous"
            )
        expected_timestamps = np.arange(length, dtype=np.float64) / fps
        if not np.allclose(
            timestamps[selection], expected_timestamps, rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"episode {expected_episode} timestamps are invalid")
        if not np.all(np.diff(source_frames[selection]) == 1):
            raise ValueError(
                f"episode {expected_episode} source frames are not contiguous"
            )
        episode_actions = atomic_actions[selection]
        if not np.isin(episode_actions, (0, 1)).all():
            raise ValueError(f"episode {expected_episode} has an unknown atomic action")
        if not np.array_equal(np.unique(episode_actions), np.asarray([0, 1])):
            raise ValueError(
                f"episode {expected_episode} does not contain both actions"
            )
        if np.any(np.diff(episode_actions) < 0):
            raise ValueError(f"episode {expected_episode} action order regresses")
        for atomic_action in (0, 1):
            matched_tasks = np.unique(
                task_indices[selection][episode_actions == atomic_action]
            )
            if len(matched_tasks) != 1:
                raise ValueError(
                    f"episode {expected_episode} action {atomic_action} has "
                    "inconsistent task indices"
                )
            previous = action_to_task.setdefault(atomic_action, int(matched_tasks[0]))
            if previous != int(matched_tasks[0]):
                raise ValueError(
                    f"atomic action {atomic_action} changes task index across episodes"
                )
        episode_state = state[selection]
        episode_valid = state_valid[selection]
        episode_action = action[selection]
        episode_action_valid = action_valid[selection]
        if not np.array_equal(episode_action[:-1], episode_state[1:]):
            raise ValueError(
                f"episode {expected_episode} next-frame actions are invalid"
            )
        if not np.array_equal(episode_action[-1], episode_state[-1]):
            raise ValueError(f"episode {expected_episode} final action is invalid")
        if not np.array_equal(episode_action_valid[:-1], episode_valid[1:]):
            raise ValueError(f"episode {expected_episode} action validity is invalid")
        if not np.array_equal(episode_action_valid[-1], episode_valid[-1]):
            raise ValueError(
                f"episode {expected_episode} final action validity is invalid"
            )

    if len(set(action_to_task.values())) != 2:
        raise ValueError("atomic actions do not map to distinct tasks")
    return {
        "rows": row_count,
        "episodes": len(episodes),
        "state_shape": [row_count, 18],
        "head_pose_shape": [row_count, 9],
        "action_shape": [row_count, 18],
        "timestamps": "PASS",
        "episode_offsets": "PASS",
        "source_frame_continuity": "PASS",
        "atomic_action_order": "PASS",
        "task_indices": "PASS",
        "next_frame_action_semantics": "PASS",
        "validity_masks": "PASS",
    }


def _decode_videos(
    root: Path, video_features: dict[str, dict[str, Any]], expected_frames: int
) -> dict[str, Any]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - optional validator dependency
        raise RuntimeError("PyAV is required for complete video validation") from exc

    checks: dict[str, Any] = {}
    for key, feature in video_features.items():
        paths = sorted((root / "videos" / key).glob("chunk-*/*.mp4"))
        if not paths:
            raise ValueError(f"no video files found for {key}")
        decoded_total = 0
        file_rows: list[dict[str, Any]] = []
        expected_height, expected_width, _ = feature["shape"]
        for path in paths:
            with av.open(str(path), mode="r") as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                decoded = 0
                for frame in container.decode(stream):
                    if frame.width != expected_width or frame.height != expected_height:
                        raise ValueError(
                            f"{path} decoded {frame.width}x{frame.height}; expected "
                            f"{expected_width}x{expected_height}"
                        )
                    decoded += 1
            decoded_total += decoded
            file_rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "frames": decoded,
                    "sha256": _sha256(path),
                }
            )
        if decoded_total != expected_frames:
            raise ValueError(
                f"{key} decoded {decoded_total} frames; expected {expected_frames}"
            )
        checks[key] = {
            "status": "PASS",
            "decoded_frames": decoded_total,
            "files": file_rows,
        }
    return checks


def validate_lerobot_dataset(root: Path) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional validator dependency
        raise RuntimeError("PyArrow is required for LeRobot validation") from exc

    info = read_json(root / "meta" / "info.json")
    export = read_json(root / "rda" / "export_manifest.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("dataset is not LeRobotDataset v3.0")
    expected_frames = int(export["frames"])
    expected_episodes = int(export["episodes"])
    if int(info.get("total_frames", -1)) != expected_frames:
        raise ValueError("LeRobot info frame count disagrees with export manifest")
    if int(info.get("total_episodes", -1)) != expected_episodes:
        raise ValueError("LeRobot info episode count disagrees with export manifest")

    data_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not data_paths or not episode_paths:
        raise ValueError("dataset is missing data or episode parquet files")
    table = pa.concat_tables([pq.read_table(path) for path in data_paths])
    episode_table = pa.concat_tables([pq.read_table(path) for path in episode_paths])
    episodes = episode_table.to_pylist()
    if len(episodes) != expected_episodes:
        raise ValueError("episode metadata row count is incorrect")
    low_dimensional = _validate_low_dimensional_data(
        table, episodes, float(info["fps"])
    )

    video_features = {
        key: feature
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    }
    if len(video_features) != 3:
        raise ValueError("dataset must contain exactly three video features")
    videos = _decode_videos(root, video_features, expected_frames)
    excluded_reports = {
        "rda/internal_validation.json",
        "rda/official_validation.json",
    }
    artifact_hashes = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded_reports
    }
    internal = {
        "schema_version": 1,
        "status": "PASS",
        "validator": "robot-dataset-annotator 0.1.0",
        "format": info["codebase_version"],
        "low_dimensional": low_dimensional,
        "videos": videos,
        "artifact_hashes": artifact_hashes,
    }
    write_json_atomic(root / "rda" / "internal_validation.json", internal)

    try:
        import av
        import pyarrow
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional validator dependency
        raise RuntimeError(
            "official LeRobot validation dependencies are missing"
        ) from exc

    dataset = LeRobotDataset(repo_id=str(export["repo_id"]), root=root)
    if len(dataset) != expected_frames or dataset.num_episodes != expected_episodes:
        raise ValueError("official LeRobot loader reported incorrect totals")
    representative = sorted(
        {0, expected_frames // 2, expected_frames - 1}
        | {int(row["dataset_from_index"]) for row in episodes}
    )
    expected_keys = set(info["features"]) | {"task"}
    indexed_shapes: dict[str, list[int]] = {}
    for index in representative:
        row = dataset[index]
        missing = sorted(expected_keys - set(row))
        if missing:
            raise ValueError(f"official loader row {index} is missing keys: {missing}")
        for key in video_features:
            indexed_shapes[key] = list(row[key].shape)
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    if int(batch["action"].shape[0]) != 2:
        raise ValueError("official DataLoader did not return a two-row batch")
    official = {
        "schema_version": 1,
        "status": "PASS",
        "loader": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
        "versions": {
            "lerobot": version("lerobot"),
            "torch": torch.__version__,
            "pyarrow": pyarrow.__version__,
            "av": av.__version__,
        },
        "rows": len(dataset),
        "episodes": dataset.num_episodes,
        "representative_indices": representative,
        "representative_video_shapes": indexed_shapes,
        "dataloader_batch_size": int(batch["action"].shape[0]),
    }
    write_json_atomic(root / "rda" / "official_validation.json", official)
    return {"internal": internal, "official": official}
