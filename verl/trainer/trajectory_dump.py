import json
import math
from collections.abc import Mapping
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is present in training envs.
    np = None


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if np is not None:
        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _append_trajectory_dump(
    trajectory_dump_path,
    *,
    solution_str,
    ground_truth,
    data_source,
    split,
    index=None,
    track_a=None,
    track_b=None,
    prompt=None,
    extra_info=None,
    reward_components=None,
):
    path = Path(trajectory_dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "data_source": data_source,
        "split": split,
    }
    if index is not None:
        row["index"] = index
    if track_a is not None:
        row["track_a"] = track_a
    if track_b is not None:
        row["track_b"] = track_b
    if prompt is not None:
        row["prompt"] = prompt
    if extra_info is not None:
        row["extra_info"] = extra_info
    if reward_components is not None:
        row["reward_components"] = reward_components

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
