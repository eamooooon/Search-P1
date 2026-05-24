import json
import os


def _to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def append_trajectory_dump(path,
                           *,
                           solution_str,
                           ground_truth,
                           data_source,
                           prompt=None,
                           extra_info=None,
                           reward_components=None,
                           split=None,
                           index=None):
    if not path:
        return

    extra_info = extra_info if isinstance(extra_info, dict) else {}
    if split is None:
        split = extra_info.get("split")
    if index is None:
        index = extra_info.get("index")

    row = {
        "solution_str": solution_str,
        "ground_truth": _to_jsonable(ground_truth),
        "data_source": _to_jsonable(data_source),
        "split": _to_jsonable(split),
        "index": _to_jsonable(index),
        "prompt": _to_jsonable(prompt),
        "extra_info": _to_jsonable(extra_info),
        "reward_components": _to_jsonable(reward_components or {}),
    }
    row = {key: value for key, value in row.items() if value is not None}

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
