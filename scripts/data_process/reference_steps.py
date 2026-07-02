import json
import re
from collections.abc import Iterable, Mapping


_TAG_PATTERN = re.compile(r"</?[^>]+>")
_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


def normalize_question(question):
    if not isinstance(question, str):
        return ""
    normalized = " ".join(question.strip().split())
    if normalized and normalized[-1] != "?":
        normalized += "?"
    return normalized


def normalize_reference_steps(steps, max_reference_steps=None):
    if isinstance(steps, str):
        steps = [steps]
    if isinstance(steps, Mapping) or not isinstance(steps, Iterable):
        return []

    normalized_steps = []
    seen = set()
    for step in steps:
        if not isinstance(step, str):
            continue
        step = " ".join(step.strip().split())
        if not step:
            continue
        if _TAG_PATTERN.search(step) or _URL_PATTERN.search(step):
            continue
        if len(step.split()) > 32:
            continue
        step_key = step.lower()
        if step_key in seen:
            continue
        seen.add(step_key)
        normalized_steps.append(step)

    if max_reference_steps is not None and len(normalized_steps) > max_reference_steps:
        return []
    return normalized_steps


def _index_keys(entry):
    keys = []
    data_source = entry.get("data_source")
    split = entry.get("split")
    index = entry.get("index")
    if data_source is not None and split is not None and index is not None:
        keys.append(("id", str(data_source), str(split), str(index)))

    question = normalize_question(entry.get("question"))
    if data_source is not None and question:
        keys.append(("source_question", str(data_source), question))
    if question:
        keys.append(("question", question))
    return keys


def load_reference_steps(path, max_reference_steps=None):
    if not path:
        return {}

    references = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            steps = normalize_reference_steps(
                entry.get("reference_steps"),
                max_reference_steps=max_reference_steps,
            )
            if not steps:
                continue
            keys = _index_keys(entry)
            if not keys:
                raise ValueError(f"Reference row {line_number} has no usable key")
            for key in keys:
                references[key] = steps
    return references


def lookup_reference_steps(references, data_source, split, index, question):
    if not references:
        return []
    question = normalize_question(question)
    candidates = [
        ("id", str(data_source), str(split), str(index)),
        ("source_question", str(data_source), question),
        ("question", question),
    ]
    for key in candidates:
        steps = references.get(key)
        if steps:
            return list(steps)
    return []


def has_reference_steps(example):
    if not isinstance(example, dict):
        return False
    reward_model = example.get("reward_model")
    if not isinstance(reward_model, dict):
        return False
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return False
    steps = ground_truth.get("reference_steps")
    if isinstance(steps, str):
        return bool(steps.strip())
    if isinstance(steps, Mapping) or not isinstance(steps, Iterable):
        return False
    return any(isinstance(step, str) and step.strip() for step in steps)
