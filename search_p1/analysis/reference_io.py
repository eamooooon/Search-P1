import json


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            yield row


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_question(question):
    if not isinstance(question, str):
        return ""
    question = " ".join(question.strip().split())
    if question and question[-1] != "?":
        question += "?"
    return question


def row_question(row):
    question = row.get("question")
    if question:
        return normalize_question(question)

    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        content = prompt[-1].get("content", "") if isinstance(prompt[-1], dict) else ""
        marker = "Question:"
        if marker in content:
            return normalize_question(content.rsplit(marker, 1)[-1])
    return ""


def row_ground_truth(row):
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, dict):
        return ground_truth

    reward_model = row.get("reward_model", {})
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict):
            return ground_truth

    target = row.get("target") or row.get("golden_answers")
    if target is not None:
        return {"target": target}
    return {}


def row_key(row):
    data_source = row.get("data_source")
    split = row.get("split")
    index = row.get("index")

    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, dict):
        split = split if split is not None else extra_info.get("split")
        index = index if index is not None else extra_info.get("index")

    question = row_question(row)
    if data_source is not None and split is not None and index is not None:
        return ("id", str(data_source), str(split), str(index))
    if data_source is not None and question:
        return ("source_question", str(data_source), question)
    if question:
        return ("question", question)
    return None


def key_metadata(row):
    data_source = row.get("data_source")
    split = row.get("split")
    index = row.get("index")
    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, dict):
        split = split if split is not None else extra_info.get("split")
        index = index if index is not None else extra_info.get("index")
    return {
        "data_source": data_source,
        "split": split,
        "index": index,
        "question": row_question(row),
    }


def solution_str(row):
    return row.get("solution_str") or row.get("trajectory") or row.get("response") or ""


def key_from_custom_id(custom_id):
    if not custom_id:
        return None
    return tuple(str(custom_id).split("|"))
