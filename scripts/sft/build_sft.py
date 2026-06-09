import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


QUESTION_PATTERN = re.compile(r"Question:\s*(.*)\s*$", re.DOTALL)
TAG_PATTERN = re.compile(r"</?[^>]+>")


def json_safe(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def extract_prompt_content(row: dict):
    prompt = json_safe(row.get("prompt"))
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return first["content"]
    if isinstance(prompt, str):
        return prompt
    return None


def extract_question(prompt_content: str):
    match = QUESTION_PATTERN.search(prompt_content)
    if match:
        return " ".join(match.group(1).split())
    return " ".join(prompt_content.split())


def extract_targets(row: dict):
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict) and "target" in ground_truth:
            target = ground_truth["target"]
        else:
            target = None
    else:
        target = None

    if target is None:
        target = row.get("target", row.get("answers", row.get("answer")))
    target = json_safe(target)

    if isinstance(target, str):
        return [target]
    if isinstance(target, list):
        return [str(item) for item in target if str(item).strip()]
    return []


def make_clean_query(question: str, max_words: int):
    query = TAG_PATTERN.sub(" ", question)
    query = re.sub(r"[?？]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip(" .,:;")
    words = query.split()
    query = " ".join(words[:max_words]).strip(" .,:;")
    if re.fullmatch(r"(?:query|search)", query, re.IGNORECASE):
        query = f"{question.strip(' ?？')} fact"
    if re.fullmatch(r"(?:search|query)-[^\s]+", query, re.IGNORECASE):
        query = query.replace("-", " ")
    return query


def build_information(answer: str):
    return f"<information>Doc 1(Title: Answer evidence) The answer is {answer}.</information>"


def build_first_assistant(query: str):
    return (
        "<plan>\n"
        f"Step 1: Search {query}\n"
        "</plan>\n"
        "<think>I need external evidence for the question.</think>\n"
        f"<search>{query}</search>"
    )


def build_final_assistant(answer: str):
    return (
        "<think>The evidence is sufficient to answer the question.</think>\n"
        f"<answer>{answer}</answer>"
    )


def build_single_assistant(query: str, answer: str):
    return (
        f"{build_first_assistant(query)}\n"
        f"{build_information(answer)}\n"
        f"{build_final_assistant(answer)}"
    )


def iter_rows(paths: Iterable[Path]):
    for path in paths:
        frame = pd.read_parquet(path)
        for row_index, row in frame.iterrows():
            payload = row.to_dict()
            payload["_source_path"] = str(path)
            payload["_source_row"] = int(row_index)
            yield payload


def build_record(row: dict, max_query_words: int, conversation_format: str):
    prompt_content = extract_prompt_content(row)
    targets = extract_targets(row)
    if not prompt_content or not targets:
        return None

    question = extract_question(prompt_content)
    answer = targets[0]
    query = make_clean_query(question, max_words=max_query_words)
    if not query:
        return None

    if conversation_format == "single_assistant":
        messages = [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": build_single_assistant(query, answer)},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": build_first_assistant(query)},
            {"role": "user", "content": build_information(answer)},
            {"role": "assistant", "content": build_final_assistant(answer)},
        ]

    return {
        "messages": messages,
        "prompt": prompt_content,
        "response": build_single_assistant(query, answer),
        "metadata": {
            "source_path": row.get("_source_path"),
            "source_row": row.get("_source_row"),
            "data_source": row.get("data_source"),
            "question": question,
            "query": query,
            "target": targets,
            "sft_type": "template_answer_stub",
        },
    }


def positive_int(value: str):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Search-P1 format cold-start SFT JSONL from existing parquet data.",
    )
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Input parquet file(s).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=positive_int, default=None, help="Maximum records to write.")
    parser.add_argument("--max-query-words", type=positive_int, default=18)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle constructed records before writing.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--conversation-format",
        choices=("multi_turn", "single_assistant"),
        default="multi_turn",
        help="multi_turn keeps information outside assistant messages; single_assistant writes a full trajectory.",
    )
    parser.add_argument(
        "--output-format",
        choices=("jsonl", "verl_parquet"),
        default="jsonl",
        help="jsonl writes messages+metadata; verl_parquet writes prompt/response columns for verl FSDP SFT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    skipped = 0
    for row in iter_rows(args.input):
        record = build_record(
            row,
            max_query_words=args.max_query_words,
            conversation_format=args.conversation_format,
        )
        if record is None:
            skipped += 1
            continue
        records.append(record)
        if args.limit is not None and not args.shuffle and len(records) >= args.limit:
            break

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)
    if args.limit is not None:
        records = records[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output_format == "verl_parquet":
        frame = pd.DataFrame(
            [
                {
                    "prompt": record["prompt"],
                    "response": record["response"],
                    "metadata": record["metadata"],
                }
                for record in records
            ]
        )
        frame.to_parquet(args.output, index=False)
    else:
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.output}")
    if skipped:
        print(f"Skipped {skipped} rows without prompt or target")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
