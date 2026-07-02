import importlib.util
import json
from pathlib import Path

import numpy as np


_REFERENCE_STEPS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data_process" / "reference_steps.py"
_SPEC = importlib.util.spec_from_file_location("reference_steps", _REFERENCE_STEPS_PATH)
reference_steps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference_steps)


def test_load_reference_steps_by_source_split_index(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text(
        json.dumps({
            "data_source": "nq",
            "split": "train",
            "index": 3,
            "reference_steps": ["Search Marie Curie Nobel Prize"],
        }) + "\n",
        encoding="utf-8",
    )

    references = reference_steps.load_reference_steps(str(path), max_reference_steps=4)
    steps = reference_steps.lookup_reference_steps(
        references,
        data_source="nq",
        split="train",
        index=3,
        question="unused",
    )

    assert steps == ["Search Marie Curie Nobel Prize"]


def test_load_reference_steps_by_question_fallback(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text(
        json.dumps({
            "question": "Who discovered radium",
            "reference_steps": ["Search who discovered radium"],
        }) + "\n",
        encoding="utf-8",
    )

    references = reference_steps.load_reference_steps(str(path), max_reference_steps=4)
    steps = reference_steps.lookup_reference_steps(
        references,
        data_source="hotpotqa",
        split="test",
        index=99,
        question="Who discovered radium?",
    )

    assert steps == ["Search who discovered radium"]


def test_invalid_reference_rows_are_filtered(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text(
        json.dumps({
            "question": "Who discovered radium?",
            "reference_steps": [
                "Search who discovered radium",
                "Search who discovered radium",
                "Search <search>bad</search>",
                "https://example.com/bad",
            ],
        }) + "\n",
        encoding="utf-8",
    )

    references = reference_steps.load_reference_steps(str(path), max_reference_steps=4)
    steps = reference_steps.lookup_reference_steps(
        references,
        data_source="nq",
        split="train",
        index=0,
        question="Who discovered radium?",
    )

    assert steps == ["Search who discovered radium"]


def test_too_many_reference_steps_drop_row(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text(
        json.dumps({
            "question": "Question?",
            "reference_steps": ["Search one", "Search two"],
        }) + "\n",
        encoding="utf-8",
    )

    references = reference_steps.load_reference_steps(str(path), max_reference_steps=1)

    assert references == {}


def test_has_reference_steps_reads_ground_truth_reference_steps():
    assert reference_steps.has_reference_steps({
        "reward_model": {
            "ground_truth": {
                "reference_steps": np.array(["Search who discovered radium"], dtype=object),
            },
        },
    })
    assert not reference_steps.has_reference_steps({
        "reward_model": {
            "ground_truth": {
                "reference_steps": [],
            },
        },
    })


def test_output_features_accept_empty_and_non_empty_reference_steps():
    train_merge_path = Path(__file__).resolve().parents[1] / "scripts" / "data_process" / "qa_search_train_merge.py"
    train_spec = importlib.util.spec_from_file_location("qa_search_train_merge", train_merge_path)
    qa_search_train_merge = importlib.util.module_from_spec(train_spec)
    try:
        train_spec.loader.exec_module(qa_search_train_merge)
    except ModuleNotFoundError as exc:
        if exc.name == "datasets":
            return
        raise

    datasets = qa_search_train_merge.datasets
    raw = datasets.Dataset.from_list([
        {
            "id": "raw-0",
            "question": "Who discovered radium?",
            "golden_answers": ["Marie Curie"],
        },
        {
            "id": "raw-1",
            "question": "What is the capital of France?",
            "golden_answers": ["Paris"],
        },
    ])

    def map_fn(example, idx):
        return {
            "data_source": "nq",
            "prompt": [{"role": "user", "content": example["question"]}],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "target": example["golden_answers"],
                    "reference_steps": ["Search who discovered radium"] if idx == 0 else [],
                },
            },
            "extra_info": {"split": "train", "index": idx},
        }

    mapped = raw.map(
        function=map_fn,
        with_indices=True,
        remove_columns=raw.column_names,
        features=qa_search_train_merge.output_features(),
    )

    assert "id" not in mapped.column_names
    assert mapped[0]["reward_model"]["ground_truth"]["reference_steps"] == ["Search who discovered radium"]
    assert mapped[1]["reward_model"]["ground_truth"]["reference_steps"] == []
