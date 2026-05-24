import importlib.util
import json
from pathlib import Path


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
                "Search <tool_call>bad</tool_call>",
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
