import importlib.util
from pathlib import Path

import numpy as np


_QA_EM_FORMAT_PATH = Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / "qa_em_format.py"
_SPEC = importlib.util.spec_from_file_location("qa_em_format", _QA_EM_FORMAT_PATH)
qa_em_format = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qa_em_format)


def test_reference_alignment_does_not_require_planner():
    solution = (
        "<think>I need evidence.</think>"
        "<search>Marie Curie Nobel Prize</search>"
        "<information>Doc</information>"
        "<think>I can answer.</think>"
        "<answer>radium</answer>"
    )
    ground_truth = {
        "target": ["radium"],
        "reference_steps": ["Search Marie Curie Nobel Prize"],
    }

    components = qa_em_format.compute_reference_alignment_components(solution, ground_truth)

    assert components["reference_alignment"] == 1.0
    assert components["ref_available"] == 1.0
    assert components["ref_n_steps"] == 1
    assert components["ref_n_actions"] == 1
    assert components["ref_n_covered"] == 1


def test_reference_alignment_accepts_parquet_array_reference_steps():
    solution = "<search>Marie Curie Nobel Prize</search>"
    ground_truth = {
        "target": np.array(["radium"], dtype=object),
        "reference_steps": np.array(["Search Marie Curie Nobel Prize"], dtype=object),
    }

    components = qa_em_format.compute_reference_alignment_components(solution, ground_truth)

    assert components["reference_alignment"] == 1.0
    assert components["ref_available"] == 1.0
    assert components["ref_n_steps"] == 1


def test_missing_reference_steps_return_zero_components():
    solution = "<search>Marie Curie Nobel Prize</search>"
    ground_truth = {"target": ["radium"]}

    components = qa_em_format.compute_reference_alignment_components(solution, ground_truth)

    assert components["reference_alignment"] == 0.0
    assert components["ref_available"] == 0.0
    assert components["ref_n_steps"] == 0
    assert components["ref_n_actions"] == 1
    assert components["ref_n_covered"] == 0


def test_duplicate_actions_do_not_inflate_reference_coverage():
    solution = (
        "<search>Marie Curie Nobel Prize</search>"
        "<search>Marie Curie Nobel Prize</search>"
    )
    ground_truth = {
        "target": ["radium"],
        "reference_steps": [
            "Search Marie Curie Nobel Prize",
            "Search Marie Curie discovery",
        ],
    }

    components = qa_em_format.compute_reference_alignment_components(solution, ground_truth)

    assert components["reference_alignment"] == 0.25
    assert components["ref_n_steps"] == 2
    assert components["ref_n_actions"] == 2
    assert components["ref_n_covered"] == 1


def test_reference_alignment_default_weight_does_not_change_final_score():
    solution = (
        "<search>Marie Curie Nobel Prize</search>"
        "<answer>radium</answer>"
    )
    without_reference = {"target": ["radium"]}
    with_reference = {
        "target": ["radium"],
        "reference_steps": ["Search Marie Curie Nobel Prize"],
    }

    no_ref_components = qa_em_format.compute_score_components(solution, without_reference)
    ref_components = qa_em_format.compute_score_components(solution, with_reference)

    assert no_ref_components["final_score"] == ref_components["final_score"]
    assert no_ref_components["reference_alignment"] == 0.0
    assert ref_components["reference_alignment"] == 1.0


def test_reference_alignment_weight_adds_track_b_bonus():
    solution = (
        "<plan>Step 1: Search Marie Curie Nobel Prize.</plan>"
        "<think>I need evidence.</think>"
        "<search>Marie Curie Nobel Prize</search>"
        "<information>Doc</information>"
        "<think>I can answer.</think>"
        "<answer>radium</answer>"
    )
    ground_truth = {
        "target": ["radium"],
        "reference_steps": ["Search Marie Curie Nobel Prize"],
    }

    components = qa_em_format.compute_score_components(
        solution,
        ground_truth,
        reference_alignment_weight=0.2,
    )

    assert components["reference_alignment"] == 1.0
    assert components["track_b_bonus"] == 0.2
    assert components["reference_alignment_weight"] == 0.2
    assert components["path_bonus"] == 0.2
    assert components["final_score"] == components["base_score"] + 0.2


def test_invalid_reference_steps_are_unavailable():
    solution = "<search>Marie Curie Nobel Prize</search>"
    ground_truth = {
        "target": ["radium"],
        "reference_steps": ["Search Marie Curie <search>Nobel Prize</search>"],
    }

    components = qa_em_format.compute_reference_alignment_components(solution, ground_truth)

    assert components["reference_alignment"] == 0.0
    assert components["ref_available"] == 0.0
