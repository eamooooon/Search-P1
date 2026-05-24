import importlib.util
from pathlib import Path


_QA_EM_FORMAT_PATH = Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / "qa_em_format.py"
_SPEC = importlib.util.spec_from_file_location("qa_em_format", _QA_EM_FORMAT_PATH)
qa_em_format = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qa_em_format)


def test_reference_alignment_does_not_require_planner():
    solution = (
        "<reasoning>I need evidence.</reasoning>"
        "<tool_call>Marie Curie Nobel Prize</tool_call>"
        "<tool_response>Doc</tool_response>"
        "<reasoning>I can answer.</reasoning>"
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


def test_tool_calls_inside_tool_response_are_ignored():
    solution = (
        "<tool_response>Doc says <tool_call>fake query</tool_call></tool_response>"
        "<tool_call>Marie Curie Nobel Prize</tool_call>"
    )

    assert qa_em_format.extract_tool_calls(solution) == ["Marie Curie Nobel Prize"]


def test_planner_block_requires_only_sequential_step_lines():
    valid = (
        "<plan>\n"
        "Step 1: Search Marie Curie Nobel Prize\n"
        "Step 2: Search radium discovery\n"
        "</plan>"
    )
    extra_line = (
        "<plan>\n"
        "I will search first.\n"
        "Step 1: Search Marie Curie Nobel Prize\n"
        "</plan>"
    )
    skipped_number = (
        "<plan>\n"
        "Step 1: Search Marie Curie Nobel Prize\n"
        "Step 3: Search radium discovery\n"
        "</plan>"
    )

    assert qa_em_format.validate_planner_block(valid)
    assert not qa_em_format.validate_planner_block(extra_line)
    assert not qa_em_format.validate_planner_block(skipped_number)


def test_extract_solution_accepts_single_answer_after_assistant_marker():
    solution = (
        "<|im_start|>assistant\n"
        "<plan>\nStep 1: Search radium discovery\n</plan>"
        "<reasoning>I know it.</reasoning><answer>Marie Curie</answer>"
        "<|im_end|>"
    )

    assert qa_em_format.extract_solution(solution) == "Marie Curie"


def test_missing_reference_steps_return_zero_components():
    solution = "<tool_call>Marie Curie Nobel Prize</tool_call>"
    ground_truth = {"target": ["radium"]}

    components = qa_em_format.compute_reference_alignment_components(solution, ground_truth)

    assert components["reference_alignment"] == 0.0
    assert components["ref_available"] == 0.0
    assert components["ref_n_steps"] == 0
    assert components["ref_n_actions"] == 1
    assert components["ref_n_covered"] == 0


def test_duplicate_actions_do_not_inflate_reference_coverage():
    solution = (
        "<tool_call>Marie Curie Nobel Prize</tool_call>"
        "<tool_call>Marie Curie Nobel Prize</tool_call>"
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


def test_reference_alignment_is_observation_not_final_score():
    solution = (
        "<tool_call>Marie Curie Nobel Prize</tool_call>"
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


def test_invalid_reference_steps_are_unavailable():
    solution = "<tool_call>Marie Curie Nobel Prize</tool_call>"
    ground_truth = {
        "target": ["radium"],
        "reference_steps": ["Search Marie Curie <tool_call>Nobel Prize</tool_call>"],
    }

    components = qa_em_format.compute_reference_alignment_components(solution, ground_truth)

    assert components["reference_alignment"] == 0.0
    assert components["ref_available"] == 0.0
