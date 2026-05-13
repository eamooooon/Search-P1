import importlib.util
from pathlib import Path

import pytest


def load_reward_module():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / "qa_em_format.py"
    spec = importlib.util.spec_from_file_location("qa_em_format_direct", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qa_em_format = load_reward_module()


PERFECT_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
Step 2: Search Albert Einstein Nobel Prize year.
</plan>
<reasoning>I need the birthplace.</reasoning>
<tool_call>Albert Einstein birthplace</tool_call>
<tool_response>Doc 1 says Ulm.</tool_response>
<reasoning>I need the Nobel year.</reasoning>
<tool_call>Albert Einstein Nobel Prize year</tool_call>
<tool_response>Doc 2 says 1921.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>Ulm and 1921</answer>"""


def test_self_consistency_perfect_match_records_components_without_bonus():
    components = qa_em_format.compute_score_components(
        PERFECT_TRAJECTORY,
        {"target": ["wrong"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        score=1.0,
    )

    assert components["self_consistency"] == 1.0
    assert components["self_r_planner"] == 1.0
    assert components["self_n_plan"] == 2
    assert components["self_n_actions"] == 2
    assert components["self_n_exec"] == 2
    assert "path_bonus" not in components
    assert components["final_score"] == components["base_score"]


def test_compute_score_em_preserves_base_score_when_self_consistency_is_positive():
    components = qa_em_format.compute_score_components(
        PERFECT_TRAJECTORY,
        {"target": ["wrong"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
    )

    score = qa_em_format.compute_score_em(
        PERFECT_TRAJECTORY,
        {"target": ["wrong"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
    )

    assert components["self_consistency"] > 0
    assert score == components["base_score"]


def test_redundant_action_lowers_self_consistency():
    redundant = PERFECT_TRAJECTORY.replace(
        "<reasoning>Now answer.</reasoning>",
        "<reasoning>Extra check.</reasoning>\n"
        "<tool_call>unrelated query</tool_call>\n"
        "<tool_response>Noise.</tool_response>\n"
        "<reasoning>Now answer.</reasoning>",
    )

    perfect_score = qa_em_format.compute_self_consistency_score(PERFECT_TRAJECTORY)
    redundant_score = qa_em_format.compute_self_consistency_score(redundant)

    assert redundant_score < perfect_score


def test_duplicate_actions_do_not_inflate_covered_steps():
    duplicate_action = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
Step 2: Search Albert Einstein Nobel Prize year.
</plan>
<reasoning>I need the birthplace.</reasoning>
<tool_call>Albert Einstein birthplace</tool_call>
<tool_response>Doc 1 says Ulm.</tool_response>
<reasoning>I repeat the same query.</reasoning>
<tool_call>Albert Einstein birthplace</tool_call>
<tool_response>Doc 1 again says Ulm.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>Ulm</answer>"""

    components = qa_em_format.compute_self_consistency_components(duplicate_action)

    assert components["self_n_plan"] == 2
    assert components["self_n_actions"] == 2
    assert components["self_n_exec"] == 1
    assert components["self_consistency"] == 0.25


def test_tool_response_content_is_not_counted_as_action():
    response_with_fake_action = PERFECT_TRAJECTORY.replace(
        "Doc 2 says 1921.",
        "<tool_call>fake response query</tool_call>",
    )

    assert qa_em_format.extract_tool_calls(response_with_fake_action) == [
        "Albert Einstein birthplace",
        "Albert Einstein Nobel Prize year",
    ]


@pytest.mark.parametrize(
    "trajectory",
    [
        PERFECT_TRAJECTORY.replace(
            "<reasoning>I need the birthplace.</reasoning>",
            "<plan>\nStep 3: Search duplicate planner.\n</plan>\n"
            "<reasoning>I need the birthplace.</reasoning>",
        ),
        PERFECT_TRAJECTORY.replace("<plan>", "intro\n<plan>", 1),
        PERFECT_TRAJECTORY.replace("Step 1: Search", "First search"),
        PERFECT_TRAJECTORY.replace("Step 2: Search", "Step 3: Search"),
    ],
)
def test_invalid_planner_blocks_zero_self_consistency(trajectory):
    components = qa_em_format.compute_self_consistency_components(trajectory)

    assert components["self_r_planner"] == 0.0
    assert components["self_consistency"] == 0.0


def test_missing_actions_zero_self_consistency():
    no_actions = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<reasoning>I will answer without search.</reasoning>
<answer>Ulm</answer>"""

    components = qa_em_format.compute_self_consistency_components(no_actions)

    assert components["self_n_actions"] == 0
    assert components["self_consistency"] == 0.0


def test_unsupported_match_strategy_fails_clearly():
    with pytest.raises(ValueError):
        qa_em_format.compute_self_consistency_score(PERFECT_TRAJECTORY, match_strategy="embedding")
