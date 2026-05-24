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


INTENT_INSTANTIATED_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search [identified actress] character in The Honeymooners.
</plan>
<reasoning>I need the character for the identified actress.</reasoning>
<tool_call>Joyce Randolph Trixie Norton The Honeymooners</tool_call>
<tool_response>Doc 1 says Joyce Randolph played Trixie Norton.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>Trixie Norton</answer>"""


INTENT_TOO_GENERIC_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search [identified winner] nationality.
</plan>
<reasoning>I need the winner nationality.</reasoning>
<tool_call>Jonas Vingegaard nationality</tool_call>
<tool_response>Doc 1 says Jonas Vingegaard is Danish.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>Danish</answer>"""


NO_SEARCH_WRONG_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<reasoning>I will answer without search.</reasoning>
<answer>wrong</answer>"""


FAKED_TOOL_RESPONSE_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<reasoning>I will fake evidence without search.</reasoning>
<tool_response>Doc 1 says Ulm.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>wrong</answer>"""


NO_ANSWER_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<reasoning>I never provide a final answer.</reasoning>"""


MALFORMED_TOOL_CALL_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<reasoning>I will emit an invalid search action.</reasoning>
<tool_call>https://example.com/einstein</tool_call>
<answer>wrong</answer>"""


INVALID_PLANNER_LEGAL_SEARCH_WRONG_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<plan>
Step 2: Search duplicate planner.
</plan>
<reasoning>I still issue a legal search.</reasoning>
<tool_call>Albert Einstein birthplace</tool_call>
<tool_response>Doc 1 says Ulm.</tool_response>
<reasoning>Now answer incorrectly.</reasoning>
<answer>wrong</answer>"""


INVALID_PLANNER_PARTIAL_STEP_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search first children's day celebration India.
Step 2: If no direct information found, search history of children's day India.
</plan>
<reasoning>I need evidence.</reasoning>
<tool_call>first children's day celebration India</tool_call>
<tool_response>Doc 1 mentions a celebration date.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>wrong</answer>"""


LONG_PLAN_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search topic one.
Step 2: Search topic two.
Step 3: Search topic three.
Step 4: Search topic four.
Step 5: Search topic five.
</plan>
<reasoning>I will execute the first search.</reasoning>
<tool_call>topic one</tool_call>
<tool_response>Doc 1 has evidence.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>wrong</answer>"""


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
    assert components["has_search"] is True
    assert components["effective_structure_format"] == 1.0
    assert components["effective_retrieval"] == 1.0
    assert components["track_a_bonus"] == 0.0
    assert components["self_consistency_weight"] == 0.0
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


def test_self_consistency_weight_adds_perfect_track_a_bonus():
    components = qa_em_format.compute_score_components(
        PERFECT_TRAJECTORY,
        {"target": ["wrong"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        self_consistency_weight=0.05,
    )
    score = qa_em_format.compute_score_em(
        PERFECT_TRAJECTORY,
        {"target": ["wrong"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        self_consistency_weight=0.05,
    )

    assert components["self_consistency"] == 1.0
    assert components["track_a_bonus"] == pytest.approx(0.05)
    assert components["self_consistency_weight"] == 0.05
    assert components["final_score"] == pytest.approx(components["base_score"] + 0.05)
    assert score == components["final_score"]
    assert "path_bonus" not in components


def test_self_consistency_weight_adds_partial_track_a_bonus():
    partial = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<reasoning>I need the birthplace.</reasoning>
<tool_call>Albert Einstein birthplace</tool_call>
<tool_response>Doc 1 says Ulm.</tool_response>
<reasoning>I also search something unrelated.</reasoning>
<tool_call>unrelated query</tool_call>
<tool_response>Noise.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>wrong</answer>"""

    components = qa_em_format.compute_score_components(
        partial,
        {"target": ["right"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        self_consistency_weight=0.05,
    )

    assert components["self_consistency"] == 0.5
    assert components["track_a_bonus"] == pytest.approx(0.025)
    assert components["final_score"] == pytest.approx(components["base_score"] + 0.025)
    assert "path_bonus" not in components


def test_self_consistency_weight_no_action_adds_zero_bonus():
    components = qa_em_format.compute_score_components(
        NO_SEARCH_WRONG_TRAJECTORY,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        self_consistency_weight=0.05,
    )

    assert components["self_consistency"] == 0.0
    assert components["track_a_bonus"] == 0.0
    assert components["final_score"] == components["base_score"]
    assert "path_bonus" not in components


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


def test_intent_lexical_matches_instantiated_intermediate_entity():
    lexical_components = qa_em_format.compute_self_consistency_components(
        INTENT_INSTANTIATED_TRAJECTORY,
        match_strategy="lexical",
    )
    intent_components = qa_em_format.compute_self_consistency_components(
        INTENT_INSTANTIATED_TRAJECTORY,
        match_strategy="intent_lexical",
    )

    assert lexical_components["self_n_exec"] == 0
    assert lexical_components["self_consistency"] == 0.0
    assert intent_components["self_n_exec"] == 1
    assert intent_components["self_consistency"] == 1.0
    assert qa_em_format.step_matches_action(
        "Search [identified actress] role in The Honeymooners.",
        "Joyce Randolph Honeymooners role",
        match_strategy="intent_lexical",
    )


def test_intent_lexical_rejects_single_generic_overlap():
    components = qa_em_format.compute_self_consistency_components(
        INTENT_TOO_GENERIC_TRAJECTORY,
        match_strategy="intent_lexical",
    )

    assert components["self_n_exec"] == 0
    assert components["self_consistency"] == 0.0


def test_intent_lexical_duplicate_actions_do_not_cover_multiple_steps():
    duplicate_intent_action = """<|im_start|>assistant
<plan>
Step 1: Search [identified actress] character in The Honeymooners.
Step 2: Search [identified actress] role in The Honeymooners.
</plan>
<reasoning>I need the character.</reasoning>
<tool_call>Joyce Randolph Trixie Norton The Honeymooners</tool_call>
<tool_response>Doc 1 says Joyce Randolph played Trixie Norton.</tool_response>
<reasoning>I repeat the same query.</reasoning>
<tool_call>Joyce Randolph Trixie Norton The Honeymooners</tool_call>
<tool_response>Doc 1 again says Joyce Randolph played Trixie Norton.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>Trixie Norton</answer>"""

    components = qa_em_format.compute_self_consistency_components(
        duplicate_intent_action,
        match_strategy="intent_lexical",
    )

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


def test_require_search_false_preserves_no_search_format_shaping():
    valid_no_search = qa_em_format.compute_score_components(
        NO_SEARCH_WRONG_TRAJECTORY,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=False,
    )
    invalid_no_search = qa_em_format.compute_score_components(
        FAKED_TOOL_RESPONSE_TRAJECTORY,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=False,
    )

    assert valid_no_search["has_search"] is False
    assert valid_no_search["effective_structure_format"] == 1.0
    assert valid_no_search["effective_retrieval"] == 1.0
    assert valid_no_search["base_score"] == 0.2
    assert invalid_no_search["base_score"] == 0.1


def test_require_search_false_preserves_invalid_sequence_final_format_shaping():
    components = qa_em_format.compute_score_components(
        INVALID_PLANNER_LEGAL_SEARCH_WRONG_TRAJECTORY,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=False,
    )

    is_valid_format, _ = qa_em_format.is_valid_sequence(
        INVALID_PLANNER_LEGAL_SEARCH_WRONG_TRAJECTORY
    )
    assert is_valid_format is False
    assert components["has_search"] is True
    assert components["effective_structure_format"] == 1.0
    assert components["effective_retrieval"] == 1.0
    assert components["base_score"] == 0.1


def test_extract_solution_keeps_standalone_single_answer_compatibility():
    assert qa_em_format.extract_solution("<answer>Ulm</answer>") is None
    assert (
        qa_em_format.compute_score_em(
            "<answer>Ulm</answer>",
            {"target": ["Ulm"]},
            structure_format_score=0.2,
            final_format_score=0.1,
        )
        == 0
    )


def test_require_search_true_blocks_no_search_wrong_answer_shaping():
    for trajectory in (NO_SEARCH_WRONG_TRAJECTORY, FAKED_TOOL_RESPONSE_TRAJECTORY, NO_ANSWER_TRAJECTORY):
        components = qa_em_format.compute_score_components(
            trajectory,
            {"target": ["Ulm"]},
            structure_format_score=0.2,
            final_format_score=0.1,
            retrieval_score=0.1,
            require_search_for_format=True,
        )

        assert components["has_search"] is False
        assert components["effective_structure_format"] == 0.0
        assert components["effective_retrieval"] == 0.0
        assert components["base_score"] == 0
        assert components["final_score"] == 0


def test_require_search_true_blocks_malformed_tool_call_shaping():
    components = qa_em_format.compute_score_components(
        MALFORMED_TOOL_CALL_TRAJECTORY,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=True,
    )

    assert components["has_search"] is False
    assert components["effective_structure_format"] == 0.0
    assert components["effective_retrieval"] == 0.0
    assert components["base_score"] == 0


@pytest.mark.parametrize(
    "tool_call",
    [
        "search",
        "query",
        "Search Albert Einstein birthplace",
        "query: Albert Einstein birthplace",
        "search(Albert Einstein birthplace)",
        "tool_call search Albert Einstein birthplace",
        "tool_call: search(Albert Einstein birthplace)",
        "search-P1",
        "query-MIob",
        '{"query": "Albert Einstein birthplace"}',
    ],
)
def test_search_query_quality_gate_blocks_pseudo_tool_calls(tool_call):
    trajectory = MALFORMED_TOOL_CALL_TRAJECTORY.replace(
        "https://example.com/einstein",
        tool_call,
    )
    components = qa_em_format.compute_score_components(
        trajectory,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=True,
        self_consistency_weight=0.05,
    )

    assert qa_em_format.is_valid_search_query(tool_call) is False
    assert components["has_search"] is False
    assert components["self_consistency"] == 0.0
    assert components["track_a_bonus"] == 0.0
    assert components["base_score"] == 0


@pytest.mark.parametrize(
    "tool_call",
    [
        "Search-P1 paper contribution",
        "Q-learning algorithm",
        "Spider-Man actor",
        "COVID-19 symptoms",
    ],
)
def test_search_query_quality_gate_allows_informative_hyphenated_queries(tool_call):
    assert qa_em_format.is_valid_search_query(tool_call) is True


def test_require_search_true_keeps_structure_shaping_with_legal_tool_call():
    components = qa_em_format.compute_score_components(
        PERFECT_TRAJECTORY,
        {"target": ["wrong"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0,
        require_search_for_format=True,
    )

    assert components["has_search"] is True
    assert components["effective_structure_format"] == 1.0
    assert components["effective_retrieval"] == 1.0
    assert components["base_score"] == 0.2


def test_require_search_true_blocks_invalid_sequence_final_format_shaping():
    components = qa_em_format.compute_score_components(
        INVALID_PLANNER_LEGAL_SEARCH_WRONG_TRAJECTORY,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=True,
    )

    is_valid_format, _ = qa_em_format.is_valid_sequence(
        INVALID_PLANNER_LEGAL_SEARCH_WRONG_TRAJECTORY
    )
    assert is_valid_format is False
    assert components["has_search"] is True
    assert components["effective_structure_format"] == 1.0
    assert components["effective_retrieval"] == 1.0
    assert components["base_score"] == 0
    assert components["final_score"] == 0


def test_invalid_planner_partial_step_cannot_take_structure_format_score():
    self_components = qa_em_format.compute_self_consistency_components(
        INVALID_PLANNER_PARTIAL_STEP_TRAJECTORY
    )
    is_valid_format, _ = qa_em_format.is_valid_sequence(
        INVALID_PLANNER_PARTIAL_STEP_TRAJECTORY
    )
    gated_components = qa_em_format.compute_score_components(
        INVALID_PLANNER_PARTIAL_STEP_TRAJECTORY,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=True,
    )
    legacy_components = qa_em_format.compute_score_components(
        INVALID_PLANNER_PARTIAL_STEP_TRAJECTORY,
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=False,
    )

    assert self_components["self_r_planner"] == 0.0
    assert is_valid_format is False
    assert gated_components["has_search"] is True
    assert gated_components["base_score"] == 0
    assert legacy_components["base_score"] == 0.1


def test_max_plan_steps_marks_long_planner_invalid_and_blocks_structure_reward():
    steps = qa_em_format.extract_plan_steps(LONG_PLAN_TRAJECTORY)
    legacy_components = qa_em_format.compute_score_components(
        LONG_PLAN_TRAJECTORY,
        {"target": ["right"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=True,
    )
    limited_self_components = qa_em_format.compute_self_consistency_components(
        LONG_PLAN_TRAJECTORY,
        max_plan_steps=4,
    )
    limited_score_components = qa_em_format.compute_score_components(
        LONG_PLAN_TRAJECTORY,
        {"target": ["right"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        require_search_for_format=True,
        max_plan_steps=4,
    )

    assert len(steps) == 5
    assert qa_em_format.validate_planner_block(LONG_PLAN_TRAJECTORY, steps) is True
    assert qa_em_format.is_valid_sequence(LONG_PLAN_TRAJECTORY)[0] is True
    assert legacy_components["base_score"] == 0.2
    assert qa_em_format.validate_planner_block(LONG_PLAN_TRAJECTORY, steps, max_plan_steps=4) is False
    assert qa_em_format.is_valid_sequence(LONG_PLAN_TRAJECTORY, max_plan_steps=4)[0] is False
    assert limited_self_components["self_r_planner"] == 0.0
    assert limited_self_components["self_consistency"] == 0.0
    assert limited_score_components["has_search"] is True
    assert limited_score_components["base_score"] == 0


def test_require_search_true_keeps_exact_match_outcome_reward_without_search():
    components = qa_em_format.compute_score_components(
        NO_SEARCH_WRONG_TRAJECTORY.replace("<answer>wrong</answer>", "<answer>Ulm</answer>"),
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        score=1.0,
        require_search_for_format=True,
    )

    assert components["has_search"] is False
    assert components["effective_structure_format"] == 0.0
    assert components["effective_retrieval"] == 0.0
    assert components["base_score"] == 1.0


def test_require_search_true_keeps_exact_match_outcome_reward_with_invalid_format():
    components = qa_em_format.compute_score_components(
        INVALID_PLANNER_LEGAL_SEARCH_WRONG_TRAJECTORY.replace(
            "<answer>wrong</answer>",
            "<answer>Ulm</answer>",
        ),
        {"target": ["Ulm"]},
        structure_format_score=0.2,
        final_format_score=0.1,
        retrieval_score=0.1,
        score=1.0,
        require_search_for_format=True,
    )

    assert components["has_search"] is True
    assert components["base_score"] == 0.8


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
    with pytest.raises(ValueError) as exc_info:
        qa_em_format.compute_self_consistency_score(PERFECT_TRAJECTORY, match_strategy="embedding")
    message = str(exc_info.value)
    assert "intent_lexical" in message
    assert "lexical" in message
