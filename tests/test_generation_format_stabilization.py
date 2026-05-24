import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")


def load_generation_module():
    module_names = ["verl", "verl.utils", "verl.utils.tracking"]
    original_modules = {name: sys.modules.get(name) for name in module_names}

    verl_module = types.ModuleType("verl")

    class DataProto:
        pass

    verl_module.DataProto = DataProto
    sys.modules["verl"] = verl_module
    sys.modules["verl.utils"] = types.ModuleType("verl.utils")

    tracking_module = types.ModuleType("verl.utils.tracking")

    class Tracking:
        pass

    tracking_module.Tracking = Tracking
    sys.modules["verl.utils.tracking"] = tracking_module

    try:
        return importlib.import_module("search_p1.llm_agent.generation")
    finally:
        for name in module_names:
            original = original_modules[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


generation = load_generation_module()


def make_manager():
    manager = generation.LLMGenerationManager.__new__(generation.LLMGenerationManager)
    manager.tokenizer = types.SimpleNamespace(pad_token_id=0, pad_token="<pad>")
    manager.batch_search = lambda queries: [f"Doc for {query}" for query in queries]
    manager._rollout_debug_samples_remaining = 0
    manager._rollout_debug_samples_emitted = 0
    return manager


def test_control_observations_are_masked_but_tool_response_is_kept():
    manager = make_manager()
    predictions = [
        "<plan>\nStep 1: Search France capital.\n</plan>\n"
        "<reasoning>I need evidence.</reasoning><tool_call>France capital</tool_call>",
        "<plan>\nStep 1: Search France capital.\n</plan>",
        "<plan>\nStep 1: Search France capital.\n</plan>\n"
        "<reasoning>I use an old action tag.</reasoning><query>France capital</query>",
    ]

    next_obs, _, _, is_search, _, reasons = manager.execute_predictions(
        predictions,
        "<pad>",
        active_mask=[True, True, True],
        planner_seen=[False, False, False],
        return_reason_stats=True,
        return_reasons=True,
    )

    assert reasons == ["valid_search", "valid_plan", "malformed_query_tag"]
    assert is_search == [1, 0, 0]
    assert "<tool_response>Doc for France capital</tool_response>" in next_obs[0]
    assert "Plan accepted." in next_obs[1]
    assert "My previous action is invalid." in next_obs[2]

    mask = manager._control_observation_mask_from_reasons(reasons, [True, True, True])
    next_obs_ids = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ])

    masked = manager._mask_control_observations_for_final_trajectory(next_obs_ids, mask)

    assert masked[0].tolist() == [1, 2, 3, 4]
    assert masked[1].tolist() == [0, 0, 0, 0]
    assert masked[2].tolist() == [0, 0, 0, 0]


def test_query_and_legacy_formats_get_specific_invalid_reasons():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        [
            "<reasoning>Old format.</reasoning><query>France capital</query>",
            "<reasoning>Old format.</reasoning><tool_query>France capital</tool_query>",
            "<reasoning>Old format.</reasoning><search>France capital</search>",
            "<think>Need evidence.</think>",
        ],
        planner_seen=[True, True, True, True],
        active_mask=[True, True, True, True],
        return_reasons=True,
    )

    assert actions == [None, None, None, None]
    assert contents == ["", "", "", ""]
    assert reasons == [
        "malformed_query_tag",
        "malformed_query_tag",
        "malformed_legacy_tag",
        "malformed_legacy_tag",
    ]


def test_malformed_tool_call_content_is_not_treated_as_plain_query():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        [
            "<reasoning>Need evidence.</reasoning>"
            "<tool_call>tool_call: search(France capital)</tool_call>",
            "<reasoning>Need evidence.</reasoning>"
            "<tool_call>{\"query\": \"France capital\"}</tool_call>",
            "<reasoning>Need evidence.</reasoning>"
            "<tool_call>Albert Einstein birthplace</tool_call>",
        ],
        planner_seen=[True, True, True],
        active_mask=[True, True, True],
        return_reasons=True,
    )

    assert actions == [None, None, "search"]
    assert contents == ["", "", "Albert Einstein birthplace"]
    assert reasons == [
        "malformed_tool_call_content",
        "malformed_tool_call_content",
        "valid_search",
    ]


def test_answer_before_any_search_is_invalid():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        [
            "<reasoning>I can answer directly.</reasoning><answer>Paris</answer>",
            "<reasoning>I can answer after evidence.</reasoning><answer>Paris</answer>",
        ],
        planner_seen=[True, True],
        search_seen=[False, True],
        active_mask=[True, True],
        return_reasons=True,
    )

    assert actions == [None, "answer"]
    assert contents == ["", "Paris"]
    assert reasons == ["answer_before_search", "valid_answer"]
