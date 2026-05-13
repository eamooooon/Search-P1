import importlib
import sys
import types

import torch


def load_generation_module():
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

    return importlib.import_module("search_p1.llm_agent.generation")


generation = load_generation_module()


def make_manager():
    manager = generation.LLMGenerationManager.__new__(generation.LLMGenerationManager)
    manager.tokenizer = types.SimpleNamespace(pad_token_id=0, pad_token="<pad>")
    manager.batch_search = lambda queries: [f"Doc for {query}" for query in queries]
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

    assert reasons == ["valid_search", "valid_plan", "malformed_action_tag"]
    assert is_search == [1, 0, 0]
    assert "<tool_response>Doc for France capital</tool_response>" in next_obs[0]
    assert "Plan accepted." in next_obs[1]
    assert "My previous action is invalid." in next_obs[2]

    mask = manager._control_observation_mask_from_reasons(reasons, [True, True, True])
    next_obs_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
    )

    masked = manager._mask_control_observations_for_final_trajectory(next_obs_ids, mask)

    assert masked[0].tolist() == [1, 2, 3, 4]
    assert masked[1].tolist() == [0, 0, 0, 0]
    assert masked[2].tolist() == [0, 0, 0, 0]


def test_query_tag_stays_invalid_and_is_reported_as_malformed_action_tag():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        ["<reasoning>Old format.</reasoning><query>France capital</query>"],
        planner_seen=[True],
        active_mask=[True],
        return_reasons=True,
    )

    assert actions == [None]
    assert contents == [""]
    assert reasons == ["malformed_action_tag"]


def test_feedback_avoids_full_xml_pair_examples():
    manager = make_manager()
    feedback_texts = [
        manager.PLAN_ACCEPTED_OBSERVATION,
        manager._invalid_action_observation(False, "missing_plan"),
        manager._invalid_action_observation(True, "malformed_action_tag"),
    ]

    forbidden_pairs = [
        "<tool_call>...</tool_call>",
        "<answer>...</answer>",
        "<plan>...</plan>",
    ]
    for feedback in feedback_texts:
        for pair in forbidden_pairs:
            assert pair not in feedback
