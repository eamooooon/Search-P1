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

    assert reasons == ["valid_search", "valid_plan", "malformed_query_tag"]
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


def test_query_formats_stay_invalid_and_are_reported_as_malformed_query_tag():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        [
            "<reasoning>Old format.</reasoning><query>France capital</query>",
            "<reasoning>Old format.</reasoning><tool_query>France capital</tool_query>",
            "/query France capital",
        ],
        planner_seen=[True, True, True],
        active_mask=[True, True, True],
        return_reasons=True,
    )

    assert actions == [None, None, None]
    assert contents == ["", "", ""]
    assert reasons == [
        "malformed_query_tag",
        "malformed_query_tag",
        "malformed_query_tag",
    ]


def test_legacy_tags_are_reported_as_malformed_legacy_tag():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        [
            "<reasoning>Old format.</reasoning><search>France capital</search>",
            "<think>Need evidence.</think>",
            "<information>Doc text.</information>",
        ],
        planner_seen=[True, True, True],
        active_mask=[True, True, True],
        return_reasons=True,
    )

    assert actions == [None, None, None]
    assert contents == ["", "", ""]
    assert reasons == [
        "malformed_legacy_tag",
        "malformed_legacy_tag",
        "malformed_legacy_tag",
    ]


def test_malformed_tool_call_content_is_reported_separately():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        [
            "<reasoning>Need evidence.</reasoning>"
            "<tool_call>tool_call: search(France capital)</tool_call>",
            "<reasoning>Need evidence.</reasoning>"
            "<tool_call>tool_response: Doc text</tool_call>",
        ],
        planner_seen=[True, True],
        active_mask=[True, True],
        return_reasons=True,
    )

    assert actions == [None, None]
    assert contents == ["", ""]
    assert reasons == [
        "malformed_tool_call_content",
        "malformed_tool_call_content",
    ]


def test_plain_tool_call_query_is_valid_search():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        ["<reasoning>Need evidence.</reasoning><tool_call>Albert Einstein birthplace</tool_call>"],
        planner_seen=[True],
        active_mask=[True],
        return_reasons=True,
    )

    assert actions == ["search"]
    assert contents == ["Albert Einstein birthplace"]
    assert reasons == ["valid_search"]


def test_trainer_action_reason_allowlist_includes_malformed_buckets():
    with open("verl/trainer/ppo/ray_trainer.py", encoding="utf-8") as trainer_file:
        trainer_source = trainer_file.read()

    for reason in [
        "malformed_query_tag",
        "malformed_legacy_tag",
        "malformed_tool_call_content",
    ]:
        assert reason in trainer_source


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
