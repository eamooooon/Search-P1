import importlib
import sys
import types
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_generation_module():
    module_names = ["verl", "verl.utils", "verl.utils.tracking"]
    previous_modules = {name: sys.modules.get(name) for name in module_names}
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
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


generation = load_generation_module()


def make_manager():
    manager = generation.LLMGenerationManager.__new__(generation.LLMGenerationManager)
    manager.tokenizer = types.SimpleNamespace(pad_token_id=0, pad_token="<pad>")
    manager.batch_search = lambda queries: [f"Doc for {query}" for query in queries]
    return manager


def test_control_observations_are_masked_but_information_is_kept():
    manager = make_manager()
    predictions = [
        "<plan>\nStep 1: Search France capital.\n</plan>\n"
        "<think>I need evidence.</think><search>France capital</search>",
        "<plan>\nStep 1: Search France capital.\n</plan>",
        "<plan>\nStep 1: Search France capital.\n</plan>\n"
        "<think>I use an invalid query tag.</think><query>France capital</query>",
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
    assert "<information>Doc for France capital</information>" in next_obs[0]
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
            "<think>Old format.</think><query>France capital</query>",
            "<think>Old format.</think><lookup>France capital</lookup>",
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
        "malformed_tool_tag",
        "malformed_query_tag",
    ]


def test_search_r1_tags_are_accepted_for_actions():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        [
            "<think>Need evidence.</think><search>France capital</search>",
            "<think>Need evidence.</think>",
            "<information>Doc text.</information>",
        ],
        planner_seen=[True, True, True],
        active_mask=[True, True, True],
        return_reasons=True,
    )

    assert actions == ["search", None, None]
    assert contents == ["France capital", "", ""]
    assert reasons == [
        "valid_search",
        "missing_action_tag",
        "missing_action_tag",
    ]


def test_malformed_search_content_is_reported_separately():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        [
            "<think>Need evidence.</think>"
            "<search>tool: search(France capital)</search>",
            "<think>Need evidence.</think>"
            "<search>information: Doc text</search>",
            "<think>Need evidence.</think>"
            "<search>search</search>",
            "<think>Need evidence.</think>"
            "<search>Search France capital</search>",
            "<think>Need evidence.</think>"
            "<search>tool search France capital</search>",
            "<think>Need evidence.</think>"
            "<search>search-P1</search>",
        ],
        planner_seen=[True, True, True, True, True, True],
        active_mask=[True, True, True, True, True, True],
        return_reasons=True,
    )

    assert actions == [None, None, None, None, None, None]
    assert contents == ["", "", "", "", "", ""]
    assert reasons == [
        "malformed_search_content",
        "malformed_search_content",
        "malformed_search_content",
        "malformed_search_content",
        "malformed_search_content",
        "malformed_search_content",
    ]


def test_plain_search_query_is_valid_search():
    manager = make_manager()

    actions, contents, reasons = manager.postprocess_predictions(
        ["<think>Need evidence.</think><search>Albert Einstein birthplace</search>"],
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
        "malformed_tool_tag",
        "malformed_search_content",
    ]:
        assert reason in trainer_source


def test_feedback_avoids_full_xml_pair_examples():
    manager = make_manager()
    feedback_texts = [
        manager.PLAN_ACCEPTED_OBSERVATION,
        manager._invalid_action_observation(False, "missing_plan"),
        manager._invalid_action_observation(True, "malformed_action_tag"),
        manager._invalid_action_observation(True, "malformed_search_content"),
    ]

    forbidden_pairs = [
        "<search>...</search>",
        "<answer>...</answer>",
        "<plan>...</plan>",
    ]
    for feedback in feedback_texts:
        for pair in forbidden_pairs:
            assert pair not in feedback


def test_malformed_search_feedback_teaches_plain_query_recovery():
    manager = make_manager()
    feedback = manager._invalid_action_observation(True, "malformed_search_content")

    assert "concrete plain query" in feedback
    assert "Good query content: Albert Einstein birthplace." in feedback
    for bad_content in [
        "Bad query content: search;",
        "Search Albert Einstein birthplace",
        "search(Albert Einstein birthplace)",
        "search-P1",
        "query-MIob",
    ]:
        assert bad_content in feedback
