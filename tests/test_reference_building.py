import importlib.util
import json
from pathlib import Path


def _load_module(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference_sampling = _load_module(
    "reference_sampling",
    Path("search_p1") / "analysis" / "reference_sampling.py",
)
reference_voting = _load_module(
    "reference_voting",
    Path("search_p1") / "analysis" / "reference_voting.py",
)
reference_llm = _load_module(
    "reference_llm",
    Path("search_p1") / "analysis" / "reference_llm.py",
)
trajectory_dump = _load_module(
    "trajectory_dump",
    Path("verl") / "trainer" / "trajectory_dump.py",
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_collect_successful_groups_filters_wrong_answers(tmp_path):
    trajectory_path = tmp_path / "trajectories.jsonl"
    _write_jsonl(trajectory_path, [
        {
            "data_source": "nq",
            "split": "train",
            "index": 0,
            "ground_truth": {"target": ["radium"]},
            "solution_str": "<tool_call>Marie Curie discovery</tool_call><answer>radium</answer>",
        },
        {
            "data_source": "nq",
            "split": "train",
            "index": 0,
            "ground_truth": {"target": ["radium"]},
            "solution_str": "<tool_call>Marie Curie discovery</tool_call><answer>polonium</answer>",
        },
    ])

    groups, stats = reference_sampling.collect_successful_groups(str(trajectory_path))

    group = groups[("id", "nq", "train", "0")]
    assert stats["total_rows"] == 2
    assert stats["correct_rows"] == 1
    assert group["correct"] == 1
    assert group["trajectories"][0]["actions"] == ["Marie Curie discovery"]


def test_consensus_reference_steps_use_repeated_successful_actions():
    group = {
        "trajectories": [
            {"actions": ["Marie Curie discovery", "radium discoverer"]},
            {"actions": ["Marie Curie discovery"]},
            {"actions": ["unrelated one-off query"]},
        ],
    }

    steps = reference_voting.build_consensus_reference_steps(
        group,
        min_vote_count=2,
        min_vote_ratio=0.2,
        max_reference_steps=4,
    )

    assert steps == ["Search Marie Curie discovery"]


def test_build_reference_rows_prefers_llm_votes():
    key = ("id", "nq", "train", "0")
    groups = {
        key: {
            "metadata": {
                "data_source": "nq",
                "split": "train",
                "index": 0,
                "question": "Who discovered radium?",
            },
            "total": 2,
            "correct": 2,
            "trajectories": [{"actions": ["Marie Curie discovery"]}],
        }
    }
    rows = reference_voting.build_reference_rows(
        groups,
        llm_votes={key: ["Search who discovered radium"]},
        max_reference_steps=4,
    )

    assert rows[0]["reference_plan_source"] == "llm_vote"
    assert rows[0]["reference_steps"] == ["Search who discovered radium"]


def test_vote_request_contains_candidate_actions():
    key = ("id", "nq", "train", "0")
    group = {
        "metadata": {"question": "Who discovered radium?"},
        "trajectories": [{"actions": ["Marie Curie discovery", "radium discoverer"]}],
    }

    request = reference_voting.build_vote_request(key, group)

    assert request["custom_id"] == "id|nq|train|0"
    content = json.loads(request["messages"][1]["content"])
    assert content["question"] == "Who discovered radium?"
    assert content["candidate_actions"][0]["action"] == "Marie Curie discovery"


def test_valid_reference_steps_normalizes_llm_output():
    steps = reference_llm.valid_reference_steps(
        ["who discovered radium", "Search who discovered radium", "<bad>"],
        max_reference_steps=4,
    )

    assert steps == ["Search who discovered radium"]


def test_run_llm_voting_skip_completed_custom_ids():
    requests = [{
        "custom_id": "id|nq|train|0",
        "messages": [],
        "metadata": {"data_source": "nq"},
    }]

    rows, stats = reference_llm.run_llm_voting(
        requests,
        base_url="unused",
        api_key="unused",
        model="unused",
        skip_custom_ids={"id|nq|train|0"},
    )

    assert rows == []
    assert stats["skipped"] == 1
    assert stats["vote_requests"] == 0


def test_trajectory_dump_serializes_multi_answer_arrays(tmp_path):
    class ArrayLike:
        def tolist(self):
            return ["answer one", "answer two"]

    path = tmp_path / "trajectories.jsonl"
    trajectory_dump.append_trajectory_dump(
        str(path),
        solution_str="<answer>answer one</answer>",
        ground_truth={"target": ArrayLike()},
        data_source="nq",
        extra_info={"split": "train", "index": 0},
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["ground_truth"]["target"] == ["answer one", "answer two"]
