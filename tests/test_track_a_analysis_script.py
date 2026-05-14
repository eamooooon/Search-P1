import json
import subprocess
import sys
from pathlib import Path


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


NO_ACTION_TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<reasoning>I will answer without search.</reasoning>
<answer>Ulm</answer>"""


def test_track_a_analysis_script_outputs_summary(tmp_path):
    jsonl_path = tmp_path / "samples.jsonl"
    rows = [
        {
            "solution_str": PERFECT_TRAJECTORY,
            "ground_truth": {"target": ["wrong"]},
        },
        {
            "solution_str": NO_ACTION_TRAJECTORY,
            "ground_truth": {"target": ["wrong"]},
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/track_a_self_consistency.py",
            str(jsonl_path),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    summary = payload["summary"]
    assert summary["samples"] == 2
    assert summary["self_consistency"]["mean"] == 0.5
    assert summary["planner_valid_rate"] == 1.0
    assert summary["failure_counts"]["complete"] == 1
    assert summary["failure_counts"]["no_actions"] == 1
    assert payload["buckets"] == []


def test_track_a_analysis_script_tail_and_bucket_json_output(tmp_path):
    jsonl_path = tmp_path / "append.jsonl"
    rows = [
        {"solution_str": NO_ACTION_TRAJECTORY},
        {"solution_str": PERFECT_TRAJECTORY},
        {"solution_str": NO_ACTION_TRAJECTORY},
        {"solution_str": PERFECT_TRAJECTORY},
    ]
    jsonl_path.write_text(
        json.dumps(rows[0]) + "\n\n" + "\n".join(json.dumps(row) for row in rows[1:]),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/track_a_self_consistency.py",
            str(jsonl_path),
            "--tail",
            "3",
            "--bucket-size",
            "2",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    summary = payload["summary"]
    buckets = payload["buckets"]

    assert summary["samples"] == 3
    assert summary["failure_counts"]["complete"] == 2
    assert summary["failure_counts"]["no_actions"] == 1
    assert set(summary["failure_counts"]) == {
        "complete",
        "no_actions",
        "invalid_planner",
        "partial_plan_coverage",
        "unmatched_actions",
        "redundant_actions",
    }

    assert len(buckets) == 2
    assert buckets[0]["index"] == 0
    assert buckets[0]["samples"] == 2
    assert buckets[0]["planner_valid_rate"] == 1.0
    assert buckets[0]["self_consistency_mean"] == 0.5
    assert buckets[0]["failure_counts"]["complete"] == 1
    assert buckets[0]["failure_counts"]["no_actions"] == 1
    assert buckets[0]["failure_counts"]["invalid_planner"] == 0
    assert set(buckets[0]["failure_counts"]) == set(summary["failure_counts"])
    assert buckets[0]["source_range"] == {
        "start": f"{jsonl_path}:3",
        "end": f"{jsonl_path}:4",
    }

    assert buckets[1]["index"] == 1
    assert buckets[1]["samples"] == 1
    assert buckets[1]["self_consistency_mean"] == 1.0
    assert buckets[1]["failure_counts"]["complete"] == 1
    assert set(buckets[1]["failure_counts"]) == set(summary["failure_counts"])
    assert buckets[1]["source_range"] == {
        "start": f"{jsonl_path}:5",
        "end": f"{jsonl_path}:5",
    }


def test_track_a_analysis_script_tail_applies_per_input_file(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_path.write_text(
        "\n".join(
            [
                json.dumps({"solution_str": PERFECT_TRAJECTORY}),
                json.dumps({"solution_str": NO_ACTION_TRAJECTORY}),
            ]
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        "\n\n".join(
            [
                json.dumps({"solution_str": NO_ACTION_TRAJECTORY}),
                json.dumps({"solution_str": PERFECT_TRAJECTORY}),
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/track_a_self_consistency.py",
            str(first_path),
            str(second_path),
            "--tail",
            "1",
            "--bucket-size",
            "1",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)

    assert payload["summary"]["samples"] == 2
    assert payload["buckets"][0]["source_range"] == {
        "start": f"{first_path}:2",
        "end": f"{first_path}:2",
    }
    assert payload["buckets"][1]["source_range"] == {
        "start": f"{second_path}:3",
        "end": f"{second_path}:3",
    }


def test_track_a_analysis_script_bucket_text_output_includes_all_failure_counts(tmp_path):
    jsonl_path = tmp_path / "append.jsonl"
    rows = [
        {"solution_str": NO_ACTION_TRAJECTORY},
        {"solution_str": PERFECT_TRAJECTORY},
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/track_a_self_consistency.py",
            str(jsonl_path),
            "--bucket-size",
            "2",
            "--sample-size",
            "0",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "bucket=0" in result.stdout
    for reason in (
        "complete",
        "no_actions",
        "invalid_planner",
        "partial_plan_coverage",
        "unmatched_actions",
        "redundant_actions",
    ):
        assert f"{reason}=" in result.stdout


def test_track_a_analysis_script_rejects_limit_with_tail(tmp_path):
    jsonl_path = tmp_path / "samples.jsonl"
    jsonl_path.write_text(json.dumps({"solution_str": PERFECT_TRAJECTORY}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/track_a_self_consistency.py",
            str(jsonl_path),
            "--limit",
            "1",
            "--tail",
            "1",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr
