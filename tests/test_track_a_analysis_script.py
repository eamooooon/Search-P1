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
