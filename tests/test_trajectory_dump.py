import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


TRAJECTORY = """<|im_start|>assistant
<plan>
Step 1: Search Albert Einstein birthplace.
</plan>
<reasoning>I need the birthplace.</reasoning>
<tool_call>Albert Einstein birthplace</tool_call>
<tool_response>Doc 1 says Ulm.</tool_response>
<reasoning>Now answer.</reasoning>
<answer>Ulm</answer>"""


def load_trajectory_dump_module():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "trajectory_dump.py"
    spec = importlib.util.spec_from_file_location("trajectory_dump_direct", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trajectory_dump = load_trajectory_dump_module()


def test_trajectory_dump_jsonl_is_readable_by_analysis_script(tmp_path):
    jsonl_path = tmp_path / "nested" / "trajectories.jsonl"

    trajectory_dump._append_trajectory_dump(
        jsonl_path,
        solution_str=TRAJECTORY,
        ground_truth={"target": np.array(["Ulm"])},
        data_source=np.str_("nq"),
        split="train",
        index=np.int64(7),
        track_a={"self_consistency": np.float32(1.0)},
    )

    row = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert row["solution_str"] == TRAJECTORY
    assert row["ground_truth"] == {"target": ["Ulm"]}
    assert row["data_source"] == "nq"
    assert row["split"] == "train"
    assert row["index"] == 7

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
    assert payload["summary"]["samples"] == 1
    assert payload["summary"]["failure_counts"]["complete"] == 1
