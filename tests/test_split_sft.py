import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_split_sft_parquet_with_fixed_val_size(tmp_path):
    input_path = tmp_path / "full.parquet"
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    pd.DataFrame(
        [{"prompt": f"q{i}", "response": f"a{i}", "metadata": {"index": i}} for i in range(10)]
    ).to_parquet(input_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/sft/split_sft.py",
            "--input",
            str(input_path),
            "--train-output",
            str(train_path),
            "--val-output",
            str(val_path),
            "--val-size",
            "3",
            "--seed",
            "123",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    train_frame = pd.read_parquet(train_path)
    val_frame = pd.read_parquet(val_path)
    assert len(train_frame) == 7
    assert len(val_frame) == 3
    assert sorted(train_frame["prompt"].tolist() + val_frame["prompt"].tolist()) == [f"q{i}" for i in range(10)]


def test_split_sft_jsonl_with_ratio_and_no_shuffle(tmp_path):
    input_path = tmp_path / "full.jsonl"
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    rows = [{"prompt": f"q{i}", "response": f"a{i}"} for i in range(5)]
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/sft/split_sft.py",
            "--input",
            str(input_path),
            "--train-output",
            str(train_path),
            "--val-output",
            str(val_path),
            "--val-ratio",
            "0.4",
            "--no-shuffle",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in val_rows] == ["q0", "q1"]
    assert [row["prompt"] for row in train_rows] == ["q2", "q3", "q4"]
