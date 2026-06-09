import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_search_p1_sft_from_parquet(tmp_path):
    parquet_path = tmp_path / "train.parquet"
    output_path = tmp_path / "sft.jsonl"
    prompt = (
        "Answer the given question. Before any search, output a plan. "
        "Question: who conducts a title search and issues a report?"
    )
    pd.DataFrame(
        [
            {
                "data_source": "nq",
                "prompt": [{"role": "user", "content": prompt}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": ["Anyone"]},
                },
            }
        ]
    ).to_parquet(parquet_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/sft/build_sft.py",
            "--input",
            str(parquet_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    messages = row["messages"]

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "<plan>" in messages[1]["content"]
    assert "Step 1: Search who conducts a title search and issues a report" in messages[1]["content"]
    assert "<search>who conducts a title search and issues a report</search>" in messages[1]["content"]
    assert messages[2]["content"] == "<information>Doc 1(Title: Answer evidence) The answer is Anyone.</information>"
    assert messages[3]["content"] == (
        "<think>The evidence is sufficient to answer the question.</think>\n"
        "<answer>Anyone</answer>"
    )
    assert row["metadata"]["sft_type"] == "template_answer_stub"


def test_build_search_p1_sft_single_assistant_format(tmp_path):
    parquet_path = tmp_path / "train.parquet"
    output_path = tmp_path / "sft.jsonl"
    pd.DataFrame(
        [
            {
                "data_source": "hotpotqa",
                "prompt": [{"role": "user", "content": "Question: Search-P1?"}],
                "reward_model": {"ground_truth": {"target": ["format"]}},
            }
        ]
    ).to_parquet(parquet_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/sft/build_sft.py",
            "--input",
            str(parquet_path),
            "--output",
            str(output_path),
            "--conversation-format",
            "single_assistant",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
    assert "<information>" in row["messages"][1]["content"]


def test_build_search_p1_sft_verl_parquet_format(tmp_path):
    parquet_path = tmp_path / "train.parquet"
    output_path = tmp_path / "sft.parquet"
    pd.DataFrame(
        [
            {
                "data_source": "nq",
                "prompt": [{"role": "user", "content": "Question: who founded Example Co?"}],
                "reward_model": {"ground_truth": {"target": ["Alice"]}},
            }
        ]
    ).to_parquet(parquet_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/sft/build_sft.py",
            "--input",
            str(parquet_path),
            "--output",
            str(output_path),
            "--output-format",
            "verl_parquet",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    frame = pd.read_parquet(output_path)
    assert list(frame.columns) == ["prompt", "response", "metadata"]
    assert frame.iloc[0]["prompt"].endswith("Question: who founded Example Co?")
    assert "<plan>" in frame.iloc[0]["response"]
    assert "<search>who founded Example Co</search>" in frame.iloc[0]["response"]
    assert "<answer>Alice</answer>" in frame.iloc[0]["response"]


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    chat_template = "dummy"

    def apply_chat_template(self, chat, add_generation_prompt=True, tokenize=False):
        assert not tokenize
        rendered = "".join(f"<|{message['role']}|>{message['content']}" for message in chat)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        import torch

        tokens = [index + 3 for index, _ in enumerate(text.split())]
        if not tokens:
            tokens = [self.eos_token_id]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_sft_dataset_masks_prompt_and_padding(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    from verl.utils.dataset import SFTDataset

    parquet_path = tmp_path / "sft.parquet"
    pd.DataFrame([{"prompt": "Question: x?", "response": "<answer>y</answer>"}]).to_parquet(parquet_path)

    dataset = SFTDataset(
        parquet_files=str(parquet_path),
        tokenizer=DummyTokenizer(),
        prompt_key="prompt",
        response_key="response",
        max_length=16,
    )
    item = dataset[0]

    assert set(item) == {"input_ids", "attention_mask", "position_ids", "loss_mask"}
    assert item["input_ids"].shape == torch.Size([16])
    assert item["attention_mask"].shape == torch.Size([16])
    assert item["position_ids"].shape == torch.Size([16])
    assert item["loss_mask"].shape == torch.Size([16])
    assert item["loss_mask"][0].item() == 0
    assert item["loss_mask"].sum().item() > 0
    assert item["loss_mask"][item["attention_mask"] == 0].sum().item() == 0
