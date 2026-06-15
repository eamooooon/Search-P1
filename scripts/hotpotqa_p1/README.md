# HotpotQA P1

Planner-format HotpotQA experiments live here so the original Search-R1 scripts
remain unchanged.

## Data

```bash
bash scripts/hotpotqa_p1/data_process.sh
```

This creates `data/hotpotqa_p1/train.parquet` from `hotpotqa` and
`data/hotpotqa_p1/test.parquet` from `2wikimultihopqa,musique,bamboogle`.

If you want Track B reference alignment to be available during training or
evaluation, pass a JSONL file with `reference_steps`:

```bash
REFERENCE_STEPS_FILE=path/to/reference_steps.jsonl \
MAX_REFERENCE_STEPS=4 \
bash scripts/hotpotqa_p1/data_process.sh
```

Each JSONL row can be keyed by `data_source + split + index`, or by `question`:

```json
{"data_source": "hotpotqa", "split": "train", "index": 0, "reference_steps": ["Search the main entity.", "Search the target fact."]}
{"question": "Who discovered radium?", "reference_steps": ["Search who discovered radium."]}
```

## Build Track B Reference Plans

Track B reference plans come from offline rollout JSONL, not from reward-time
LLM calls. The current HotpotQA pipeline uses the direct LLM builder:

```bash
bash scripts/hotpotqa_p1/build_reference_steps_llm.sh
```

This writes a JSONL file with `reference_steps`, for example:

```text
data/hotpotqa_p1/reference_steps_v22_corrected_llm.jsonl
```

Then regenerate parquet:

```bash
REFERENCE_STEPS_FILE=data/hotpotqa_p1/reference_steps_v22_corrected_llm.jsonl \
bash scripts/hotpotqa_p1/data_process.sh
```

## Train

```bash
bash scripts/hotpotqa_p1/train_grpo.sh
```

The script uses `verl.trainer.main_ppo_format`, with Track B reference
alignment enabled by default through `REFERENCE_ALIGNMENT_WEIGHT=0.05`.

## Evaluate

```bash
BASE_MODEL=checkpoints/<experiment>/actor/global_step_<step> bash scripts/hotpotqa_p1/evaluate.sh
```
