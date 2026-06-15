# NQ HotpotQA P1

Planner-format experiments live here so the original Search-R1 scripts under
`scripts/nq_hotpotqa/` remain unchanged.

## Data

```bash
bash scripts/nq_hotpotqa_p1/data_process.sh
```

This creates `data/nq_hotpotqa_p1/train.parquet` from `nq,hotpotqa` and
`data/nq_hotpotqa_p1/test.parquet` from
`nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle`.

If you want Track B reference alignment to be available during training or
evaluation, pass a JSONL file with `reference_steps`:

```bash
REFERENCE_STEPS_FILE=path/to/reference_steps.jsonl \
MAX_REFERENCE_STEPS=4 \
bash scripts/nq_hotpotqa_p1/data_process.sh
```

Each JSONL row can be keyed by `data_source + split + index`, or by `question`:

```json
{"data_source": "nq", "split": "train", "index": 0, "reference_steps": ["Search the main entity.", "Search the target fact."]}
{"question": "Who discovered radium?", "reference_steps": ["Search who discovered radium."]}
```

## Build Track B Reference Plans

Track B reference plans come from offline rollout JSONL, not from reward-time
LLM calls. The standalone two-stage reference-building scripts were removed;
generate a JSONL file with `reference_steps` using the active
`search_p1/analysis` pipeline, then pass it into data processing.

Then regenerate parquet:

```bash
REFERENCE_STEPS_FILE=data/nq_hotpotqa_p1/reference_steps.jsonl \
bash scripts/nq_hotpotqa_p1/data_process.sh
```

## Train

```bash
bash scripts/nq_hotpotqa_p1/train_grpo.sh
```

or:

```bash
bash scripts/nq_hotpotqa_p1/train_ppo.sh
```

Both scripts use `verl.trainer.main_ppo_format` so the planner trajectory
format is included in reward scoring.

## Evaluate

```bash
BASE_MODEL=checkpoints/<experiment>/actor/global_step_<step> bash scripts/nq_hotpotqa_p1/evaluate.sh
```
