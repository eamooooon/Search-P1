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
