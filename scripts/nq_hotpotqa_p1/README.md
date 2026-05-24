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

To inject Track B reference plans, pass a JSONL file with `reference_steps`:

```bash
REFERENCE_STEPS_FILE=path/to/reference_steps.jsonl \
MAX_REFERENCE_STEPS=4 \
bash scripts/nq_hotpotqa_p1/data_process.sh
```

Each JSONL row can be keyed by `data_source` + `split` + `index`, or by
`question`:

```json
{"data_source": "nq", "split": "train", "index": 0, "reference_steps": ["Search the main entity.", "Search the target fact."]}
{"question": "Who discovered radium?", "reference_steps": ["Search who discovered radium."]}
```

After generating parquet, verify the coverage before training:

```bash
bash scripts/nq_hotpotqa_p1/check_reference_steps.sh
```

## Build Track B Reference Plans

Reference building needs a real rollout trajectory JSONL. The P1 GRPO smoke
script now writes one by default:

```text
logs/<experiment>-trajectories.jsonl
```

Run `train_grpo.sh` first if you do not already have a trajectory dump. You can
override the path and dump limit:

```bash
TRAJECTORY_DUMP_PATH=logs/my-trajectories.jsonl \
TRAJECTORY_DUMP_LIMIT=512 \
bash scripts/nq_hotpotqa_p1/train_grpo.sh
```

Given a rollout trajectory JSONL with `solution_str`, `ground_truth`, and either
`data_source/split/index` or `question`, build reference plans with rejection
sampling plus voting prompts:

```bash
TRAJECTORY_JSONL=logs/my-trajectories.jsonl \
bash scripts/nq_hotpotqa_p1/build_reference_steps.sh
```

This filters correct trajectories by exact match, groups successful actions per
question, and writes two files:

```text
data/nq_hotpotqa_p1/reference_steps.jsonl
data/nq_hotpotqa_p1/reference_vote_requests.jsonl
```

The first file is a conservative consensus fallback for smoke tests. The second
file is sent to a stronger LLM for voting.

For OpenAI-compatible chat-completions APIs, copy `.env.example` to the repo
root `.env.llm` or `.env`, then fill in runtime config:

```text
LLM_API_KEY=...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
LLM_LIMIT=200
LLM_PROGRESS_EVERY=25
```

Then run:

```bash
bash scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh
```

Before running the full voting file, test one sample request against the
configured model:

```bash
bash scripts/nq_hotpotqa_p1/test_reference_llm_connection.sh
```

The connectivity test prints the raw model response, parsed JSON, validated
`reference_steps`, and a final status.

To replay a real voting request, pass the request file and `custom_id`:

```bash
bash scripts/nq_hotpotqa_p1/test_reference_llm_connection.sh \
  --vote_requests data/nq_hotpotqa_p1/reference_vote_requests.jsonl \
  --custom_id 'id|hotpotqa|train|46035'
```

`LLM_LIMIT` is optional and is useful for the first smoke test. The voting
script writes three observable files by default:

```text
data/nq_hotpotqa_p1/reference_vote_results.jsonl
data/nq_hotpotqa_p1/reference_vote_failures.jsonl
data/nq_hotpotqa_p1/reference_llm_voting.log
```

It resumes by default by skipping `custom_id` rows already present in
`reference_vote_results.jsonl`. Set `LLM_RESUME=0` to overwrite prior results.
The script automatically loads `.env.llm` first, then `.env`. You can also
override the file explicitly:

```bash
LLM_ENV_FILE=.env.llm bash scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh
```

```bash
TRAJECTORY_JSONL=logs/my-trajectories.jsonl \
LLM_VOTES_FILE=data/nq_hotpotqa_p1/reference_vote_results.jsonl \
bash scripts/nq_hotpotqa_p1/build_reference_steps.sh
```

This second build uses the same trajectory JSONL plus `LLM_VOTES_FILE` to
rewrite `reference_steps.jsonl` with `reference_plan_source=llm_vote` where LLM
votes are available. It does not rewrite `reference_vote_requests.jsonl` by
default. Set `WRITE_VOTE_REQUESTS=1` if you intentionally want to regenerate
vote requests in the same run.

Then regenerate parquet with:

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
