# Track B Reference-Alignment 设计思路

## 目标

Track B 衡量的是：

```text
模型实际执行的搜索动作，是否覆盖了外部参考计划中的关键步骤？
```

边界必须清楚：

- Track A 比较模型自己的 `<plan>` 和 `<tool_call>`，得到 `S_self`。
- Track B 比较模型自己的 `<tool_call>` 和 `ground_truth.reference_steps`，得到 `S_ref`。
- Track B 不读取 `<plan>`。
- Track A 不读取 `reference_steps`。
- 双轨组合只在后续聚合层做：`R_path = max(S_self, S_ref)`。

## 公式

```text
S_ref = (n_covered / |R_ref|) * (n_covered / n_actions)
```

含义：

- `R_ref`：外部参考计划步骤集合，即 `reference_steps`。
- `n_actions`：模型实际发出的合法 `<tool_call>` 数量。
- `n_covered`：actions 覆盖了多少个 reference step。

边界：

- 缺少 `reference_steps`：`S_ref = 0.0`，`ref_available = 0.0`。
- 没有合法 `<tool_call>`：`S_ref = 0.0`。
- 重复 action 不重复增加 `n_covered`，但仍计入 `n_actions` 分母。
- 一个 action 第一版最多覆盖一个 reference step。

## 数据格式

训练/验证数据中，参考计划放在：

```json
{
  "reward_model": {
    "ground_truth": {
      "target": ["answer"],
      "reference_steps": [
        "Search the main entity.",
        "Search the target fact."
      ]
    }
  }
}
```

离线 reference JSONL 支持两种 key：

```json
{"data_source": "nq", "split": "train", "index": 0, "reference_steps": ["Search ..."]}
{"question": "Who discovered radium?", "reference_steps": ["Search who discovered radium."]}
```

优先使用 `data_source + split + index`，缺失时按 question fallback。

## 离线生成流程

1. 拒绝采样：对同一问题采样多条完整 trajectory，只保留最终答案 EM 正确且包含合法 `<tool_call>` 的轨迹。
2. LLM voting：把正确轨迹中的搜索动作整理成 voting request，让强 LLM 输出短的 consensus search intent list。
3. 质量门控：过滤空 step、重复 step、含 trajectory tag、URL、过长步骤、超过 `max_reference_steps` 的计划。
4. 注入 parquet：把 `reference_steps.jsonl` 合并进 `reward_model.ground_truth.reference_steps`。

当前实现提供 provider-agnostic 文件接口，不把具体 LLM API 写死进训练流程。

## 当前实现状态

截至 2026-05-20，已实现：

- `verl/utils/reward_score/qa_em_format.py`
  - `compute_reference_alignment_components`
  - `reference_alignment`
  - `ref_available`
  - `ref_n_steps`
  - `ref_n_actions`
  - `ref_n_covered`
- `verl/trainer/main_ppo_format.py` 和 `verl/trainer/ppo/ray_trainer.py`
  - 透出 Track B reward metrics。
- `verl/trainer/config/ppo_trainer.yaml`
  - 新增 `reward_model.max_reference_steps`。
- `scripts/data_process/reference_steps.py`
  - 从外部 JSONL 读取和清洗 `reference_steps`。
- `scripts/data_process/qa_search_train_merge.py`
  - 支持 `--reference_steps_file` 和 `--max_reference_steps`。
- `scripts/data_process/qa_search_test_merge.py`
  - 支持 `--reference_steps_file` 和 `--max_reference_steps`。
- `scripts/nq_hotpotqa_p1/data_process.sh`
  - 支持 `REFERENCE_STEPS_FILE` 和 `MAX_REFERENCE_STEPS`。
- `search_p1.analysis.build_reference_steps`
  - 从 trajectory JSONL 做拒绝采样。
  - 导出 LLM voting requests。
  - 读回 LLM voting 结果。
  - 没有 LLM 结果时，用保守 consensus fallback 生成 smoke-test 用 reference plan。
- `search_p1.analysis.check_reference_steps`
  - 检查 parquet 中 reference plan 覆盖率。
- `search_p1.analysis.run_reference_llm_voting`
  - 调用 OpenAI-compatible chat-completions API，把 voting request 转成 voting result。
- `scripts/nq_hotpotqa_p1/build_reference_steps.sh`
  - P1 实验目录下的 reference 构建入口。
- `scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh`
  - P1 实验目录下的 LLM voting 入口。
- `scripts/nq_hotpotqa_p1/check_reference_steps.sh`
  - P1 实验目录下的 reference 覆盖率检查入口。

第一版仍保持：

```text
final_score = existing_score
```

Track B 只做旁路观测，不改变 scalar reward。

## 使用流程

从 trajectory JSONL 构建 reference：

```bash
TRAJECTORY_JSONL=logs/trajectories.jsonl \
bash scripts/nq_hotpotqa_p1/build_reference_steps.sh
```

如果已有 LLM voting 结果：

```bash
LLM_API_KEY=... \
LLM_BASE_URL=https://api.openai.com/v1 \
LLM_MODEL=gpt-4.1-mini \
LLM_LIMIT=200 \
bash scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh
```

```bash
TRAJECTORY_JSONL=logs/trajectories.jsonl \
LLM_VOTES_FILE=data/nq_hotpotqa_p1/reference_vote_results.jsonl \
bash scripts/nq_hotpotqa_p1/build_reference_steps.sh
```

注入 parquet：

```bash
REFERENCE_STEPS_FILE=data/nq_hotpotqa_p1/reference_steps.jsonl \
bash scripts/nq_hotpotqa_p1/data_process.sh
```

训练前检查：

```bash
bash scripts/nq_hotpotqa_p1/check_reference_steps.sh
```

只有看到 `reference_available_ratio > 0` 后，Track B 的 `reference_alignment` 才可能出现非零值。

## 暂不包含

- 在线 LLM judge。
- reward-time LLM voting。
- `R_path = max(S_self, S_ref)` 双轨 reward 组合。
- 把具体 LLM provider 绑定进训练入口。

## 验收标准

- 没有 `<plan>` 的轨迹仍能计算 `S_ref`。
- 缺少 `reference_steps` 时稳定返回 0。
- 没有合法 `<tool_call>` 时稳定返回 0。
- 完全覆盖 reference 且无冗余 action 时 `S_ref = 1.0`。
- 冗余 action 会降低 `S_ref`。
- 重复 action 不会虚增 `ref_n_covered`。
- `<tool_response>` 中的文本绝不会被当作 action。
- 第一版保持 `final_score = existing_score`。
- `scripts/nq_hotpotqa_p1/*.sh` 是实验入口，Python 实现放在 `search_p1` 包或 data_process 专用目录中。
