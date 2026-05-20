# Track B Reference-Alignment 设计思路

## 设计动机

Track B 只回答一个问题：

```text
模型实际执行的搜索动作，是否覆盖了外部参考计划中的关键步骤？
```

它和 Track A 的参照物不同。Track A 比较的是模型自己的 `<plan>` 与自己的 `<tool_call>`，衡量 self-consistency；Track B 比较的是模型自己的 `<tool_call>` 与外部 `reference_steps`，衡量 reference-alignment。

因此 Track B 第一版必须和 Track A 解耦：

- Track B 不读取模型 `<plan>`。
- Track B 不复用 `self_consistency` 作为中间量。
- Track B 不要求模型 planner 合法才给 `S_ref`。
- Track B 只依赖实际搜索动作和外部参考步骤。

后续如果需要双轨聚合，应在单独的组合层完成：

```text
R_path = max(S_self, S_ref)
```

不要在 Track B scorer 内部调用 Track A，也不要在 Track A scorer 内部读取 reference plan。

## 核心定义

Track B 分数定义为：

```text
S_ref = (n_covered / |R_ref|) * (n_covered / n_actions)
```

变量含义：

- `R_ref`：外部参考计划步骤集合，也就是 `reference_steps`。
- `n_actions`：模型实际发出的合法 `<tool_call>` 数量。
- `n_covered`：模型 actions 覆盖了多少个 reference step。

两个比例分别表达：

- `n_covered / |R_ref|`：参考计划覆盖率。模型有没有做专家认为必要的搜索。
- `n_covered / n_actions`：动作效率。模型有没有做大量参考计划之外的冗余搜索。

边界条件：

- `reference_steps` 缺失或为空时，`S_ref = 0.0`，并记录 `ref_missing = 1`。
- 没有合法 `<tool_call>` 时，`S_ref = 0.0`，并记录 `ref_n_actions = 0`。
- `reference_steps` 中的非法空步骤应在数据生成阶段过滤；scorer 第一版仍做防御性过滤。
- 重复 action 不应重复增加 `n_covered`。
- 一个 action 第一版最多覆盖一个 reference step；一个 reference step 第一版最多被一个 action 覆盖。

## Track B 与 Track A 的边界

Track A 输入：

```text
solution_str -> planner steps + actions -> S_self
```

Track B 输入：

```text
solution_str + ground_truth.reference_steps -> actions + reference_steps -> S_ref
```

Track B 不应该依赖：

- `<plan>` 是否存在。
- `<plan>` 是否合法。
- `self_r_planner`、`self_n_plan`、`self_n_exec`。
- Track A analysis 脚本。
- Track A debug 归因。

Track B 可以复用的通用能力：

- 从 assistant trajectory 中抽取 `<tool_call>`。
- 搜索 query 的合法性校验。
- deterministic lexical matcher 的基础 normalization。
- trajectory JSONL dump 的通用字段：`solution_str`、`ground_truth`、`data_source`、`split`、`index`。

如果当前代码里这些能力带有 Track A 命名，Track B 实现时应先抽出中立命名层，例如：

```python
extract_tool_calls(solution_str) -> list[str]
normalize_path_text(text) -> str
step_matches_action(step, action, match_strategy="lexical") -> bool
count_covered_steps(reference_steps, actions, match_strategy="lexical") -> int
```

## Reference Plan 数据结构

训练或分析样本中建议把参考计划放在 `ground_truth.reference_steps`：

```json
{
  "solution_str": "<plan>...</plan>...",
  "ground_truth": {
    "target": ["answer"],
    "reference_steps": [
      "Search the main entity mentioned in the question.",
      "Search the specific attribute needed for the final answer."
    ]
  },
  "data_source": "hotpotqa",
  "split": "train",
  "index": 7
}
```

`reference_steps` 的语义是 search intent list，不是 exact query list。它应该和 planner step 处在同一抽象层级，但来源是离线专家计划，而不是当前模型输出。

约束：

- 每个 step 应是一个可执行搜索目标。
- 不写 fallback branches。
- 不写 year-by-year、episode-by-episode、长枚举式 exhaustive list。
- 可以使用 `[identified actor]`、`[identified film]`、`[target entity]` 这类 placeholder。
- 不包含 `<plan>`、`<tool_call>`、`<tool_response>`、`<answer>` 等 trajectory tag。

## Reference Plan 生成流程

参考计划生成不放在 reward-time 路径里。它是离线数据构建流程，建议分三步：

### 1. 拒绝采样

对同一个问题生成 `N` 条完整轨迹，论文设定可参考 `N=64`。每条轨迹至少需要保存：

- question / prompt id
- `solution_str`
- final answer
- ground-truth answer
- extracted actions
- answer correctness

筛选最终答案正确的轨迹，形成 candidate successful trajectories。

### 2. LLM Voting

对正确轨迹中的搜索动作和必要推理步骤进行归纳，抽取反复出现的关键 search intent。

LLM voting 的输出必须是结构化 JSON，建议格式：

```json
{
  "reference_steps": [
    "Search ...",
    "Search ..."
  ],
  "evidence_count": 12,
  "vote_model": "model-name",
  "source_trajectory_count": 64,
  "accepted_trajectory_count": 18
}
```

第一版只把 voting 放在离线脚本中，不在 reward scorer 中调用 LLM。

### 3. 质量门控

生成后的 `reference_steps` 需要通过 deterministic validator：

- 非空。
- 步骤数不超过 `max_reference_steps`，建议第一版与 `max_turns=4` 对齐。
- 每行不包含 trajectory tag。
- 每行不是 URL。
- 每行长度不过长。
- 步骤之间不完全重复。

未通过质量门控的样本可以：

- 不写入 `reference_steps`，让 Track B 得到 `ref_missing = 1`。
- 或写入 `reference_plan_status = "invalid"`，训练时不启用 Track B。

第一版建议采用保守策略：无高质量 reference plan 就不计算正向 `S_ref`。

## Scorer 数据流

Track B scorer 的第一版数据流：

```text
serialized trajectory
  -> extract model tool calls
  -> load ground_truth.reference_steps
  -> validate / filter reference steps
  -> match reference steps against actions
  -> compute S_ref
  -> log reference-alignment components
```

第一版只做观测信号，不改 scalar reward：

```text
final_score = existing_score
```

也就是说，第一版新增 metrics，但不启用：

```text
final_score = existing_score + lambda_p * R_path
```

这样可以先验证 reference plan 的质量、matcher 的稳定性、`S_ref` 与正确答案之间的关系，再决定是否进入 reward composition。

## 建议接口

Reference 数据接口：

```python
extract_reference_steps(ground_truth) -> list[str]
validate_reference_steps(steps, max_reference_steps=None) -> bool
```

Action 解析接口：

```python
extract_tool_calls(solution_str) -> list[str]
validate_actions(actions) -> bool
```

匹配接口：

```python
reference_step_matches_action(reference_step, action, match_strategy="lexical") -> bool
count_reference_covered_steps(reference_steps, actions, match_strategy="lexical") -> int
```

评分接口：

```python
compute_reference_alignment_components(
    solution_str,
    ground_truth,
    match_strategy="lexical",
    max_reference_steps=None,
) -> dict
```

建议返回字段：

```python
{
    "reference_alignment": S_ref,
    "ref_available": 1.0 or 0.0,
    "ref_n_steps": int,
    "ref_n_actions": int,
    "ref_n_covered": int,
}
```

后续进入双轨聚合时，再新增：

```python
{
    "path_score": max(self_consistency, reference_alignment)
}
```

不要把 `reference_alignment` 命名为 `path_bonus`，避免把“度量值”和“奖励增量”混在一起。

## Matcher 策略

第一版使用 deterministic lexical matching。理由和 Track A 一致：可解释、可测试、可复现。

基础规则：

- lowercase。
- 去标点。
- 折叠空白。
- 移除低信息 token。
- 支持 containment。
- 支持 token overlap。
- 一对一 coverage。

Track B 与 Track A 的差异：

- Track B 的 reference step 来自离线 voting，表达通常更规范。
- Track B 不需要处理模型 planner 中的格式错误。
- Track B 更需要防止 reference step 过抽象导致误匹配。

因此第一版建议比 Track A 更保守：

- 单 token overlap 不作为匹配依据。
- action 或 reference step 只有泛词时不匹配。
- `date`、`role`、`nationality`、`location` 这类字段词不能单独触发匹配。
- 不默认启用在线 LLM judge。

只有当离线分析证明 lexical matcher 明显低估有效覆盖时，再考虑 `intent_lexical`、embedding 或离线 LLM judge。

## Metrics 与分析

训练或离线分析至少记录：

```text
reference_alignment
ref_available
ref_n_steps
ref_n_actions
ref_n_covered
```

推荐派生分布：

```text
mean(reference_alignment)
mean(ref_available)
mean(ref_n_steps)
mean(ref_n_actions)
mean(ref_n_covered)
ref_n_covered / ref_n_steps
ref_n_covered / ref_n_actions
```

失败归因建议：

| 类别 | 含义 | 可能动作 |
| --- | --- | --- |
| `missing_reference` | 样本没有可用 `reference_steps` | 检查离线 reference generation |
| `invalid_reference` | reference step 为空、过长、含 tag 或重复严重 | 收紧 LLM voting 输出 validator |
| `no_actions` | 轨迹没有合法 `<tool_call>` | 检查 rollout / prompt / no-search shortcut |
| `unmatched_actions` | 有 action，但没有覆盖 reference | 检查 matcher 或 reference 抽象层级 |
| `partial_reference_coverage` | 覆盖部分 reference steps | 分析缺失步骤是否真必要 |
| `redundant_actions` | 覆盖 reference 后仍有多余 action | 检查模型是否过度搜索 |

## 与 Reward 的关系

第一版 Track B 是 observation，不是 reward composition：

```text
existing_score = format/retrieval/outcome score
final_score = existing_score
```

Track B 只新增：

```text
reference_alignment = S_ref
```

进入双轨 reward 前至少需要满足：

- `reference_steps` 覆盖率足够高。
- `reference_steps` validator 稳定。
- `S_ref` 分布可解释。
- 低分样本归因清楚。
- `S_ref` 与高质量轨迹有正相关。
- 不奖励无意义 query stuffing。
- 和 Track A 聚合后的 `R_path = max(S_self, S_ref)` 不会掩盖明显坏轨迹。

如果启用最终奖励组合，应新开一个明确的组合层：

```text
R_total = lambda_p * R_path + lambda_a * R_outcome + lambda_f * R_format
R_path = max(S_self, S_ref)
```

不要把 Track B 直接塞进 `compute_score_em` 的 outcome / format 逻辑里。

## 第一版实施范围

包含：

- 定义 `reference_steps` 数据契约。
- 定义 `S_ref` 计算方式。
- 定义 Track B 与 Track A 的解耦边界。
- 定义离线 reference plan 生成流程。
- 定义 metrics、失败归因与验收标准。

不包含：

- 在线 LLM judge。
- reward-time LLM voting。
- 立即修改 scalar reward。
- 立即实现 `R_path = max(S_self, S_ref)`。
- 把 Track B 写进 Track A analysis 脚本。

## 当前实现状态

截至 2026-05-20，第一版已实现到 reward 旁路观测：

- `qa_em_format.py` 已新增 `compute_reference_alignment_components`。
- `compute_score_components` 已返回 `reference_alignment`、`ref_available`、`ref_n_steps`、`ref_n_actions`、`ref_n_covered`。
- `main_ppo_format.py` 已把 Track B components 写入 `reward_components`。
- `ray_trainer.py` 已把 Track B components 暴露为训练 / 验证 metrics。
- `ppo_trainer.yaml` 已新增 `reward_model.max_reference_steps`，默认 `null`。
- 第一版仍保持 Track B 不改变 `final_score`；当前 `final_score` 仍只受既有 `base_score + path_bonus` 逻辑影响。

尚未实现：

- 离线拒绝采样和 LLM voting 生成 `reference_steps`。
- Track B 专用 JSONL analysis 脚本。
- `R_path = max(S_self, S_ref)` 双轨组合层。
- trajectory dump 的中立字段重命名。

## 设计验收

Track B 第一版设计是否成立，可以用这些问题验收：

- 没有 `<plan>` 的轨迹是否仍能计算 `S_ref`？
- 缺失 `reference_steps` 时是否稳定返回 `0.0` 并记录原因？
- 没有 `<tool_call>` 时是否稳定返回 `0.0`？
- 完全覆盖 reference 且无冗余 action 时是否得到 `S_ref = 1.0`？
- 冗余 action 是否会降低 `S_ref`？
- 重复 action 是否不会虚增 `ref_n_covered`？
- `<tool_response>` 中的文本是否绝不会被当作 action？
- 第一版是否保持 `final_score = existing_score`？
- Track B scorer 是否完全不读取 planner？
- Track A scorer 是否完全不读取 `reference_steps`？

## 后续版本记录规则

每次 Track B 相关修改都必须同步更新：

- `docs/track_b_reference_alignment_plan.md`：记录当前设计状态、接口或边界变化。
- `docs/track_b_debug_journal.md`：追加“现象 / 根因 / 调整 / 验证 / 后续观察”。

如果某次修改只修实现 bug，也至少要在 debug journal 里写清楚问题和解决方案，避免后续只看到最终代码而看不到排障路径。
