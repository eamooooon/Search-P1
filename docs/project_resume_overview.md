# Search-P1 项目简历与面试讲解文档

一句话定位：Search-P1 是在 Search-R1 reasoning-search 强化学习框架上，面向“可规划、可解析、可度量搜索路径”的 LLM 搜索智能体训练改造项目。

## 1. 项目从 Search-R1 baseline 开始

### 1.1 Search-R1 想解决什么问题

Search-R1 的核心目标是训练一个会“边推理、边搜索、再回答”的语言模型。普通问答模型只能依赖参数内知识，遇到开放域、多跳事实、冷门实体或时效性问题时容易编造。Search-R1 把搜索引擎作为环境工具接入 RL rollout，让模型学习什么时候需要外部知识、应该搜什么、如何根据 observation 继续推理，最后给出答案。

从 repo README 可以确认，Search-R1 是一个基于 veRL 的强化学习框架，支持 PPO、GRPO、reinforce 等 RL 方法，支持 Qwen、Llama 等模型，也支持本地 sparse/dense retriever 或在线搜索引擎。它不是单纯做 RAG inference，而是把 search/tool call 放进训练闭环，让模型通过 reward 学会使用工具。

### 1.2 基本交互闭环

Search-R1 的训练闭环可以概括为：

1. 模型根据 question 生成一段 assistant response。
2. response 中包含 reasoning、search/tool action 或 final answer。
3. rollout 环境解析模型输出，如果发现搜索动作，就调用 retriever。
4. retriever 返回 observation，环境把 observation 拼回上下文。
5. 模型基于新 observation 继续生成下一步。
6. 轨迹结束后，reward function 根据答案正确性、格式合法性、检索证据等信号给分。
7. PPO/GRPO 根据 reward 更新策略模型。

也就是说，模型并不是一次性回答，而是在多轮 rollout 中和搜索环境交互。训练时真正优化的是“生成可执行 search action 并最终答对”的策略，而不是只优化静态文本相似度。

### 1.3 Search-R1 的典型轨迹结构和 tag 风格

Search-R1 常见口径使用如下模型可见标签：

```text
<think>...</think>
<search>query</search>
<information>retrieval observation</information>
<answer>final answer</answer>
```

含义是：

- `<think>`：模型自己的推理过程。
- `<search>`：模型发起的搜索 query。
- `<information>`：环境返回的检索结果。
- `<answer>`：最终答案。

在本 repo 的当前 Search-P1 迁移中，这套旧标签已经被明确替换为新的训练轨迹合约：

| Search-R1 常见标签 | Search-P1 标签 | 含义 |
| --- | --- | --- |
| `<think>` | `<reasoning>` | 工具调用或最终答案前的推理 |
| `<search>` | `<tool_call>` | 模型发出的搜索动作，内容是 query |
| `<information>` | `<tool_response>` | 环境注入的检索 observation |
| `<answer>` | `<answer>` | 最终答案 |

需要注意：上面对 Search-R1 的描述来自 repo README 与该类方法的 baseline 口径；具体 Search-R1 论文中的全部实验细节不是本文重点，也不在这里扩写成已验证事实。

### 1.4 Search-R1 的训练目标

baseline 训练目标大致由几类信号组成：

- 答案正确性：例如 QA exact match，最终 `<answer>` 是否命中 ground truth。
- 格式合法性：模型是否输出可解析的 reasoning/search/answer 结构。
- 检索环境信号：搜索 observation 中是否包含答案或支持性证据。
- RL 优化方式：基于 veRL 的 PPO/GRPO 等策略优化，rollout 后把 reward 回填到 response token 上。

在 Search-P1 的当前 P1 训练脚本 `scripts/nq_hotpotqa_p1/train_grpo.sh` 中，训练入口是 `verl.trainer.main_ppo_format`，advantage estimator 设为 `grpo`，reward 配置包括：

```text
reward_model.structure_format_score=0.2
reward_model.final_format_score=0.1
reward_model.retrieval_score=0
reward_model.trajectory_dump_path=logs/$EXPERIMENT_NAME-tracka-v2.jsonl
reward_model.trajectory_dump_limit=200
```

这说明当前 Search-P1 分支已经不是只跑原始 Search-R1 格式，而是在 planner-format reward manager 下训练，并额外开启了 reward-time trajectory dump 供 Track A 分析。

### 1.5 Search-R1 baseline 的限制

从 Search-P1 的设计动机看，Search-R1 baseline 有几个工程上的限制：

- 没有显式前置计划：模型可以边想边搜，但没有一个独立的 plan block 声明“我准备搜哪些步骤”。
- 路径质量难度量：最终答对并不代表搜索路径好；答错也不容易知道是计划错、搜索错、证据错，还是最后综合错。
- 搜索行为容易漂移：模型可能多搜、少搜、重复搜，或者从一个 query 游走到另一个 query，缺少可解释约束。
- reward 多集中在 outcome/format：答案 EM、格式分、retrieval evidence 分更像结果或局部格式信号，不直接度量“是否按计划执行”。
- parser 和环境边界容易混淆：如果模型标签、环境 observation 标签、reward parser 标签没有同步迁移，训练时很容易出现 rollout 能跑但 reward 解析口径不一致的问题。

Search-P1 的改造目标就是把这些隐含路径行为变成显式结构和可观测指标。

## 2. Search-P1 做了哪些改进

### 2.1 轨迹结构升级

Search-P1 把一次训练时 assistant trajectory 规范为：

```text
Planner -> Reasoning -> Tool Call -> Tool Response -> Answer
```

序列化形式是：

```text
<plan>
Step 1: Search ...
Step 2: Search ...
</plan>
<reasoning>...</reasoning>
<tool_call>...</tool_call>
<tool_response>...</tool_response>
<reasoning>...</reasoning>
<answer>...</answer>
```

更抽象地说：

```text
T = (p, r_1, a_1, o_1, ..., r_n, a_n, o_n, r_final, a_hat)
```

其中 `p` 是前置 planner，`r_i` 是搜索前 reasoning，`a_i` 是 tool call，`o_i` 是环境 observation，`a_hat` 是最终答案。

这个结构的关键不是换标签，而是把“模型说它准备怎么搜”和“模型实际怎么搜”放进同一条可解析轨迹里，为后续 path reward 做准备。

### 2.2 Tag 迁移和 parser 同步

当前 repo 已经在设计文档和核心代码中统一了 Search-P1 标签：

- `<think>` -> `<reasoning>`
- `<search>` -> `<tool_call>`
- `<information>` -> `<tool_response>`

同步点包括：

- 数据 prompt：`scripts/data_process/qa_search_train_merge.py`、`qa_search_test_merge.py`、`nq_search.py` 已提示模型先输出 `<plan>`，后续使用 `<reasoning>/<tool_call>/<tool_response>/<answer>`。
- rollout parser：`search_p1/llm_agent/generation.py` 的 `postprocess_predictions` 只解析 `<tool_call>` 和 `<answer>` 作为 action tag；旧的 `<query>/<search>/<think>/<information>` 会进入 `malformed_action_tag`。
- 环境 observation：搜索结果被注入为 `<tool_response>...</tool_response>`。
- reward parser：`verl/utils/reward_score/qa_em_format.py` 的状态机校验 `<plan>/<reasoning>/<tool_call>/<tool_response>/<answer>` 顺序，并且 `extract_tool_calls` 会先移除 `<tool_response>` block，避免把 observation 里的伪 `<tool_call>` 当作模型 action。
- masking 配置：`verl/trainer/config/ppo_trainer.yaml` 使用 `<tool_response>` 作为 state masking marker。

这部分是很重要的工程工作，因为 rollout、reward、trainer 三层只要有一层仍按旧标签解析，训练数据流就会出现隐性错配。

### 2.3 Front-loaded planner、plan-once 和 planner_seen

Search-P1 引入了单个前置 planner：

```text
<plan>
Step 1: Search ...
Step 2: Search ...
</plan>
```

约束包括：

- planner 必须出现在 assistant trajectory 最前面。
- 只能出现一次。
- 每个非空 step 必须符合 `Step N: Search ...`。
- step 编号从 1 开始连续递增。
- planner 内不能嵌套 `<reasoning>/<tool_call>/<tool_response>/<answer>` 等标签。

在 rollout 侧，`LLMGenerationManager.run_llm_loop` 初始化 `planner_seen`，用它追踪每个样本是否已经接受过合法 plan。这个状态会影响 action parser 和 invalid feedback：

- `planner_seen=False` 时，模型必须先给出合法 `<plan>`。
- `planner_seen=True` 时，重复输出 `<plan>` 会被视为 `duplicate_plan`。
- final rollout step 调用 `execute_predictions(..., allow_plan_only=False)`，不再接受只输出 plan 的样本，因为没有后续 step 消费该 plan。

### 2.4 Plan-only 第一阶段

当前设计允许模型第一轮只输出 planner：

```text
<plan>
Step 1: Search ...
Step 2: Search ...
</plan>
```

如果该 plan 合法，并且本轮没有合法 `<tool_call>` 或 `<answer>`，rollout 会把它作为 `valid_plan` 接受：

- 不触发搜索。
- 不结束 trajectory。
- `valid_action=1`。
- `is_search=0`。
- 下一轮 rolling prompt 注入短控制指令，要求不要再输出 plan，并继续输出一个 `<reasoning>` 后接 `<tool_call>` 或 `<answer>`。

工程上还有一个细节：plan-only follow-up instruction 是控制文本，不是 `<tool_response>`。因此它可以进入 rolling prompt 帮助模型继续生成，但在最终用于 reward parsing 的 serialized trajectory 中会被 mask/pad 掉，避免污染 reward parser。

### 2.5 Invalid action feedback、action_reason_stats 和 debug samples

Search-P1 不只是把非法 action 判掉，还给了更细粒度的原因分类。`search_p1/llm_agent/generation.py` 中当前稳定 reason buckets 包括：

- valid：`valid_search`、`valid_answer`、`valid_plan`、`inactive`
- invalid：`missing_plan`、`duplicate_plan`、`missing_or_invalid_plan_steps`、`action_before_plan`、`missing_reasoning`、`invalid_tool_call`、`missing_action_tag`、`empty_prediction`、`malformed_action_tag`、`unknown_invalid`

这些 reason 会被聚合到 `DataProto.meta_info["action_reason_stats"]`，再由 `verl/trainer/ppo/ray_trainer.py` 输出成训练或验证指标，例如 `val/env/action_reason/...`。这对排查训练早期非常有用：如果模型经常失败，不再只能看到 reward 低，而是能知道失败集中在缺 plan、重复 plan、旧标签、缺 reasoning，还是 query 不合法。

同时，rollout debug sampling 通过环境变量控制：

```text
SEARCH_P1_ROLLOUT_DEBUG_SAMPLES=N
```

默认不开启。开启后会打印有限数量的样本，包含 reason、planner state、截断 prediction 和 observation。这个设计能支持定位 parser 行为，同时避免把完整 prompt/query 大量刷进日志。

### 2.6 Reward parser 和 trajectory parser 的同步改造

当前 Search-P1 的 reward parser 主要在 `verl/utils/reward_score/qa_em_format.py`：

- `is_valid_sequence` 用状态机校验完整结构。
- `extract_plan_steps` 从 `<plan>` 中提取 numbered search steps。
- `extract_tool_calls` 提取模型 `<tool_call>`，并显式移除 `<tool_response>` 内容。
- `count_actions` 返回可计数 tool call 数量。
- `validate_planner_block` 校验单个前置 plan、合法 step、连续编号。
- `is_valid_search_query` 拒绝空 query、嵌套标签、URL-like query 和超长 query。

rollout parser 和 reward parser 的职责不同：

- rollout parser 负责在线执行，决定本轮是 search、answer、plan-only 还是 invalid。
- reward parser 负责离线解析完整 decoded trajectory，给 format/outcome/path components 打分。

Search-P1 的改造点在于两者共享同一套 tag 语义和 planner 约束，但不把 trainer 变成第三个 parser。trainer 只消费 reward manager 写入的 `reward_components` 和 rollout 写入的 `action_reason_stats`。

有一个需要明确的当前边界：PRD 和 spec 中提到 `reward_model.require_search_for_format`，用于阻止“有合法 plan/reasoning/answer 但完全不搜索”的错误答案拿到结构轨迹格式分。当前代码检索中没有看到该参数在 `qa_em_format.py` 和 `main_ppo_format.py` 中实际透传，`train_grpo.sh` 也未设置该项。因此本文把它归为设计要求/待核对实现点，而不是已完成能力。

### 2.7 Track A Self-Consistency：已实现/设计的旁路观测信号

Track A 回答的问题很窄：

```text
模型是否按自己的 Planner 执行？
```

它不判断 planner 是否全局最优，不判断是否覆盖外部 reference steps，也不判断最终答案是否正确。

当前设计公式是：

```text
S_self = r_planner * (n_exec_self / n_plan) * (n_exec_self / n_actions)
```

变量含义：

- `r_planner`：planner 有效性门控。缺失、重复、非前置、无 numbered search step 时为 0。
- `n_plan`：planner 中声明的 search step 数。
- `n_actions`：模型实际发出的 `<tool_call>` 数。
- `n_exec_self`：实际 tool call 覆盖了多少 planner step。

两个比例分别度量：

- `n_exec_self / n_plan`：计划完成度。
- `n_exec_self / n_actions`：执行简洁性，冗余搜索越多越低。

当前实现位于 `qa_em_format.py`：

- `compute_self_consistency_components`
- `compute_self_consistency_score`
- `step_matches_action`
- `count_covered_steps`

第一版 matcher 是 deterministic lexical matching：lowercase、去标点、去低信息 token、containment/token overlap 判断。它不是最聪明的 matcher，但可解释、可复现、容易测试。

第一版 Track A 只记录，不改 reward：

```text
final_score = base_score
```

也就是说，`self_consistency` 只是进入 `reward_components` 和 trajectory dump，不作为 bonus 加到 scalar reward。这样做的原因是先观察信号分布和失败模式，避免一个还没校准的 path score 直接影响 RL 训练。

当前 RewardManager 记录的字段包括：

- `base_score`
- `self_consistency`
- `self_r_planner`
- `self_n_plan`
- `self_n_actions`
- `self_n_exec`
- `final_score`

注意：spec 中还提到 `has_search`、`effective_structure_format`、`effective_retrieval` 等字段，但当前代码的 `reward_components` 还没有这些字段，属于后续补齐或分支核对点。

### 2.8 Track B Reference Alignment：后续规划

Track B 和 Track A 必须解耦。Track A 看模型是否执行自己的 plan；Track B 看模型 action 是否覆盖外部参考步骤：

```text
actions <-> reference_steps
```

后续 Track B 可以包含：

- reference steps 生成：从参考答案、标准解析路径或强模型轨迹中提炼必要搜索步骤。
- 拒绝采样：生成多个候选 reference path，过滤掉不支持答案、过长、不可执行或重复的路径。
- LLM voting：让多个候选或多个 judge 对 reference steps 的必要性、一致性进行投票。
- 和 Track A 解耦：Track B 不读取模型 planner，Track A 不读取 `ground_truth.reference_steps`。

这样可以保证两个指标含义清楚：

- `S_self` 高：模型执行了自己的计划。
- `S_ref` 高：模型覆盖了外部参考必要步骤。

### 2.9 Aggregator 后续规划

后续路径奖励聚合层建议使用：

```text
R_path = max(S_self, S_ref)
```

直觉是：如果模型很好地执行了自己的合理计划，或者覆盖了外部参考必要步骤，都可以认为路径有价值。`max` 聚合也避免强行要求两条轨道同时高分。

但这仍是后续计划，不是当前已启用 reward。真正接入 scalar reward 时还需要额外设计：

```text
final_score = existing_score + weight * R_path
```

并配套做消融实验，验证它不会奖励 no-search shortcut、不会鼓励冗余搜索、不会牺牲最终答案。

### 2.10 离线分析脚本

当前 repo 已有离线分析入口：

```bash
python scripts/analysis/track_a_self_consistency.py samples.jsonl
```

输入 JSONL 至少包含：

```json
{"solution_str": "<plan>...</plan>...", "ground_truth": {"target": ["answer"]}}
```

脚本会读取 `solution_str`，调用 reward module 计算 Track A components，并输出：

- `self_consistency`、`self_r_planner`、`self_n_plan`、`self_n_actions`、`self_n_exec` 的 mean/min/p50/p90/max。
- planner valid rate。
- mean plan coverage：`self_n_exec / self_n_plan`。
- mean action efficiency：`self_n_exec / self_n_actions`。
- failure attribution：`invalid_planner`、`no_actions`、`unmatched_actions`、`partial_plan_coverage`、`redundant_actions`、`complete`。
- low-score samples 的文本片段。

也支持机器可读输出：

```bash
python scripts/analysis/track_a_self_consistency.py samples.jsonl --json
```

这个脚本和 `trajectory_dump_path` 配合，可以把训练/验证中的 decoded trajectory 抽样落盘，再离线观察 self-consistency 分布。

## 3. 项目架构和数据流

### 3.1 数据处理

P1 数据入口在 `scripts/nq_hotpotqa_p1/README.md` 中说明：

- train：从 `nq,hotpotqa` 合并生成 `data/nq_hotpotqa_p1/train.parquet`。
- test：从 `nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle` 合并生成 `data/nq_hotpotqa_p1/test.parquet`。
- 数据处理脚本在 prompt 中注入 Search-P1 标签规范，要求先输出完整 `<plan>`，再进行 `<reasoning>` 和 `<tool_call>`。

数据样本保留 QA reward 所需字段，例如 `data_source`、`prompt`、`ability`、`reward_model.ground_truth` 和 `extra_info`。

### 3.2 Rollout

训练和验证中的搜索 rollout 由 `search_p1/llm_agent/generation.py` 管理：

1. 根据当前 rolling prompt 调用 actor rollout worker 生成 response。
2. `_truncate_at_first_action` 在第一个 `</tool_call>` 或 `</answer>` 处截断，避免单轮生成多个 action。
3. `postprocess_predictions` 解析 action：
   - `<tool_call>` -> internal `search`
   - `<answer>` -> terminal answer
   - legal plan-only -> `plan`
   - 其他 -> invalid reason
4. `execute_predictions` 根据 action 调 retriever、注入 `<tool_response>`、返回 valid/done/search 标记。
5. `planner_seen` 更新 plan-once 状态。
6. 控制 observation 被用于 rolling prompt，但从最终 reward trajectory 中 mask 掉。

### 3.3 Retriever

retriever 是独立服务，训练脚本通过：

```text
retriever.url="http://127.0.0.1:8000/retrieve"
retriever.topk=3
```

把搜索请求发到本地检索服务。README 说明 Search-R1 支持本地 sparse/dense retriever 和在线 search engine。Search-P1 当前没有改 search backend 行为，主要改的是模型可见 action 标签、rollout 控制和 reward 解析。

### 3.4 Reward

P1 脚本使用 `verl.trainer.main_ppo_format`，其中 `RewardManager`：

1. decode prompt + response 得到完整 `sequences_str`。
2. 按 data source 选择 `qa_em_format.compute_score_components`。
3. 计算 base score：答案 EM、结构格式、最终答案格式、retrieval evidence。
4. 计算 Track A components。
5. 当前 `final_score = base_score`，Track A 不改 scalar reward。
6. 把 score 写到 response 最后一个有效 token。
7. 把 `reward_components` 写入 `data.meta_info`。
8. 如果开启 `trajectory_dump_path`，追加 JSONL trajectory dump。

### 3.5 Trainer metrics

`verl/trainer/ppo/ray_trainer.py` 聚合两类指标：

- reward components：`reward/<component>/mean|max|min`，验证时是 `val/reward/...`。
- action reason stats：`env/action_reason/...`，验证时是 `val/env/...`。

这样训练日志能同时看到结果分、路径观测信号和 rollout parser 失败原因。

### 3.6 Analysis

离线分析脚本 `scripts/analysis/track_a_self_consistency.py` 读取 trajectory JSONL，复用 reward parser 计算 Track A，并输出分布和失败归因。它是当前阶段连接训练日志和人工分析的工具，用于回答：

- planner 合法率是否足够高。
- 模型是否经常 no-search。
- `n_actions` 是否明显大于 `n_plan`。
- 低 `S_self` 是 parser 问题、matcher 问题，还是模型没有执行 plan。

## 4. 当前进度与可验证成果

### 已实现

- Search-P1 训练轨迹结构文档：`docs/trajectory_structure_design.md`。
- Track A self-consistency 设计文档：`docs/track_a_self_consistency_plan.md`。
- P1 专用训练脚本和 README：`scripts/nq_hotpotqa_p1/`。
- rollout tag 迁移：`<tool_call>/<tool_response>/<reasoning>`。
- `planner_seen`、plan-only 第一阶段、final step 禁止 plan-only。
- invalid action feedback 和 `action_reason_stats`。
- opt-in rollout debug samples。
- reward parser 的 planner/action/helper 和 Track A components。
- reward-time trajectory dump。
- trainer 对 reward components 和 action reason metrics 的聚合。
- Track A 离线分析脚本。

### 已设计但未完全接入 scalar reward

- Track A 作为 path signal 的定义和记录。
- Track B reference alignment。
- `R_path = max(S_self, S_ref)` 聚合层。
- 基于 path reward 的最终 scalar reward composition。

### 需要核对或后续补齐

- `require_search_for_format` 在 PRD/spec 中是要求，但当前代码中未看到完整透传和训练脚本配置。
- spec 中提到的 `has_search`、`effective_structure_format`、`effective_retrieval` 当前未出现在 `RewardManager.reward_components`。
- `infer.py` 仍能看到旧 `<think>/<search>/<information>` 示例，若后续要让 inference 也走 Search-P1 口径，需要单独迁移。
- Track B 的 reference data 生产、拒绝采样、LLM voting 还未落地。

### 可以跑的测试和检查

当前 repo 中与该项目相关的测试包括：

```bash
python -m pytest tests/test_track_a_self_consistency.py
python -m pytest tests/test_track_a_analysis_script.py
python -m pytest tests/test_trajectory_dump.py
python -m pytest tests/test_generation_control_observations.py
```

可观察指标包括：

- `reward/base_score/mean|max|min`
- `reward/final_score/mean|max|min`
- `reward/self_consistency/mean|max|min`
- `reward/self_r_planner/mean|max|min`
- `reward/self_n_plan/mean|max|min`
- `reward/self_n_actions/mean|max|min`
- `reward/self_n_exec/mean|max|min`
- `val/env/action_reason/<reason>`
- trajectory dump JSONL 中的 `track_a` 字段
- 离线分析脚本输出的 planner valid rate、plan coverage、action efficiency、failure attribution

## 5. 简历可写法

可以写成工程成果，不夸大实验效果：

- 基于 Search-R1/veRL 构建 Search-P1 搜索增强 RL 训练流程，将原 reasoning-search 轨迹升级为 front-loaded planner + tool-call 结构，统一 rollout、reward parser 与 trainer metric 的标签合约。
- 设计并实现 Search-P1 轨迹解析与校验机制，引入 `<plan>/<reasoning>/<tool_call>/<tool_response>/<answer>` 状态机、plan-once 约束、plan-only 首阶段和非法 action 原因统计，提升训练行为可观测性。
- 设计 Track A Self-Consistency 路径质量指标，基于 planner steps 与实际 tool calls 计算 `S_self`，以旁路 reward component 方式记录，不改变现有 scalar reward，降低早期 reward shaping 风险。
- 打通 reward-time trajectory dump、trainer 指标聚合和离线分析脚本，支持对 planner 合法率、计划覆盖率、搜索冗余和低分失败原因进行样本级分析。
- 规划 Track B reference alignment 与双轨路径奖励聚合方案，将模型自声明计划执行度和外部 reference step 覆盖度解耦，为后续 path reward 接入和消融实验打基础。

## 6. 面试讲解提纲

### Q1：这个项目和普通 RAG 有什么区别？

普通 RAG 多数是在 inference 前或中间检索文档，然后让模型读上下文回答。Search-P1 继承 Search-R1 的思路，把 search action 放进 RL rollout：模型要自己决定是否搜索、搜什么、何时停止，并通过 reward 学会这种策略。Search-P1 进一步要求模型先写 plan，再执行 tool calls，使搜索路径可解析、可度量。

### Q2：为什么要加 front-loaded planner？

没有 planner 时，只能看到模型实际搜了什么，很难判断路径质量。加 planner 后，模型先声明搜索策略，再执行 query，我们就能比较“计划”和“行动”是否一致。这样可以区分几类问题：模型计划无效、计划合理但没执行、执行过度搜索、或者 action 和 plan 匹配不上。

### Q3：为什么要从 `<think>/<search>/<information>` 迁移到 `<reasoning>/<tool_call>/<tool_response>`？

这不是单纯改名。新标签把模型动作和环境响应边界讲清楚：`<tool_call>` 是模型输出的可执行动作，`<tool_response>` 是环境注入的 observation。reward parser 会移除 `<tool_response>` 再提取 `<tool_call>`，避免把检索结果里的文本误判为模型 action。这个边界对训练稳定性很重要。

### Q4：plan-only 第一阶段解决什么问题？

要求模型一轮内同时输出完整 plan 和第一个 action，早期训练可能很难。plan-only 允许第一轮只输出合法 plan，rollout 接受后给下一轮控制指令，让模型继续输出 reasoning + action。这样既保留 front-loaded planner 约束，又降低第一步格式学习难度。最终 rollout step 不接受 plan-only，因为没有下一轮继续执行。

### Q5：Track A 的 `S_self` 为什么不直接加到 reward？

因为第一版 matcher 是 lexical matching，自然语言 plan 和 query 的匹配有噪声。如果直接把它加到 scalar reward，可能错误惩罚语义等价但词面不同的 action，也可能引入新的 reward hacking。当前做法是只记录 components，先看分布和失败样本，确认信号可靠后再讨论 reward composition。

### Q6：invalid action reason metrics 有什么价值？

RL 训练中 reward 低只能说明“结果不好”，不能告诉我们为什么。`action_reason_stats` 可以告诉我们失败集中在哪里，比如缺 plan、重复 plan、旧标签、缺 reasoning、query 不合法。这样调 prompt、parser 或训练配置时有明确方向，而不是盲目看样本。

### Q7：Track A 和 Track B 为什么要解耦？

Track A 的参照物是模型自己的 planner，回答“有没有按自己说的做”。Track B 的参照物是外部 reference steps，回答“有没有覆盖参考必要步骤”。如果 Track A 读取 reference，或者 Track B 读取 planner，指标含义会混乱。先解耦，后续再用 `R_path=max(S_self,S_ref)` 聚合，更容易解释和做消融。

### Q8：你在这个项目里最工程化的取舍是什么？

第一，先统一 tag/parser/masking/reward 的跨层合约，避免训练暗错。第二，Track A 先旁路记录，不改 scalar reward，降低 reward shaping 风险。第三，用 deterministic lexical matcher 起步，牺牲一部分语义召回，换来可解释、可复现、可测试，等分析证明需要时再升级 matcher。

## 7. 后续路线图

1. Track A 小样本分析：用 `trajectory_dump_path` 抽取训练/验证样本，跑 `scripts/analysis/track_a_self_consistency.py`，观察 `self_consistency`、planner valid rate、plan coverage、action efficiency 和失败归因。
2. Matcher 校准：人工检查高/低 `S_self` 样本，判断 lexical matcher 是低估有效执行，还是模型确实没有按 plan 行动。必要时增强 normalization、keyword extraction 或 containment 规则。
3. Track B reference 生成：定义 `reference_steps` 数据格式，探索强模型生成、多候选拒绝采样和 LLM voting，确保 reference steps 可执行、必要、不过长。
4. 双轨 reward 接入：保持 `S_self` 和 `S_ref` 解耦，新增聚合层 `R_path=max(S_self,S_ref)`，再设计 `final_score = existing_score + weight * R_path` 的接入策略。
5. 消融实验：比较无 planner、planner-only、Track A 旁路、Track A reward、Track A + Track B、不同 matcher、不同 path weight 对答案 EM、搜索次数、格式合法率和路径指标的影响。
6. no-search shortcut 防护：补齐或核对 `require_search_for_format`，确保错误答案且没有 `<tool_call>` 的轨迹不能通过结构格式分获益。
7. inference 口径收敛：如果要对外 demo Search-P1，需要把 `infer.py` 等残留旧标签示例迁移到新标签，保证训练和推理口径一致。

## 8. 简短总结

Search-P1 当前最适合在简历中描述为“Search-R1 的训练轨迹和路径度量工程化升级”。已完成的价值不在于声称刷新指标，而在于把原本难解释的搜索行为拆成可解析结构、可观测 metrics 和可离线分析样本，为后续 reference alignment 和 path reward 接入建立了稳定基础。
