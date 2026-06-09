# Track A Self-Consistency 设计思路

## 设计动机

Search-P1 的 planner 只有在训练中能被度量时才有意义。模型先写 `<plan>`，再执行 `<search>`，这件事本身提供了一个很直接的自监督信号：模型有没有按自己声明的搜索计划行动。

Track A 只回答这个问题：

```text
模型是否按自己的 Planner 执行？
```

它不回答：

- 这个 planner 是否是全局最优路径。
- 这个 planner 是否覆盖人工或参考模型给出的必要步骤。
- 最终答案是否正确。
- 检索结果是否充分支持答案。

因此 Track A 是 Self-Consistency，不是 Reference-Alignment，也不是 Outcome Reward。

## 核心定义

Track A 分数定义为：

```text
S_self = r_planner * (n_exec_self / n_plan) * (n_exec_self / n_actions)
```

变量含义：

- `r_planner`：planner 有效性门控。planner 缺失、重复、位置错误，没有 numbered `Step N: Search ...`，或超过启用的 `max_plan_steps` 上限时为 `0`。
- `n_plan`：planner 中声明的搜索步骤数。
- `n_actions`：模型实际发出的 `<search>` 数量。
- `n_exec_self`：实际 `<search>` 覆盖了多少 planner step。

这个公式有两个乘法比例：

- `(n_exec_self / n_plan)` 衡量计划完成度：声明的步骤有多少真的被执行。
- `(n_exec_self / n_actions)` 衡量执行简洁性：实际搜索动作里有多少是在落实 planner，而不是额外游走。

直觉上，一个高分轨迹应该满足三件事：

1. planner 本身是合法的。
2. planner 里写的搜索步骤基本被执行。
3. 执行动作没有大量 planner 之外的冗余搜索。

## 为什么 Track A 必须和 Track B 解耦

Track A 的参照物是模型自己的 planner。Track B 的参照物是外部 `reference_steps`。这两个信号含义不同，如果混在一起，训练指标会变得难解释。

Track A 应只依赖：

- assistant 轨迹中的 `<plan>`。
- assistant 轨迹中的 `<search>`。
- planner step 与 action 的匹配规则。

Track A 不应依赖：

- `ground_truth.reference_steps`。
- 离线 reference trajectory。
- 参考模型生成的 reasoning path。
- 最终答案字符串。

Track B 后续只依赖：

- `actions`
- `reference_steps`

Aggregator 后续再统一组合：

```text
R_path = max(S_self, S_ref)
```

这个拆法的好处是口径清楚：

- `S_self` 高：模型执行了自己的计划。
- `S_ref` 高：模型覆盖了外部参考必要步骤。
- `R_path` 高：模型至少在一条路径奖励轨道上表现好。

如果 Track A 读取 `reference_steps`，那 `S_self` 就不再是 self-consistency；如果 Track B 读取 planner，`S_ref` 也不再是纯 reference-alignment。两条轨道需要先保持干净，后面 aggregator 才有意义。

## 第一版范围

第一版 Track A 只做观测信号，不改变训练奖励。

包含：

- 计算 `S_self`。
- 记录 Track A 相关组件。
- 验证 planner/action 解析是否足够稳定。
- 验证 self-consistency 信号是否能区分好坏轨迹。

不包含：

- 改变 scalar reward。
- 引入 `path_bonus`。
- 引入 `S_ref`。
- 引入 `R_path = max(S_self, S_ref)`。
- 使用 `reference_steps`。
- 使用 embedding matcher 或 LLM judge。

第一版的 reward 行为必须保持：

```text
final_score = existing_score
```

也就是说，`S_self` 只是被计算和记录，不给额外奖励，也不对原有奖励做惩罚。

## 为什么不保留 path_bonus 口径

`path_bonus` 这个名字暗示路径分已经作为 bonus 加到了最终 scalar reward。Track A 第一版并不会这么做，所以继续记录 `path_bonus` 会造成实验解读混乱。

更清晰的命名是直接记录自一致性本身：

```text
self_consistency
self_r_planner
self_n_plan
self_n_actions
self_n_exec
```

后续如果要加入双轨聚合，再显式记录：

```text
S_self
S_ref
R_path
```

这样可以避免把“度量值”和“奖励增量”混成一个字段。

## Planner 与 Action 的关系

Planner 是模型提前声明的搜索策略：

```text
<plan>
Step 1: Search ...
Step 2: Search ...
</plan>
```

这里的 planner step 是 search intent，不是 exact query list。每个 step 表示一个可执行搜索目标，后续 `<search>` 可以把中间检索结果实例化为具体 plain query。对于依赖未知中间结果的多跳问题，planner 可以写 `[identified actor]`、`[identified film]`、`[target entity]` 这类 placeholder；不应写 fallback branches、year-by-year searches、episode-by-episode searches 或长枚举式 exhaustive lists。

Action 是模型实际执行的搜索：

```text
<search>...</search>
```

Track A 的关键设计点是把两者放在同一个语义空间里比较：

```text
planner step  <->  search query
```

第一版不追求完美语义匹配，而是追求可解释、可复现、容易测试。基础策略使用 deterministic lexical matching：

- lowercase。
- 去掉标签和常见标点。
- 折叠空白。
- 可去掉低信息 token。
- 用 containment 或 token overlap 判断是否匹配。

这会漏掉一些改写表达，但它有一个重要优点：当分数变化时，我们能解释原因。对于 reward wiring 的第一版，这是比“看起来更聪明但不稳定”的匹配器更重要的性质。

## Coverage 设计

`n_exec_self` 不应该被重复 action 或宽松匹配虚高。第一版建议采用一对一 coverage：

- 一个 planner step 最多被一个 action 覆盖。
- 一个 action 最多覆盖一个 planner step。
- 重复 action 不重复增加 `n_exec_self`。
- 冗余 action 仍计入 `n_actions`。
- 匹配不要求顺序一致。

这样设计后，Track A 同时体现“按 planner 行动”和“少做废动作”。

示例 1：完全按 planner 行动。

```text
n_plan = 2
n_actions = 2
n_exec_self = 2
S_self = 1.0 * (2 / 2) * (2 / 2) = 1.0
```

示例 2：执行了计划，但有冗余搜索。

```text
n_plan = 2
n_actions = 3
n_exec_self = 2
S_self = 1.0 * (2 / 2) * (2 / 3) = 0.666...
```

示例 3：写了 planner，但实际没有执行。

```text
n_plan = 2
n_actions = 2
n_exec_self = 0
S_self = 0.0
```

## 数据流视角

Track A 的数据流应保持单向、旁路：

```text
serialized trajectory
  -> extract planner steps
  -> extract model tool calls
  -> match planner steps against actions
  -> compute S_self
  -> log self-consistency components
```

它不应该反向影响：

- rollout 是否继续。
- answer reward。
- format reward。
- retrieval reward。
- scalar reward。

这能让第一版承担一个清晰职责：先观察 self-consistency 是否是有用信号，再决定后续是否把它纳入训练目标。

## 接口边界建议

这些接口的意义不是规定实现步骤，而是划清模块边界，避免 trainer、reward parser、logger 各自重新解析轨迹。

解析边界：

```python
extract_plan_steps(solution_str) -> list[str]
extract_search_calls(solution_str) -> list[str]
validate_planner_steps(steps) -> bool
validate_planner_block(solution_str, steps=None, max_plan_steps=None) -> bool
```

匹配边界：

```python
step_matches_action(step, action, match_strategy="lexical") -> bool
count_covered_steps(steps, actions, match_strategy="lexical") -> int
```

评分边界：

```python
compute_self_consistency_score(solution_str, match_strategy="lexical", max_plan_steps=None) -> float
```

边界要求：

- `<information>` 中的文本不能被计为 action。
- planner 必须是单个、前置 block；重复 planner 或非前置 planner 应使 `r_planner = 0`。
- planner 的非空行都必须是 `Step N: Search ...`，并且编号应从 `1` 开始连续递增。
- planner step 中嵌套标签应视为 planner 无效。
- 如果启用 `max_plan_steps` 且 planner step 数超过上限，planner 应视为无效；训练诊断使用 `reward_model.max_plan_steps=4` 与 `max_turns=4` 对齐。
- 非 `lexical` 的 matcher 策略在第一版不应静默 fallback。
- `compute_self_consistency_score` 不接收 `reference_steps`。

## 与现有 Reward 的关系

Track A 第一版是 reward component observation，不是 reward composition。

现有分数继续由现有逻辑决定：

```text
existing_score = format/retrieval/outcome score
final_score = existing_score
```

Track A 只额外产生：

```text
self_consistency = S_self
```

后续如果要启用路径奖励，应单独设计组合层：

```text
R_path = max(S_self, S_ref)
final_score = existing_score + weight * R_path
```

这个组合层属于后续阶段。第一版不通过 `path_reward_weight` 改变训练行为。

## 设计验收

Track A 设计是否成立，可以用以下问题验收：

- 没有 `reference_steps` 的样本是否仍能计算 `S_self`？
- 缺失或非法 planner 是否稳定得到 `S_self = 0.0`？
- 没有 `<search>` 的轨迹是否稳定得到 `S_self = 0.0`？
- 完全执行 planner 且没有冗余 action 时是否得到 `S_self = 1.0`？
- 冗余 action 是否会降低分数？
- 重复 action 是否不会虚增 `n_exec_self`？
- `<information>` 是否绝不会被当作 action？
- 记录指标是否只表达 self-consistency，而不是 bonus？
- 开启 Track A 记录前后，scalar reward 是否完全一致？

如果这些问题都能回答“是”，Track A 第一版就是一个干净的自一致性观测轨道。

## 后续补充方向

第一版落地后，Track A 不应该马上进入 reward composition。更稳的路线是先补观测、校准和失败样本分析，把这个信号的脾气摸清楚。

建议继续补充三类内容。

### 1. 观测面补全

当前 `self_consistency` 只能告诉我们总分，但后续分析需要看它为什么高或低。因此 Track A 应继续保留并观察分解字段：

```text
self_r_planner
self_n_plan
self_n_actions
self_n_exec
```

这些字段回答的问题分别是：

- planner 合法性是不是主要瓶颈？
- 模型是不是写了过长或过短的 plan？
- 模型是不是搜索动作过多？
- 搜索动作到底覆盖了多少自声明步骤？

如果只看 `self_consistency`，很容易把“planner 无效”“没有搜索”“搜索冗余”“匹配失败”混成同一种低分。

### 2. 分布校准

Track A 第一版不改 scalar reward，所以它最有价值的用途是做离线/训练旁路分析。建议至少看这些分布：

```text
mean(self_consistency)
mean(self_r_planner)
mean(self_n_plan)
mean(self_n_actions)
mean(self_n_exec)
self_n_exec / self_n_plan
self_n_exec / self_n_actions
```

重点不是追求某个固定阈值，而是确认信号是否符合直觉：

- 合法 planner 的比例是否足够高。
- `n_plan` 是否集中在合理区间。
- `n_actions` 是否明显大于 `n_plan`，说明模型在 planner 之外游走。
- 高 answer reward 样本是否通常也有更高 `S_self`。
- 低 `S_self` 样本主要是 parser 问题、matcher 问题，还是模型真的没有执行 planner。

这一步决定后续要优化 parser、prompt、matcher，还是 reward 组合。

### 3. 失败样本归因

后续需要给低分样本建立简单归因，避免只看到一个数字。

建议归因类别：

| 类别 | 含义 | 可能动作 |
| --- | --- | --- |
| `invalid_planner` | planner 缺失、重复、非前置或没有 numbered search step | 优先看 prompt/rollout 约束 |
| `no_actions` | 没有 `<search>` | 检查 no-search shortcut |
| `unmatched_actions` | 有搜索，但和 planner steps 匹配不上 | 看 matcher 或 planner 表达 |
| `redundant_actions` | 覆盖 planner 后仍有多余搜索 | 看模型是否过度搜索 |
| `overabstract_plan` | planner 太抽象，难以和 query 对齐 | 调整 planner prompt |

这些归因可以先作为分析脚本或 debug 输出，不一定立刻写进 reward components。核心是让 Track A 的失败可解释。

## 下一步计划

### Step 1: 固化第一版实现

先把当前第一版变成可靠基线：

- 保持 `final_score = existing_score`。
- 保持无 `path_bonus`。
- 保持 Track A 不读取 `reference_steps`。
- 保证 `<information>` 内伪 `<search>` 不计入 action。
- 保证重复/非前置 planner 时 `S_self = 0`。
- 保证 planner 中混入非 `Step N: Search ...` 行或编号不连续时 `S_self = 0`。

验收标准：现有 reward scalar 不变，component metrics 能稳定输出 Track A 字段。

### Step 2: 增加最小测试覆盖

把现在手工验证过的样例沉淀成自动测试或轻量验证脚本：

- 完全匹配时 `S_self = 1.0`。
- 冗余 action 降分。
- 重复 action 不虚增 `n_exec_self`。
- 缺 planner、重复 planner、非前置 planner 均为 `0`。
- planner 混入非法行或编号不连续时为 `0`。
- planner 超过 `max_plan_steps=4` 时为 `0`，并且在 `require_search_for_format=true` 下错误答案不能拿结构格式分。
- `<information>` 中的伪 `<search>` 不计数。
- `compute_score_em` 返回值不受 `S_self` 影响。

验收标准：以后改 parser、matcher、logger 时不会悄悄改变 Track A 口径。

### Step 3: 做一轮样本分布分析

用一小批训练或验证 rollout 样本统计 Track A 分布：

- `self_consistency` 分布。
- planner 合法率。
- `n_plan` 和 `n_actions` 的关系。
- 高/低 `S_self` 样本各抽样若干条人工看。

验收标准：能回答“低分主要来自模型行为，还是来自匹配器太弱”。

当前离线分析入口：

```bash
python scripts/analysis/track_a_self_consistency.py samples.jsonl
```

输入 JSONL 的最小格式：

```json
{"solution_str": "<plan>...</plan>...", "ground_truth": {"target": ["answer"]}}
```

其中 `ground_truth` 可用于同时输出 `base_score/final_score`；如果只关心 Track A，也可以只提供 `solution_str`。

脚本会输出：

- `self_consistency`、`self_r_planner`、`self_n_plan`、`self_n_actions`、`self_n_exec` 的分布。
- planner 合法率。
- 平均 plan coverage：`self_n_exec / self_n_plan`。
- 平均 action efficiency：`self_n_exec / self_n_actions`。
- 低分样本的初步归因和文本片段。

如果需要机器可读输出：

```bash
python scripts/analysis/track_a_self_consistency.py samples.jsonl --json
```

### Step 4: 决定是否增强 matcher

只有当样本分析证明 lexical matcher 明显低估有效执行时，再考虑增强 matcher。

可选方向：

- 更好的 normalization。
- query keyword extraction。
- step/action containment 规则细化。
- 离线 embedding 或 LLM judge。

第一阶段不建议上在线 LLM judge，因为 reward 计算需要稳定、低成本、可复现。

### Step 5: 再讨论是否进入奖励组合

只有满足下面条件，才值得讨论把 Track A 纳入 scalar reward：

- Track A 分布稳定。
- 低分归因基本可解释。
- `S_self` 和高质量轨迹有正相关。
- 不会奖励 no-search shortcut。
- Track B 的 `reference_steps` 口径已经明确，或明确决定先只启用 Track A。

如果进入这一阶段，应该新开组合设计，而不是复活 `path_bonus` 旧口径。更清晰的方向是：

```text
S_self = Track A
S_ref = Track B
R_path = max(S_self, S_ref)
final_score = existing_score + weight * R_path
```

这属于下一阶段，不是 Track A 第一版的一部分。

## 风险与取舍

Track A 的最大风险是 planner 和 action 都是自然语言，匹配天然会有噪声。第一版选择 lexical matching，是为了让信号可解释、可测试、可替换。

这个取舍意味着：

- 分数可能低估语义等价但词面不同的执行。
- planner 太抽象时，匹配会比较困难。
- 一个 action 实际可能覆盖多个 planner step，但第一版不允许多重覆盖。

这些限制是可接受的，因为 Track A 第一版不改变 scalar reward。它先作为观测信号存在，等分布和失败模式被看清楚后，再决定是否升级 matcher 或接入路径奖励。

## 2026-05-24 - v13 small-weight Track A reward experiment

Track A has moved from observation-only logging to an opt-in small-weight reward experiment. Backward compatibility remains the default:

```text
reward_model.self_consistency_weight = 0.0
final_score = base_score
```

When the weight is positive, the scalar reward is:

```text
track_a_bonus = self_consistency_weight * self_consistency
final_score = base_score + track_a_bonus
```

The v13 diagnostic run uses `reward_model.self_consistency_weight=0.05` with the existing Track A settings: `path_match_strategy=intent_lexical`, `max_plan_steps=4`, and `require_search_for_format=true`.

This is still not Track B. It does not read `reference_steps`, does not compute `S_ref`, and does not use `R_path = max(S_self, S_ref)`. The component name is `track_a_bonus`; the old `path_bonus` name remains forbidden because it blurs Track A self-consistency with future path aggregation.
