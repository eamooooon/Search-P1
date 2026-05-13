# Search-P1 轨迹结构设计

## 目标

本文定义 Search-P1 训练时使用的模型侧轨迹结构，目标是把一次问答 rollout 表达为可解析、可校验、可计算路径信号的序列：

```text
Planner -> Search -> Think/Reasoning -> Answer
```

在序列化文本中，规范轨迹写作：

```text
T = (p, r_1, a_1, o_1, ..., r_n, a_n, o_n, r_final, a_hat)
```

其中：

- `p` 是单个前置 planner，对应 `<plan>...</plan>`。
- `r_i` 是第 `i` 次工具调用前的推理，对应 `<reasoning>...</reasoning>`。
- `a_i` 是第 `i` 次搜索动作，对应 `<tool_call>...</tool_call>`。
- `o_i` 是环境返回的搜索结果，对应 `<tool_response>...</tool_response>`。
- `r_final` 是最终作答前的推理。
- `a_hat` 是最终答案，对应 `<answer>...</answer>`。

本文只描述训练时的结构合约，不定义新的奖励权重，也不要求生成离线 reference trajectory。

## Tag 合约

Search-P1 的模型可见标签固定为：

| 标签 | 来源 | 含义 |
| --- | --- | --- |
| `<plan>...</plan>` | 模型 | 单个前置 planner，必须在任何 search/action 之前出现 |
| `<reasoning>...</reasoning>` | 模型 | 工具调用或最终答案前的推理 |
| `<tool_call>...</tool_call>` | 模型 | 搜索动作，内容是 query |
| `<tool_response>...</tool_response>` | 环境 | 检索系统返回给模型的 observation |
| `<answer>...</answer>` | 模型 | 简洁最终答案 |

旧的 Search-R1 标签不再作为合法模型输出：

- `<think>` 替换为 `<reasoning>`。
- `<search>` 替换为 `<tool_call>`。
- `<information>` 替换为 `<tool_response>`。

内部实现仍可把 `<tool_call>` 映射到现有 search action，但模型侧与 reward parser 只能看到 Search-P1 标签。

## Planner 结构

Planner 是单个前置 `<plan>` block。它必须：

- 出现在完整 assistant trajectory 的最前面。
- 只出现一次。
- 包含编号步骤。
- 每个步骤使用 `Step N: Search ...` 形式。
- 描述完整搜索策略，而不是只描述当前一步。

示例：

```text
<plan>
Step 1: Search the main entity mentioned in the question.
Step 2: Search the specific attribute needed for the final answer.
</plan>
```

不合法情况：

- 缺失 `<plan>`。
- 出现多个 `<plan>`。
- planner 在 `<tool_call>` 或 `<answer>` 后出现。
- planner 中没有 numbered `Step N: Search ...`。
- planner step 内嵌套 `<reasoning>`、`<tool_call>`、`<tool_response>`、`<answer>` 等标签。

## 序列化语法

完整合法轨迹的常见形式：

```text
<plan>
Step 1: Search ...
Step 2: Search ...
</plan>
<reasoning>...</reasoning>
<tool_call>...</tool_call>
<tool_response>...</tool_response>
<reasoning>...</reasoning>
<tool_call>...</tool_call>
<tool_response>...</tool_response>
<reasoning>...</reasoning>
<answer>...</answer>
```

抽象为：

```text
<plan>p</plan>
(<reasoning>r_i</reasoning><tool_call>a_i</tool_call><tool_response>o_i</tool_response>)*
<reasoning>r_final</reasoning><answer>a_hat</answer>
```

约束：

- `<tool_response>` 只能由环境注入，不能作为模型 action 解析。
- `<tool_call>` 必须由模型输出，并由 rollout 环境执行。
- 每次 `<tool_call>` 前必须有一个 `<reasoning>`。
- 最终 `<answer>` 前必须有一个 `<reasoning>`。
- `<answer>` 结束后不应再有 `<tool_call>` 或 `<tool_response>`。
- 标签外的自由文本应视为格式风险，reward parser 应按现有严格状态机处理。

## Plan-only 第一阶段

在线 rollout 允许一个特殊的第一阶段：当样本还没有接受过 planner 时，模型可以先只输出合法 `<plan>...</plan>`，本轮不触发搜索、不结束 trajectory。

接受条件：

- `planner_seen == false`。
- 当前输出包含且只包含一个合法 front-loaded `<plan>`。
- 当前输出没有合法 `<tool_call>` 或 `<answer>`。
- 当前 rollout 后续还有可用 generation step。

接受后的行为：

- 标记该样本 `planner_seen = true`。
- 将该轮视为有效 planner action。
- 向下一轮 rolling prompt 注入短控制指令，要求模型不要再输出 `<plan>`，继续输出一个 `<reasoning>` 后接一个 `<tool_call>` 或 `<answer>`。
- 该控制指令不是 `<tool_response>`，不得写入最终用于 reward parsing 的序列化 trajectory。

最终 rollout step 不接受 plan-only 输出，因为没有后续 step 消费该 plan。

## 数据流

### Prompt 侧

数据处理脚本需要在系统/用户提示中明确：

- 先输出完整 `<plan>`。
- planner 使用 numbered `Step N: Search ...`。
- 后续每次搜索使用 `<reasoning>` 加 `<tool_call>`。
- 检索结果会由环境放入 `<tool_response>`。
- 不再使用 `<think>`、`<search>`、`<information>`。

### Rollout 侧

rollout parser 的职责：

- 只把 `<tool_call>` 解析为可执行搜索动作。
- 只把 `<answer>` 解析为终止动作。
- 在 `planner_seen == false` 时强制等待合法 planner。
- 在 `planner_seen == true` 时拒绝重复 planner。
- 对空 query、嵌套标签 query、URL-like 或超长 query 继续按非法 action 处理。

环境 observation 注入的职责：

- 搜索返回必须包在 `<tool_response>...</tool_response>` 中。
- observation masking 也应使用 `<tool_response>` 边界。
- invalid-action feedback 不是 `<tool_response>`，应按现有控制文本路径处理。

### Reward 侧

reward parser 的职责：

- 校验单个前置 `<plan>`。
- 校验 planner numbered search steps。
- 校验 `<reasoning>/<tool_call>/<tool_response>/<answer>` 状态顺序。
- 提取 planner steps 与 tool calls，供后续 path reward 使用。
- 当 `require_search_for_format == true` 时，零 `<tool_call>` 的错误答案轨迹不能获得结构轨迹格式分。

## 接口建议

建议 reward parsing 层提供稳定 helper，而不是让 trainer 重复解析文本：

```python
extract_plan_steps(solution_str) -> list[str]
extract_tool_calls(solution_str) -> list[str]
count_actions(solution_str) -> int
validate_planner_steps(steps) -> bool
is_valid_format(solution_str) -> bool
```

接口边界：

- `extract_plan_steps` 只读取 assistant trajectory 中的 `<plan>` 内容，不能读取 prompt 示例。
- `extract_tool_calls` 只读取模型输出的 `<tool_call>` block，不能从 `<tool_response>` 中提取 query。
- `count_actions` 等价于合法或可计数的 `<tool_call>` 数量。
- helper 应保持确定性，不调用在线 LLM 或外部 embedding 服务。

## 测试验收

文档对应的实现应至少覆盖这些验收用例：

- 合法完整轨迹通过格式校验。
- 缺失 planner、重复 planner、planner 在 action 后出现均失败。
- planner 没有 `Step N: Search ...` 时失败。
- `<think>/<search>/<information>` 不被兼容为合法新标签。
- `<tool_response>` 内出现看似 query 的文本不会被计为 action。
- plan-only 第一阶段在非最终 rollout step 可接受，在最终 rollout step 不可接受。
- `require_search_for_format == true` 时，错误答案且没有 `<tool_call>` 的轨迹不获得结构轨迹格式分。

## 范围边界

本设计包含：

- Search-P1 训练轨迹标签合约。
- planner 前置与 numbered search step 合约。
- rollout、environment、reward parser 的职责边界。
- 后续路径奖励所需的基础提取接口。

本设计不包含：

- 离线 reference trajectory 生成。
- Track B reference steps 数据生产。
- path reward 权重接入策略。
- hosted reference LLM provider 选择。
- 对现有 search backend 的行为修改。
