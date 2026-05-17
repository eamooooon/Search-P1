# Track A / Trajectory 工程复盘日志

本文档用于记录 Track A / trajectory 相关更新的工程复盘。后续每次相关改动都追加一条记录，固定格式为：时间 / 现象 / 根因 / 调整 / 验证 / 后续观察。重点写清楚“遇到什么问题 + 怎么解决”，避免后续只看到最终设计而看不到排障路径。

## 当前状态快照

- 已解决：
  - Track A 已明确为旁路观测信号，只记录 self-consistency components，不改变 scalar reward。
  - `path_bonus` 口径已移除，避免误读为路径分已经进入最终 reward。
  - analysis 脚本已兼容 Python 3.9，不再使用 `int | None` 这类 3.10+ type union 写法。
  - reward-time decoded `solution_str` 已支持写入 trajectory JSONL，用于离线 Track A 分析。
  - plan-only 控制文本污染 final trajectory 的问题已修复：rolling prompt 保留 feedback，final trajectory / dump mask 掉 control observation。
  - 旧 `<query>` / 非法格式不会被兼容成合法动作，而是归入 `malformed_action_tag`。
- 未解决：
  - v2 dump 中 planner/action 格式仍不稳定，planner 合法率很低。
  - `<query>` / 旧格式残留下降后，模型仍会产生 no action、unmatched action、partial coverage 等失败样本。
- 下一步：
  - 优先修 prompt 和 action format guidance，让模型稳定输出 `<plan>`、`<reasoning>`、`<tool_call>`、`<answer>` 结构。
  - 暂不调整 Track A scorer 或把 Track A 纳入 scalar reward，先继续观察格式稳定性和失败归因。

## 2026-05-14 - Track A 解耦与 `path_bonus` 口径移除

- 现象：
  - Track A 初版容易被误读成已经参与最终 reward composition。
  - `path_bonus` 这个名字暗示路径分已经作为 bonus 加到 scalar reward。
- 根因：
  - Track A 的语义是 self-consistency：比较模型自己的 planner steps 和实际 tool calls 是否一致。
  - Track B 的语义是 reference-alignment：比较模型 action 和外部 reference steps 是否一致。
  - 两者参照物不同，如果混用或提前合成，训练指标含义会变得难解释。
- 调整：
  - Track A 只记录 components / metrics，不改变 scalar reward。
  - 去掉 `path_bonus` 口径，避免日志字段暗示 reward 已被加权。
  - 保持 Track A 不读取 `ground_truth.reference_steps`，Track B 不读取模型 planner。
- 验证：
  - 设计口径确认：Track A 第一版是 observation，不是 reward composition。
  - 测试覆盖确认：`path_bonus` 不再出现在 Track A components 中。
- 后续观察：
  - 只有当 Track A 分布稳定、失败归因清楚、prompt 格式稳定后，再讨论是否进入 reward composition。

## 2026-05-14 - analysis 脚本 Python 3.9 兼容

- 现象：
  - analysis 脚本在 Python 3.9 下运行报错，问题出在 `int | None` 类型注解。
- 根因：
  - `X | Y` 的类型联合写法是 Python 3.10+ 语法；当前环境需要兼容 Python 3.9。
- 调整：
  - 将 `int | None` 改成 `Optional[int]`。
  - 保持脚本逻辑不变，只修类型注解兼容性。
- 验证：
  - Python 3.9 环境可以解析脚本。
  - Track A analysis 输入输出语义不变。
- 后续观察：
  - 后续新增脚本时继续避免使用 3.10+ 语法，除非项目明确升级 Python baseline。

## 2026-05-14 - analysis 输入不是原始 parquet，而是 rollout trajectory JSONL

- 现象：
  - 直接拿原始 parquet 数据跑 Track A analysis 没有可分析样本。
- 根因：
  - Track A analysis 需要的是 rollout 后的 assistant trajectory，也就是包含 decoded `solution_str` 的 JSONL。
  - 原始 parquet 只提供训练输入和标签，不包含模型实际生成的 `<plan>`、`<tool_call>`、`<answer>` 序列。
- 调整：
  - 明确 analysis 脚本读取 rollout trajectory JSONL。
  - 新增 / 使用 trajectory dump，把 reward-time decoded `solution_str` 落盘。
- 验证：
  - JSONL 行中包含 `solution_str` 后，analysis 可以读取并计算 Track A metrics。
- 后续观察：
  - 后续排查 Track A 分布时，先确认输入文件是 trajectory dump，不是原始训练 parquet。

## 2026-05-14 - trajectory dump 落盘语义确认

- 现象：
  - 需要把训练 / 验证中的 decoded trajectory 抽样保存，供离线 Track A 分析。
  - dump 行数和 step 数的语义容易混淆。
- 根因：
  - Track A 的分析对象是完整 assistant trajectory，而不是 rollout 的单个 step。
  - `trajectory_dump_limit` 限制的是样本行数，不是每条样本中的 generation step 数。
- 调整：
  - reward-time decoded `solution_str` 自动写 JSONL。
  - dump 默认关闭，需要配置 `trajectory_dump_path` 才启用。
  - `trajectory_dump_limit` 按样本行数计数；`0` 表示不 dump，负数可表示不限量。
- 验证：
  - dump JSONL 可被 analysis 脚本读取。
  - 每行包含完整 `solution_str`，而不是单步片段。
- 后续观察：
  - 训练中只开小样本 dump 做诊断，避免大规模写盘影响运行。

## 2026-05-14 - v1 dump 全部 `invalid_planner`

- 现象：
  - v1 dump analysis 结果显示样本几乎全部归因为 `invalid_planner`。
  - 这不像真实 planner 能力问题，更像 trajectory 串被污染。
- 根因：
  - invalid feedback 控制文本混入了 final trajectory。
  - 这些控制文本不是模型生成的合法 `<plan>` / `<tool_call>` / `<answer>` 内容，却进入了 reward parser 和 dump。
  - 污染文本影响 `<plan>`、`<tool_call>`、`<answer>` tag 计数，导致 planner 校验失败。
- 调整：
  - 区分 rolling prompt 的控制 feedback 和最终 serialized trajectory。
  - 控制文本可以继续给下一轮 prompt 使用，但不能进入 final trajectory / dump。
- 验证：
  - 复盘确认 v1 的 `invalid_planner` 高占比来自控制文本污染，而不应直接归因于 Track A scorer。
- 后续观察：
  - 之后看到大面积 `invalid_planner` 时，先检查 final trajectory 是否含有非模型输出的控制 observation。

## 2026-05-14 - 修复控制文本污染 final trajectory

- 现象：
  - rollout 需要把 feedback 放回 rolling prompt，引导模型继续输出正确格式。
  - 但 reward parser / trajectory dump 只应看到模型真实生成内容和真实工具返回。
- 根因：
  - 同一段 observation 同时服务 rolling prompt 和 final trajectory，缺少对 control observation 的 mask 边界。
- 调整：
  - rolling prompt 仍保留 invalid / plan-only feedback，帮助模型下一步纠偏。
  - final trajectory / dump mask 掉 control observation。
  - 真实搜索结果仍以 `<tool_response>` 保留，因为它是环境对合法 tool call 的实际响应，属于训练 trajectory 合约的一部分。
- 验证：
  - v2 dump 中控制文本污染下降。
  - 真实 `<tool_response>` 不被误删，trajectory 仍能表达工具调用后的环境返回。
- 后续观察：
  - 后续新增任何 feedback / control message 时，都要明确它是否允许进入 final trajectory。

## 2026-05-14 - `<query>` / 旧格式残留归因

- 现象：
  - v2 dump 中仍能看到模型输出 `<query>` 或旧格式残留。
  - 这些输出不符合 Search-P1 当前 action tag 合约。
- 根因：
  - 模型仍受旧格式或 prompt 示例影响，未稳定迁移到 `<tool_call>`。
  - 如果把 `<query>` 兼容成合法动作，会掩盖 action format guidance 的真实问题。
- 调整：
  - 不把 `<query>` / 旧格式兼容成合法动作。
  - 将这类样本归入 `malformed_action_tag`，保留失败可见性。
- 验证：
  - v2 显示控制文本污染下降。
  - 同时仍能通过 `malformed_action_tag` 看见模型格式不稳的问题。
- 后续观察：
  - 下一步应修 prompt / action format guidance，而不是放宽 parser 兼容旧标签。

## 2026-05-14 - v2 dump 分析结论

- 现象：
  - v2 dump 分析 200 条样本后，Track A 指标仍然很低。
  - 具体结果：
    - `planner_valid_rate = 0.075`
    - `self_consistency mean = 0.00375`
    - `invalid_planner = 185`
    - `no_actions = 7`
    - `unmatched_actions = 6`
    - `partial_plan_coverage = 2`
- 根因：
  - 控制文本污染下降后，主问题转移为模型自身格式不稳定。
  - 大量样本仍不能稳定产出合法 planner / action trajectory。
- 调整：
  - 结论从“修 Track A scorer”转为“修 prompt / action format guidance”。
  - 暂不把低分解读为 Track A 指标无效；当前它主要暴露格式生成问题。
- 验证：
  - 200 条 dump 的失败归因支持该判断：`invalid_planner` 仍占绝对多数，少量样本进入 no action / unmatched / partial coverage。
- 后续观察：
  - prompt 修复后重新 dump 同等规模样本，对比 planner valid rate、self-consistency mean 和失败归因分布。

## 后续追加模板

```md
## YYYY-MM-DD - 标题
- 现象：
- 根因：
- 调整：
- 验证：
- 后续观察：
```

## 2026-05-14 - Search-P1 prompt / malformed action reason 细化

- 现象：
  - v2 dump 共 200 条样本，`planner_valid_rate = 0.075`，`self_consistency mean = 0.00375`。
  - 控制文本污染下降后，模型仍大量输出 `<query>`、`<tool_query>`、嵌套 `<tool_call>`、`tool_call: search(...)`、`/query` 等旧格式或伪工具格式。
- 根因：
  - 数据 prompt 中仍有 `<tool_call> query </tool_call>` 这类占位式示例，容易让模型把 `query` 或旧 Search-R1 格式当成可复制格式。
  - rollout parser 之前把多类旧格式统一记为 `malformed_action_tag`，无法区分是旧 query tag、Search-R1 legacy tag，还是合法 wrapper 内部内容不合法。
- 调整：
  - Search-P1 数据 prompt 改为单个干净正例，`<tool_call>` 内使用具体 plain query，并加入强禁止列表。
  - action parser 新增 `malformed_query_tag`、`malformed_legacy_tag`、`malformed_tool_call_content`，继续保持 `<query>` 等旧格式非法。
  - trainer action reason allowlist 同步新增 reason，确保训练 / 验证指标能记录细分桶。
- 验证：
  - 增加 parser 测试覆盖 `<query>`、`<tool_query>`、`/query`、`<search>`、`<think>`、`<information>`、伪工具调用内容和合法 plain query。
  - 运行相关 Track A / trajectory / generation 测试与 `py_compile`、`git diff --check`。
- 后续观察：
  - 重新生成训练 parquet 或确认数据生成流程会重跑这些 scripts 后，再跑同规模 dump，对比 planner valid rate、self-consistency mean 和 malformed reason 分布。

## 2026-05-14 - v3 append dump 与 warm-up 趋势验证

- 现象：
  - `logs/...tracka-v3.jsonl` 是 append 文件，总计约 600 条；前 400 条混入旧 run，最后 200 条才对应新 prompt。
  - 直接分析整文件会把旧 prompt 的失败分布混进新 prompt 结果，导致 planner valid rate / self-consistency 判断被稀释或误导。
- 根因：
  - trajectory dump 采用追加写入，同一路径复用时不会自动截断历史 run。
  - v3 的有效观察窗口应限定到最后 200 条，而不是整文件 600 条。
- 调整：
  - 最后 200 条确认新 prompt 已生效，应作为 v3 prompt 修复后的主要分析样本。
  - 为验证“训练初始 step 是否需要 warm-up 才稳定格式”的假设，analysis 脚本新增 `--tail` 读取 append 文件尾部样本，并新增 `--bucket-size` 输出按时间顺序的 bucket trend。
- 验证：
  - 后续分析 v3 / v4 / v5 时优先使用 `--tail 200` 隔离当前 run，再用 bucket trend 观察每桶 planner_valid_rate、self_consistency、complete / no_actions / invalid_planner / partial_plan_coverage / unmatched_actions / redundant_actions 是否随时间改善。
- 后续观察：
  - v4 / v5 dump 继续按桶对比早期和后期样本；如果后期 bucket 明显改善，说明 warm-up 假设有支持，否则应继续排查 prompt、rollout feedback 或 reward parser 边界。

## 2026-05-14 - warm-up 10-step 诊断短跑

- 现象：
  - 需要验证 Track A 格式是否会随着训练 step warm-up 逐步变稳。
  - 现有较短 dump 容易混入 val-before-train、save/test 周期或 append 历史样本干扰，难以只看训练初期趋势。
- 根因：
  - 如果训练刚开始的格式不稳定来自 warm-up，单看前几百条混合样本无法判断后续 bucket 是否改善。
  - 诊断 run 需要固定训练步数、独立 dump 路径和足够样本上限，避免与 checkpoint / test / val-before-train 行为混杂。
- 调整：
  - 将 GRPO 脚本改为 10-step 短跑，`data.train_data_num=3840`，并显式设置 `trainer.total_training_steps=10`。
  - 禁用 `val_before_train`，把 `save_freq` / `test_freq` 调大到 `999999`，减少诊断过程干扰。
  - 独立 dump 到 `logs/$EXPERIMENT_NAME-tracka-v4-10steps.jsonl`，并把 `trajectory_dump_limit` 提高到 `1000`。
- 验证：
  - 运行 `bash -n scripts/nq_hotpotqa_p1/train_grpo.sh` 检查脚本语法。
  - 运行 `git diff --check` 检查空白和补丁格式。
- 后续观察：
  - 短训练完成后，用 analysis 脚本加 `--bucket-size 100` 查看 planner valid rate、self-consistency 和失败归因是否随 step 呈改善趋势。

## 2026-05-15 - 30-step warm-up 与 final validation 限流

- 现象：
  - v4 10-step 诊断证明后 1000 条 dump 是 final validation，不是训练过程崩溃。
  - `no_actions` 在 10 step 内没有呈现单调下降趋势，无法确认是否只是 warm-up 不足。
- 根因：
  - 10 step 观察窗口偏短，模型可能还没有获得足够格式探索机会。
  - final validation 默认全量运行，诊断 run 结束后会花很久跑验证并写入大量 val 样本。
- 调整：
  - 将诊断 run 扩展为 30 step，`data.train_data_num=11520`，`trainer.total_training_steps=30`。
  - 将 dump 路径改为 `logs/$EXPERIMENT_NAME-tracka-v5-30steps.jsonl`，`trajectory_dump_limit=3000`，保留更长训练窗口。
  - 设置 `data.val_data_num=1000`，限制 final validation 规模，避免全量验证过久。
  - 继续保留 `val_before_train=false` 与 `save_freq` / `test_freq=999999`，避免训练前验证和中途 checkpoint / test 干扰诊断。
- 验证：
  - 运行 `bash -n scripts/nq_hotpotqa_p1/train_grpo.sh` 检查脚本语法。
  - 运行 `git diff --check` 检查空白和补丁格式。
- 后续观察：
  - v5 完成后优先分析 train split，并使用 `--limit 3000 --bucket-size 100` 观察 `no_actions`、planner valid rate、self-consistency 和失败归因是否随训练 step 改善。

## 2026-05-17 - require_search_for_format 启用 no-search shortcut 防护

- 现象：
  - v6 full dump 证明继续增加训练 step 没有解决 `no_actions`，反而出现 invalid planner 坍塌。
  - 大量样本没有合法 `<tool_call>`，但会伪造 `<tool_response>` 并直接给 `<answer>`，仍可能拿到格式 shaping。
- 根因：
  - `require_search_for_format` 只存在于 PRD/spec 设计里，reward scorer 和 P1 GRPO 脚本没有实际透传启用。
  - no-search / 伪造 tool response 的错误答案仍能通过结构或 final 格式分获得正反馈，形成 shortcut。
- 调整：
  - 在 `qa_em_format` 中实现 `require_search_for_format` gate：没有合法 `<tool_call>` 搜索查询时，错误答案或无答案轨迹不能获得 structure / retrieval / final format shaping。
  - 保留 exact-match outcome 奖励：正确答案仍按原有高分逻辑处理，避免过度惩罚已知答案。
  - 在 RewardManager、Hydra 默认配置和 P1 GRPO 脚本中透传并启用该开关，同时记录 `has_search`、`effective_structure_format`、`effective_retrieval`。
- 验证：
  - 覆盖 no-search 兼容模式、gate 模式、合法 tool call 错误答案、no-search exact match 和组件字段。
- 后续观察：
  - 用 v7 dump 观察 `no_actions`、invalid planner、EM 和 format shaping 分布是否改善，重点确认 no-search wrong-answer 不再靠格式分维持。
