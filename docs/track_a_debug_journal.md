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

## 2026-05-18 - v8 invalid planner shortcut 收紧

- 现象：
  - v7 证明 `require_search_for_format=true` 已经压住 no-actions / no-search shortcut。
  - 但部分 `invalid_planner` / invalid sequence 轨迹因为包含合法 `<tool_call>` 或 `<answer>`，错误答案仍能拿到 `final_format_score=0.1`。
- 根因：
  - 上一轮 gate 只要求错误答案拿 format shaping 时有合法 search，没有区分 structural / retrieval shaping 与 final-format shaping。
  - 对 Search-P1 来说，错误答案想拿任何 format shaping，不仅需要合法 search，还必须满足整体轨迹格式合法。
- 调整：
  - 在 `qa_em_format` 中将 `final_format_score` 的 gate 收紧为：`require_search_for_format=true` 时必须同时满足有合法 search 和 `is_valid_format=True`。
  - 保持 backward compatibility：`require_search_for_format=false` 时，invalid sequence + wrong answer 仍保留旧的 final-format shaping。
  - 保留 exact-match outcome reward：invalid format 的 EM 正确答案仍按现有 `score - structure_format_score` 逻辑给分。
- 验证：
  - 增加 invalid planner + legal `<tool_call>` + wrong `<answer>` 覆盖，确认 false 模式为 `0.1`、true 模式为 `0`。
  - 继续覆盖 valid Search-P1 + legal search + wrong answer 在 true 模式下保留 `structure_format_score=0.2`。
- 后续观察：
  - 用 v8 dump 观察 `invalid_planner` 样本的 `base_score` 分布，重点确认错误答案 invalid-format 轨迹不再靠 `final_format_score` 维持正反馈。

## 2026-05-18 - v8 诊断窗口收窄到 20 step

- 现象：
  - v8 诊断需要继续观察 `require_search_for_format=true` 和 invalid-format final gate 收紧后的早期训练行为。
  - 30-step 短跑耗时偏长，而 v6/v7 的关键问题已经在前 10-20 step 内足够显现。
- 根因：
  - 当前要验证的是 early stability 和 mid-run shortcut 是否仍然出现，不需要完整保留第 21-30 step 的训练窗口。
  - 在相同 batch / agent 设置下，20 step 仍覆盖足够多 reward-time trajectory：`data.train_data_num = 384 * 20 = 7680`，`trajectory_dump_limit = 384 * 3 * 20 = 23040`。
- 调整：
  - 将 P1 GRPO v8 诊断从 30 step 改为 20 step：`trainer.total_training_steps=20`。
  - 将 dump 路径改为 `logs/$EXPERIMENT_NAME-tracka-v8-20steps.jsonl`，避免和历史 30-step 文件混用。
  - 保留 `reward_model.require_search_for_format=true` 和其他 v8 gating 设置。
- 验证：
  - 运行 `bash -n scripts/nq_hotpotqa_p1/train_grpo.sh` 检查脚本语法。
  - 运行 `git diff --check` 检查补丁空白格式。
- 后续观察：
  - 优先用 20-step dump 对比前 10-20 step 的 no-search / invalid-format 分布，确认 early stability 和 mid-run shortcut 是否已经可判定，同时省掉后 10 step 时间。

## 2026-05-18 - v8 invalid planner structure shaping 对齐

- 现象：
  - v8 日志里仍能看到少量 `invalid_planner + base_score=0.2` 样本。
  - 典型轨迹里 planner 第一行是合法 `Step N: Search ...`，后续非空行不是严格的 `Step N: Search ...`，Track A 已判为 `self_r_planner=0`，但 format reward 仍把 full sequence 判为 valid。
- 根因：
  - `is_valid_sequence` 在 plan 数量检查后只调用 `extract_plan_steps` 判断“至少有一个合法 step”。
  - `extract_plan_steps` 会跳过非法 planner 行，导致“部分合法 planner”绕过结构格式校验，进而在 wrong answer + legal search 时拿到 `structure_format_score=0.2`。
- 调整：
  - `is_valid_sequence` 改为复用完整 `validate_planner_block` 校验。
  - planner block 必须单个前置、每个非空行都匹配 `Step N: Search ...`，且 step 编号连续。
- 验证：
  - 增加回归测试覆盖 partial planner step：`self_r_planner=0`、`is_valid_sequence=False`、`require_search_for_format=true` 时 `base_score=0`。
  - 保持 legacy 兼容：`require_search_for_format=false` 下 invalid format wrong answer 只拿 `final_format_score=0.1`，不再拿 structure 0.2。
- 后续观察：
  - 继续用 v8/v9 dump 检查 `invalid_planner` 样本的 `base_score` 分布，确认 `self_r_planner=0` 与 format reward 合法性不再分裂。

## 2026-05-20 - Track A intent-aware lexical matcher

- 现象：
  - `lexical` matcher 只能比较 planner 字面 step 和 `<tool_call>` query，遇到“先识别实体，再用搜索结果实例化后续查询”的轨迹时会低估执行覆盖。
  - 典型样本是 planner 写 `Search [identified actress] character in The Honeymooners.`，实际 action 写 `Joyce Randolph Trixie Norton The Honeymooners`，两者语义连续但字面 overlap 被 placeholder 稀释。
- 根因：
  - planner step 里有 `[identified ...]` 这类中间实体占位符，以及 identified / target / specific / information 等低信息胶水词。
  - action query 往往已经替换成检索得到的实体名，旧 matcher 没有 intent normalization，只能按完整 token overlap 判断。
- 调整：
  - 新增可选 `intent_lexical` 策略，默认仍为 `lexical`，并保留旧 matcher 行为不变。
  - `intent_lexical` 先尝试原 lexical；失败后移除 bracket placeholders 和低信息 intent glue，再按 intent tokens 与 action tokens 的覆盖率匹配。
  - 增加保护：intent tokens 少于 2 个不匹配；单个 overlap 如果只是 `role`、`date`、`nationality` 等泛词不匹配，避免 `[identified winner] nationality` 这类样本被过宽放行。
  - 训练脚本显式设置 `reward_model.path_match_strategy=intent_lexical`，dump 切到 `tracka-v10-intent-20steps.jsonl`，保留 20-step 和 `require_search_for_format=true`。
- 验证：
  - 增加单元测试覆盖 lexical 低估、intent 覆盖、泛词 negative case、重复 action 不重复覆盖多个 plan step、unsupported strategy 报错提示包含新策略。
- 后续观察：
  - v10 dump 优先比较 `intent_lexical` 后 `partial_plan_coverage` / `unmatched_actions` 是否下降，同时确认 `nationality`、`date`、`role` 等单泛词没有引入明显误匹配。

## 2026-05-20 - v11 planner intent prompt and max plan steps

- 现象：
  - v9/v10 暴露出 planner quality 和超长 plan 问题：模型容易把 planner 写成 exact query 清单、fallback branches、year-by-year / episode-by-episode expansion，导致计划不可执行或显著超过 `max_turns=4` 的执行预算。
  - 这类超长 planner 即使格式行都合法，也不应该拿结构轨迹格式分；Track A 也应将其视为 invalid planner。
- 根因：
  - 数据 prompt 只强调 numbered `Step N: Search ...` 和完整搜索策略，没有明确 planner 是 search intent list，而不是 exact query list。
  - reward / Track A 缺少 plan step 上限，无法把超过执行预算的 planner 判 invalid。
- 调整：
  - 数据 prompt 明确 planner step 是一个可执行 search intent，允许 `[identified actor]`、`[identified film]`、`[target entity]` placeholder。
  - prompt 禁止 fallback branches、year-by-year searches、episode-by-episode searches 和长 exhaustive lists，并替换为多跳 intent 示例。
  - reward / Track A 增加 `max_plan_steps` 参数；训练脚本设置 `reward_model.max_plan_steps=4`，与 `max_turns=4` 对齐。
  - dump 切到 `tracka-v11-intent-planlimit-20steps.jsonl`，避免和 v10 intent-only dump 混写。
- 验证：
  - 增加 5-step planner 回归测试：默认不设置上限时旧行为保持；`max_plan_steps=4` 时 `validate_planner_block` / `is_valid_sequence` 均失败，`self_r_planner=0`，`require_search_for_format=true` 下错误答案不拿结构格式分。
- 后续观察：
  - v11 dump 优先观察 `self_n_plan` 分布、`invalid_planner` 占比和 `base_score` 分布，确认 plan limit 没有误伤短多跳 intent planner。
## 2026-05-23 - v12 assistant/environment boundary prompt

- Phenomenon: v11 `no_actions` rose sharply, with samples copying or faking `<tool_response>` instead of stopping after `<tool_call>`.
- Root cause: the prompt example showed one continuous full trajectory, so the model treated environment observations as assistant output to copy.
- Adjustment: v12 changes data prompts to role-separated assistant/environment turns. Assistant examples stop at `</tool_call>` and environment-only examples return `<tool_response>`.
- Verification: run py_compile for data_process prompts, `bash -n` for the GRPO script, and `git diff --check`.

## 2026-05-24 - v13 Track A small-weight reward

- Phenomenon:
  - v12 fixed the assistant/environment prompt boundary, but self-consistency remains low.
  - Main failures are still partial plan coverage and actions that do not execute the declared plan.
- Root cause:
  - Track A has been observation-only, so it exposes the failure mode but does not reward plan-following behavior.
- Adjustment:
  - Add opt-in `reward_model.self_consistency_weight`.
  - Keep the default at `0.0`, so `final_score == base_score` for compatibility.
  - For v13 diagnostics set `self_consistency_weight=0.05`, with `final_score = base_score + self_consistency_weight * self_consistency`.
  - Keep `path_match_strategy=intent_lexical`, `max_plan_steps=4`, and `require_search_for_format=true`.
  - Do not implement Track B, `S_ref`, or `max(S_self, S_ref)`.
- Verification:
  - Add tests for zero weight, perfect Track A bonus, partial Track A bonus, no-action zero bonus, and absence of `path_bonus`.
- Follow-up observation:
  - Use `logs/$EXPERIMENT_NAME-tracka-v13-reward-20steps.jsonl` to compare whether a 0.05 Track A signal reduces `partial_plan_coverage` and unmatched action failures without destabilizing base reward behavior.

## 2026-05-24 - v13 reward run diagnosis and v14 action quality gate

- 现象：
  - v13 接入 `self_consistency_weight=0.05` 后，Track A 信号明显生效：训练集 `self_consistency` 均值约从 v12 的 `0.039` 提升到 `0.137`，后段 window 提升到约 `0.391`。
  - 但 `base_score` 仍很低，训练集均值约 `0.0025`；大量 complete self-consistent 样本仍是错误答案。
  - complete 样本主要集中在 1-step planner，说明模型可能在学习“短 plan + 一个匹配 action”的 Track A 捷径，而不是稳定提升答案正确性。
  - action 内容仍不干净，日志中有大量裸 `search`、`Search ...` 前缀、`search(...)`、`tool_call search ...`、嵌套标签或 JSON/function-call 风格内容。
- 根因：
  - Track A reward 已经给了路径自洽的正反馈，但合法 action 的质量边界不够一致。
  - rollout parser 只拦住部分伪工具写法；reward parser 的 `is_valid_search_query` 更宽，可能把一些低质量 `<tool_call>` 当作合法 search 参与 `has_search` 和 self-consistency。
  - 如果继续单纯提高 Track A 权重，模型会更容易优化格式/短路径分，而不是优化检索有效性和最终答案。
- 调整：
  - v14 不提高 `self_consistency_weight`，继续保持 `0.05`。
  - 收紧 rollout parser 和 reward parser 的 action quality gate：裸 `search` / `query`、`Search ...` / `query: ...` 前缀、`search(...)`、`tool_call search ...`、`tool_call: search(...)`、`tool_response:`、JSON-like 内容都视为非法 search query。
  - 保留 plain query，例如 `Albert Einstein birthplace` 仍是合法 `<tool_call>` 内容。
  - 训练脚本 dump 路径切到 `logs/$EXPERIMENT_NAME-tracka-v14-action-clean-gate-20steps.jsonl`，避免和 v13 混写。
- 验证：
  - 增加 reward parser 回归测试，确认伪工具 `<tool_call>` 不产生 `has_search`、`self_consistency`、`track_a_bonus` 或格式 shaping。
  - 扩展 rollout parser 测试，确认裸 `search`、`Search ...` 和 `tool_call search ...` 归入 `malformed_tool_call_content`。
  - 用 v13 dump 离线重算新 gate，`self_consistency` 从约 `0.150` 降到 `0.067`，complete 从 `2718` 降到 `1297`；这说明 v13 中相当一部分 Track A 分来自低质量 action，新 gate 会先压低但净化信号。
  - 本地已通过 `tests/test_track_a_self_consistency.py`；`tests/test_generation_control_observations.py` 在当前 Windows Python 环境因缺少 `torch` 无法收集，但相关文件已通过 `py_compile`。
- 后续观察：
  - v14 重点看 `has_search` 和 `self_consistency` 是否短期下降但更可信。
  - 如果 `bare_search` / `search_prefix` 明显下降，同时 `plain_query` 占比上升，说明 action-clean gate 生效。
  - 如果 `base_score` 仍无改善，再考虑降低 Track A 权重到 `0.02`，或引入更严格的 action informativeness / outcome-aware gate。

## 2026-05-24 - v14 action-clean gate log observation

- 现象：
  - v14 训练集 `self_consistency mean = 0.074`，低于 v13 原口径，但高于用 v13 dump 离线套新 gate 的 `0.067`。
  - 训练后段仍明显上升：step 1-5 约 `0.013`，step 16-19 约 `0.209`；val 约 `0.379`。
  - `has_search` 被 gate 压到训练集约 `0.291`，但同样随训练推进从约 `0.112` 升到 `0.511`。
  - action 内容分布显示模型仍大量输出裸 `search`，但 plain query 随训练增加：step 1-5 plain `689` / bare `2897`，step 16-19 plain `2468` / bare `697`。
- 根因：
  - v14 gate 没有阻断学习；它把旧的伪工具调用从 reward 信号中清掉，使早期分数下降。
  - 模型仍在迁移格式习惯，部分输出从裸 `search` 转为 `Search ...` 前缀或 `search(...)`，这些仍被判非法。
  - 少量 `plain_query` 仍是低信息字符串，如 `search-P1`、`search-MIob`，说明仅靠 prefix gate 还不能完全保证 query informativeness。
- 调整：
  - 暂不提高 Track A 权重；v14 说明 0.05 已能推动 action 质量迁移。
  - 下一步优先考虑继续保留 v14 gate，并观察更长 run 或加强 prompt 中 plain query 的正例。
  - 如果需要 v15，可以增加低信息 query 规则，例如拒绝 `search-...` 这类只有搜索词缀但无自然语言实体/属性的内容。
- 验证：
  - v14 dump 共 `22656` 条，其中 train `21888`、val `768`。
  - train complete `1386`，val complete `248`；complete 轨迹仍大多 `base_score=0`，说明 Track A 仍主要改善路径，不直接改善答案正确性。
- 后续观察：
  - 比较 v15 时重点看 `plain_query / total tool_call`、`bare_search`、`search_prefix`、`base_score` 是否同步改善。
  - 如果 `base_score` 持续不动，需要引入 outcome-aware gate 或降低 Track A 权重，避免路径分压过答案分。

## 2026-05-24 - v15 low-information search-prefix query gate

- 现象：
  - v14 的 `plain_query` 增加是好趋势，但抽样里出现了 `search-P1`、`search-MIob` 这类低信息内容。
  - 这些内容没有违反 v14 的裸 `search`、`Search ...`、`search(...)`、JSON 或嵌套 tag 规则，因此会被当作合法 plain query。
- 根因：
  - 模型正在从旧的 action 习惯迁移，可能把 `search` 当成词缀拼到随机 token 或任务名上。
  - 简单禁止所有连字符会误伤合法实体，如 `Q-learning algorithm`、`Spider-Man actor`、`COVID-19 symptoms`。
- 调整：
  - 只拒绝整体形如 `search-xxx` / `query-xxx` 且没有空格的短伪 query。
  - 保留信息量更高的连字符 query，例如 `Search-P1 paper contribution`、`Q-learning algorithm`、`Spider-Man actor`。
  - rollout parser 和 reward parser 同步应用该规则，避免训练环境和 reward 口径不一致。
  - 训练 dump 路径切到 `tracka-v15-low-info-query-gate-20steps.jsonl`。
  - analysis 脚本增加 action quality 汇总，直接输出 `plain_query`、`bare_search`、`search_prefix`、`low_info_search_prefix`、`function_search` 等分布，减少每次诊断都临时写脚本。
  - 数据 prompt 同步提示不要输出短 `search-xxx` / `query-xxx` 伪 query；若重跑 `scripts/nq_hotpotqa_p1/data_process.sh`，新 parquet 会带上该约束。
- 验证：
  - 增加伪 query negative cases：`search-P1`、`query-MIob`。
  - 增加合法连字符 positive cases：`Search-P1 paper contribution`、`Q-learning algorithm`、`Spider-Man actor`、`COVID-19 symptoms`。
- 后续观察：
  - v15 dump 重点看 `plain_query` 中低信息 `search-*` / `query-*` 是否消失，同时确认合法 hyphenated entity 没有被误伤。

## 2026-05-24 - v16 rollout feedback for clean query recovery

- 现象：
  - v15 新 run 后期 Track A 明显学习，`no_actions` 从约 500 降到几十，`complete` 上升到约 200，`self_consistency` 后期可到 `0.25+`。
  - 但 `unmatched_actions` 仍是最大失败项，action quality 中仍有大量 `bare_search`、`search_prefix`、`low_info_search_prefix`、`function_search` 和 `nested_tag`。
- 根因：
  - parser / reward gate 已经能判错，但 invalid feedback 对模型来说还不够操作化。
  - 模型知道当前 action 错了，却没有被明确示范下一次 `<tool_call>` 内部应该只写具体 query 内容。
- 调整：
  - 强化 `malformed_tool_call_content` 的 rollout feedback，直接给出 good / bad query content。
  - good 示例只写内容本身：`Albert Einstein birthplace`。
  - bad 示例覆盖当前高频错误：`search`、`Search Albert Einstein birthplace`、`search(Albert Einstein birthplace)`、`search-P1`、`query-MIob`。
  - 不在 feedback 里放完整 XML 对，避免模型复制 `<tool_call>...</tool_call>` 占位示例。
  - dump 路径切到 `tracka-v16-feedback-clean-query-20steps.jsonl`，避免和 v15 append 混写。
- 验证：
  - 增加 feedback 回归测试，确认包含 concrete plain query 指令、good 示例和高频 bad 示例。
  - 保留 full XML pair 防复制测试。
- 后续观察：
  - v16 优先看 `bare_search`、`search_prefix`、`low_info_search_prefix`、`function_search` 是否下降。
  - 如果 action quality 改善但 `unmatched_actions` 仍高，再考虑 matcher 或 plan/action intent 表达问题。

## 2026-05-31 - SFT 冷启动数据与 verl SFT 训练入口

- 现象：
  - Search-P1 直接做 RL 时，Track A 后期能推动格式和路径一致性，但 `base_score` 长期偏低，answer correctness 没有同步改善。
  - v16 100-step 训练还出现中后期格式崩塌，说明仅靠在线 RL 从零学习 Planner -> Search -> Think -> Answer 轨迹成本高、稳定性差。
  - 用户希望先做 SFT 冷启动，但当前仓库的 `fsdp_sft_trainer.py` 已经引用 `verl.utils.dataset.SFTDataset`，而本地 `verl/utils/dataset` 没有导出该 Dataset，直接跑 verl SFT 会报错。

- 根因：
  - 当前 Search-P1 数据是 RL parquet，字段核心是 `prompt` 和 `reward_model.ground_truth`，不是 verl SFT trainer 期望的 `prompt / response` 监督学习格式。
  - SFT trainer 的训练逻辑需要 `input_ids`、`attention_mask`、`position_ids`、`loss_mask`，其中 loss 只能打在 assistant response 上；缺少 Dataset 时无法建立这个边界。
  - 之前只讨论了 A 类模板数据，没有把它转换成 verl 可直接消费的 parquet，也没有提供启动 SFT 的脚本和 swanlab 监控入口。

- 调整：
  - 新增/整理 `scripts/sft/build_sft.py`，从现有 `data/nq_hotpotqa_p1/*.parquet` 抽取 `prompt[0].content` 和 `reward_model.ground_truth.target`，构造 Search-P1 冷启动 response。
  - `build_sft.py` 支持两类输出：
    - `jsonl`：保留 `messages + metadata`，方便人工检查。
    - `verl_parquet`：输出 `prompt / response / metadata`，用于 verl FSDP SFT。
  - 补齐 `verl/utils/dataset/sft_dataset.py` 并在 `verl/utils/dataset/__init__.py` 导出 `SFTDataset`；实现口径对齐上游 verl：prompt 套 chat template，拼接 `response + eos`，padding/truncation 后只对 response 区间计算 `loss_mask`。
  - 新增 `scripts/sft/build_sft.sh`，默认生成 `data/nq_hotpotqa_p1/search_p1_sft_format_10k.parquet`。
  - 新增 `scripts/sft/train_sft.sh`，默认先 build train/val SFT parquet，再通过 `torchrun -m verl.trainer.fsdp_sft_trainer` 启动训练，并设置 `trainer.logger=['swanlab']`。
  - 用户将脚本名从 `build_search_p1_sft.py / train_search_p1_sft.sh` 收敛为 `build_sft.py / build_sft.sh / train_sft.sh` 后，同步修正 `train_sft.sh` 和测试里的旧路径引用，避免运行时找不到旧文件。

- 验证：
  - 本地执行 `bash -n scripts/sft/train_sft.sh scripts/sft/build_sft.sh` 检查 shell 语法。
  - 本地执行 `python -m py_compile scripts/sft/build_sft.py verl/utils/dataset/sft_dataset.py` 检查 Python 语法。
  - 本地执行 `python -m pytest tests/test_build_search_p1_sft.py`，结果为 `3 passed, 1 skipped`；skip 原因是当前 Windows Python 没安装 `torch`，服务器 search 环境有 torch 时应执行 Dataset loss_mask 测试。

- 后续观察：
  - 第一阶段先用模板 A 数据做冷启动，建议训练后用小规模 rollout 评估：planner_valid_rate、plain_query 占比、no_actions、unmatched_actions、base_score。
  - 如果 SFT 后格式稳定但答案仍弱，再补 B 类 clean rollout 轨迹；C 类 LLM synthetic 多跳数据暂时作为第三步，不应先引入过多噪声。
  - SFT 达到可进入 RL 的最低标准建议是：合法 planner 稳定高于 90%，plain query 占比明显高于 RL 冷启动初期，no_actions 显著下降，并且短 rollout 中不再大面积复制 `<tool_response>` 或伪造工具结果。

## 2026-05-31 - SFT 数据 train/test 切分工具补齐

- 现象：
  - 已经有 `build_sft.py` 能从现有 RL parquet 构造 Search-P1 SFT 数据，也有 `train_sft.sh` 能启动 verl SFT。
  - 但缺少一个独立的 SFT 数据切分工具；如果后续把 A 类模板数据、B 类 clean rollout、C 类 synthetic 数据先合并成一个大文件，就没有稳定方式切出 train/val。

- 根因：
  - 之前默认沿用原始 `data/nq_hotpotqa_p1/train.parquet` 和 `test.parquet` 分别生成 SFT train/val。
  - 这个方式适合第一版模板 A 数据，但不适合后续混合多来源数据，因为混合后需要在同一分布上重新切分，避免 train/val 来源不一致。

- 调整：
  - 新增 `scripts/sft/split_sft.py`，支持 `jsonl` 和 `parquet` 两种 SFT 数据格式。
  - 支持 `--val-size` 固定验证集大小，也支持 `--val-ratio` 按比例切分；默认 shuffle，并可用 `--no-shuffle` 保留原顺序。
  - 新增 `scripts/sft/split_sft.sh`，提供 shell 入口，默认把 `search_p1_sft_format_10k.parquet` 切成 `search_p1_sft_train.parquet` 和 `search_p1_sft_val.parquet`。
  - `train_sft.sh` 新增 `SPLIT_FROM_FULL=1` 路径：先 build 一个 full SFT parquet，再调用 `split_sft.py` 生成 train/val，然后启动 SFT。

- 验证：
  - 新增 `tests/test_split_sft.py`，覆盖 parquet 固定 `val_size` 切分和 jsonl 按 `val_ratio` + `--no-shuffle` 切分。
  - 保留 `tests/test_build_search_p1_sft.py`，确认 build 输出与 SFTDataset 兼容逻辑不受影响。

- 后续观察：
  - 后续引入 B/C 数据时，推荐先合并成一个 full SFT 文件，再用该工具切分，保证验证集能反映混合数据整体分布。
  - 如果需要严格按数据来源分层切分，可以在 `metadata.sft_type` 或 `metadata.data_source` 上再扩展 stratified split；当前第一版只做随机切分。
