# Search-P1 简历模板与面试准备

## 信息边界 / 使用方式

本文用于 Search-P1 项目的简历投递和面试准备，默认表述为“已经设计并实现 Search-P1 核心训练链路与工程化落地”。所有指标仍使用 `[x%]`、`[+y points]`、`[N]`、`[A -> B]` 这类占位符，面试前必须用本地真实实验日志、trajectory dump 分析结果和训练记录替换。

- 可写成完成态：主导实现面向 Agentic RL / Tool-Use Agent RL 的 Search-Agent 训练链路，覆盖显式 planner 轨迹构造、双轨 path reward、轨迹诊断与消融实验入口。
- 可以说明项目面向检索增强问答场景，但简历主定位应放在 Agentic RL、Tool-Use Agent RL 或 Search-Agent RL，而不是 Agentic RAG。
- 不要把论文摘要中的平均 `7.7 points` 增益固定写成本地结果。除非本地全量实验确实得到对应日志，否则只能写“论文报告”或用 `[+y points]` 占位。
- 简历中优先写可自证内容：训练轨迹合约、rollout/reward/parser/masking/metrics 同步、Track A self-consistency、Track B reference-alignment、trajectory dump、invalid action reason、no-search shortcut、reference_steps 数据链路和消融设计。
- 项目名、时间、模型、数据集、GPU、训练步数、benchmark、最终分数和提升幅度均需要按实际日志改写。

简历定位一句话：

> 主导设计并实现 Search-P1 的 Agentic RL / Tool-Use Agent RL 训练链路，基于 Search-R1/veRL 将检索增强问答中的搜索行为改造为显式 planner + tool-call 轨迹，并构建双轨 path reward、reference_steps 数据链路、轨迹诊断和消融实验体系。

## 简历项目条目

**项目名**：Search-P1：面向 Tool-Use Agent RL 的路径中心强化学习训练

**时间**：`YYYY.MM - YYYY.MM`，按真实项目周期填写

**角色**：强化学习 / LLM 算法工程，按真实职责填写

**技术栈**：Python、PyTorch、veRL、PPO/GRPO、Transformers、Qwen/Llama、HuggingFace Datasets、Parquet/Arrow、JSONL、FAISS/Pyserini 或本地 retriever、W&B/SwanLab/本地日志、pytest，按真实环境删改

**核心成果 / 简历 Bullets**：

- 主导构建 Search-Agent RL 轨迹合约，将训练输出组织为 front-loaded `<plan>`、后续 `<think>/<search>/<information>/<answer>` 的 plan-once 序列；支持 plan-only 首阶段接收合法 planner 但不触发检索，并严格区分模型 action `<search>` 与环境 observation `<information>`，同步 rollout parser、reward parser、response masking、trainer metrics 与 trajectory dump，降低奖励解析和真实执行路径不一致的风险。
- 设计双轨 path reward：Track A self-consistency 比较模型自声明 planner steps 与实际 `<search>`，度量计划执行度和 action efficiency；Track B reference-alignment 比较实际 `<search>` 与外部 `reference_steps`，度量关键搜索步骤覆盖率；两条信号保持解耦，后续通过 `R_path=max(S_self,S_ref)` 聚合，并配套 `intent_lexical`、`max_plan_steps`、`require_search_for_format` 等约束抑制 no-search shortcut 和低质量工具调用。
- 建立工程诊断与实验闭环，输出完整 trajectory dump、ground truth、data source、split、Track A/Track B 组件分和 invalid action reason，定位 `missing_plan`、`duplicate_plan`、`malformed_search_content`、no-search shortcut 等失败模式；构建 `reference_steps` 的候选生成、清洗、parquet 注入和 coverage check 流程，并用 `[N]` 组 short-run / 消融实验跟踪 `planner_valid_rate`、`valid_search ratio`、`invalid_action ratio`、`self_consistency`、`reference_alignment` 与 `[EM/F1/Score]`。

## 可量化指标占位清单

以下数值应从训练日志、validation 日志、trajectory dump 分析或消融表中替换，简历里只选择最能支撑项目贡献的 2-3 个：

- 实验范围：数据集数量 `[N]`、模型规模 `[B]`、训练步数 `[K]`、GPU 数量 `[N]`、总样本数 `[N]`。
- 最终效果：`EM/F1/Accuracy/Score = [x]`，相对 Search-R1 或 base GRPO 提升 `[+y points]`。
- 结构质量：`planner_valid_rate [a% -> b%]`、`valid_search ratio [a% -> b%]`、`invalid_action ratio [a% -> b%]`、no-search wrong-answer format reward 命中率 `[x% -> y%]`。
- Track A / Track B：`self_consistency mean [a -> b]`、`action efficiency [a -> b]`、`reference_available_ratio [x%]`、`reference_alignment mean [a -> b]`。
- 消融实验：baseline、planner-format、Track A observation、Track A reward、Track B reference、dual-track reward、matcher/weight variants 的对比表。

## 面试问题与回答

### 1. 你怎么理解 Search-P1？

Search-P1 面向的是 Tool-Use Agent RL 里的路径学习问题。Search-R1 已经把搜索动作放进 rollout，让模型学习何时搜索、搜什么、何时回答，但 reward 主要看最终答案和格式，失败样本通常缺少路径级学习信号。Search-P1 的关键是把搜索路径显式化：先要求模型写 front-loaded planner，再执行 tool call，并用 path-centric reward 判断路径是否合理。

我的设计拆成三层：轨迹结构层、路径评分层和实验诊断层。轨迹层保证 `<plan>/<think>/<search>/<information>/<answer>` 能被训练闭环稳定解析；评分层实现 Track A self-consistency 和 Track B reference-alignment；诊断层通过 trajectory dump、invalid action reason 和消融指标判断分数变化来自路径质量、格式改善、检索质量还是答案本身。

### 2. Search-P1 和 Search-R1 / 普通 RAG 的区别是什么？

普通 RAG 多数是在推理时检索上下文再回答，检索过程通常不是 RL 训练中的可学习 action。Search-R1 把搜索动作接入 RL rollout，让模型在生成过程中主动调用搜索工具。Search-P1 在 Search-R1 基础上进一步要求模型先给出可解析 planner，再执行搜索动作，使路径从自由文本变成可校验、可评分、可诊断的训练对象。

工程上，Search-P1 不只是换 prompt。它要求数据 prompt、rollout action parser、环境 observation 注入、reward parser、masking marker 和 trainer metrics 使用同一套标签。尤其要区分 `<search>` 是模型动作，`<information>` 是环境返回；如果两者边界混乱，reward 读到的就不是模型实际执行的路径。

### 3. 轨迹构造为什么要用 front-loaded planner 和 plan-once？

front-loaded planner 让模型在搜索前先声明搜索意图，Track A 才能比较“计划里说要查什么”和“实际 tool call 查了什么”。plan-once 则避免模型每轮重写 planner 来追逐已经拿到的检索结果，否则 self-consistency 会被事后修正的计划污染。

rollout 里保留 plan-only 首阶段：第一轮如果只有合法 `<plan>` 且还有后续 step，就接受为有效 planner，但不触发检索，也不把控制提示写进最终 trajectory。后续轮次必须输出 `<think>` 加 `<search>` 或 `<answer>`。这样既允许模型先完整规划，又保证 reward parser 看到的最终序列仍是干净的 `<plan>/<think>/<search>/<information>/<answer>` 合约。

### 4. Track A 和 Track B 分别解决什么问题？

Track A 是 self-consistency，参照物是模型自己的 planner。它问的是：模型是否按自己声明的计划执行搜索。公式可以概括为：

```text
S_self = r_planner * (n_exec_self / n_plan) * (n_exec_self / n_actions)
```

Track B 是 reference-alignment，参照物是外部 `reference_steps`。它问的是：模型实际搜索动作是否覆盖了解题所需参考步骤。公式可以概括为：

```text
S_ref = (n_covered / |R_ref|) * (n_covered / n_actions)
```

两者必须解耦。Track A 不依赖外部参考，适合在线诊断模型是否遵守自己的 planner；Track B 依赖离线构建的 reference_steps，适合约束“自洽但低质量”的 planner。最终路径奖励可以考虑 `R_path=max(S_self,S_ref)`，但权重和接入时机需要通过消融验证。

### 5. reward 公式怎么设计？为什么不能直接加大路径奖励？

完整奖励可以写成：

```text
R_total = lambda_p * R_path + lambda_a * R_outcome + lambda_f * R_format
R_path = max(S_self, S_ref)
```

我会先让路径组件可观测，再小权重接入。原因是 Track A 高并不必然代表答案正确，模型可能学到“一步短 plan + 一个泛化 query”的 reward hacking。路径奖励如果太强，会压过 outcome reward，让模型优化格式和自洽而不是答案质量。

更稳的做法是先记录 `base_score`、`self_consistency`、`has_search`、`effective_structure_format`、`reference_alignment` 等组件，确认 invalid action 和 no-search shortcut 被压住，再通过 `[0.01, 0.03, 0.05, ...]` 这类权重消融决定是否接入 scalar reward。

### 6. 训练闭环是怎么跑起来的？

数据侧先构造要求模型输出 `<plan>` 的 prompt。rollout 阶段模型先生成 planner，再生成 `<think>` 和 `<search>`；环境读取 tool call 内容请求 retriever，把真实检索结果包装为 `<information>` 注入下一轮上下文。最后模型输出 `<answer>`，reward parser 从完整 trajectory 中解析 planner、actions、observations 和 answer。

关键状态是 `planner_seen`。如果第一轮只输出合法 planner 且还有后续 step，系统接受为 plan-only 阶段，不触发搜索；后续轮次必须输出 reasoning 加 tool call 或 answer。invalid action 会被分桶统计，例如 `missing_plan`、`duplicate_plan`、`missing_or_invalid_plan_steps`、`malformed_search_content`，用于定位训练失败来源。

### 7. 你如何设计实验？

我会分阶段推进。第一阶段是 smoke test：小数据、小步数、短 context，确认数据、retriever、rollout、reward、dump 都能跑通。第二阶段是组件验证：固定 baseline，分别打开 planner-format、Track A observation、Track A reward、Track B reference steps，确认每个组件的指标方向。第三阶段是全量实验：在 `[dataset]`、`[model]`、`[GPU]` 上跑足训练步数，与 Search-R1 或原始 GRPO baseline 对比 EM/F1/Score。

实验表至少包括 baseline、planner only、Track A no scalar、Track A small reward、Track B observation、dual-track reward，以及 matcher 和 reward weight 消融。每个结果都要配套 trajectory dump 抽样，否则只看最终 EM 很难判断改进来自路径质量、格式奖励还是检索质量。

### 8. 失败案例有哪些？你怎么定位？

常见失败分五类。第一类是 planner 失败，比如缺失、重复、非前置、step 不连续或超过 `max_plan_steps`。第二类是 action 失败，比如旧标签、缺 reasoning、JSON/function-call 风格 tool call、裸 `search` 或低信息 query。第三类是 no-search shortcut，模型写了 plan 但不搜索直接回答。第四类是路径自洽但答案错，说明 Track A 没有保证 retrieval 和 reasoning 质量。第五类是 reference_steps 数据质量差，Track B 覆盖率没有实际意义。

定位时先看 invalid reason 分布和 trajectory dump，再抽样低分、高 self-consistency 但错答、高 EM 但低路径分的样本。这样可以区分是 parser 太严、matcher 低估、retriever 返回差、planner 抽象、模型偏离 plan，还是 reward weight 导致 hacking。

### 9. 工程取舍是什么？

最重要的取舍是先保证跨层合约一致，再追求复杂 reward。Search-P1 的标签、masking 和 parser 如果不同步，reward 信号会失真。第二个取舍是先用 deterministic matcher，而不是在线 LLM judge。lexical / intent_lexical 可能低估语义等价动作，但成本低、可重复验证、可测试，适合在线 RL loop。

第三个取舍是把昂贵和不稳定的 LLM voting 放到离线 Track B 数据准备阶段。训练时只读取结构化 `reference_steps`，避免把外部 API 延迟、失败和随机性引入 reward-time。

### 10. 你如何证明项目有效？

我不会只用“代码能跑”证明项目有效。至少需要四类证据：第一，功能证据，数据处理、retriever、rollout、reward、trajectory dump 和训练脚本都能端到端运行；第二，结构证据，合法 planner、valid search、invalid action、self-consistency、reference coverage 等指标符合预期；第三，效果证据，最终 EM/F1/Score 相对 baseline 有 `[+y points]` 或至少不退化；第四，消融证据，关闭 Track A/Track B 或改变 matcher/weight 后，指标变化能解释。

简历里如果没有全量效果日志，就只写工程实现和诊断链路，不写论文级性能结论。如果有全量日志，才把 `[benchmark]`、`[score]`、`[+y points]` 放进成果 bullet。

## 后续全量实验可能出现的问题与解决方法

### 1. 环境与依赖

**问题**：CUDA、PyTorch、FlashAttention、veRL、Transformers、Ray 或检索库版本不匹配，表现为 import error、kernel 编译失败、训练启动后 worker 崩溃。

**解决方法**：固定 conda/pip lock 或记录 `pip freeze`；先跑最小 import 和单卡 smoke；把训练、retriever、数据处理的环境分开验证；遇到 CUDA 扩展问题先降级到已知可用组合，再考虑重新编译。

### 2. retriever 服务

**问题**：retriever endpoint 配错、服务未启动、topk 不一致、返回 schema 与环境解析不匹配，导致 `<information>` 为空或 rollout 大量超时。

**解决方法**：训练前用固定 query 调 `/retrieve` 做健康检查；记录 `url/topk/index/version`；对空结果和超时做日志分桶；确保环境只把真实检索结果包装为 `<information>`，不要把控制提示误当 observation。

### 3. 数据处理 / Arrow schema

**问题**：新增 `ground_truth.reference_steps` 时，HuggingFace Datasets / Arrow 自动推断出错，例如部分样本缺字段、空列表被推断为 `list<null>`、后续 `list<string>` cast 失败。

**解决方法**：每条样本都写入 `reference_steps`，未命中时写空列表；显式声明 `Features`；必要时 `remove_columns` 去掉原始列；数据处理后单独跑 schema check，确认 parquet 中 `ground_truth.reference_steps` 类型稳定。

### 4. trajectory dump

**问题**：dump 没有写出、只写了模型响应片段、缺少 `split` 或 `ground_truth`，导致离线分析无法还原 reward-time 解析。

**解决方法**：确认 `trajectory_dump_path` 和 `trajectory_dump_limit` 已启用；dump 行必须包含完整 `solution_str`、`ground_truth`、`data_source`、`split`；训练和验证 split 分开标记；写入时用 UTF-8 JSONL 并处理 numpy 标量和数组。

### 5. planner / 标签格式

**问题**：模型混用旧标签 `<think>/<search>/<information>`，重复输出 `<plan>`，planner 不前置，或 planner step 不是 `Step N: Search ...` 格式。

**解决方法**：同步更新数据 prompt、rollout parser、reward parser 和 masking marker；invalid feedback 明确指出旧标签非法；用 `planner_seen` enforce plan-once；把 `max_plan_steps` 与 rollout 最大搜索步数对齐。

### 6. invalid action

**问题**：模型生成空 action、缺 `<think>`、tool call 内含 JSON/function-call、URL、嵌套标签、`Search ...` 前缀或低信息 query，导致大量 `invalid_action`。

**解决方法**：把 invalid reason 分成稳定 bucket，例如 `missing_think`、`missing_action_tag`、`malformed_tool_tag`、`malformed_search_content`；按 bucket 调 prompt 和 parser feedback；避免 feedback 中包含完整可复制标签，降低模型照抄控制文本的概率。

### 7. no-search shortcut

**问题**：模型写合法 planner 后不调用搜索，直接输出 answer，却仍可能拿到结构格式分。

**解决方法**：训练脚本启用 `require_search_for_format=true`；wrong-answer 且无 `<search>` 时不给 structure / retrieval / final-format shaping；监控 `has_search`、`effective_structure_format` 和 no-search wrong-answer 样本；必要时在 prompt 中强调至少一次必要搜索。

### 8. Track A 分数上升但 EM 不升

**问题**：`self_consistency` 提升，最终 EM/F1 不变甚至下降。模型可能学到短 planner、泛化 query 或冗余但匹配的搜索路径。

**解决方法**：同时看 `base_score`、answer EM、query 质量、retriever 命中文档、action efficiency 和高 Track A 错答样本；降低 `self_consistency_weight` 或先只记录不加分；引入 Track B reference coverage、evidence quality 或 outcome-aware gate 做约束。

### 9. Track B reference_steps 生成 / 注入

**问题**：拒绝采样样本太少、reference_steps 过长或含标签、parquet 注入位置错误，训练时 `reference_available_ratio` 为 0。

**解决方法**：先从最终答案正确且包含合法 `<search>` 的轨迹生成候选；对 reference_steps 做长度、重复、URL、标签和空 step 清洗；统一写入 `reward_model.ground_truth.reference_steps`；训练前跑 coverage check，确认样本级 available ratio 和字段 schema。

### 10. LLM voting 失败

**问题**：外部 API 超时、返回非 JSON、生成步骤不稳定、多个候选投票冲突，导致 reference_steps 噪声大。

**解决方法**：把 LLM voting 放在离线阶段；请求和响应全量落盘；设置重试、超时和失败样本跳过；对结果做 deterministic 清洗；保留 consensus reference 作为 fallback；报告 voting 成功率 `[x%]` 和过滤率 `[y%]`。

### 11. reward shaping 过强导致 reward hacking

**问题**：模型优化格式、planner 自洽或 reference 覆盖，但答案质量不升，甚至学会固定模板和泛化搜索。

**解决方法**：路径奖励先观察后加权；从小权重开始消融；监控 `final_score` 与 `base_score` 的相关性；设置 no-search gate、query quality gate 和 action efficiency 惩罚；抽样检查高 reward 错答，必要时降低路径权重或引入 outcome-aware 条件。

### 12. 分布式训练显存 / 并发 / 日志

**问题**：Ray worker OOM、rollout 并发过高、retriever 被打满、日志写入竞争、trajectory dump 文件过大。

**解决方法**：先单机单卡或低并发 smoke；逐步增加 rollout batch、并发和 topk；限制 dump 行数或按 split/worker 分片；日志指标只记录低基数 bucket，避免把 query 文本写入 metric name；定期检查 GPU memory、retriever QPS 和 worker restart。

### 13. 消融实验与指标解释

**问题**：最终分数有波动，但无法判断是 planner、Track A、Track B、retriever、数据还是随机种子造成的。

**解决方法**：消融表必须逐项打开功能：baseline、planner-format、Track A observation、Track A reward、Track B reference、dual-track reward、matcher/weight variants；每组固定数据、模型、retriever、训练步数和 seed；报告均值和方差；同时展示 EM/F1、path metrics、invalid action、has_search 和 reference coverage。

### 14. 论文结果与本地结果不一致

**问题**：本地结果没有达到论文报告提升，或只在部分数据集有效。

**解决方法**：先确认 baseline 是否一致，包括模型、数据切分、retriever index、topk、训练步数、reward 权重、评测脚本；再用 trajectory dump 看失败类型。如果差异仍存在，简历中写本地真实结果和可解释原因，不把论文结果冒充本地成果。
