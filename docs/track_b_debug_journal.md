# Track B / Reference-Alignment 工程复盘日志

本文档用于记录 Track B 相关更新的工程复盘。后续每次修改都追加一条记录，固定格式为：时间 / 现象 / 根因 / 调整 / 验证 / 后续观察。

重点写清楚：

- 遇到了什么问题。
- 为什么会发生。
- 本次怎么解决。
- 用什么方式验证。
- 后续还需要观察什么。

## 当前状态快照

- 已明确：
  - Track B 是 reference-alignment，不是 self-consistency。
  - Track B 第一版只做旁路观测，不改变 scalar reward。
  - Track B scorer 只读取 actions 和 `ground_truth.reference_steps`。
  - Track B 不读取模型 `<plan>`，也不依赖 Track A 的 `self_*` 字段。
  - `R_path = max(S_self, S_ref)` 属于后续组合层，不属于 Track B scorer 第一版。
- 待实现：
  - `reference_steps` 数据契约落地。
  - reference plan 离线生成脚本：拒绝采样 + LLM voting + validator。
  - `compute_reference_alignment_components`。
  - Track B analysis 脚本。
  - trajectory dump 的中立字段设计，避免写死 `track_a` 或 `track_b`。
- 风险：
  - 当前代码中已有 Track A / `path_bonus` 相关实现痕迹，后续实现 Track B 时需要先清理边界，避免耦合。
  - `reference_steps` 质量如果不稳定，`S_ref` 会变成噪声。
  - matcher 如果过宽，会奖励无意义 query stuffing；如果过窄，会低估有效搜索。

## 2026-05-20 - Track B 初始设计文档

- 现象：
  - 当前分支准备开始实现 Track B，但需要先明确它和 Track A 的边界。
  - Track A 文档中已经强调 self-consistency 只比较 planner 与 action，不读取 reference plan。
  - Track B 如果从 Track A 代码或最新实验状态直接延伸，容易把 `planner`、`self_consistency`、`path_bonus` 混入 reference-alignment。
- 根因：
  - 双轨路径评分有共同的 matcher 和 trajectory parser，但两个 track 的参照物不同。
  - Track A 的参照物是模型自己的 `<plan>`；Track B 的参照物是外部 `reference_steps`。
  - 如果不先写清楚接口边界，后续实现很容易为了复用代码而引入语义耦合。
- 调整：
  - 新增 `docs/track_b_reference_alignment_plan.md`，定义 Track B 的目标、公式、数据契约、离线 reference plan 生成流程、scorer 接口、metrics 和验收标准。
  - 新增本日志文件，要求后续每版修改都记录问题和解决方案。
  - 明确 Track B 第一版保持 `final_score = existing_score`，只记录 `reference_alignment` 及组件。
  - 明确 Track B scorer 不读取 `<plan>`，Track A scorer 不读取 `reference_steps`。
- 验证：
  - 文档层面已对齐 Track A 的设计风格：先做 observation，再决定是否进入 reward composition。
  - 文档层面已明确双轨聚合只能在后续组合层完成：`R_path = max(S_self, S_ref)`。
- 后续观察：
  - 实现前先检查当前分支中 Track A 相关代码残留，决定哪些 parser / matcher 可抽为中立工具。
  - 先实现离线 analysis 和 metrics，确认 `reference_steps` 与 matcher 质量，再考虑是否接入 reward。

## 后续追加模板

```md
## YYYY-MM-DD - 标题

- 现象：
  - 
- 根因：
  - 
- 调整：
  - 
- 验证：
  - 
- 后续观察：
  - 
```
