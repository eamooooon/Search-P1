# Track B / Reference-Alignment 工程复盘日志

本文档记录 Track B 的设计边界、数据链路和当前实现状态。

## 当前状态快照

- Track B 已接入 reward components，但不改变 scalar reward。
- Track B 数据入口已支持从 JSONL 注入 `ground_truth.reference_steps`。
- 拒绝采样 + LLM voting 的离线构建工具已放入 `search_p1.analysis`。
- P1 目录下已有 bash wrapper：
  - `scripts/nq_hotpotqa_p1/build_reference_steps.sh`
  - `scripts/nq_hotpotqa_p1/check_reference_steps.sh`
- 训练 smoke run 已支持通过 `reward_model.trajectory_dump_path` 落盘 trajectory JSONL。
- 当前版本只做 observation，不做 `R_path = max(S_self, S_ref)`。

## 设计边界

Track B 只读：

- `ground_truth.reference_steps`
- 模型实际 `<search>`

Track B 不读：

- `<plan>`
- Track A 的 planner 统计

Track A 也不读 `reference_steps`。

## 现在最重要的判断标准

### 数据链路

- `reference_available_ratio > 0` 才说明 reference 注入成功。
- `mean_reference_steps > 0` 才说明清洗没有把数据全过滤掉。
- `ref_n_actions > 0` 才说明模型真的发出了合法搜索动作。

### 训练链路

- `reference_alignment` 非零不代表 reward 已经改了，只代表观测可用了。
- `final_score` 仍然只由 base reward 和 Track A bonus 构成。
- `invalid_action/ratio` 太高时，不要拿 Track B 指标解释效果。

## 已验证内容

- `qa_em_format.py` 能独立计算 reference-alignment components。
- `qa_search_train_merge.py` / `qa_search_test_merge.py` 能按需注入 `reference_steps`。
- `check_reference_steps.sh` 能检查 parquet 中 reference 覆盖率。
- `build_reference_steps.sh` 能从 trajectory JSONL 生成 reference 候选。
- 新增的 Track B 单测通过。

## 现在的结论

Track B 的设计是成立的，但它目前仍是“观测层设计”，不是“训练目标设计”。

这意味着：

- 适合用来做诊断、复盘、样本分析。
- 适合做后续双轨 reward 的准备。
- 不适合现在就把它直接并入 reward，尤其是在 reference 质量和 matcher 还没做充分消融之前。
