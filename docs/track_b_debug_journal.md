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

## 2026-05-20 - Track B scorer 旁路观测接入

- 现象：
  - 需要根据设计文档开始落地 Track B，但当前 reward 代码只有 `self_consistency` 和 `path_bonus` 相关组件。
  - Track B 第一版必须能在没有合法 planner 的情况下计算 `S_ref`，并且不能改变 scalar reward。
- 根因：
  - Track B 的参照物是 `ground_truth.reference_steps`，不是模型 `<plan>`。
  - 如果直接复用 Track A 的 `compute_self_consistency_score`，会把 planner 合法性误作为 reference-alignment 的前置条件。
- 调整：
  - 在 `qa_em_format.py` 中新增 `extract_reference_steps`、`validate_reference_steps`、`reference_step_matches_action`、`count_reference_covered_steps` 和 `compute_reference_alignment_components`。
  - `compute_score_components` 新增返回 `reference_alignment`、`ref_available`、`ref_n_steps`、`ref_n_actions`、`ref_n_covered`。
  - `main_ppo_format.py` 和 `ray_trainer.py` 透出 Track B metrics。
  - 新增 `reward_model.max_reference_steps` 配置项，默认 `null`，保持兼容。
  - 新增最小测试覆盖无 planner、缺失 reference、重复 action、非法 reference、以及 `final_score` 不受 Track B 影响。
- 验证：
  - `python -m pytest tests/test_track_b_reference_alignment.py -q` 未运行成功：当前本机 Python 环境缺少 `pytest`。
  - 由于当前本机 Python 环境也缺少 `numpy`，测试文件改为按路径加载 `qa_em_format.py`，避免导入 `verl.__init__` 时被全仓库依赖阻塞。
  - 通过 `python -c "...调用 tests/test_track_b_reference_alignment.py 中所有 test_* 函数..."` 验证 Track B 断言。
  - 通过 `python -m py_compile verl/utils/reward_score/qa_em_format.py verl/trainer/main_ppo_format.py verl/trainer/ppo/ray_trainer.py tests/test_track_b_reference_alignment.py`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 当前 Track B matcher 使用保守 lexical 规则；后续需要用真实 dump 检查是否低估 reference coverage。
  - 当前 `path_bonus` 仍是既有 Track A 口径，Track B 没有进入 reward composition；后续如启用双轨，需要单独设计组合层。

## 2026-05-20 - GRPO 启动脚本补充 Track B 观测配置

- 现象：
  - Track B scorer 和 metrics 已接入，但 `scripts/nq_hotpotqa_p1/train_grpo.sh` 没有显式传入 Track B 相关配置。
  - 实验名仍是 `plan-format`，不容易从日志路径判断本次 run 是否包含 Track B observation。
- 根因：
  - 第一版代码只改了默认配置和 trainer 透传，遗漏了项目实际使用的 GRPO 启动脚本。
  - 虽然 `ppo_trainer.yaml` 中已有默认值，但训练脚本是实验可复现入口，应显式记录关键开关。
- 调整：
  - 将 `EXPERIMENT_NAME` 后缀改为 `trackb-observe`。
  - 在脚本中显式设置 `reward_model.path_reward_weight=0`，强调 Track B 第一版不改变 scalar reward。
  - 显式设置 `reward_model.path_match_strategy=lexical`。
  - 显式设置 `reward_model.max_reference_steps=4`，与 `max_turns=4` 对齐。
- 验证：
  - `bash -n scripts/nq_hotpotqa_p1/train_grpo.sh` 未运行成功：当前 Windows/WSL Bash 启动被 `E_ACCESSDENIED` 拦截。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 如果训练数据还没有 `ground_truth.reference_steps`，`reference_alignment` 会稳定为 `0.0` 且 `ref_available=0.0`；下一步需要实现 reference plan 数据生成或注入。

## 2026-05-20 - Track B 诊断 run 收敛到 10 step

- 现象：
  - `train_grpo.sh` 仍按 `trainer.total_epochs=1` 和完整 `data.train_data_num=null` 运行，实际会跑完整训练集。
  - `val_before_train=true` 会在训练前先跑验证，进一步拉长 Track B smoke test。
- 根因：
  - 第一版 Track B 只是验证 `reference_alignment` metrics 是否打通，不需要完整 epoch。
  - 当前训练数据很可能还没有 `ground_truth.reference_steps`，长跑也只会得到大量 `ref_available=0`，诊断价值有限。
- 调整：
  - 将实验名后缀改为 `trackb-observe-10steps`。
  - 设置 `trainer.total_training_steps=10`。
  - 设置 `data.train_data_num=3840`，按 `train_batch_size=384` 对齐 10 个训练 batch。
  - 设置 `data.val_data_num=256`，限制最终验证规模。
  - 设置 `trainer.val_before_train=false`，跳过训练前验证。
  - 设置 `trainer.save_freq=999999`、`trainer.test_freq=999999`，避免中途 checkpoint / test 干扰。
- 验证：
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 10-step run 只用于确认 Track B metrics wiring；如果 `ref_available` 全为 0，下一步优先补 reference plan 数据生成，而不是延长训练 step。

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
