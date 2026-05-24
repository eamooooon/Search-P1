# Track B / Reference-Alignment 工程复盘日志

本文档记录 Track B 每版修改中的问题、根因、调整、验证和后续观察。

## 当前状态快照

- Track B scorer 已接入 reward components，但不改变 scalar reward。
- Track B 数据入口已支持从 JSONL 注入 `ground_truth.reference_steps`。
- 拒绝采样 + LLM voting 离线构建工具已放入 `search_p1.analysis`。
- P1 目录下已有 bash wrapper：
  - `scripts/nq_hotpotqa_p1/build_reference_steps.sh`
  - `scripts/nq_hotpotqa_p1/check_reference_steps.sh`
- 训练 smoke run 已支持通过 `reward_model.trajectory_dump_path` 落盘 trajectory JSONL。
- 下一步重点是先生成真实 trajectory JSONL，再用它构建 `reference_steps.jsonl`。

## 下一步决策依据

后续不要只看“脚本是否跑完”，要按下面这些日志信号决定下一步。

### 1. 拒绝采样日志

来源：

```bash
TRAJECTORY_DUMP_PATH=logs/my-trajectories.jsonl \
TRAJECTORY_DUMP_LIMIT=512 \
bash scripts/nq_hotpotqa_p1/train_grpo.sh

TRAJECTORY_JSONL=logs/my-trajectories.jsonl \
bash scripts/nq_hotpotqa_p1/build_reference_steps.sh
```

脚本会在 stdout 打印 JSON stats，重点看：

- `total_rows`：输入 trajectory 总数。
- `correct_rows`：最终答案 EM 正确的 trajectory 数。
- `groups`：成功按 question / id 聚合出的样本组数。
- `skipped_no_key`：没有可用 question / id，无法生成 reference 的行数。
- `skipped_correct_no_actions`：答案正确但没有合法 search action 的行数。
- `reference_rows`：最终生成 reference plan 的样本数。
- `llm_vote_rows`：读回的 LLM voting 结果数。

决策：

- `total_rows = 0`：先检查 trajectory dump 路径和格式。
- `correct_rows = 0`：不要跑 LLM voting，先检查答案抽取 / EM 判断 / rollout 质量。
- `skipped_no_key` 很高：先修 trajectory dump 的 `question`、`data_source`、`split`、`index` 字段。
- `skipped_correct_no_actions` 很高：先检查 `<tool_call>` 格式和 search action validator。
- `reference_rows = 0`：先降低 `MIN_SUCCESSFUL` / 检查 consensus 阈值，或者直接跑 LLM voting。
- `llm_vote_rows = 0` 且 `reference_plan_source` 多为 `consensus`：只能作为 smoke test，不作为正式 Track B reference。

### 2. LLM voting 运行日志

来源：

```bash
bash scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh
```

主要文件：

- `data/nq_hotpotqa_p1/reference_llm_voting.log`
- `data/nq_hotpotqa_p1/reference_vote_results.jsonl`
- `data/nq_hotpotqa_p1/reference_vote_failures.jsonl`

日志重点看：

- `progress processed=... success=... failures=... skipped=... rate=.../s`
- 最终 JSON summary 中的 `vote_requests`、`vote_rows`、`failures`、`skipped`
- failures JSONL 中的 `error` 类型

决策：

- `failures / vote_requests` 很高：先看 `reference_vote_failures.jsonl`，通常是 key/base_url/model、超时、返回非 JSON、没有合法 `reference_steps`。
- `skipped` 很高且 `vote_rows` 没增长：说明 resume 生效，若想重跑设置 `LLM_RESUME=0` 或换输出文件。
- `vote_rows` 正常增长：继续跑更大的 `LLM_LIMIT` 或全量。
- 返回经常 “no valid reference_steps”：先改 prompt / validator，不要急着进 data process。

### 3. reference 注入检查日志

来源：

```bash
REFERENCE_STEPS_FILE=data/nq_hotpotqa_p1/reference_steps.jsonl \
bash scripts/nq_hotpotqa_p1/data_process.sh

bash scripts/nq_hotpotqa_p1/check_reference_steps.sh
```

`check_reference_steps.sh` 会分别检查 train / test parquet，重点看：

- `rows`
- `reference_available`
- `reference_available_ratio`
- `mean_reference_steps`
- `examples`

决策：

- `reference_available_ratio = 0`：不要跑训练，说明 reference 没注入进去；检查 JSONL key 是否和 parquet 的 `data_source/split/index/question` 对得上。
- `reference_available_ratio` 很低：可以跑小样本 smoke，但不能解释正式 Track B 指标。
- `mean_reference_steps = 0`：reference 清洗把 steps 全过滤了，检查 step 长度、tag、URL、空字符串。
- `examples` 正常展示 steps：可以进入 10-step training smoke test。

### 4. 训练 / 验证日志

来源：GRPO 训练 stdout / wandb / swanlab 中的 reward 和 env metrics。

Track B 必看字段：

- `val/reward/reference_alignment/mean`
- `val/reward/reference_alignment/max`
- `val/reward/ref_available/mean`
- `val/reward/ref_n_steps/mean`
- `val/reward/ref_n_actions/mean`
- `val/reward/ref_n_covered/mean`
- `val/reward/path_bonus/mean`
- `val/reward/final_score/mean`

格式和环境必看字段：

- `val/env/invalid_action/ratio`
- `val/env/action_reason/valid_search/ratio`
- `val/env/action_reason/valid_plan/ratio`
- `val/env/action_reason/missing_action_tag/ratio`
- `val/env/action_reason/duplicate_plan/ratio`

决策：

- `ref_available/mean = 0`：问题在数据注入，不在 scorer。
- `ref_available/mean > 0` 但 `reference_alignment/mean = 0`：检查 matcher、search query 质量、reference 抽象层级。
- `ref_n_actions/mean = 0`：模型没有发合法 search，先修 trajectory/action 格式。
- `ref_n_covered/mean = 0` 但 `valid_search/ratio` 正常：优先人工抽样检查 reference 与 action 是否语义层级不一致。
- `path_bonus/mean = 0` 且 `final_score = base_score`：符合当前 Track B observation 设计；Track B 尚未进入 reward。
- `invalid_action/ratio` 很高：先修 Planner/Search/Answer 格式，不要用当前 run 判断 Track B 好坏。

### 5. 下一步状态机

按顺序推进：

1. 没有 trajectory JSONL：先跑能落盘 trajectory 的 rollout / smoke。
2. 有 trajectory，但 `correct_rows = 0`：先修答案抽取或提升 rollout 质量。
3. 有正确 trajectory，但 `reference_rows = 0`：先修 action extraction / consensus / LLM voting。
4. 有 `reference_steps.jsonl`，但 `reference_available_ratio = 0`：先修 data_process 注入 key。
5. `reference_available_ratio > 0`，但 `reference_alignment = 0`：先做样本级 matcher 诊断。
6. `reference_alignment` 有非零分布，且 invalid action 不高：再讨论是否进入 `R_path = max(S_self, S_ref)`。

## 2026-05-20 - Track B 初始设计

- 现象：
  - 需要实现 Track B，但不能和 Track A 的 planner self-consistency 耦合。
- 根因：
  - Track A 的参照物是模型自己的 `<plan>`。
  - Track B 的参照物是外部 `reference_steps`。
- 调整：
  - 明确 Track B scorer 不读取 `<plan>`。
  - 明确第一版只做 observation，不改 `final_score`。
- 验证：
  - 设计文档记录了 `S_ref` 公式、数据契约和边界。
- 后续观察：
  - 双轨组合必须单独设计，不要塞进 Track B scorer。

## 2026-05-20 - Track B scorer 旁路观测接入

- 现象：
  - reward 代码原先只有 `self_consistency` / `path_bonus`。
- 根因：
  - 缺少 reference-alignment components。
- 调整：
  - 在 `qa_em_format.py` 中新增 reference step 提取、校验、匹配和 `compute_reference_alignment_components`。
  - 在 trainer metrics 中透出 `reference_alignment`、`ref_available`、`ref_n_steps`、`ref_n_actions`、`ref_n_covered`。
  - 新增 `reward_model.max_reference_steps`。
- 验证：
  - 手动 runner 调用 Track B 测试断言通过。
  - `py_compile` 通过。
  - `git diff --check` 通过。
- 后续观察：
  - 当前 `path_bonus` 仍是既有 Track A 口径，Track B 没有进入 reward composition。

## 2026-05-20 - GRPO 启动脚本补充 Track B 配置

- 现象：
  - `train_grpo.sh` 没有显式传入 Track B 观测配置。
- 根因：
  - 只改默认 config 不够，项目实际入口是 P1 bash 脚本。
- 调整：
  - 实验名加 `trackb-observe-10steps`。
  - 设置 `reward_model.path_reward_weight=0`。
  - 设置 `reward_model.path_match_strategy=lexical`。
  - 设置 `reward_model.max_reference_steps=4`。
- 验证：
  - `git diff --check` 通过。
- 后续观察：
  - 该脚本用于 smoke test，不是完整训练配置。

## 2026-05-20 - Track B 诊断 run 收敛到 10 step

- 现象：
  - 原脚本会跑完整 epoch，成本太高。
- 根因：
  - Track B 第一版只需要验证 metrics wiring，不需要训练收敛。
- 调整：
  - `trainer.total_training_steps=10`。
  - `data.train_data_num=3840`。
  - `data.val_data_num=256`。
  - `trainer.val_before_train=false`。
  - `trainer.save_freq=999999`、`trainer.test_freq=999999`。
- 验证：
  - `git diff --check` 通过。
- 后续观察：
  - 如果 `ref_available=0`，不要延长训练，先补 reference 数据。

## 2026-05-20 - reference_steps 数据注入入口

- 现象：
  - val 输出 `ref_available=0`、`ref_n_steps=0`。
- 根因：
  - parquet 里只有 `ground_truth.target`，没有 `ground_truth.reference_steps`。
- 调整：
  - 新增 `scripts/data_process/reference_steps.py`。
  - `qa_search_train_merge.py` / `qa_search_test_merge.py` 支持 `--reference_steps_file`。
  - `data_process.sh` 支持 `REFERENCE_STEPS_FILE` 和 `MAX_REFERENCE_STEPS`。
  - 新增 reference 覆盖率检查能力。
- 验证：
  - 手动 runner 调用 reference data process 测试通过。
  - `py_compile` 通过。
  - `git diff --check` 通过。
- 后续观察：
  - 训练前必须先检查 `reference_available_ratio`。

## 2026-05-20 - 拒绝采样 + LLM voting 离线构建

- 现象：
  - 已有数据注入入口，但还没有从 rollout trajectory 生成 `reference_steps.jsonl` 的工具。
- 根因：
  - Reference plan 应来自离线流程，不应在 reward-time 调 LLM。
- 调整：
  - 新增 `search_p1.analysis.build_reference_steps`。
  - 用 EM 做拒绝采样，只保留答案正确且包含合法 `<tool_call>` 的轨迹。
  - 导出 provider-agnostic LLM voting request JSONL。
  - 支持读回 LLM voting result JSONL。
  - 没有 LLM result 时，用保守 consensus fallback 生成 smoke-test reference plan。
- 验证：
  - 手动 runner 调用 `tests/test_reference_building.py` 通过。
  - `py_compile` 通过。
  - `git diff --check` 通过。
- 后续观察：
  - consensus fallback 只适合端到端 smoke test，正式实验优先用 LLM voting 输出。

## 2026-05-20 - P1 bash 入口补齐

- 现象：
  - Python 工具已有，但 P1 实验目录下没有对应 bash 入口。
- 根因：
  - 工程实际 workflow 通过 `scripts/nq_hotpotqa_p1/*.sh` 启动。
- 调整：
  - 新增 `scripts/nq_hotpotqa_p1/build_reference_steps.sh`。
  - 新增 `scripts/nq_hotpotqa_p1/check_reference_steps.sh`。
  - README 改为统一使用 bash wrapper。
- 验证：
  - `bash -n scripts/nq_hotpotqa_p1/build_reference_steps.sh` 通过。
  - `bash -n scripts/nq_hotpotqa_p1/check_reference_steps.sh` 通过。
  - `bash -n scripts/nq_hotpotqa_p1/data_process.sh` 通过。
  - `git diff --check` 通过。
- 后续观察：
  - P1 wrapper 只编排文件输入输出，不把具体 LLM provider 写死进训练入口。

## 2026-05-20 - analysis Python 位置修正

- 现象：
  - `build_reference_steps.py` 和 `check_reference_steps.py` 最初放在 scripts 下的分析目录，不符合当前工程组织。
- 根因：
  - `scripts` 应主要放实验入口；Search-P1 相关 Python 逻辑应在 `search_p1` 包下。
- 调整：
  - 将两个 Python 模块迁移到 `search_p1/analysis/`。
  - bash wrapper 改为 `python -m search_p1.analysis.*`。
  - 测试路径和文档同步更新。
- 验证：
  - 通过显式文件列表 `python -m py_compile`。
  - 手动 runner 调用 `tests/test_reference_building.py`、`tests/test_reference_steps_data_process.py`、`tests/test_track_b_reference_alignment.py` 通过。
  - 通过 `bash -n scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 后续新增 Track B 分析 Python 代码优先放在 `search_p1/analysis`。

## 2026-05-20 - OpenAI-compatible LLM voting 入口

- 现象：
  - 已经能生成 `reference_vote_requests.jsonl`，但还没有仓库内的 LLM voting 调用入口。
  - `api-key`、`base_url`、`model` 没有明确放置位置。
- 根因：
  - 之前只做了 provider-agnostic 文件接口，还没补具体 API 调用层。
- 调整：
  - 新增 `search_p1.analysis.run_reference_llm_voting`。
  - 新增 `scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh`。
  - 使用环境变量传入敏感和可变配置：
    - `LLM_API_KEY`
    - `LLM_BASE_URL`
    - `LLM_MODEL`
    - `LLM_LIMIT`
  - 不把 key 写进代码或配置文件。
- 验证：
  - 通过 `python -m py_compile search_p1/analysis/run_reference_llm_voting.py search_p1/analysis/build_reference_steps.py search_p1/analysis/check_reference_steps.py`。
  - 通过 `bash -n scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh`。
  - 通过 `bash -n scripts/nq_hotpotqa_p1/build_reference_steps.sh`。
  - 通过 `bash -n scripts/nq_hotpotqa_p1/check_reference_steps.sh`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 第一次建议用 `LLM_LIMIT=100` 或 `200` 做 smoke test，确认 JSON 输出稳定后再全量跑。

## 2026-05-20 - analysis 模块职责拆分

- 现象：
  - `build_reference_steps.py` 同时承担拒绝采样、voting request、voting result 合并和 consensus fallback，CLI 文件过重。
  - `run_reference_llm_voting.py` 里也混着 API 调用、响应解析、step 校验和 CLI 参数。
- 根因：
  - 第一版优先把链路打通，导致实现集中在少数脚本里，读起来像一锅粥。
- 调整：
  - 新增 `reference_io.py`：统一 JSONL 读写、question/key/metadata 提取。
  - 新增 `reward_format.py`：隔离对 `qa_em_format.py` 的按路径加载。
  - 新增 `reference_sampling.py`：只负责拒绝采样和正确轨迹分组。
  - 新增 `reference_voting.py`：只负责 consensus、vote request、LLM vote 读回和 reference rows 生成。
  - 新增 `reference_llm.py`：只负责 OpenAI-compatible API 调用、JSON 响应解析和 LLM 输出 step 校验。
  - `build_reference_steps.py` 和 `run_reference_llm_voting.py` 变成薄 CLI 编排层。
  - 测试改为 `tests/test_reference_building.py`，直接测拆出来的模块。
- 验证：
  - 通过显式文件列表 `python -m py_compile`。
  - 手动 runner 调用 `tests/test_reference_building.py`、`tests/test_reference_steps_data_process.py`、`tests/test_track_b_reference_alignment.py` 通过。
  - `bash -n` 已覆盖 P1 wrapper。
  - `git diff --check` 通过。
- 后续观察：
  - 后续如果接入具体 LLM provider SDK，也应放到新模块，不让 CLI 再变厚。

## 2026-05-20 - LLM voting 可观测性补强

- 现象：
  - `run_reference_llm_voting` 只有最终 summary，长跑时看不到进度、失败样本和中间结果。
  - bash 脚本里一度有 `LLM_API_KEY=...` 这类占位，容易覆盖外部环境变量，也容易诱导把真 key 写进脚本。
- 根因：
  - 第一版只验证 API 调用闭环，缺少长任务运行时的观测和恢复机制。
- 调整：
  - voting 成功一条就 append 到 `reference_vote_results.jsonl`，不等进程结束。
  - 失败写入 `reference_vote_failures.jsonl`。
  - 每 `LLM_PROGRESS_EVERY` 条打印一次进度。
  - 默认 resume：跳过结果文件里已经存在的 `custom_id`。
  - bash 脚本支持 `LLM_ENV_FILE`，但不再内置 key/base_url/model 占位。
  - bash 脚本默认写 `reference_llm_voting.log`。
- 验证：
  - 待运行测试和语法检查。
- 后续观察：
  - 第一次跑 100-200 条时，重点看 log 里的 failure rate 和 failures JSONL 的错误类型。

## 2026-05-23 - LLM 配置改为自动读取 .env

- 现象：
  - 运行 LLM voting 时需要 `api-key`、`base_url`、`model`，但每次命令行手动传容易漏，也不适合写进脚本。
- 根因：
  - 这些配置是运行时 secret / endpoint，不应该进 YAML 或 git tracked 文件。
- 调整：
  - `run_reference_llm_voting.sh` 默认先读取仓库根目录 `.env.llm`，找不到再读取 `.env`。
  - 仍支持 `LLM_ENV_FILE=...` 显式指定配置文件。
  - 使用 `set -a` source env 文件，支持 `LLM_API_KEY=...` 和 `export LLM_API_KEY=...` 两种写法。
  - env 文件先于脚本默认值加载，确保 `.env.llm` / `.env` 可以覆盖 `LLM_LIMIT`、`LLM_LOG`、输出路径等运行参数。
  - `.gitignore` 新增 `.env` / `.env.*`，保留 `!.env.example`。
  - 新增 `.env.example`，只记录占位配置，不包含真实 key。
- 验证：
  - 通过 `bash -n scripts/nq_hotpotqa_p1/run_reference_llm_voting.sh`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 不要把真实 key 写入 README、脚本或 tracked config。

## 2026-05-23 - 补充下一步日志决策依据

- 现象：
  - 已有每版修改日志，但缺少“后续根据哪些运行日志决定下一步”的操作层判断。
  - 只写“后续观察”不够，下一轮容易不知道该先看 LLM voting、data injection、还是训练 metrics。
- 根因：
  - Track B 链路跨越 trajectory dump、拒绝采样、LLM voting、reference 注入、训练验证多个阶段；每个阶段的失败信号不同。
- 调整：
  - 在本文档顶部新增“下一步决策依据”。
  - 明确 4 类日志来源：
    - `build_reference_steps.sh` stdout stats。
    - `reference_llm_voting.log` / `reference_vote_results.jsonl` / `reference_vote_failures.jsonl`。
    - `check_reference_steps.sh` 输出。
    - 训练 / 验证 reward 与 env metrics。
  - 补充状态机：从 trajectory 是否存在，到 `reference_alignment` 是否有非零分布，再到是否能讨论双轨 reward。
- 验证：
  - 文档已覆盖当前可观测字段：`correct_rows`、`reference_rows`、`vote_rows`、`failures`、`reference_available_ratio`、`ref_available/mean`、`reference_alignment/mean`、`invalid_action/ratio`。
- 后续观察：
  - 如果后续脚本新增字段或改名，必须同步更新“下一步决策依据”，否则日志会再次变成只能复盘、不能指导下一步。

## 2026-05-24 - 明确两次 build_reference_steps 的覆盖语义

- 现象：
  - `build_reference_steps.sh` 在第一次生成 voting request、第二次读回 LLM votes 时使用同一个脚本，容易误解为重复构建或无意识覆盖。
  - 原脚本第二次带 `LLM_VOTES_FILE` 时仍会默认重写 `reference_vote_requests.jsonl`。
- 根因：
  - 两次 build 的输入不同：第一次只有 `TRAJECTORY_JSONL`，第二次是 `TRAJECTORY_JSONL + LLM_VOTES_FILE`。
  - 第二次的核心目标是重写最终 `reference_steps.jsonl`，不是重新生成 voting request。
- 调整：
  - `build_reference_steps.sh` 新增 `WRITE_VOTE_REQUESTS` 控制。
  - 未设置 `LLM_VOTES_FILE` 时，默认 `WRITE_VOTE_REQUESTS=1`，生成 `reference_vote_requests.jsonl`。
  - 设置 `LLM_VOTES_FILE` 时，默认 `WRITE_VOTE_REQUESTS=0`，只根据 LLM votes 重建 `reference_steps.jsonl`。
  - README 补充两次 build 的输入、输出和覆盖语义。
- 验证：
  - 通过 `bash -n scripts/nq_hotpotqa_p1/build_reference_steps.sh`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 如果后续希望完全避免覆盖 fallback reference，可把第一次输出设为 `REFERENCE_STEPS_OUTPUT=.../reference_steps_consensus.jsonl`，第二次输出设为 `.../reference_steps.jsonl`。

## 2026-05-24 - 补齐 trajectory dump 作为 reference 构建输入

- 现象：
  - 直接运行 `TRAJECTORY_JSONL=logs/trajectories.jsonl bash scripts/nq_hotpotqa_p1/build_reference_steps.sh` 会报 `FileNotFoundError`。
  - 该路径只是示例，当前分支此前没有自动生成 trajectory JSONL。
- 根因：
  - `build_reference_steps.sh` 的输入必须是真实 rollout trajectory。
  - Track B 离线 reference 构建依赖 `solution_str`、`ground_truth`、`data_source`、`split`、`index` 等字段，但训练脚本没有先把这些字段落盘。
- 调整：
  - 新增 `verl/trainer/trajectory_dump.py`，以 JSONL 形式写出 trajectory rows。
  - `RewardManager` 支持 `reward_model.trajectory_dump_path` 和 `reward_model.trajectory_dump_limit`。
  - `train_grpo.sh` 默认写 `logs/$EXPERIMENT_NAME-trajectories.jsonl`，默认最多写 512 条。
  - `build_reference_steps.sh` 增加输入文件存在性检查，缺文件时给出先跑 trajectory-producing smoke run 的提示。
  - README 和设计文档改为先生成真实 trajectory dump，再运行 reference build。
- 验证：
  - 通过 `python -m py_compile verl/trainer/trajectory_dump.py verl/trainer/main_ppo_format.py search_p1/analysis/build_reference_steps.py search_p1/analysis/reference_sampling.py search_p1/analysis/reference_io.py`。
  - 通过 `bash -n scripts/nq_hotpotqa_p1/train_grpo.sh`。
  - 通过 `bash -n scripts/nq_hotpotqa_p1/build_reference_steps.sh`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 第一轮 dump 后先检查 `build_reference_steps.sh` 输出的 `total_rows`、`correct_rows`、`skipped_no_key` 和 `skipped_correct_no_actions`，再决定是否进入 LLM voting。

## 2026-05-24 - 修复 trajectory dump 多答案数组序列化

- 现象：
  - 训练进入 reward dump 时抛出 `ValueError: can only convert an array of size 1 to a Python scalar`。
  - 栈在 `trajectory_dump._to_jsonable()`，处理 `ground_truth.target` 时触发。
- 根因：
  - QA 数据的 `ground_truth.target` 可能是多答案数组。
  - 原实现看到对象有 `.item()` 就直接转标量，只适用于单元素 array / tensor，不适用于多元素答案列表。
- 调整：
  - `_to_jsonable()` 优先处理 tensor 的 `detach().cpu()`。
  - 对有 `tolist()` 的对象先转 list，再递归 JSON 化。
  - `.item()` 只作为单标量 fallback，遇到 `ValueError` 不再中断。
  - 新增测试覆盖多答案 array-like 对象落盘。
- 验证：
  - 通过 `python -m py_compile verl/trainer/trajectory_dump.py tests/test_reference_building.py`。
  - 通过手动调用 `test_trajectory_dump_serializes_multi_answer_arrays`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 如果后续 dump 中出现不可 JSON 序列化的自定义对象，再统一在 `_to_jsonable()` 增加保守 fallback，而不是在 reward loop 里特殊处理。

## 2026-05-24 - 新增 LLM voting 连通性测试

- 现象：
  - `run_reference_llm_voting.sh` 能调用到模型，但失败信息是 `LLM response has no valid reference_steps`。
  - 仅看批量 voting 日志，无法区分是 API 不通、模型不支持 JSON mode、返回字段名不对，还是 prompt 约束不够。
- 根因：
  - 批量 voting 脚本为了长跑恢复和失败记录，只输出失败原因，不输出每次原始模型响应。
  - 调试 provider / model 时需要一个单样本、可观察、低成本的连通性测试。
- 调整：
  - 新增 `search_p1.analysis.test_reference_llm_connection`。
  - 新增 `scripts/nq_hotpotqa_p1/test_reference_llm_connection.sh`，复用 `.env.llm` / `.env` 配置加载逻辑。
  - 测试脚本打印 `base_url`、`model`、raw response、parsed JSON、validated `reference_steps` 和最终状态。
  - README 补充在全量 LLM voting 前先跑连通性测试。
- 验证：
  - 通过 `python -m py_compile search_p1/analysis/test_reference_llm_connection.py search_p1/analysis/reference_llm.py`。
  - 通过 `bash -n scripts/nq_hotpotqa_p1/test_reference_llm_connection.sh`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 如果连通测试 `status=OK`，再跑 `run_reference_llm_voting.sh`。
  - 如果 `CONNECTED_BUT_NO_VALID_REFERENCE_STEPS`，优先查看 raw response 决定是改 prompt、改 parser，还是换模型。

## 2026-05-24 - 支持复现真实 LLM voting 失败样本

- 现象：
  - 单样本连通测试返回 `status=OK`，但批量 `run_reference_llm_voting.sh` 仍出现 `LLM response has no valid reference_steps`。
  - 这说明 provider/model 连通没问题，失败来自某些真实 vote request 的内容或模型对该请求的返回格式。
- 根因：
  - 原连通测试只使用固定 toy prompt，不能代表真实 `reference_vote_requests.jsonl` 中的复杂候选 action。
  - 批量失败记录只写 error，没有保存 raw response 和 parsed response，无法判断 validator 为什么拒绝。
- 调整：
  - `run_llm_voting()` 在失败记录中写入 `raw_content` 和 `parsed_response`。
  - `test_reference_llm_connection.py` 支持 `--vote_requests`、`--custom_id`、`--request_index`，可直接复现真实失败样本。
  - bash wrapper 改为透传 CLI 参数。
  - README 增加按 `custom_id` replay 真实 voting request 的命令。
- 验证：
  - 通过 `python -m py_compile search_p1/analysis/test_reference_llm_connection.py search_p1/analysis/reference_llm.py`。
  - 通过 `bash -n scripts/nq_hotpotqa_p1/test_reference_llm_connection.sh`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 对失败样本先运行 replay 命令，看 raw response；如果只是字段名或嵌套结构不同，优先增强 parser；如果内容质量差，再调整 prompt。

## 2026-05-24 - 修复 vote request 中 question 为空

- 现象：
  - 单样本连通测试正常，但 replay 真实 `reference_vote_requests.jsonl` 时，失败样本 metadata 显示 `"question": ""`。
  - 模型对真实样本返回 `{"reference_steps": []}`，因此 validator 报 `CONNECTED_BUT_NO_VALID_REFERENCE_STEPS`。
- 根因：
  - LLM voting 的真实请求依赖 `reference_io.row_question()` 提供 question。
  - trajectory dump 中可能没有可直接读取的 `prompt` 字段，或 prompt 结构不是普通 list[dict]。
  - 原逻辑只从 `row.question` 或 `prompt[-1].content` 中找 `Question:`，没有从完整 `solution_str` 兜底解析。
- 调整：
  - `row_question()` 新增从 `solution_str` 兜底解析最后一个 `Question:` 的逻辑。
  - 解析时会在 `<|im_end|>`、assistant marker、`<plan>`、`<reasoning>`、`<tool_call>`、`<answer>` 前截断，避免把后续轨迹内容拼进 question。
  - 新增测试覆盖从 `solution_str` 提取 question。
- 验证：
  - 通过 `python -m py_compile search_p1/analysis/reference_io.py tests/test_reference_building.py`。
  - 通过手动调用 `test_row_question_falls_back_to_solution_str_question_marker`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 需要重新运行 `build_reference_steps.sh` 生成新的 `reference_vote_requests.jsonl`，旧文件里的空 question 不会自动修复。

## 2026-05-24 - 修复 reference_steps 注入时 Arrow schema 不一致

- 现象：
  - 运行 `REFERENCE_STEPS_FILE=... bash scripts/nq_hotpotqa_p1/data_process.sh` 时，`datasets.map` 在约 10% 处报错：
    - `Couldn't cast array of type struct<reference_steps: list<item: string>, target: list<item: string>> to {'target': List(Value('string'))}`
- 根因：
  - 前面的样本没有命中 reference，只返回 `ground_truth.target`。
  - 后面某个样本命中 reference 后，`ground_truth` 新增 `reference_steps` 字段。
  - HuggingFace Datasets / Arrow 已经根据前面样本推断出 schema 为 `{'target': List(string)}`，中途新增字段会导致 cast 失败。
- 调整：
  - `qa_search_train_merge.py` 和 `qa_search_test_merge.py` 每条样本都写 `ground_truth.reference_steps`。
  - 未命中 reference 时写空列表 `[]`，命中时写具体 steps。
  - 这样 `ground_truth` schema 从第一条样本开始就是稳定的。
- 验证：
  - 通过 `python -m py_compile scripts/data_process/qa_search_train_merge.py scripts/data_process/qa_search_test_merge.py scripts/data_process/reference_steps.py`。
  - 通过 `git diff --check`，仅有 Git 换行符提示，无 whitespace error。
- 后续观察：
  - 重新运行 data process 后，继续用 `check_reference_steps.sh` 看 `reference_available_ratio` 和 examples。
