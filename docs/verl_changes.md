# Search-P1 对 verl 的改动清单（面试用）

> 范围说明
> - **上游 verl** = volcengine/verl
> - **Search-R1 baseline** = 本 repo `main` 分支（已经在 verl 之上做了一层改造）
> - **Search-P1** = 本 repo `feature/plan-once`（= HEAD），即"我自己"做的改动
>
> 改动统计
> - `git diff 068516b..HEAD`（vs 仓库 initial commit）：全仓 **+25,658 / -1,012 行**，其中 `verl/` 子目录约 **+5,500 / -956 行**
> - `git diff main..HEAD`（vs Search-R1 baseline，= 我自己做的）：68 个文件，**+21,336 / -167 行**

文档结构：
1. 分层总览（上游 verl → Search-R1 → Search-P1）
2. Search-R1 layer 已经动过 verl 的地方（继承自 baseline）
3. Search-P1 自己对 verl 的改动（按文件逐个说清"做了什么、为什么、怎么实现"）
4. 面试问答 cheat sheet

---

## 1. 分层总览

```
upstream verl (volcengine)
     ↓  ─── Search-R1 改造（main 分支祖先里就有）
Search-R1 baseline (= main)
     ↓  ─── Search-P1 改造（feature/plan-once）
HEAD
```

Search-P1 对 verl 的整体定位可以一句话概括：

> **把 Search-R1 "能跑通"的 multi-turn search RL framework，升级为"能复现、能断点、能观测、能离线分析、能承载 path-level reward"的工程。**

具体动了 verl 的哪些子目录：

| 子目录 | Search-R1 已改 | Search-P1 我改 |
|---|---|---|
| `verl/trainer/main_ppo.py` | ✅ data_source 路由、ray init env vars | — |
| `verl/trainer/main_ppo_format.py` | （baseline 已加 212 行版本） | ✅ 扩到 441 行，加 plan/path-aware RewardManager |
| `verl/trainer/ppo/ray_trainer.py` | ✅ 嵌入 `LLMGenerationManager` + `info_mask` loss masking | ✅ resume + rollout-only + reward/action 指标聚合 (+491 / -156) |
| `verl/trainer/config/ppo_trainer.yaml` | ✅ max_turns / retriever / state_masking | ✅ reward manager 新参数 + seed + resume |
| `verl/trainer/trajectory_dump.py` | — | ✅ 全新 (+88) |
| `verl/utils/reward_score/qa_em.py` | ✅ extract_solution / EM | （微调） |
| `verl/utils/reward_score/qa_em_format.py` | — | ✅ 全新 (+741) |
| `verl/utils/dataset/sft_dataset.py` | — | ✅ 全新 (+171) |
| `verl/utils/tracking.py` | — | ✅ swanlab + finish (+31) |
| `verl/workers/fsdp_workers.py` | ✅ 多轮 rollout 接口适配 | ✅ rng/optim/lr 状态持久化 + load_checkpoint (+127) |

---

## 2. Search-R1 已经在 verl 上动过的部分（我必须能讲清楚，但不归功给自己）

### 2.1 `verl/trainer/ppo/ray_trainer.py` —— Search-R1 的核心改造

baseline 在 PPO 训练循环里嵌入了 `LLMGenerationManager`，把 verl 原本"一发一答"的 PPO rollout 改成 **multi-turn search rollout**：

1. 模型生成 `<search>query</search>`
2. rollout 截停，调 retriever，把检索结果拼成 `<information>...</information>` 注入回 prompt
3. 模型基于新 prompt 继续生成，直到出 `<answer>` 或达到 `max_turns`
4. 整条 trajectory 拿去算 reward 并更新策略

伴随两个关键 trick：

- **`info_mask`**：`apply_kl_penalty` 和 `_create_loss_mask` 都用 `info_mask` 代替 `attention_mask`，把环境注入的 `<information>` token 的 loss/KL 屏蔽掉。只对模型自生 token 反传梯度。
  - 数学意义：策略梯度只更新 `π(a_t | s_t)` 中 `a_t` 是 actor 自己输出的位置，避免让模型学"复述检索结果"。

- **state masking marker**：早期 baseline 用 `<information>...</information>` 作为 marker，由 `_create_loss_mask` 通过 `re.finditer` 在 decoded response 上反查 token position 来构造 mask（Search-P1 已经把这段拆掉，直接用 rollout 阶段产出的 `info_mask`，见 §3.4）。

### 2.2 `verl/trainer/main_ppo.py`

```python
def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa',
                       '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
```

按 dataset 名路由到 EM reward；`ray.init` 注入 SWANLAB / TOKENIZERS_PARALLELISM / NCCL_DEBUG 等环境变量。

### 2.3 `verl/utils/reward_score/qa_em.py`

`extract_solution` 只在 `<|im_start|>assistant` 之后找 `<answer>`，并去掉"My previous action is invalid... Let me try again."这类 retry 模板。EM / sub-EM 用 normalize（去冠词、去标点、lower）后做字符串相等。

### 2.4 `verl/trainer/config/ppo_trainer.yaml`

baseline 加了：
- `max_turns: 10`、`do_search: true`
- `retriever.url / retriever.topk`
- `algorithm.state_masking.{start_state_marker, end_state_marker}`
- `algorithm.no_think_rl`

### 2.5 `verl/workers/fsdp_workers.py`

baseline 让 actor rollout worker 支持 multi-turn search rollout 的接口（让 `LLMGenerationManager` 能通过 dispatch 调进来）。

---

## 3. Search-P1 自己对 verl 的改动（重头戏）

按"为什么要做 → 做了什么 → 怎么实现 → 关键细节"四段式。

---

### 3.1 `verl/trainer/main_ppo_format.py`（+441 行，从 baseline 212 行版本重写）

#### 为什么
baseline 的 RewardManager 只算"EM + 简单格式分"。Search-P1 要的是：
- 按"plan / search / answer"结构状态机校验
- 把 reward 拆成可观测的 components（base / structure_format / final_format / retrieval / Track A / Track B）
- 在不改 scalar reward 的前提下，把 path-level signal（self-consistency、reference alignment）以旁路 component 形式记录

#### 做了什么
1. **数据源扩展**：`_select_rm_score_fn` 增加 `web_questions / strategyqa`，统一走 `qa_em_format.compute_score_em`
2. **`RewardManager.__init__` 多了 11 个参数**：
   ```python
   structure_format_score, final_format_score, retrieval_score, format_score,
   path_match_strategy, require_search_for_format,
   max_plan_steps, max_reference_steps,
   self_consistency_weight, reference_alignment_weight,
   trajectory_dump_path, trajectory_dump_limit,
   trajectory_dump_full_solution, trajectory_dump_split
   ```
3. **`__call__` 流程**：
   - decode `prompt + response` → `sequences_str`
   - 抽 question 文本（`_extract_question` 从 `Question:` 之后到第一个 stop tag）
   - 调 `qa_em_format.compute_score_components` 算 base + Track A + Track B
   - **当前 `final_score = base_score`**，Track A/B 只记录、不影响 scalar reward
   - 把 score 写到 response 最后一个有效 token（非 `<information>` 的最后一个 token）
   - 把 `reward_components` 写进 `data.meta_info`
   - 如果开启 `trajectory_dump_path`，append JSONL
4. **`validate_path_match_strategy`**：启动时校验 `path_match_strategy ∈ {lexical, intent_lexical}`，未实现的策略（embedding / offline LLM）直接 raise

#### 关键细节（面试可被追问）
- **为什么 final_score 等于 base_score？**
  当前 Track A 的 matcher 是 deterministic lexical matching，自然语言 plan 步骤 vs 实际 query 的匹配有噪声；先把信号写出来观察分布，避免给一个没校准的 path score 直接驱动策略，落入 reward hacking。

- **score 为什么不写到 `attention_mask` 的最后一个 token？**
  最后一个有效 token 可能落在 `<information>` block 里，那是环境注入的，写在那里 reward 就附给了不该被更新的位置；要写到模型自生的"真正答案/结束"token。

---

### 3.2 `verl/utils/reward_score/qa_em_format.py`（+741 行，全新）

#### 为什么
要把 P1 trajectory 的"语义合约"以可测试模块沉淀下来：rollout parser 和 reward parser 共享同一套 tag 语义与 planner 约束，并且独立可单测。

#### 做了什么（核心 API）

| 函数 | 作用 |
|---|---|
| `normalize_answer` | EM 的标准化（lower / 去冠词 / 去标点 / 折叠空白） |
| `em_check` / `subem_check` | EM / 包含式 EM |
| `_extract_assistant_content` | 取 `<\|im_start\|>assistant` 之后的内容 |
| `extract_plan_steps` | 从 `<plan>...</plan>` 用 `_STEP_LINE_PATTERN` 抽 `Step N: Search ...` |
| `validate_planner_block` | 校验 plan 唯一、在最前、step 编号连续从 1 开始、无嵌套 tag |
| `extract_search_calls` | **先删 `<information>` block，再抽 `<search>`** |
| `count_actions` | 数有效 tool call |
| `is_valid_search_query` | 拒绝空 query / 嵌套 tag / URL-like / 超长 |
| `is_valid_sequence` | 状态机校验完整 `<plan>/<think>/<search>/<information>/<answer>` 顺序 |
| `compute_self_consistency_components` | Track A：`r_planner, n_plan, n_actions, n_exec` |
| `compute_self_consistency_score` | `S_self = r_planner * (n_exec/n_plan) * (n_exec/n_actions)` |
| `step_matches_action` / `count_covered_steps` | plan step ↔ search query 的 lexical / intent_lexical 匹配 |
| `compute_score_em` / `compute_score_components` | 入口，组合上述 |

#### 关键细节
- **`extract_search_calls` 为什么先移除 `<information>`？**
  检索结果里完全可能字面出现 `<search>`（比如知识库的某段文本就含这个字符串），不去掉会被误判为模型 action，污染 `n_actions` 和 Track A 分母 `n_exec/n_actions`。

- **`is_valid_sequence` 为什么用状态机不用 regex？**
  P1 trajectory 是嵌套 + 顺序约束的（plan 唯一在最前 → 接着只能 think → 然后 search 或 answer → search 后必须有 information → 终止只能在 answer）。regex 表达不出"顺序合法"这种语义，状态机能精确给出哪一步开始非法。

- **`_MATCH_STOPWORDS / _INTENT_STOPWORDS / _INTENT_PLACEHOLDER_PATTERN`**
  Track A matcher 的 noise filter：lowercase → 去标点 → 去 `[占位符]` → 去 stopword/intent stopword → 剩下的 token 做 containment / overlap 判断。
  这是 deterministic 的选择，牺牲一部分语义召回，换可解释、可复现、可单测。

---

### 3.3 `verl/trainer/trajectory_dump.py`（+88 行，全新）

#### 为什么
reward 阶段是唯一同时拿到 `(prompt, full decoded response, ground truth, reward components, plan/search calls)` 的地方；把它落 JSONL 后才能离线跑 `scripts/analysis/track_a_self_consistency.py` 这类分析。

#### 做了什么
- 暴露 `_append_trajectory_dump(...)`，按 `schema_version=2` 写入：
  `solution_str / ground_truth / data_source / split / index / question / trajectory / plan_steps / search_calls / final_answer / track_a / track_b / prompt / extra_info / reward_components`
- `_json_safe` 递归把 numpy scalar / ndarray / mapping / list 转成 JSON 安全类型，对 `inf/nan` 转 str 保留信息

#### 关键细节
- **为什么自己写 `_json_safe` 不用 `json.dumps(default=str)`？**
  `default` fallback 会把整个嵌套结构里第一个非 JSON-native 类型转 str，丢掉结构；递归处理可以保留 dict / list 嵌套，只把叶子节点转换。

- **schema_version 为什么写到 row 里？**
  下游分析脚本演进时不至于读到 v1 dump 就崩；可以按 version 走兼容分支。

---

### 3.4 `verl/trainer/ppo/ray_trainer.py`（+491 / -156 行）

这是我自己改动最多的文件。功能上分四块：**断点续训、可复现 dataloader、rollout-only 模式、metrics 聚合**。

#### 3.4.1 断点续训

##### 为什么
verl 上游 + Search-R1 baseline 没有完整 resume：optimizer / lr scheduler / RNG / dataloader 顺序 / KL controller / global_step 全都不持久化，挂掉重跑等于从零训练，loss 曲线对不上。

##### 做了什么
新增 trainer 级 state：
```python
TRAINER_STATE_FILENAME = 'trainer_state.pt'

def _build_trainer_state(next_global_step, next_epoch, next_batch):
    return {
        'global_steps': next_global_step,
        'epoch': next_epoch,
        'batch_in_epoch': next_batch,
        'total_training_steps': self.total_training_steps,
        'seed': self.seed,
        'rng_state': self._get_rng_state(),
        'kl_ctrl_state': {'value': getattr(self.kl_ctrl, 'value', None)},
    }
```
配套：
- `_get_rng_state / _set_rng_state`：同时存 python / numpy / torch / torch.cuda 四套 rng
- `_resolve_resume_dir`：支持 `resume_mode ∈ {none, latest}` 和显式 `resume_path`
- `_discover_latest_checkpoint_dir`：用 `re.compile(r'^global_step_(\d+)$')` 找最大 step
- `_load_resume_state`：把 actor / critic 的 checkpoint dir 解出来，并写回到 `config.actor_rollout_ref.actor.path / config.critic.model.path`（所以 worker init 时会从 resume 路径加载模型）
- `_save_checkpoint(epoch, batch_idx, steps_in_epoch)`：先存 actor/critic 权重，再存 trainer_state，next_batch 跨 epoch 自动滚到下一 epoch 的 0
- `fit()` 开头：如果有 resume_state，恢复 `global_steps / start_epoch / start_batch / rng / kl_ctrl.value`，并跳过 val_before_train

##### 关键细节
- **为什么 trainer_state 和 actor/critic 权重分开存？**
  actor/critic 是 fsdp shard 出来的 huggingface 权重，结构不能塞 trainer 元数据；分开存让权重目录依然能被 HF transformers 加载做推理。

- **`batch_in_epoch` 为什么要存？**
  否则 resume 后会从当前 epoch 的 batch 0 重跑前 N 个 batch，相当于重复训练。我用 `_get_epoch_indices(seed + epoch)` + `Subset` + `enumerate(dataloader)` 跳过 `batch_idx < start_batch`，配合 RNG state 保证完全对齐。

- **RNG 为什么存四套？**
  dataloader shuffle 用 torch.Generator（torch rng），numpy 用在 dataset filter / sample，python random 用在 `uuid` 等地方，cuda rng 影响 dropout / sampling kernels；缺一个 resume 后训练就会偏。

#### 3.4.2 可复现 dataloader

##### 为什么
原 baseline 把 `DataLoader(shuffle=True)` 直接绑一次性，没法做 epoch-level seed 控制；resume 时不知道当前 epoch 的 permutation。

##### 做了什么
```python
def _get_epoch_indices(self, epoch):
    if not self.config.data.shuffle_train_dataloader:
        return list(range(dataset_size))
    generator = torch.Generator()
    generator.manual_seed(self.seed + epoch)
    return torch.randperm(dataset_size, generator=generator).tolist()

def _get_train_dataloader_for_epoch(self, epoch):
    train_subset = self._subset_cls(self.train_dataset, self._get_epoch_indices(epoch))
    return DataLoader(train_subset, batch_size=..., shuffle=False, drop_last=True, ...)
```
val_dataloader 改成 `shuffle=False`，保证验证完全可复现。

##### 关键细节
- **`seed + epoch` 而不是固定 `seed`**：保证不同 epoch 是不同 permutation，又能 resume 时复现
- **为什么不用 `DistributedSampler.set_epoch`？**
  这里 DataLoader 是单进程从 dataset 取数据，再 dispatch 到 ray worker；分布式 sampler 不直接适用

#### 3.4.3 `rollout_only` 模式

##### 为什么
- 训练前/中需要纯评估，但又想跑训练 stack（reward manager / parser / trajectory dump 全套，包括 `n_plan / n_actions / S_self` 等指标）
- 不想浪费 ref policy / critic / actor update 的开销

##### 做了什么
`fit()` 里：
```python
rollout_only = self.config.trainer.get('rollout_only', False)
...
if rollout_only:
    with _timer('reward', timing_raw):
        reward_tensor = self.reward_fn(batch)
        batch.batch['token_level_scores'] = reward_tensor
    metrics.update(compute_rollout_only_metrics(batch, reward_tensor))
    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
    print_rollout_only_timing(self.global_steps, batch, timing_raw, metrics)
    logger.log(data=metrics, step=self.global_steps)
    self.global_steps += 1
    if self.global_steps >= self.total_training_steps:
        logger.finish(); return
    continue  # 跳过 ref / critic / actor update
```
配套 `compute_rollout_only_metrics`（只算 reward + env metrics，不依赖 critic）、`print_rollout_only_timing`（输出 `sec/traj` 和 `traj/s`）。

#### 3.4.4 reward components + action reason metrics 聚合

##### 为什么
原 baseline 训练日志只有 `critic/score` 和长度类指标，看不到"failure attribution"。Search-P1 引入了 21 个 reward components 和 12 类 action reason，需要批量化聚合。

##### 做了什么
模块顶层定义白名单：
```python
REWARD_COMPONENT_KEYS = ('base_score', 'has_search', 'effective_structure_format',
    'effective_retrieval', 'track_a_bonus', 'self_consistency_weight',
    'track_b_bonus', 'reference_alignment_weight', 'path_bonus',
    'self_consistency', 'self_r_planner', 'self_n_plan', 'self_n_actions',
    'self_n_exec', 'reference_alignment', 'ref_available', 'ref_n_steps',
    'ref_n_actions', 'ref_n_covered', 'final_score')

VALID_ACTION_REASON_KEYS = {"valid_search", "valid_answer", "valid_plan", "inactive"}
INVALID_ACTION_REASON_KEYS = {"missing_plan", "duplicate_plan",
    "missing_or_invalid_plan_steps", "action_before_plan", "missing_think",
    "invalid_search", "missing_action_tag", "empty_prediction",
    "malformed_action_tag", "malformed_query_tag", "malformed_tool_tag",
    "malformed_search_content", "unknown_invalid"}
```
两个聚合函数：
- `_compute_reward_component_metrics(meta_info, prefix='reward')` → `reward/<comp>/{mean,max,min}`
- `_compute_action_reason_metrics(meta_info, prefix='env')` → `env/invalid_action/{total,ratio}`、`env/action_reason/<reason>/{count,ratio}`、对 invalid reason 再额外发 `env/invalid_action/<reason>/{count,ratio}`

`_validate()` 里把这两类指标都按 `val/` 前缀汇总：
```python
metric_dict.update(_compute_reward_component_metrics(
    {'reward_components': reward_component_values}, prefix='val/reward'))
metric_dict.update(_compute_action_reason_metrics(
    {'action_reason_stats': action_reason_stats}, prefix='val/env'))
```

#### 3.4.5 其它顺手做的改动

- `apply_kl_penalty` 已经被 baseline 改成用 `info_mask`，我把它和 `_create_loss_mask` 的实现简化：直接读 rollout 阶段产出的 `batch.batch['info_mask']`，不再在 trainer 里用 `re.finditer(state_marker)` 在 decoded response 上反查 token boundary。
  - **原因**：那种基于 marker 反查的方式假设 tokenizer 是 prefix-stable 的，对 BPE/Byte-level tokenizer 边界容易差 1；rollout 阶段就知道哪些是 observation token，直接打 mask 更可靠。

- 验证 dataloader `shuffle=True` → `False`（验证可复现）

- `num_gpus = n_gpus_per_node * nnodes`（多节点修正）

- `batch.non_tensor_batch['uid']` 从随机 uuid 改成 `batch.non_tensor_batch['index'].copy()`
  - **原因**：用 dataset index 作 uid 后，trajectory dump 跨 step 可对齐到原始样本，方便做 per-sample 训练曲线

- 把 baseline 中所有 `try/except: print(...)` 的"吞异常打 batch" pattern 删掉，让真实错误能 raise 出来

- 删掉 baseline 中 `_create_loss_mask` 里几十行调试 print

---

### 3.5 `verl/workers/fsdp_workers.py`（+127 行）

#### 为什么
对应 §3.4.1 的 trainer-level resume，actor / critic 也必须能持久化和回灌：optimizer state、lr scheduler state、各 rank 的 RNG。

#### 做了什么
在 `ActorRolloutRefWorker` 和 `CriticWorker` 两个类里都加一组：
```python
def _get_rng_state(self): ...         # python / numpy / torch / torch.cuda
def _set_rng_state(self, ...): ...
def _get_worker_state_path(self, ckpt_dir):
    return os.path.join(ckpt_dir, 'worker_state', f'rank_{self.rank}.pt')
def _save_worker_state(self, ckpt_dir):
    torch.save({'optimizer': ..., 'lr_scheduler': ..., 'rng_state': ...}, path)
def _load_worker_state(self, ckpt_dir): ...
```
注册新的 dispatch 接口：
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def load_checkpoint(self, local_path):
    # offload-aware: 必要时把 optimizer load 回 cuda 再 offload 回 cpu
    self._load_worker_state(local_path)
    torch.distributed.barrier()
```
`save_checkpoint` 里改成"权重存 rank 0 → 各 rank 存 worker_state → barrier → rank 0 上传 HDFS"，避免 HDFS 上传和 worker_state 写盘竞争。

`_build_model_optimizer` 接受 `actor.path / ref.path` 的覆盖：
```python
actor_model_path = self.config.actor.get('path') or self.config.model.path
ref_model_path   = self.config.ref.get('path')   or self.config.model.path
```
配合 trainer 在 resume 时把 checkpoint path 写进 config，让 worker init 时就从 resume 权重 load。

#### 关键细节
- **为什么 actor 和 critic 各存一份 RNG state？**
  它们运行在不同的 ray actor 进程里，各自有独立的 python / numpy / torch rng；trainer 进程那份只覆盖单 controller 进程。

- **`offload_fsdp_optimizer` 的处理**
  resume 之前如果 optimizer 已经被 offload 到 CPU，先 `load_fsdp_optimizer` 拉回 GPU 再 load state，load 完再 offload 回 CPU。否则 `optimizer.load_state_dict` 会找不到对应 device 的 tensor。

---

### 3.6 `verl/utils/dataset/sft_dataset.py`（+171 行，全新）

#### 为什么
SFT 起步阶段需要 in-memory parquet 数据集，原版 verl 的 SFT dataset 接口对我的数据格式不直接兼容（我们的 SFT parquet 是 P1 trajectory 的 prompt/response pair）。

#### 做了什么
- 读取 verl-style parquet：`copy_local_path_from_hdfs` → `pd.read_parquet` → concat
- `prompt_key / response_key` 支持嵌套字典访问（`prompt_dict_keys`）
- 用 tokenizer 的 `chat_template` 渲染 prompt，response 自带 EOS
- 拼成定长 `input_ids / attention_mask / position_ids / loss_mask`，padding 走右侧
- **`loss_mask` 把 prompt token 全部置 0**，只让 response span 参与 loss
- truncation 支持 `error / left / right`
- 在 `verl/utils/dataset/__init__.py` 里 export

#### 关键细节
- **为什么 loss_mask 的 prompt 区是 `[:prompt_length-1] = 0` 而不是 `[:prompt_length] = 0`？**
  最后一个 prompt token 的 next-token-prediction 目标是第一个 response token，所以那个位置的 loss 是需要的；从 prompt_length-1 开始向前置 0 才对。

- **最后一个 token loss_mask 也置 0**
  最后一个 response token 没有 next token 可预测，loss 没有意义。

---

### 3.7 `verl/utils/tracking.py`（+31 行）

#### 为什么
1. 我们用 swanlab 不是 wandb（公司/集群环境）
2. ray 在 sigkill / OOM 时 wandb 进程经常残留，run 状态停在 running

#### 做了什么
- 加 `swanlab` backend：
  ```python
  if 'swanlab' in default_backend:
      import swanlab
      if SWANLAB_API_KEY: swanlab.login(api_key=SWANLAB_API_KEY)
      swanlab.init(project=project_name, experiment_name=experiment_name, config=config)
      self.logger['swanlab'] = _SwanlabLoggingAdapter()
  ```
- 加 `finish()` 方法 + `atexit.register(self.finish)`，按 backend 优雅关闭（wandb.finish / swanlab.finish / mlflow.end_run）
- trainer 多处主动调 `logger.finish()`（正常退出、val_only return、超训练步数 return、final return）

#### 关键细节
- **`_finished` 标志位**：防止 atexit 和主动调用各执行一次造成重复 finish 报错
- **每个 backend 的 finish 用 try/except**：一个 backend 挂掉不能影响别的 backend cleanup

---

### 3.8 `verl/trainer/config/ppo_trainer.yaml`（+19 行）

新增字段：
```yaml
actor_rollout_ref:
  actor:
    path: null              # resume 时覆盖 model.path
  ref:
    path: null

reward_model:
  structure_format_score: 0
  final_format_score: 0
  retrieval_score: 0
  path_match_strategy: lexical
  require_search_for_format: false
  max_plan_steps: null
  max_reference_steps: null
  self_consistency_weight: 0.0
  reference_alignment_weight: 0.0
  trajectory_dump_path: null
  trajectory_dump_limit: 0
  trajectory_dump_full_solution: true

trainer:
  seed: 1
  resume_mode: none
  resume_path: null
```

---

### 3.9 `verl/utils/reward_score/qa_em.py`（小改）

baseline 的 `extract_solution`：`if len(matches) <= 1: return None`（要求至少 2 个 `<answer>`，第一个被当作 demonstration）。

我改成 `if len(matches) == 0: return None`，并显式去掉 retry 模板：
```python
solution_str = re.sub(
    r"My previous action is invalid\.[^\n]*Let me try again\.",
    "", solution_str)
```
**原因**：P1 prompt 里不再有 demonstration `<answer>`，只有 final answer 一个；保留 `<= 1` 会把单 answer 直接判 None，导致 EM 永远 0。

---

## 4. 面试 cheat sheet

### 4.1 三句话总结

1. **底层（继承自 Search-R1）**：在 `ray_trainer.py` 的 PPO 循环里嵌入 `LLMGenerationManager` 做多轮 search rollout，并用 `info_mask` 屏蔽环境注入的 `<information>` token 的 loss/KL，只对模型自生 token 反传梯度。
2. **中层（我的工程基础设施）**：补齐**断点续训**（trainer + actor + critic + 四套 RNG + dataloader 顺序）、**rollout-only 模式**、**21 项 reward components + 12 类 action reason 指标聚合**、**SFT 通路**、**swanlab + atexit finish**、**reward-time trajectory dump**。
3. **上层（我的研究侧改造）**：把单层 reasoning-search 升级为 **front-loaded planner + plan-once + plan-only 首阶段**；rollout parser / reward parser / trainer metrics / 离线分析四层共享同一套 tag 与 planner 语义合约；实现 Track A self-consistency 与 Track B reference alignment，**当前只作为旁路 component 记录，不入 scalar reward**，避免 matcher 未校准时的 reward hacking。

### 4.2 高频追问 → 标准答

| 问 | 答 |
|---|---|
| 为什么 reward parser 要先去掉 `<information>` 再抽 `<search>`？ | 检索结果文本里可能字面含 `<search>`，不删会被误判成模型 action，污染 `n_actions` 和 Track A 分母 |
| 为什么 `loss_mask` / KL 用 `info_mask` 而非 `attention_mask`？ | observation 是环境注入的，不是模型生成；对它反传梯度等于让模型学复述检索结果，污染策略；KL 同理只该约束 actor 自生 token |
| 为什么 resume 要存四套 RNG？ | torch rng 影响 dataloader shuffle + dropout，cuda rng 影响 sampling kernels，numpy rng 影响 dataset filter，python rng 影响 uuid 等；任一缺失 resume 后训练就偏 |
| resume 怎么保证 dataloader 顺序对齐？ | `_get_epoch_indices(seed + epoch)` 重建 permutation；`Subset + DataLoader(shuffle=False)` + `enumerate` 跳过 `batch_idx < start_batch`；val_dataloader 直接 `shuffle=False` |
| 为什么 Track A 暂不入 scalar reward？ | 第一版 lexical matcher 噪声大，直接入 reward 可能错罚语义等价但词面不同的 action，引入新 reward hacking；先记录、观察分布、确认信号可靠再 composition |
| plan-once 怎么在四层一致地实现？ | rollout 用 `planner_seen` 状态机 + 在 `</search>/</answer>` 处截断；reward parser 用状态机校验 plan 唯一在最前；最后一轮 `allow_plan_only=False` 拒只 plan；config 里 `<information>` 是 state masking marker |
| rollout-only 模式有什么用？ | 用训练 stack 跑纯评估，能拿到 21 个 reward components 和 12 类 action reason 的分布，但不更新参数、不跑 ref/critic，方便做 model checkpoint sweep 或 prompt 消融 |
| `score` 为什么写在"最后一个有效 token"而不是 `attention_mask` 末尾？ | `attention_mask` 末尾可能落在 `<information>` block 里（环境注入），把 reward 写在那里相当于附给被 mask 掉 loss 的位置，等于丢分；要写到模型真正生成的"答案/结束"token |
| Track A / Track B 为什么解耦？ | A 参照模型自己 plan，B 参照外部 reference steps；互读对方数据指标语义会混；解耦后可单独消融，再用 `R_path = max(S_self, S_ref)` 聚合 |
| `uid` 为什么改成 `index.copy()`？ | 用 dataset index 当 uid 后 trajectory dump 跨 step 可对齐到原始样本，能做 per-sample 训练曲线；random uuid 跨 step 无法配对 |
| baseline 的 `_create_loss_mask` 你为什么改？ | 它用 `re.finditer(state_marker)` 在 decoded response 上反查 token boundary，对 BPE/Byte-level tokenizer 容易差 1 个 token；rollout 阶段就知道哪些是 observation token，直接产出 `info_mask`，trainer 直接读即可 |

### 4.3 一定要主动提的"工程取舍"

1. **先统一四层（rollout / reward / trainer / 离线分析）tag 与 planner 语义合约**，否则训练能跑但解析口径错位，loss 曲线漂亮但策略学错。
2. **Track A 旁路记录不入 scalar reward**，降低 early reward shaping 风险。
3. **deterministic lexical matcher 起步**，牺牲一部分语义召回，换可解释、可复现、可单测；等离线分析证明需要再升级。
4. **resume 把 trainer/actor/critic/RNG/dataloader 顺序全对齐**，而不是只 dump 模型权重——后者本质等于"换初始化重训"。
5. **rollout-only 模式复用训练 stack**，避免离线评估和训练评估口径分裂。
