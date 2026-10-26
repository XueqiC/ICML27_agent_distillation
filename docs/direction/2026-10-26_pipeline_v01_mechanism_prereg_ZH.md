# Pipeline v0.1 第一步机制验证：预注册草案（待用户确认）

日期：2026-10-26（CDT）
依据：`docs/direction/2026-10-26_agent_distillation_pipeline_v0_1_ZH.md`（用户 10-26 01:39 CDT 提供，sha256 423a43e2…），以第 14 节为准。
范围：只验证 H1（保留学生路线的联合修复是否可行）和 H2（同题、同正确性下，近学生路线的监督是否更快改变学生行为并带来独立完成）。不涉及 H3/H4、任务生成、梯度分组或多轮训练。
状态：草案。工程部分（M0）不用 GPU、不调 teacher，可先做；M1 起需用户确认后才消耗 GPU 和 teacher。

## 0. 对接现有资产（不重建 harness、不重跑 baseline）

| 项 | 取值 | 来源 |
|---|---|---|
| Teacher | gpt-6-luna，黑盒，现有 OpenAI key 与 rate limiter | configs/base_e4b.yaml |
| Student | google/gemma-4-E4B-it @ ee0ef602，bf16 LoRA | 同上 |
| 首个 K | **K=20**（文档默认：K20 已有 checkpoint 时优先） | 主网格 |
| θ₀ | **e4b-tsft-k20-e32-lr5e-5（轨迹SFT，seed 0，rule-A rung）**，dev140 = 78/140 | 10-26 00:07 CDT 结果 |
| Seed 任务 | K20 view（full-TRAIN chain 的嵌套子集，K10⊂K20⊂K30） | 现有 manifest，不重选 |
| Harness | H0：ALFWorld 0.4.2，`react_v2_last_action_line`（Thought/Action），thinking off | 现有 |
| 预注册的 harness 覆盖 | **H = 40**（文档默认 50）；单轮输出上限 **1024 token**（文档默认 256）；上下文上限沿用现有 serving 配置 | 文档规定"已有 baseline 不同则统一沿用并记录" |
| Replay | `harness.alfworld_env` + `sed.runner.replay_episode`：reset 指定任务后重放，逐步精确校验（`ReplayMismatch`） | 现有 |
| Teacher 调用 | `teachers.stage`（预算、账本、429 重试、halt 即失败） | 现有 |
| 训练 | 沿用 tsft E4B 的 LoRA 范围、optimizer、lr 5e-5、loss normalization；只换 target 数据 | 文档 14.0 |
| 评估硬件 | 与 baseline 相同的 hpg B200，evaluation tree 46b8527（含 k_eq_v 修复、TRITON_ATTN） | 现有 |
| dev140 | valid_seen，长期用于选模 → 记为 development/reference set | 文档 2.2-6 |

## 1. 步骤与门槛

### M0 工程（不用 GPU、不调 teacher；Codex 实现，我审）
只新增文档 14.0 列出的模块，不另写 parser、成功判定或统计：
1. Patch 语法与编译（6.3 全部规则）、动作 Levenshtein 距离与 DP 对齐（8.4 平局顺序）。
2. 候选执行：reset 原任务后执行编译后的动作序列，每步读取真实 observation，成功即停；缓存键含环境版本、task fingerprint、动作序列、seed。
3. 最小性检查（7.3）与联合性记录（7.4），状态枚举（4.3），24 次 replay 总上限。
4. 账本：三类预算（2.3），失败、重试、被拒候选全部记账。
5. 前缀一致的 reasoning 标签构造（8.2，≤64 token），自动检查引用 step 早于当前动作、实例 ID 已公开。
6. LearningEpisode → 现有 SFT trainer 的数据格式（full-trajectory 视图）。
7. 三个 probe：决策 probe（14.9）、接续 probe（14.10）、独立评估（14.11）。
8. 单元测试：18.2 的索引、观察、删除、成功、信息泄漏、归一化、预算、评估检查。
9. 确定性验收（3.3）：5 个训练侧任务，teacher 原成功示范各重放 2 遍，核对 fingerprint、逐步 observation、成功判定和三种结束原因。

### M1 覆盖检查（GPU，约 0.5 GPU·h）
- θ₀ 在自己的 20 个训练任务上各跑 2 次：第 1 次 greedy，第 2 次 T=0.7、top_p=0.95（需给 harness 加一个 sampled 解码模式，其余不变）。
- 同时、用同一设置跑 base E4B（K=0）在这 20 个任务上的 2 次 rollout，供分支用。
- **预注册分支规则（文档 14.0 / 5.3）：**
  - θ₀ 失败的不同任务 ≥ 6 → P0，在 K20 上直接做 E1/E2。
  - < 6 → 再追加至多 2 次随机 rollout（T=0.7）；
  - 仍 < 6 → 按文档默认改用 K=0 起点（base E4B），在同样 20 个任务上验证学习机制。这一支只回答"修复监督是否更好学"，不回答"能否改善 SFT-K20"，两者分开报告。
- 预期：θ₀ 的训练 loss 已到 1e-4 量级，很可能把 20 个训练题几乎全做对，大概率进入 K=0 分支。这本身是一个发现：只重复原题的失败驱动循环，在收敛的 SFT 上可能没有信号。

### M2 冻结划分（在任何 teacher 标签之前）
- D_probe：60 个训练侧任务，按任务类型分层，固定 seed 抽取。排除：所有 K（10/20/30/100）seed 任务的并集；保守起见，再排除 K500 chain 各 view 中出现过的全部任务（任何 baseline 训练用过的题）、dev140 和 valid_unseen。
- D_unseen_quick：D_probe 中固定的 24 个，不送 teacher，不用于调 prompt、挑 target 或训练。
- D_repair_probe：从失败任务中按固定 seed 选 2 个，在 teacher 产生标签前固定。
- 所有清单落盘并记录 sha256，写进 RunManifest。

### M3 E1：执行干预（teacher + 仅 CPU 的环境 replay，不训练）
- 失败池：按任务类型分层，每任务取 1 条，最多 20 条（5.3）。
- 每条失败：1 次 diagnosis，至多 2 轮 proposal × 3 个候选，最小性撤销测试 ≤12 次，replay 总计 ≤24 次。
- 对选中修复：原轨迹失败 / 单个 block / 全部 block / 撤销再恢复，四种执行结果都由环境实际给出（14.4 证据表）。
- 同时构造 E2 所需的 N/F 配对：两条都实际成功；对象实例相同；必需操作类型相同；动作长度比和 supervised token 比都在 [0.8, 1.25]；F 的编辑距离至少比 N 大 max(2, ⌈0.15·max(L_N, L_F)⌉)；按 14.7 的确定性规则选取前 4 对。
- 进入 E2 的门槛（14.5）：至少 4 个训练对加 2 个 probe 有环境验证的修复。另外，至少有 2 个来自不同任务的多 block 必要性案例，才检验"联合修复"这一主张；不足时如实写明本次没有验证联合性。

### M4 E2：学习干预（GPU，约 1–2 GPU·h）
- 两臂 E2-N / E2-F：同一个 θ₀，相同的可训练参数、optimizer 状态和采样顺序，最多 20 次 update，effective batch = 4 个 episode。
- 混合权重：repair 0.75 / seed replay 0.25，seed replay 只用 D_repair_train 对应任务的示范；loss 为 full-trajectory，沿用现有 SFT loss；training_seed = 1。
- 不做 checkpoint 选择，固定报告 0/1/5/20 四个点。
- 观测 1（14.9）：2 个 probe × 4 个 checkpoint，各采样 8 个 turn，计算动作落入 A⁺ 的比例 m，报告 Δm_N − Δm_F。
- 观测 2（14.10）：每个 probe 每个 checkpoint 取前 2 个采样 seed，重放到 h_i 后学生自主执行到底，记录无辅助接续成功率。
- 观测 3（14.11）：D_unseen_quick 24 题，greedy，在 θ₀ 和两臂的 checkpoint 20 上各跑一次，保存逐题配对结果。
- 结论按 14.12 的对照表下。允许"机制不确定"或"F 更好"这类反证结果，不加模块掩盖。

## 2. 预算（硬上限，到达即停；实际用量另报）
- Teacher（M3，K20 一个起点）：每条失败按文档 8.6 设上限（输入 250k、输出 24k token、调用 54 次）。全实验硬上限 **5M 输入 / 0.5M 输出 token**；预计实际约 1.5M / 0.15M。
- 环境：每条失败 replay ≤ 24 次，每次 ≤ 40 步；评估步数单独记账。
- GPU：M1 约 0.5 GPU·h，M4 约 1–2 GPU·h，与主网格共用 hpg 的组配额，排在网格剩余 eval 之后。
- 优化：每臂 20 次 update；记录 processed / supervised token。

## 3. 与现有进度的关系
- 主网格（K10/K20 的 seed 0–2、K100 的 seed 2、SAD K20）照常推进，不改任何已有 cell。
- 机制实验只依赖已经存在的 θ₀（tsft K20 s0），不等网格跑完。
- 14.14-1 的"在同一 K 最强 baseline 上做增量确认"，等 AD K20、SmartAD K20 的结果出来再定起点（预计今早）。
- K10/K30/K100 的扩展：K20 有方向后再原样铺开（14.0 第 2 条）。

## 4. 需要用户确认
1. 若 M1 失败覆盖不足：按文档默认改用 K=0 起点。是否**同时**加一个 U_mech（P1）分支？做法是取 20 个训练侧额外任务采集 θ₀ 的失败，直接回答"能否在 SFT-K20 上继续提升"。我建议加，teacher 预算约翻倍，仍在上面的硬上限内。
2. Teacher 硬上限 5M 输入 / 0.5M 输出 token 是否可以。
3. GPU 阶段放在 hpg（与 baseline 同为 B200，数据和 checkpoint 都已在那里）。rai 磁盘只剩 157G（99%），不在 rai 落大文件。
4. 文档 8.2 要求对入训 reasoning 标签逐条做人工前缀检查：我先跑自动检查，再逐条审读，审核表发你。你是否也要亲自过一遍？

## 5. 用户决定（2026-10-26 10:40 CDT，"可以"；10:40 CDT "我们现在先在RAI上完成新方法机制实验吧"）
1. 加 U_mech（P1）分支：θ₀ 在 data/manifest/repair_v01_splits.json 的 20 个 U_mech 任务上采集失败，直接回答"能否在 SFT-K20 上继续提升"；与 K20 原题分支（或 K=0 分支）分开报告。
2. Teacher 全实验硬上限 5M 输入 / 0.5M 输出 token（工具在每次调用前检查）。
3. 机制实验全部在 rai 上做（覆盖检查、E1、E2 训练、probe），θ₀ / N / F 的所有比较用同一种卡（rai Blackwell RTX PRO 6000，CUDA_DEVICE_ORDER=PCI_BUS_ID）；主表数字仍全部来自 hpg B200，不与 pilot 数字混用。
4. 入训 reasoning 标签：我先跑自动检查并逐条审读，审核表发用户抽查。

预注册的实现细节（工程 M0 完成后确定，未看任何结果）：
- 代码：eval/采集/probe = 分支 repair-v01-eval（46b8527 + repair，c30e411）；训练 = 分支 repair-v01-train（2506fa9 + repair，511dc5e）。
- 划分：data/manifest/repair_v01_splits.json（排除所有 baseline 实际训练过的任务组；D_probe 60 / D_unseen_quick 24 / U_mech 20，无缺额）。
- D_repair_probe：失败任务中 sha256([20261029, task_id]) 最小的 2 个，在任何 teacher 调用前固定。
- 失败顺序：按任务类型，再按 sha256([20261028, task_id])。
- θ₀ = e4b-tsft-k20-e32-lr5e-5（LONI 训练），rai 副本 reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5（19 个文件，与 LONI sha256 一致）。

## 6. M1 第一组结果与由此确定的分支细节（2026-10-26 11:1x CDT，看任何修复结果之前写下）
- θ₀ 在自己的 20 个 K20 训练题上 greedy + T=0.7 共 40 个 episode 全部成功（LoRA 检查通过，adapter 误差 5e-6）。按规则补 2 次 T=0.7 rollout；仍不足 6 个失败任务就走 K=0 分支。两个分支与 U_mech 分别报告，不混成一条曲线。
- **U_mech 分支（P1）**：失败来自 θ₀ 在 20 个 U_mech 任务上的 rollout。E2 的 seed 回放（0.25）用 θ₀ 原来的 K20 teacher 示范，对应文档 11.2 的 D 桶，因为 U_mech 任务没有属于 θ₀ 的 seed 示范。U_mech 任务在 full-TRAIN 采集里的 teacher 成功轨迹**不读取**，既不作 Far target，也不回放，避免引入额外的 teacher 示范。新任务的监督单独记账。
- **K=0 分支**：θ₀ = base E4B（无 adapter）。E2 两臂用继承的 tsft LoRA 配方（r64/α128/dropout 0.05、lr 5e-5）从同一个初始化开始。seed 回放用 D_repair_train 任务的 K20 teacher 示范。原 teacher 示范仍不作 Far target（保守选择）。
