# V2 预注册：把有效交互变成训练数据，检验学生在无提示下能否保留

日期：2026-10-29（CDT）。依据方法完善方案 v1 §4（`docs/method/experiment_plan_v1_ZH.md`）和设计文档 §2.4、§3.3。本文写于任何 V2 采集、训练或评测之前。

前置结果：V1b 按预注册规则通过（`reports/analysis/method_v1/V1b_joint_guidance_ZH.md`）。G 0.658，U 0.253，前缀 29/4，teacher 示范步数占 3.4%。

## 1. 问题

把 V1b 中验证过的联合指导交互构造成监督数据，从 θ₀ 继续训练后，学生在**没有任何指导**时，能否在新任务和原困难历史上超过以下两组？
- **OFFLINE**：同一训练池上 teacher 自己完成的轨迹；
- **PREFIX-EXPERT**：从学生前缀出发、由 teacher 接管完成的数据。

## 2. 固定的输入

| 项目 | 值 |
|---|---|
| θ₀ | `e4b-tsft-k20-e32-lr5e-5`（主网格 K20 trajectory SFT，seed 0），与 CG、V1a、V1b 相同 |
| 训练池 | `method_v1_splits.json` 的 V2_pool：200 个 TRAIN 任务组，每类 33–34 个，每组一个固定 trial，各有一条已验证的 teacher 成功轨迹 |
| 采集代码 | repair-v01-eval 7748da5（固定 worktree），R 的提示与 V1b 完全相同 |
| 训练代码 | repair-v01-train 2050d69（E0.5，固定 worktree） |
| 诊断/评测集合 | U_dev、U_mech、CG、D_probe、D_unseen_quick 都不进入训练；不使用 dev140 和 valid_unseen |

V1b 发现的 heat 短板（R 不提示加热命令）**不在本轮修改**。V2 用的是 V1b 验证过的 R，修复另起一个小实验（§8）。

## 3. 采集

**C1. θ₀ 覆盖**
- 在 V2_pool 200 个任务上，1 次 greedy 加 3 次 T=0.7 采样（与 U_dev 预筛相同）。
- rai GPU 1，不调用 teacher。
- 至少失败一次的任务记为"失败任务"。

**C2. 决策点前缀**
- 用 `tools.prefix_points build --set V2_pool` 和 dp/1 规则，每个失败任务取一个前缀：按 greedy、seed 0、1、2 的顺序取第一个失败的 rollout，在它最早的事件处分支，事件依次为被拒绝、看到目标、拿到目标，都没有则取第 10 步之后。
- 分支后不足 10 步的不取。
- 得到的前缀集合记为 P。

**C3. 两种起点**：P 中每个任务都有两个起点：
- dp：决策点前缀；
- start：同一任务从第 0 步开始（`--from-start`）。

两种起点在 JOINT 和 PREFIX-EXPERT 之间完全相同。

**C4. G 采集（JOINT 的来源）**
- 每个起点跑 G，即每一步由 R 审阅，1 次 greedy 加 4 次采样（seed 0–3）。
- 不跑 U 和 G1。

**C5. teacher 接管（PREFIX-EXPERT 的来源）**
- 每个起点由 teacher 接管 5 次（`tools.teacher_takeover --runs 5`；start 起点加 `--from-start`），与 G 每个起点 5 个 episode 对齐。

**预算**
- 不设设计上的上限。全局计数器 `reports/runs/method_v1/v2_global_teacher_budget.jsonl`。
- 防失控上限 800M input / 100M output，只用于防止程序失控。
- 估计：G 约 100M / 12M，接管约 50M / 1M，标签约 5M / 1M。

## 4. 训练数据（E0.4，`tools.build_v2_dataset`）

| 组 | 来源 | 样本类型 |
|---|---|---|
| CONTINUE | K20 的 20 条原示范 | 轨迹 |
| OFFLINE | V2_pool 200 条已验证的 teacher 成功轨迹 | 轨迹 |
| PREFIX-EXPERT | C5 中成功的接管：分支后 teacher 执行的每一步（去掉被拒绝的步） | 决策 |
| JOINT | C4 的局部有效决策，加上成功 episode 的完整行为 | 决策 |
| JOINT_COMPLETE_ONLY | C4 中只取成功 episode 的完整行为 | 决策 |

- **规则**：
  - 所有组都按 E0.4 和接口规范 `v2-train-interchange/1` 构造；
  - 历史中的 Guidance 全部删除；
  - 依赖指导的 Thought 替换为前缀一致的 L 标签（不超过 64 token，L 调用走同一个全局计数器）；
  - 没有有效标签时只监督 Action。
- **K20 replay**：每组都附带 `replay/`。
- **记录**：每组的样本数、监督 turn 数、监督 token 数，以及按任务类型和局部事件的分布。

## 5. 训练（E0.5，`tools.train_v2_arm`）

所有组都从 θ₀ 出发，设置完全相同：
- **配方**：沿用 θ₀ 的 LoRA（r64、alpha 128、dropout 0.05、相同 target modules）和 AdamW；学习率在 θ₀ 的 5e-5 处按原形状重启。
- **batch**：每次 update 16 个样本，其中 replay 比例 0.25，即 4 个 K20 replay 加 12 个本组样本。在同一个训练 seed（0）下，各组抽到的 replay 完全相同。
- **长度**：500 次 update，所有组相同（等计算量）。
- **checkpoint**：第 1、5、20、50、100、200、300、500 次 update。
- **延长规则**：如果某组按 §6 选出的 checkpoint 是第 500 次，就把该组延长到 1,000 次（新增 700、1,000），重新选择。这条规则对所有组相同。
- **硬件**：hpg B200，每组 2 张卡（E0.5 的 `--gpus 2`，与主网格相同的 FSDP 启动方式），所有组用同一硬件，单训练 seed。
  - 方案 v1 第 4 条原定在 rai 上训练。按 θ₀ 的实测吞吐（约 1,060 token/s/卡），每组约 40M token，在 rai 单卡上每组约 10 小时。5 组依次训练要两天多，还会长期占用共享机器，所以改到 hpg。
  - 这个改动只出于吞吐考虑，与任何结果无关。

## 6. 评测（全部无提示）

所有评测都通过同一个服务一致性检查。同一个评测集合里的所有组（含 θ₀）都在同一种卡上跑（rai Blackwell，或者 hpg B200 上用对应的 twin 脚本），不混用。θ₀ 作为第 0 组，在同一批评测里重新跑。

**E-sel：选择 checkpoint**
- **集合**：D_unseen_quick 24 题，从起点开始，1 次 greedy 加 3 次采样。
- **规则**：每组在第 20–500 次 update 的 checkpoint 中，选成功率最高的一个；平局时取较早的。第 1、5 次只做短程诊断，不参与选择。

**E-short：短程诊断**
- 在第 1、5、20 次 update 的 checkpoint 上，在 38 个 V1b 主前缀上跑 U，只跑 greedy。
- 报告：成功、被拒绝的动作、第一步相对 θ₀ 的改变比例。

**用选出的 checkpoint 做三种主评测**
- **(a) 原困难历史续跑**：38 个 V1b 主前缀，U 模式，1 次 greedy 加 8 次采样（seed 0–7）。
- **(b) 同任务从起点执行**：(a) 的 38 个任务从第 0 步开始，1 次 greedy 加 8 次采样。
- **(c) 新任务**：D_probe 60 题从起点开始，1 次 greedy 加 3 次采样。**主比较只用其中不属于 quick 的 36 题**，因为 quick 24 已经用来选 checkpoint；60 题的结果作为描述。

## 7. 判定规则（预先写定）

**主判定**："JOINT 优于 X"（X 为 OFFLINE 或 PREFIX-EXPERT）须同时满足：
1. **(a) 原困难历史**：采样成功率（先按前缀算，再平均）之差 ≥ 0.10，且前缀层面"JOINT 更高"的个数比"更低"的至少多 4 个；
2. **(c) D_probe 非 quick 36 题**：采样成功率之差 ≥ 0.05，且任务层面"更高"比"更低"至少多 3 个。

所有差都附按前缀或按任务的 cluster bootstrap 95% CI（10,000 次，seed 0）。

**V2 通过**：
- JOINT 同时优于 OFFLINE 和 PREFIX-EXPERT；
- 无提示下的行为指标也改善：每个 episode 被拒绝的动作数、其中重复已拒绝动作的数、拿到目标的比例，在 (a) 和 (c) 上都不差于 OFFLINE 和 PREFIX-EXPERT。

**附加比较**（只描述，不作门槛）：
- JOINT 对 JOINT_COMPLETE_ONLY：局部有效决策的价值；
- JOINT 对 CONTINUE，以及各组对 θ₀；
- (b) 同任务从起点执行；
- 按任务类型和 V1b 的层分别报告；
- greedy 单独报告。

**监督量归因**：
- 如果 JOINT 的监督 turn 数超过 OFFLINE 或 PREFIX-EXPERT 的 2 倍，就从 JOINT 中按任务分层抽一个等监督 turn 数的子集（seed 0），用同样设置训练、选择，并评测 (a) 和 (c)。
- 结论里同时报告全量和等量两种结果。

**结果映射**：

| 结果 | 下一步 |
|---|---|
| JOINT 优于两者 | V3（条件变化练习），然后扩展到其他 K，并在 dev140 上与 baseline 配对比较 |
| JOINT ≈ PREFIX-EXPERT，两者都优于 OFFLINE | 学生状态分布是关键，执行者是谁不关键；检查 JOINT 的标签和局部决策 |
| JOINT 不优于 OFFLINE | 指导下的行为没有转化为独立能力；检查 E-short 的行为变化，以及历史中替换 Thought 的影响 |
| 训练后 (a) 提高但 (c) 不提高 | 学到的是修复特定历史，不是迁移到新任务；进入 V3 的条件练习 |

## 8. 并行的小实验：变换前提条件（不影响 V2）

- **改动**：在事实表和 R 的提示里加入 heat/cool/clean 的前提条件：手里拿着目标、在对应器具旁，以及合法的命令形式。代码由 Codex 在 repair-v01-eval 上只新增文件实现。
- **复测**：只在 V1b 的 9 个 heat 前缀上跑 G，1 次 greedy 加 8 次采样，与 V1b 的 G 和 U 比较。
- **判定**：heat 前缀的 G 成功率提高 ≥ 0.15，且前缀层面多 ≥ 3 个，才采用新版 R。即使采用，也只用于以后的采集，不回改本轮 V2 的数据。

## 9. 不做的事

- 不在 U_dev、U_mech、CG、D_probe、D_unseen_quick、dev140、valid_unseen 上训练；
- 不用评测结果调提示、调数据或调训练设置；
- teacher 只看公开历史和事实表，不读任务文件、游戏文件或这些任务的 teacher 示范；
- 第一轮只用一个训练 seed，有正向结果后再换 seed 复核。

## 修订

### 修订 1（2026-10-29 23:5x CDT；C4/C5 采集开始之前，非 heat 前缀的 joint-review/2 结果出来之前）

**§8 的结果**
- 运行 `reports/runs/method_v1/v2_heat_r2`：代码 1f23a37，rai GPU 3，81 次，0 作废，2.33M input / 0.18M output。
- 9 个 heat 前缀，采样成功率：U 0.222（V1b）；G（joint-review/1，V1b）0.349；G（joint-review/2）0.931。
- 差 +0.581，95% CI [0.431, 0.718]，前缀 9 / 0 / 0。按 §8 规则通过。
- 完成加热的比例从 0.37 升到 0.94。teacher 示范步数占 12.4%，仍低于 30%。

**改动**
- 预注册原文写的是新版 R "只用于以后的采集"。但 V2 的 G 采集（C4）还没有开始，这里改为：先检查新版 R 在其余任务类型上不变差，再决定 C4 用哪一版。
- **检查**：在 V1b 其余 29 个非 heat 前缀上，用 joint-review/2 跑 G，1 次 greedy 加 8 次采样（`reports/runs/method_v1/v2_nonheat_r2`）。
- **规则**：如果 29 个非 heat 前缀上 joint-review/2 的 G 成功率不低于 V1b G（joint-review/1）减 0.05，并且 38 个前缀合计的 teacher 示范步数占比 ≤ 30%，C4 就用 joint-review/2；否则 C4 用 joint-review/1，与原文一致。
- C5（teacher 接管）不涉及 R，不受影响。
- 训练组、评测和判定规则都不变。数据集清单记录 JOINT 用的是哪个 R 版本。

### 修订 2（2026-10-30 21:0x CDT；只看到 OFFLINE 第 20 次 checkpoint 的 E-sel 结果，其余 checkpoint 和其他组都还没有结果）

**E-sel 的"成功率"定义**（§6 原文只写"选成功率最高的一个"，没有说 greedy 和采样怎么合并）：
- 每个 checkpoint 在 quick 24 题上跑 4 次：1 次 greedy 加 3 次采样（seed 0、1、2）。
- **选择指标** = 先算每题在有效 run 上的成功比例，再对 24 题平均。没有作废时，就是 96 次里的成功数除以 96。
- 作废（unknown/error）的 run 不计入分母，也不算失败；如果某个 checkpoint 有作废，同时报告"作废算失败"的下界。
- 平局取较早的 checkpoint，与原文一致。greedy 和采样的数字分开报告，但不单独用于选择。

**理由**：E-sel 本身就是为选择设计的 1 次 greedy 加 3 次采样，合并后每个 checkpoint 有 96 次，比只用 3 次采样（72 次）更稳。

**噪声参考**：同一个 OFFLINE 第 20 次 checkpoint、同一张卡、同样的服务设置，greedy 跑了两次（第一次因为工具的检查失败，结果没有使用）：24 题中 22 题的成功与步数完全相同，1 题从成功变成失败。所以 greedy 单次的 ±1 题在噪声以内。
