# 真实模型实验 R0–R3：预注册（2026-10-12）

**状态.**
- 用户于 2026-10-11 22:20 CDT 批准 R0–R3（约 40 B200 GPU 小时）；R2 的查询预算由我选定。
- 本文在任何作业提交之前写定：配置、预测、判据、算力与 teacher 预算。运行后不改动本文；偏离之处另写在结果页并注明。
- 对应理论：框架 [§6 Theorem A](../../theoretical_framework/proofs/06_learner_relative_information_ZH.md)、[§7 Theorem B](../../theoretical_framework/proofs/07_second_moment_channel_ZH.md)、[§8 Theorem C](../../theoretical_framework/proofs/08_verification_ZH.md)，以及 [§5](../../theoretical_framework/proofs/05_optimizer_state_ZH.md) 的 warmth 与裁剪结果。

---

## 0. 共同设定

- **配方.** 主表配方：ALFWorld K20 seed 0 的 Seed-SFT；student `google/gemma-4-12B-it`，LoRA r16/α32、dropout 0；AdamW，lr 1e-5 常数，$`\epsilon=10^{-8}`$；全局梯度裁剪 1.0；每个 update 至少 512 个监督 token（中位数 835）；共 469 个 update。
- **起点状态 $`z_{10}`$.** pilot cold 臂在 104,070 token（第 115 个 update，记为 p10）的 checkpoint：adapter、AdamW 的 $`m,v`$、步数与 CPU RNG。cold 臂与主表 k20s0 的 adapter 逐字节相同；dropout 为 0，恢复是确定性的（pilot 已验证恢复后与真实训练逐位一致）。
- **surrogate $`\tilde F`$.** 留出 teacher 成功轨迹上的平均 teacher-forced log-likelihood：K100 s0 中不属于 K20 s0 的 80 条（每种任务类型 13–14 条）。只用于打分，不进入任何训练。
- **评测集 dev300.**
  - 组成：dev140（valid_seen）加 160 个 train split 的留出游戏。
  - 这 160 个游戏与 K100 s0 的所有 group 不相交，与下面的 rollout 池也不相交。
  - **sealed 的 valid_unseen 永不使用**。
- **rollout / 查询池.** 200 个 train split 游戏，按 group 与 K100 s0、dev300 不相交。所有学生 rollout 与 teacher 查询只在这个池上进行。
- **噪声参照.** 把 cold p40 adapter 在第二个 server 上再评一次 dev300，报告翻转数。此前在 dev140 上约为 ±4 题。
- **单 seed.** 探索阶段每个配置只跑一个 seed。
- **CRN 配对替换分数（Theorem 8.8 的测量）.**
  - 对候选 $`j`$，第 $`\ell`$ 个副本做三件事：从 $`z_{10}`$ 出发；把下一个参考 update 的 batch 换成候选 $`j`$ 的一个 batch（候选之间用共同的随机下标抽取）；再走 $`T`$ 个参考 update，数据顺序由种子 $`\omega_\ell`$ 对 K20 计划重新洗牌得到。最后计算 $`\tilde F`$。
  - 同一副本内，所有候选共用 $`\omega_\ell`$。$`Z^{jk}_\ell=\tilde F_\ell(j)-\tilde F_\ell(k)`$，$`\hat S^{jk}`$ 为副本均值。
  - 半径：empirical Bernstein，$`\delta=0.05`$。严格版本用 $`\tilde F`$ 的先验区间；诊断版本用观测区间，并明确标注为诊断。
  - **尾段长度取 $`T=10`$.** 每个尾段约 1.4 分钟（11 个 update 加一次 surrogate 打分）。$`T`$ 截断带来的偏差由 $`\varepsilon_{\rm sur}`$ 吸收（Remark 8.11）。

---

## R0：pilot 的再分析（约 4 GPU 小时，不调用 teacher）

**目的.** 补上审查指出的两个缺口：用敏感度加权的量代替 warmth 分位数（[Remark 5.11](../../theoretical_framework/proofs/05_optimizer_state_ZH.md#s5-11)）；测 **response** 的非可加性，而不只是 update map 的（[Proposition 5.9](../../theoretical_framework/proofs/05_optimizer_state_ZH.md#s5-9)）。

**R0a 敏感度加权的冻结分母误差.**
- 在 pilot 的 9 个 probe 状态上，逐坐标重算 $`\kappa_i`$ 与 $`u_i`$。敏感度用 $`\hat L_i=\lvert\partial\tilde F/\partial\theta_i\rvert`$ 作 proxy，并注明它是 proxy。
- 报告加权误差 $`E_{\rm fd}=\eta\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$、其中最大的单项，以及一阶项 $`\lvert\sum_i\hat L_i\Delta\theta_i\rvert`$。
- **判据（预先写定）.** 只有当 $`E_{\rm fd}\le0.1\times`$ 一阶项时，才说"在该状态上 warm Adam 的冻结分母描述成立"。否则只报告数值，不下这个结论。

**R0b response 的非可加性.**
- 状态：update 8 与 p10。各取 4 对互不相交的 batch $`(A,B)`$，与 pilot probe 相同。
- 对分支 $`\lbrace0,A,B,A\cup B\rbrace`$ 各做 CRN 尾段（$`T=10`$，$`r=4`$），记 $`\delta\tilde F(X)=\tilde F(X)-\tilde F(0)`$。
- 定义 $`\nu_{\rm resp}=\lvert\delta\tilde F(A\cup B)-w_A\delta\tilde F(A)-w_B\delta\tilde F(B)\rvert/\lvert\delta\tilde F(A\cup B)\rvert`$，$`w`$ 为 token 权重。
- **预测.** 裁剪在 p10 上始终激活，每个 batch 的梯度被归一化，所以 $`\nu_{\rm resp}`$ 显著大于 0。具体地，4 对的中位数落在已测 update map 的 $`\nu`$（裁剪 AdamW 在 p10 为 0.631）的 $`\pm0.25`$ 内。
- **判据.** 中位数大于 0.2 记为"response 层面非可加"。与 update map 的差距如实报告。

**代价.** R0a 约 1 GPU 小时（9 个状态，每个一次 surrogate 梯度）；R0b 共 2×4×4×4 = 128 个尾段，约 3 GPU 小时。

---

## R3：Theorem B 在真实模型上（约 3 GPU 小时，不调用 teacher）

**目的.** 检验 [Corollary 7.2](../../theoretical_framework/proofs/07_second_moment_channel_ZH.md#s7-2) 的幅度定律和 [Remark 7.4](../../theoretical_framework/proofs/07_second_moment_channel_ZH.md#s7-4) 的裁剪机制。还要检验只看参数的分数与余弦分数（[Proposition 7.3](../../theoretical_framework/proofs/07_second_moment_channel_ZH.md#s7-3)）在真实模型上的表现。

**设计.**
- 在 p10 上固定一个参考 batch $`b`$。消息 = 用 $`b`$ 的损失乘以倍数 $`c`$ 做一步更新，$`c\in\lbrace0.5,1,2,4,8\rbrace`$。
- 两种设定：
  - 这一步**关闭裁剪**，让幅度真正变化；
  - 保持裁剪（真实配方），只在 $`c\in\lbrace1,4\rbrace`$ 上跑。
- 对照为零梯度消息。之后走 $`T=10`$ 的参考尾段，$`r=16`$ 个 CRN 副本，测 $`\delta\tilde F(c)`$。

**事先计算的预测（运行前用 $`z_{10}`$ 的 optimizer state 算好并写入结果页）.**
- 对每个坐标，取 $`\rho_i=\lvert m_i\rvert/\sqrt{v_i}`$、$`\lambda_i(c)=c\,g_i\,\mathrm{sign}(m_i)/\sqrt{v_i}`$、$`\alpha_i=-\partial_i\tilde F\,\mathrm{sign}(m_i)`$，代入 Corollary 7.2(a) 的 mean-field 公式（$`T=10`$）后求和，得到预测曲线 $`\widehat{\delta F}(c)`$ 与它的峰值位置 $`\hat c^\star`$。
- 关闭裁剪时：
  - 实测曲线关于 $`c`$ 凹；
  - 峰值位置在 $`\hat c^\star`$ 的 2 倍以内；
  - 若 $`\widehat{\delta F}`$ 在网格内变负，实测也变负，且变负点在预测的 2 倍以内。
- 保持裁剪时：pre-clip 范数约为 4，$`c=1`$ 与 $`c=4`$ 的 consumed gradient 都被裁到范数 1，所以两者的 $`\delta\tilde F`$ 之差不超过 CRN 半径。
- 排序器：
  - 余弦分数对所有 $`c`$ 相同，无法区分；
  - 只看参数的分数 $`\langle\nabla_\theta\tilde F,\Delta\theta(c)\rangle`$ 在 Proposition 7.3(b) 的阈值以下随 $`c`$ 递增；
  - 若实测价值在大 $`c`$ 处下降，这两种分数在相应的 $`c`$ 对上排错。报告排错对数。

**判据.** 关闭裁剪时，凹性与峰值位置两项都满足，记为"支持 Corollary 7.2"；只满足一项，记为"部分支持"；都不满足，记为"不支持"。结论按实际情况写。

**代价.** 7 种消息加 1 个零消息，共 8 种，乘以 16 个副本，共 128 个尾段，约 3 GPU 小时。

---

## R1：Theorem C：同状态、同预算的反馈排序（约 10 GPU 小时；teacher ≤ 640 次调用）

**候选反馈律（都在 $`z_{10}`$ 上，即一个 update 所用的训练 episode 的分布）.**
- $`P_1`$ **replay**：K20 示范。这就是参考律。
- $`P_2`$ **teacher 修正**（DAgger 式）：用 p10 学生在池中 40 个游戏上做 greedy rollout，在学生访问的每个状态上查询 teacher 的动作，得到"学生状态 + teacher 动作"的训练样本。上限 600 次调用。
- $`P_3`$ **自生成成功轨迹**：p10 学生在池中 80 个游戏上 rollout，只保留环境判定成功的轨迹，不调用 teacher。
- $`P_4`$ **teacher 改写**：teacher 改写每条 K20 示范的 thought，动作保持不变。上限 40 次调用。

**证书.**
- CRN 配对替换，$`r=64`$，$`T=10`$，共 4×64 = 256 个尾段，约 6 GPU 小时。
- 报告每一对的 $`\hat S`$、严格半径（先验区间）与诊断半径（观测区间）。
- **在任何续训之前，把按 $`\hat S`$ 给出的排序写入 PROJECT_STATE**，这就是预先注册的排序。

**任务结果（用于校准与最终比较）.**
- 从 $`z_{10}`$ 出发续训到 p40 的 token 预算。$`P_2`$–$`P_4`$ 各一个臂：剩余 update 中每 4 个有 1 个的 batch 换成该候选的 batch，其余照参考计划。参考臂就是现有的 cold p40。
- 3 次续训约 1.6 GPU 小时；4 个 adapter 的 dev300 评测约 1.3 GPU 小时；rollout 约 1 GPU 小时。
- 用这些全尾段结果估计 $`(\alpha,\varepsilon_{\rm sur})`$（Remark 8.11）。只有 3 个候选对，校准迁移在这里**不可检验**，结果页如实写明（Remark 8.12 的开放问题）。

**对照排序器（全部是我们按定义实现的分数，不称为 baseline）.**
- 余弦分数（LESS 式：候选 batch 的梯度与 $`\tilde F`$ 的梯度的余弦）；
- 梯度范数；
- 只看参数的一阶分数 $`\langle\nabla_\theta\tilde F,\Delta\theta\rangle`$；
- 静态 proxy：来源的 teacher 一致率、学生在来源样本上的平均 log-likelihood。
- 官方 LESS / GRACE 的完整代码要在 Gemma-4-12B 上跑完整流程，本轮不做；如果以后要作为 baseline 报告，单独按"只用完整官方代码"的规则处理。

**预测（理论依据见括号）.**
- $`\hat S`$ 的排序：$`P_2`$ 第一。依据：Theorem A 的覆盖模型，查询补的是学生到达而 teacher 不覆盖的状态（Proposition 6.10）。
- $`P_4`$ 与 $`P_1`$ 的差在诊断半径之内。依据：改写是 transcript 的后处理，不增加信息（Corollary 6.7）。它只可能通过优化改变结果，预计很小。
- $`P_3`$ 位于两者之间。
- 最终 dev300：$`P_2`$ 的成功数高于 $`P_1`$。

**判据.**
- 证书排序与最终 dev300 排序的 Kendall $`\tau\ge0.67`$（4 个候选中至多一对颠倒）。
- 被认证（严格或诊断）的候选对，在 dev300 上全部同号；但只对 dev300 差距超过噪声参照的候选对做这个要求。
- 报告各对照排序器的 Kendall $`\tau`$。

---

## R2：Theorem A：查询分配（约 7 GPU 小时；teacher ≤ 1,500 次调用）

**状态簇.**
- 用 p10 学生在池中 120 个游戏上的 rollout（与 R1 共用一部分）收集访问到的状态。
- 按（任务类型 × 步数段 {0–4, 5–9, 10–19, 20+}）分成至多 24 个簇。
- 簇 $`s`$ 的学生访问频率记为 $`d(s)`$，teacher 覆盖 $`k_s`$ 为 K20 示范在该簇中的访问数。

**学习器权重（Proposition 6.11 的闭式，作为可计算的 proxy）.**
- SGD 权重 $`W^{\rm SGD}_s\propto d(s)^2\sigma_s^2`$；warm Adam（频繁访问区间）权重 $`W^{\rm Adam}_s\propto d(s)\sigma_s^2`$。
- $`\sigma_s^2`$ 是簇内单个样本梯度的二阶矩：在 $`z_{10}`$ 上对每簇约 10 个样本做前向反向，便宜。
- 真实模型的参数在簇之间共享，Proposition 6.11 的 state-owned 假设不成立，所以这里检验的是"闭式权重作为分配规则是否有用"，不是闭式本身是否精确。

**各臂.**
- A0：不查询（用 replay 补齐）；
- A1：均匀分配到学生访问的状态（DAgger 式）；
- A2：按 $`W^{\rm SGD}`$ water-filling；
- A3：按 $`W^{\rm Adam}`$ water-filling，即学习器相对的分配：$`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$。

**查询预算.**
- $`n\in\lbrace100,400\rbrace`$。小预算时分配规则的差别最明显；大预算用来看差别是否收窄。
- 被查询的状态在簇内随机抽取。查询结果按状态缓存、在各臂之间共享，所以 teacher 调用总数不超过各臂所需状态的并集。上限 1,500 次。

**训练与评测.**
- 每个臂从 $`z_{10}`$ 续训到 p40 的 token 预算，查询得到的样本按固定比例插入剩余计划。共 7 个臂（A0 加 3 个分配 × 2 个预算），约 3.7 GPU 小时。
- dev300 评测约 2.3 GPU 小时；簇统计约 0.5 GPU 小时。

**预测（运行前计算并写入结果页）.**
- 对每个臂，用 Proposition 6.10 的 $`\sum_sW^{\rm Adam}_s/(k_s+n_s)`$ 算预测误差，并给出各臂的预测顺序。
- 预期 $`n=100`$ 时 A3 ≥ A1 ≥ A2 > A0；$`n=400`$ 时 A1–A3 之间的差收窄。

**判据.**
- 主判据：$`n=100`$ 时 A3 − A1 超过噪声参照，并且配对 McNemar 检验 $`p<0.05`$。
- 次判据：7 个臂上，预测误差与 dev300 成功数的 Spearman 秩相关为负，且 $`\lvert\rho\rvert\ge0.6`$。
- A3 对 A2 的比较直接检验 Theorem A 的学习器分离：同样的数据预算，按 Adam 权重分配是否优于按 SGD 权重分配。

---

## 预算与顺序

| 实验 | GPU 小时（B200） | teacher 调用上限 | 顺序 |
|---|---:|---:|---|
| 噪声参照 | 0.3 | 0 | 1 |
| R0 | 4 | 0 | 1 |
| R3 | 3 | 0 | 2 |
| R1 | 10 | 640 | 3 |
| R2 | 7 | 1,500 | 4 |
| 合计 | 约 24（硬上限 40） | 2,140 | |

- **teacher 规则.**
  - 每个 collector 在客户端设硬上限，超过即停止，不重试；
  - 按状态哈希去重并缓存；
  - 每批先读结算后的 ledger 再继续；
  - HTTP 状态码不是 200 时停止并报告，不把空结果当成发现。
- **算力规则.**
  - 先在 rai 上用小模型做 CPU 或单卡冒烟测试；
  - 在 hpg 上用干净 worktree 的代码树；
  - 作业在 GPU 空闲阶段只占首卡；
  - pending 作业不超过 4 个；
  - 先查队列再选账户。

## 需要先实现的代码（Codex 实现，我审查）

1. CRN 尾段工具：从 checkpoint 恢复；用给定 batch 做一步消息更新（损失倍数、是否裁剪可选）；按种子洗牌的参考计划走 $`T`$ 步；计算 surrogate 打分；多卡分片；做确定性测试。
2. 逐坐标的 warmth 与敏感度加权导出（R0a）。
3. 留出 surrogate 集（K100 s0 \ K20 s0）、dev300 与 rollout 池的 manifest（按 group 不相交；带 sealed 保护）。
4. p10 学生在池上的 rollout（复用评测 harness）。
5. teacher 修正与改写的 collector，带硬上限。
6. 从恢复的 checkpoint 续训，计划中混入候选或查询样本。
7. R2 的簇统计、water-filling 分配，以及 R1 的对照排序器。
