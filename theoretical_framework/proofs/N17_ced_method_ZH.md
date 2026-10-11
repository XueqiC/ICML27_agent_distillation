# N17 · Consequence-Calibrated Experience Distillation（CED） — 推导

[← 框架总图](../README_ZH.md) · [实验](../experiments/N17_ced_method_ZH.md)

本页从框架推出 CED 的六条设计规则。每条规则先写成一个命题，说明任何声称保持或改进学习价值 $`L_E`$ 的方法必须做到什么，再给出迫使它成立的定理。这些规则是获得所述保证的必要条件，它们本身并不说明某个 CED 实例优于某个给定的 baseline。

## 设定与假设

- **少样本 agent 蒸馏.** 学生从预训练信息 $`I_{\rm pre}`$ 出发，每个任务族有 $`K`$ 条可信的 teacher 示范。
- **获取预算.** 其余允许的实验（teacher 调用、环境分支、重置）构成一笔单独核算的预算。
- **Provider.** provider $`j`$ 是一条可执行规则，它在完整学习者状态 $`z`$ 处发出反馈消息 $`Y\sim P^j_{t,z}`$（训练单元、packet、目标）。
- **配方.** 训练使用声明好的配方：更新 $`U_t`$、优化器（例如 AdamW）、时域 $`R`$ 和参照 suffix。最终学生单独部署，价值为 $`F_E(z)=J_E(\pi_z)`$。
- **response 与 provider 价值.** 同 N05，

  ```math
  g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi)),\qquad G(z,j)=\mathbb E_{Y\sim P^j_{t,z}}g_{t,z}(Y),
  ```

  带号的局部优势为 $`A_t(z,j)=G(z,j)-G(z,0)`$。

## 陈述

**(a)** **规则 1（先获取，再编译）.**

- 设 $`D`$ 为 transcript，$`D'=C(D)`$ 为它的任意随机化后处理：replay、改写、转述或合成派生。则 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。
- 因此，剩余的决策歧义只能通过新的、被允许的实验扩充 $`D`$ 来降低。获取量由下列结果决定：
  - 两世界族中 $`\mathbb E n_{\rm query}\ge\mathrm{kl}(1-\delta,\delta)/\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)`$；
  - 隐藏位模型中 $`R_{\rm info}(K,n)=\tfrac12\mathbb E\sum_{r>n}w_{(r)}`$；
  - 一般两世界模型中 $`K\kappa_D+\kappa_X\mathbb E N\ge\mathrm{kl}(1-\delta,\delta)`$。

**(b)** **规则 2（按实测后果选择）.**

- 在一个声明的族上，选择规则与学习价值的排序一致，仅当它是 $`\lbrace G(z,j)\rbrace _j`$ 的函数，或者是某个在该族上已证明对 $`G`$ 充分的统计量的函数。
- 只依赖 teacher 价值和每个学生的静态策略价值的规则，或者只依赖标签边际和平均消耗梯度的规则，都不可能在每个实例上选中更好的 provider。

**(c)** **规则 3（记录优化器状态）.**

- 决定 $`G(z,\cdot)`$ 的反馈律统计量，依赖于完整状态 $`z=(\theta,m,v,t,\dots)`$。
- 在冷启动的 Adam 状态上，它是消耗梯度的联合符号律，加上保留在 $`m`$ 中的幅值；在暖度为 $`\kappa`$ 的温热状态上，它是平均的预条件化消耗梯度，误差在 [N12(W1)](N12_optimizer_warmth_ZH.md) 的残差 $`r(y)`$ 之内。
- 因此 response 的测量及其 transport 必须按完整状态及其暖度来索引，而不能只按参数。

**(d)** **规则 4（拟合稳定的 response 模型）.**

- 考虑在特征上的 response 模型 $`c+b^\top\phi`$，特征以同时成立的容差 $`\epsilon_j`$ 被测量；response 估计 $`\widehat g`$ 满足 $`\lVert\widehat g-g\rVert_\infty\le\eta`$。
- 最小化 $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$，即可使被认证的未消解差异最小；其最优值加上 $`2\eta`$ 就是精确的界。
- 一个残差更小、但 $`\sum_j\epsilon_j\lvert b_j\rvert`$ 更大的模型，只要两项之和更大，给出的 certificate 就严格更差。

**(e)** **规则 5（计入覆盖）.** 对被选中 provider 的 $`L_E`$ 给出的保证，必须采取以下三种形式之一：

- (i) 冻结 audit 的 certificate [N14(b)](N14_frozen_audit_transport_ZH.md)，包括实测的 transport $`a_{t,j}(z)`$ 和退出概率 $`\rho`$；
- (ii) 实际状态上基于下置信界的保证 [N15(c)](N15_adaptive_audit_ZH.md)；
- (iii) 用整次训练的随机对照，比较冻结方法的总体平均，并标注为经验结果。

用未经修正的点估计来选择，或者不含 transport 的外推，都不带有这样的保证。

**(f)** **规则 6（先冻结，再测迁移）.**

- 选择器、特征、cell 和 provider 必须在评估它们的数据到来之前冻结。
- 迁移到目标任务族，只有通过目标 response 的分解 [N06(c)](N06_sufficiency_duality_ZH.md) 才有保证；否则必须在留出的任务族、配方和状态上实测。
- 方法迁移和策略迁移是两个不同的量，分开评估。

## 证明

**(a)** **规则 1.**

- [N01(d)](N01_decomposition_and_ambiguity_ZH.md)：对 $`D`$ 的随机化后处理 $`D'`$，$`\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D]`$。这来自塔性质和最大值的凸性。因此 $`\mathcal R_{\rm info}(D')=\mathbb EJ^\Pi_E-\mathbb E B(D')\ge\mathcal R_{\rm info}(D)`$。编译只能降低 $`\mathcal R_{\rm alg}`$。
- [N02(a)–(d)](N02_passive_vs_active_acquisition_ZH.md)（N02）：仅有被动数据时，最坏情形的 regret 至少为 $`\Delta`$；查询的上下界 [N02(b)](N02_passive_vs_active_acquisition_ZH.md)–[N02(c)](N02_passive_vs_active_acquisition_ZH.md) 量化了所需的获取量。
- [N03(a)](N03_few_shot_information_floor_ZH.md)、[N03(b)](N03_few_shot_information_floor_ZH.md)（N03）：给定 transcript，未揭示的比特仍是均匀的，所以 replay 已揭示的内容不改变任何东西。查询权重最大的未见比特可以达到 $`R_{\rm info}(K,n)`$。
- [N03(c)](N03_few_shot_information_floor_ZH.md)（N03）：示范的信息和查询的信息加在同一个 KL 预算里。

这四点合起来给出规则 1。

**(b)** **规则 2.**

- *必须看 response.* 由 [N05(b)](N05_continuation_response_ZH.md)，对每一次接声明 suffix 的单边界替换，两个消息律可以互换，当且仅当对声明 response 族中所有 $`g`$ 有 $`\mathbb E_Pg=\mathbb E_Qg`$。在状态 $`z`$ 处用 provider $`j`$ 替换参照，其价值恰为 $`A_t(z,j)`$；由带号恒等式 [N15(b)](N15_adaptive_audit_ZH.md)（N15），一个 provider 策略的总效应是 $`\sum_t\mathbb E_{d^S_t}A_t`$。所以任何与该族上 $`L_E`$ 排序一致的规则，必须按 $`\lbrace G(z,j)\rbrace`$ 来排序 provider。
- *静态统计量.* [N04(a)](N04_static_vs_learning_sufficiency_ZH.md)（N04）给出两个来源 $`S_1,S_2`$：它们的 teacher 价值都是 $`3/5`$，对每个混叠学生都有 $`J(\pi_p)=\tfrac12+p/20`$，但从它们做频率蒸馏分别得到 $`43/80`$ 和 $`41/80`$。一个只依赖这些公共统计量的规则，对实例 $`(S_1,S_2)`$ 和它的重标号 $`(S_2,S_1)`$ 会做出相同的选择（按位置，或按同一个随机化）。所以它至少在其中一个实例上选错，或者以 $`1/2`$ 的概率选中每一个。
- *平均梯度.* [N04(b)](N04_static_vs_learning_sufficiency_ZH.md)（N04）给出两个 packet：标签边际相同，平均梯度也相同，在 $`J(p)=p^2`$ 下学习价值相差 $`\eta^2/8`$；在 $`J(p)=1-p^2`$ 下偏好反转。这些标签统计量与 $`J`$ 无关，所以只依赖它们的规则对两个任务做出相同的选择，必在其中一个上出错。
- *packet 的组织属于 provider.* 由 [N16(a)–(b)](N16_coupling_width_ZH.md)（N16），在边际固定时重新组合单元，恰好通过联合 packet response $`H_z`$ 中不可加的部分改变学习价值。所以测得的 response 必须是对实际发出的 packet 的 response，而不是逐单元得分之和。
- *不存在通用的小统计量.* 对不受限的学习者，[N06(d)](N06_sufficiency_duality_ZH.md)（N06）表明，任何少于 $`m-1`$ 个统计量的固定族都不能决定所有 response 均值。充分统计量只在受限的族上存在，例如仿射学习者的 $`N`$ 阶以内联合矩（N07）。

**(c)** **规则 3.**

- [N11(a)–(c)](N11_adam_cold_start_ZH.md)（N11）：在冷启动状态、$`\epsilon\to0`$ 时，$`\theta_1=c-\eta\mathrm{sign}(G)`$，有界的即时效用恰好由联合符号律决定。
- [N11(e)](N11_adam_cold_start_ZH.md)（N11）：完整的下一状态包含 $`m_1=(1-\beta_1)G`$。符号序列相同的两段历史，第二步方向却相反，所以符号律不能决定后续过程。
- [N12(W1)](N12_optimizer_warmth_ZH.md)（N12）：在温热状态上，$`\lvert\psi(y)-c-b^\top G(y)\rvert\le r(y)`$，$`r`$ 由 $`\kappa`$ 和曲率 $`H_G`$ 控制。因此平均消耗梯度相等，意味着 response 差距不超过 $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$。
- [N12(W2)](N12_optimizer_warmth_ZH.md)（N12）：在冷启动状态上，均值相等的律在 $`\Theta(\eta)`$ 量级上被区分开。
- 所以起识别作用的统计量会随 $`(m,v,t)`$ 变化。
- N14 中的见证（$`\theta`$ 相同、$`(m,v)`$ 不同、下一个一阶矩为 $`\mp1/8`$）表明，参数一致的两个状态可以有不同的下一状态。因此 transport 条件 [N14(b)](N14_frozen_audit_transport_ZH.md) 必须在完整状态上陈述。

**(d)** **规则 4.** [N13](N13_stability_tradeoff_ZH.md)（N13）给出 $`W_{\mathbf{\epsilon}}(g)=\min_b[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert]`$。由 N13 的[N13(d)](N13_stability_tradeoff_ZH.md)，在同时成立的事件上，实际的律对满足 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$。对每个固定的 $`b`$，界为 $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert+2\eta`$，所以取最小化的 $`b`$ 就得到最好的被认证界。N13 的[N13(c)](N13_stability_tradeoff_ZH.md)（$`\phi=\alpha g`$，残差 $`0`$，$`W=\min\lbrace 1,\epsilon/\alpha\rbrace`$）说明，残差为零与 certificate 毫无意义可以同时出现。

**(e)** **规则 5.**

- 由 N14，冻结 audit 通过 [N14(b)](N14_frozen_audit_transport_ZH.md) 认证 $`L_E`$。其第 6 部分表明，去掉 transport 项可能对一个差距为 $`1`$ 的情形给出 $`0.0296`$ 的界，所以形式 (i) 必须包含 $`a`$ 和 $`\rho`$。
- 由 N15，在实际状态上按下置信界做 audit，得到 [N15(c)](N15_adaptive_audit_ZH.md)，即形式 (ii)。由[N15，注记 1](N15_adaptive_audit_ZH.md#注记)，按点估计选择可能有负的期望增益。
- 两种形式都不能保证单个训练 seed 的结果，而且代价都很高：由 N15 第 4 部分，每个候选、每个状态需要 $`10^3`$ 量级的配对完整尾部。
- 关于"冻结方法在声明的任务/seed 总体上的平均结果"的结论，是另一个被估计量。它由整次训练的随机对照支撑（形式 iii），必须如实标注，也不会因此变成局部 certificate。

**(f)** **规则 6.**

- *冻结.* 由 [N14(a)–(d)](N14_frozen_audit_transport_ZH.md)（N14），audit 事件对一个在 audit *之前*固定的候选集同时成立。用 audit 样本构造或训练 provider，会改变 provider 的律，certificate 就需要一次新的独立 audit（[N14，注记 3](N14_frozen_audit_transport_ZH.md#注记)）。用评估数据调过的选择器也是如此。
- *迁移.* 由 [N06(c)](N06_sufficiency_duality_ZH.md)，若目标族的每个 response 都满足 $`g=c+b^\top\phi+r`$，$`\lVert b\rVert_1\le B`$，$`\lVert r\rVert_\infty\le\tau`$，则 $`\sup_g\lvert(P-Q)g\rvert\le B\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau`$。反过来，由 [N06(a)](N06_sufficiency_duality_ZH.md)，若某个目标 response 落在 $`H_\phi`$ 之外，就存在特征匹配、却被它区分开的律 $`P,Q`$，此时仅在训练族上做 audit 无法认证目标族。在没有经过验证的目标族 $`(B,\tau)`$ 时，迁移是一个要在留出族上实测的经验量。
- *两种迁移.* 方法迁移评估 $`L_{E'}(S)`$，其中 $`S`$ 是冻结的选择流程，用在新任务族 $`E'`$ 及其自己的 $`K`$ 条示范上；它的 response 类是通过更新定义的 $`\mathcal G_{E'}`$。策略迁移评估已蒸馏学生 $`z`$ 的 $`J_{E'}(\pi_z)`$，不再做任何更新；它的检验函数是 $`F_{E'}`$。两者是检验类不同的不同泛函，一个的界不能推出另一个的界，所以分开报告。$`\square`$

## 注记

**推导出的流程.** 这些规则组合成一个循环：

1. 暴露剩余歧义，在声明的预算内获取新证据（规则 1）。用一本账把新证据与 replay、合成派生分开记录。
2. 把证据编译成有效单元，构建一个包含原始 teacher 参照的有限 provider 菜单。
3. 在可恢复、并记录了暖度的完整状态上，测量每个 provider 的实际"更新加尾部" response（规则 2–3）。
4. 在 [N13](N13_stability_tradeoff_ZH.md) 的目标下，拟合系数有界的 response 模型（规则 4）。
5. 用规则 5 的形式 (i) 或 (ii) 选择；或者经验地选择，再用形式 (iii) 验证。
6. 冻结，然后在留出的任务族、配方和状态上，分别评估方法迁移和策略迁移（规则 6）。

**受控实验补充了什么**（见实验页）：

- 静态更新统计量不能替代实测 response，这加强了规则 2。
- 短的实测尾部是好的 proxy，除非部署任务带阈值；所以尾部长度本身就是被估计量的一部分。
- moment warm-up 会压缩 provider 之间的差异，但不会让静态统计量变得有预测力。按规则 3，暖度改变的是*哪一个*统计量起作用，而不是让某个静态统计量变得充分。

## 依赖与来源

- 依赖 [N01](N01_decomposition_and_ambiguity_ZH.md)–[N06](N06_sufficiency_duality_ZH.md) 与 [N11](N11_adam_cold_start_ZH.md)–[N16](N16_coupling_width_ZH.md)，具体见各条规则中的引用。

**出处：** [v0.14.1 §13](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#13-the-derived-few-shot-method-consequence-calibrated-experience-distillation); [v0.15 §6](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#6-method-consequence-an-optimizer-aware-instance-of-ced); [warm-Adam §4](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md#4-how-warmth-evolves-and-a-free-way-to-warm-the-state); [OTLR §5](../../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#5-using-the-statistic-at-the-actual-states) — v0.14.1 §13; addendum §6; warm-Adam §4; OTLR §5. [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §13，[Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §5–§6，[CED_Method_EN.md](../../docs/direction/v014_handoff/CED_Method_EN.md) §§3–6，以及 [Warm_Adam_Response_EN.md](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md) §§2–4。
