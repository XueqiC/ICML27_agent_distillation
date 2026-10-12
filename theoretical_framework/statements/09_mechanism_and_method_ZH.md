# §9 机制与方法 — 精确陈述

[← Framework](../README_ZH.md) · [Proofs](../proofs/09_mechanism_and_method_ZH.md) · [Experiments](../experiments/09_mechanism_and_method_ZH.md) · [English](09_mechanism_and_method.md)

### §9 · 机制与方法：精确陈述

[证明](../proofs/09_mechanism_and_method_ZH.md) · [实验](../experiments/09_mechanism_and_method_ZH.md)

**本节做什么.**

- 分离出一个带精确零假设的学习后果机制：边缘分布固定的反馈单元怎样联合组织（[Proposition 9.1](../proofs/09_mechanism_and_method_ZH.md#s9-1)）。
- 把 §2–§8 推出的方法 Consequence-Calibrated Experience Distillation（CED）写成编号的步骤：每一步都引用支撑它的结论，每项输出都带有明确的保证，或标为经验结论（[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)）。
- 推导这一流程背后的六条设计规则（[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)），并把受控证据与计划中的真实模型检验分开（[Remark 9.4](../proofs/09_mechanism_and_method_ZH.md#s9-4)）。

**设定.**

- 训练 packet 的两个槽位分别从有限单元集 $`\mathcal Y_1,\mathcal Y_2`$ 中填充，边缘分布 $`\mu,\nu`$ 严格为正。耦合多面体（coupling polytope）$`\mathcal C(\mu,\nu)`$ 是行和为 $`\mu`$、列和为 $`\nu`$ 的非负表的集合。
- $`H_z(i,j)`$ 是在完整状态 $`z`$ 上对联合 packet $`(i,j)`$ 做实际更新、再走参考尾段之后的期望最终回报，即以 packet 为消息的一轮边界 response。耦合宽度为 $`w(H_z)=\max_\Gamma\langle\Gamma,H_z\rangle-\min_\Gamma\langle\Gamma,H_z\rangle`$，$`\Gamma`$ 取遍 $`\mathcal C(\mu,\nu)`$。
- 候选 feedback law（provider）$`j\in\lbrace0,\dots,M\rbrace`$ 发出消息 $`Y\sim P^j_t(\cdot\mid z)`$，候选 $`0`$ 是参考。它的平均 response 为 $`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$，带号优势为 $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$。

#### Proposition 9.1 · 耦合宽度

**(a)** $`w(H_z)=0`$ 当且仅当存在向量 $`a,b`$ 使 $`H_z(i,j)=a_i+b_j`$。

**(b)** 对所有耦合 $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$，

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty ,
```

且不等式可以严格成立。

**(c)** 存在禁止配对时，可行宽度为零并不推出可加。

**(d)** 在同一参数处、对所有 packet 的并集、以相同逐样本权重做一步 SGD，对重组不变。连续多步则不然；对二次损失，差别为 $`\eta^2H_{P_2}g_{P_1}(\theta_0)`$。

*推导要点.*

- (a) 可加表与每个耦合的内积都是 $`\mu^\top a+\nu^\top b`$。反过来，乘积耦合 $`\mu\nu^\top`$ 在内部，所以每个矩形移动 $`e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$ 在两个方向上都可行；宽度为零迫使 $`H_z`$ 的每个矩形和都为零，这就是可加性。
- (b) 两个耦合之差的行和与列和为零，所以它消去所有可加表，且 $`\ell_1`$ 范数不超过 2；由 Hölder 不等式得到上界。
- (c) 禁止一个配对后，该配对的指示表对每个可行耦合都不可见，但它不可加。
- (d) 一步更新是在同一参数处把逐样本梯度相加，与划分无关。两步时，第二个 packet 的梯度在第一次更新之后才计算，于是出现交叉项 $`\eta^2H_{P_2}g_{P_1}`$。

*示例.* 取 $`\mu=\nu=(0.9,0.1)`$，$`H=e_{11}`$。每个耦合都有 $`\Gamma_{11}=x\in[0.8,0.9]`$，宽度为 $`0.1`$；最佳可加逼近留下大小为 $`1/4`$ 的棋盘形残差，所以界 (b) 为 $`0.5`$。界成立但不紧。

*用到* [Definition 1.4](../README_ZH.md#s1-4)、[Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)。*被用于* [Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)（第 3 步）、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)（Rule 2）。

[证明](../proofs/09_mechanism_and_method_ZH.md#s9-1) · [实验](../experiments/09_mechanism_and_method_ZH.md#c9-1) · 已证明

#### Algorithm 9.2 · Consequence-Calibrated Experience Distillation（CED）

*输入：* 初始完整学习器状态、声明的训练配方与部署任务、每个任务族 $`K`$ 条可信示范、$`n`$ 次允许实验的预算、候选生成器与参考律 $`P^0`$，以及 audit 预算（检查点、$`r`$ 个 CRN 副本、surrogate $`\tilde F`$、少量全尾段校准 audit、置信水平 $`\delta`$）。*输出：* 蒸馏后的学生、证据台账、每个检查点的分数与半径以及被选中的 feedback law（是否被认证）、冻结的选择器，以及分开报告的方法迁移与策略迁移结果。

1. **记录完整学习器状态**：每个检查点都保存，并记录 warmth（[Remark 5.11](../proofs/05_learner_statistics_ZH.md#s5-11)）；学习器的统计量取决于 optimizer state（[Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)）。
2. **先获取，再编译**：在 pilot 中估计学习器相对权重 $`W_s`$ 与 teacher 覆盖 $`k_s`$，剩余查询按 water-filling $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ 分配（[Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)），即 [Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5) 的 c-optimal design；维护证据台账，只有新证据计数（[Proposition 2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4)）。
3. **编译候选**：以 packet 律的形式给出，因为分组是 response 的一部分（[Proposition 9.1](../proofs/09_mechanism_and_method_ZH.md#s9-1)）。
4. **记录每个候选的学习器统计量并控制幅度**：平均 optimizer 增量、裁剪激活率，以及逐坐标的 $`\lambda`$ 与 $`\rho`$；越过 [Corollary 7.2](../proofs/07_second_moment_channel_ZH.md#s7-2) 价值峰值 $`\lambda^\star`$ 的候选被标记，它们的排序只依赖第 5 步（[Proposition 7.3](../proofs/07_second_moment_channel_ZH.md#s7-3)、[Remark 7.5](../proofs/07_second_moment_channel_ZH.md#s7-5)）。
5. **在实际状态上测量后果**：用带半径的 CRN 配对替换分数（[Theorem 8.8](../proofs/08_certificate_ZH.md#s8-8)），并用全尾段配对 audit 校准（[Theorem 8.3](../proofs/08_verification_ZH.md#s8-3)、[Remark 8.11](../proofs/08_certificate_ZH.md#s8-11)）；一阶分数只在 margin 以上充当筛选（[Corollary 8.9](../proofs/08_certificate_ZH.md#s8-9)）。
6. **按下置信界选择、训练、重新 audit、冻结、检验迁移**：若没有正的下界就保留参考；在未 audit 的状态上使用选择时计入 transport（[Theorem 8.2](../proofs/08_verification_ZH.md#s8-2)）；方法迁移与策略迁移分开评测。

*保证.* (G1) 被认证的选择以至少 $`1-\delta`$ 的概率不降低被 audit 状态上一轮替换的任务价值（[Theorem 8.8](../proofs/08_certificate_ZH.md#s8-8)）；(G2) 若在实际状态上用全尾段 audit，整次训练满足 $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$（[Theorem 8.3](../proofs/08_verification_ZH.md#s8-3)）；(G3) 获取在 [Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5) 的常数因子内达到学习器相对的信息界；(G4) 其余都是经验结论，并如此标注。使用先验分数区间时，证书半径可能超过典型差距，这一点尚未解决（[Remark 8.12](../proofs/08_certificate_ZH.md#s8-12)）。

*示例.* 设参考是 $`K`$ 条示范的 replay，第二个候选在学生的长尾状态上加入 teacher 修正。对访问足够频繁、处于 [Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11) 频繁访问区间的状态，warm Adam 下学习器相对权重按访问率 $`d(s)`$ 缩放，而不像 SGD 那样按 $`d(s)^2`$ 缩放。所以相对常见状态，长尾状态的权重更大，第 2 步分给它们的查询比 SGD 式分配更多；随后第 5 步按实测后果判断：编译成 packet 的修正，在当前状态上是否真的优于 replay。

*用到* 各步所引的 §2–§8 结论。*被用于* 计划中的真实模型检验（[Remark 9.4](../proofs/09_mechanism_and_method_ZH.md#s9-4)）。

[证明](../proofs/09_mechanism_and_method_ZH.md#s9-2) · [实验](../experiments/09_mechanism_and_method_ZH.md#c9-4) · 推导出的设计规则

#### Remark 9.3 · 六条设计规则

1. **先获取，再编译.** 后处理不能降低信息 regret；即使 teacher 最优，被动数据也可能留下缺口；获取多少、在哪里获取由学习器决定（[Proposition 2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4)、[Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)、[Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3) 与 [6.5](../proofs/06_learner_relative_information_ZH.md#s6-5)、[Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)）。
2. **按实测后果选择.** response 是不变量（[Theorem 3.2](../proofs/03_learning_consequence_ZH.md#s3-2)）；静态价值与平均梯度确定不了排序（[Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1)）；task-blind 统计量、只看参数的分数与余弦分数不一致（[Proposition 7.3](../proofs/07_second_moment_channel_ZH.md#s7-3)）；在受控学习器上测试过的其他静态 proxy 未能可靠地选对。
3. **匹配学习器的统计量，而不是原始梯度.** 一阶统计量是完整状态的平均增量（[Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)、[Theorem 5.10](../proofs/05_learner_statistics_ZH.md#s5-10)）；warm Adam 下，相对 running RMS 的幅度也是它的一部分（[Theorem 7.1](../proofs/07_second_moment_channel_ZH.md#s7-1)、[Corollary 7.2](../proofs/07_second_moment_channel_ZH.md#s7-2)）。
4. **拟合系数稳定的 response 模型.** 一个特征上的测量误差，代价是它的系数大小（[Theorem 8.1](../proofs/08_verification_ZH.md#s8-1)）。
5. **计入覆盖；用配对替换认证；一阶 audit 只用在 margin 占优处**（[Theorem 8.2](../proofs/08_verification_ZH.md#s8-2)、[8.3](../proofs/08_verification_ZH.md#s8-3)、[8.8](../proofs/08_certificate_ZH.md#s8-8)，[Corollary 8.9](../proofs/08_certificate_ZH.md#s8-9)）。
6. **先冻结，再检验迁移.** audit 事件只对事先固定的候选同时成立；迁移只能通过目标 response 的已核实分解来保证（[Theorem 4.2](../proofs/04_complexity_of_sufficiency_ZH.md#s4-2)(c)）；方法迁移与策略迁移是不同的泛函。

这些规则是所述保证的必要条件；它们本身并不说明某个 CED 实例优于某个给定的 baseline。

[证明](../proofs/09_mechanism_and_method_ZH.md#s9-3) · [实验](../experiments/09_mechanism_and_method_ZH.md#c9-4) · 推导出的设计规则

#### Remark 9.4 · 证据的范围

- Proposition 9.1 与选择规则的证据，是精确有限计算和一个受控小网络。不作任何真实模型的结论。
- 计划中的真实模型检验：R0（pilot 的再分析：敏感度加权的 warmth 与 response 的不可加性）；R1（Theorem C：同状态、同预算下，用 CRN 证书给四个候选 feedback law 排序，预先注册后再训练到底）；R2（Theorem A：按学习器相对 water-filling 分配查询，对照均匀分配与 SGD 式分配）；R3（Theorem B：余弦分数相同、幅度不同的消息批）。

[证明](../proofs/09_mechanism_and_method_ZH.md#s9-4) · [实验](../experiments/09_mechanism_and_method_ZH.md) · 未解决

---
