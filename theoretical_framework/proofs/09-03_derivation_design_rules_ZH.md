# Remark 9.3 · 设计规则的推导

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/09-03_derivation_design_rules_ZH.md) · [证明](09-03_derivation_design_rules_ZH.md) · [实验](../experiments/09-03_derivation_design_rules_ZH.md) · [English](09-03_derivation_design_rules.md)

**本页推导什么.** 对 [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) 背后的六条设计规则中的每一条，给出迫使它成立的先前陈述。每一条推导要么应用一个该规则所遵守的已证明的界，要么借助一个先前的陈述给出一个实例：违反该规则的方法在其中失去保证。这些规则是获得所述保证的必要条件，而不是 CED 优于任何 baseline 的证明。

**本页用到的设定与记号.**

- *学习器与候选.* 完整学习器状态 $`z=(\theta,o)`$，即参数加 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)）；候选 feedback law $`P^0,\dots,P^M`$ 发出消息 $`Y\sim P^j_t(\cdot\mid z)`$，$`P^0`$ 是参考；一条消息产生完整状态的增量 $`\Delta`$。
- *response.* $`g_{t,z}(Y)`$ 是一轮边界 response（[Definition 1.4](../README_ZH.md#s1-4)）；$`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$；$`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$ 是在 $`z`$ 上把一轮参考换成 $`j`$、之后走参考尾段的价值；$`d^S_t`$ 是协议 $`S`$ 下第 $`t`$ 轮状态的分布。
- *信息.* $`D`$ 是 transcript；$`\mathcal R_{\rm info}(D)`$ 是它的信息 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2)）；$`\mathfrak n_{K,n}`$ 是 $`K`$ 条示范与 $`n`$ 次查询下的学习器相对信息范数，$`c=\sum_tJ_t^*g_t`$ 是学习器相对的目标向量：学习器相对泛函的梯度，把学习器平均增量关于环境参数的 Jacobian $`J_t`$ 与任务方向 $`g_t`$ 合在一起（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。
- *平均增量.* $`\mu_t^P(z)=\mathbb E_P\Delta`$；warmth 度量一个 Adam 状态离它的冻结分母描述有多远（[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11)）；$`p_{\rm act}`$ 是 [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9) 中的裁剪激活率，$`c`$ 是裁剪阈值。
- *response 模型.* 特征 $`\phi_1,\dots,\phi_d`$，矩阵为 $`\Phi`$；response 模型 $`c+b^\top\phi`$，常数为 $`c`$、系数为 $`b`$；容差 $`\epsilon_j`$；response 估计 $`\widehat g`$ 满足 $`\lVert\widehat g-g\rVert_\infty\le\eta`$；$`\mathrm{osc}(h)=\max h-\min h`$。
- *记号说明.* 与所引陈述一致，字母 $`c`$ 在 Rule 1 中是目标向量，在 Rule 3 中是裁剪阈值，在 Rule 4 与 Rule 6 中是 response 模型的常数；含义由所在的规则确定。
- *迁移.* 方法迁移评测在新任务族 $`E'`$ 上用它自己的 $`K`$ 条示范重跑冻结协议 $`S`$ 得到的 $`L_{E'}(S)`$；策略迁移评测蒸馏好的学生 $`z`$ 不再更新时的 $`J_{E'}(\pi_z)`$。

**陈述（[Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3)）.** Algorithm 9.2 实现了六条规则，每条都是获得相应保证的必要条件：

- **Rule 1（先获取，再编译）.** transcript 无法确定、但与决策相关的事实，只能由新的允许实验提供；获取多少、在哪里获取，由学习器决定。
- **Rule 2（按实测后果选择）.** 在声明的族上，选择规则与学习价值的排序一致，仅当它是平均 response $`\lbrace\bar g_t(z,j)\rbrace_j`$ 的函数，或是某个在该族上已证明对它们充分的统计量的函数。
- **Rule 3（匹配学习器的统计量，而不是原始梯度）.** 候选在一阶上必须匹配的，是完整学习器状态的平均增量在 continuation value 梯度方向上的部分。因此 response 及其统计量要按完整状态及其 warmth 来索引。
- **Rule 4（拟合稳定的 response 模型）.** 特征以容差 $`\epsilon_j`$ 测得时，response 模型 $`c+b^\top\phi`$ 按最小化 $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$ 来拟合。
- **Rule 5（计入覆盖；用配对替换做认证；一阶 audit 只用在 margin 占优处）.** 对被选中候选的学习价值给出的保证，取以下四种形式之一：
  - (i) 冻结 audit 的证书，含实测的 transport 与退出概率；
  - (ii) 实际状态上、基于全尾段配对 audit 的下置信界保证；
  - (iii) 被 audit 状态上、带校准 surrogate 的 CRN 配对替换证书，沿训练推广只能靠 transport 项或重新 audit；
  - (iv) 冻结方法的整次训练随机对照，标注为经验结论。
- **Rule 6（先冻结，再检验迁移）.** 选择器、特征、cell、surrogate 与候选都在评测它们的数据之前冻结；迁移只能通过目标 response 的已核实分解来保证，否则就要测量。方法迁移与策略迁移分开报告。

这些规则本身并不说明某个 CED 实例优于某个给定的 baseline。

**推导.**

**Rule 1（先获取，再编译）.**

- 设 $`D'=C(D)`$ 是 transcript 的任意随机化后处理：replay、改写、转述或合成派生。则 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$（[Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)）。由 [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2) 的分解（regret = 信息项 + 算法项 + teacher 项），编译只能降低算法项。
- 即使 teacher 最优，被动数据也可能留下缺口：在两世界族中最坏 regret 至少为 $`\Delta`$（[Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)(a)）。更一般地，若两个环境的示范分布相同、学习器相关泛函不同，则对每个 $`K`$，任何蒸馏器（包括 replay）的最坏误差至少是两者差的一半（[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)）。
- 需要获取的量，在显式模型中由 [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)(b)（主动下界）、[Theorem 2.10](../statements/02-10_few_shot_information_floor_ZH.md#s2-10)（few-shot 信息下限）与 [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11)（共用一份预算）给出下界，一般情形下的下界是 $`\tfrac14\mathfrak n_{K,n}`$（[Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)）。c-optimal design 达到它（[Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)），在状态覆盖模型中就是 water-filling（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）。
- 目标向量 $`c=\sum_tJ_t^*g_t`$ 取决于 optimizer，所以同一批示范可以对一个学习器充分、对另一个不充分（[Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)）。获取必须针对实际使用的学习器来规划。
- 把获取到的事实编译进学生另有代价（[Proposition 2.7](../statements/02-07_compilation_cost_ZH.md#s2-7)），与获取分开核算。

**Rule 2（按实测后果选择）.**

- *response 是不变量.* 对每一次一轮边界替换加声明尾段，两个消息律可以互换，当且仅当它们在声明类的每个 response 下均值相等（[Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)(b)）。在 $`z`$ 上把参考换成候选 $`j`$ 的价值恰为 $`A_t(z,j)`$，一个候选策略的总效应为 $`\sum_t\mathbb E_{d^S_t}A_t`$（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)(b)）。
- *静态价值与平均梯度确定不了排序.* [Proposition 3.1](../statements/03-01_static_value_not_learning_value_ZH.md#s3-1)(a) 给出两个数据源：它们的 teacher 价值相同，每个混叠学生的静态策略价值也相同，蒸馏出的学生价值却不同。只依赖这些统计量的规则，对这个实例和它交换标签后的实例做出同样的选择，所以在其中一个上选到较差的源。[Proposition 3.1](../statements/03-01_static_value_not_learning_value_ZH.md#s3-1)(b) 给出两个 packet：标签边缘分布与平均梯度都相同，学习价值之差的符号却在两个任务之间相反；只依赖这些统计量的规则至少在一个任务上出错。
- *两类 proxy 不一致.* task-blind 统计量不能对所有任务都排对；只看参数的一阶分数和余弦分数在 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2) 描述的区域里会排错或打平（[Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)）。对其他静态 proxy 没有证明不一致性；在受控学习器上测试过的静态 proxy 未能可靠地选对（[Check 9.4](../experiments/09-03_derivation_design_rules_ZH.md#c9-4)）。
- *packet 的组织是候选的一部分.* 在边缘分布固定时重组单元，恰好通过 $`H_z`$ 的不可加部分改变学习价值（[Proposition 9.1](../statements/09-01_coupling_width_ZH.md#s9-1)）。所以测量的 response 必须是对实际发出的 packet 的 response，而不是逐单元分数之和。
- *不存在通用的小统计量.* 对不受限制的学习器，任何少于 $`m-1`$ 个统计量的固定族都不能确定 $`m`$ 条消息的字母表上的全部 response 均值（[Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(d)）。小的充分统计量只在受限的族上存在，例如仿射学习器的 $`N`$ 阶以内联合矩（[Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)）。

**Rule 3（匹配学习器的统计量，而不是原始梯度）.**

- 一阶表示通过完整状态的平均增量写出结果差（[Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)：$`(K_t^P-K_t^Q)V(z)=\langle\nabla V(z),\mu_t^P(z)-\mu_t^Q(z)\rangle+r_1`$）。[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10) 对每种学习器算出这个统计量：
  - SGD 是平均梯度；
  - 裁剪下是裁剪后的均值，只看平均方向的误差不超过 $`c(1-p_{\rm act})`$（[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)）；
  - cold Adam 一步是符号分布，幅度保留在 $`m`$ 中（[Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)）；
  - warm 状态下是平均 Adam 增量，只有在冻结分母近似下才化为 $`(\mathbb EG,\mathbb E\,G\odot G)`$（[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)、[Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6)）。
- 在 cold 状态上，均值相同的分布在一阶上就分开（[Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7)）。$`\beta_2`$ 固定时，warmth 收敛到平台 $`(1-\beta_2)/\beta_2`$，而不是趋于零（[Proposition 5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8)）。所以冻结分母描述每一步都有一个误差下限，warmth 必须测量，不能假定它很小。
- warm Adam 下，消息还通过二阶矩通道起作用（[Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1)）；在所述区间内，它的价值关于相对 running RMS 的幅度是凹的，超过一个显式阈值后变负（[Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)）。所以相对于 $`\sqrt v`$ 的幅度也是统计量的一部分。
- 参数相同、optimizer 矩不同的两个状态会走到不同的下一状态，response 也可以不同，所以 transport 条件要在完整状态上陈述（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)）。

**Rule 4（拟合稳定的 response 模型）.**

- [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)(a) 表明，这个目标就是被认证的未分辨差距的精确值：特征均值之差至多为 $`\epsilon_j`$ 的两个分布之间最大的 response 差，等于 $`\min_b[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert]`$。若 response 估计在 sup 范数下误差不超过 $`\eta`$，被认证的界就是它在 $`\widehat g`$ 处的值加 $`2\eta`$（Theorem 8.1(d)）。对每个固定的 $`b`$，界是方括号在 $`b`$ 处的值，所以取最小的 $`b`$ 给出最好的证书。
- 残差更小、但 $`\sum_j\epsilon_j\lvert b_j\rvert`$ 更大的模型，只要两项之和更大，给出的证书就严格更差。特征是 response 的小倍数时，残差为零的模型也可能给出空洞的证书（[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)(c)：$`\phi_1=\alpha g`$ 时差距为 $`\min\lbrace1,\epsilon/\alpha\rbrace`$）。

**Rule 5（计入覆盖；用配对替换做认证；一阶 audit 只用在 margin 占优处）.** 四种形式是：

- (i) 冻结 audit 的证书，含实测的 transport 与退出概率（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(c)）；
- (ii) 实际状态上、基于全尾段配对 audit 的下置信界保证（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)(c)）；
- (iii) 被 audit 状态上、带校准 surrogate 的 CRN 配对替换证书（[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)），沿训练推广只能靠 transport 项或重新 audit（[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）；
- (iv) 冻结方法的整次训练随机对照，标注为经验结论。

为什么是这四种、且只有这四种：

- 去掉 transport 项是无效的：audit 可以报告零差距，而学习差距为 $`1`$（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(f)）。
- 实际状态上的全尾段 audit 保证损失不超过 $`\delta`$，但每个候选需要上千个配对副本（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)(c)–(d)：$`M=4`$、$`\delta=0.05`$、半径 $`0.1`$ 时为 1,016 个）。
- 配对替换分数在期望下精确，不需要 continuation value 梯度，也不需要曲率，半径只含统计项与校准项（[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)）。在 CRN 下，它的方差由两个候选增量之差决定（[Proposition 8.10](../statements/08-10_cost_and_crn_ZH.md#s8-10)）。
- 只有当候选对的一阶项超过二阶余项界时，一阶分数才能认证它。所以平均增量沿 surrogate 梯度方向相同的候选对，无论真实差距多大，一阶上都认证不了（[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)）；Theorem 8.8 可以认证这类候选对。
- 按点估计选择、或不带 transport 的外推，都不带 (i)–(iii) 中的任何保证。

**Rule 6（先冻结，再检验迁移）.**

*冻结.* [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(a) 的 audit 事件是对 audit 之前固定的候选集同时成立的。复用 audit 样本来构造或训练候选会改变候选的分布，证书就需要一次新的独立 audit。在评测数据上调过的选择器同理。

*迁移.* 若目标族的每个 response 都满足 $`g=c+b^\top\phi+r`$，$`\lVert b\rVert_1\le B`$，$`\lVert r\rVert_\infty\le\tau`$，则由 [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(c)，

```math
\sup_g\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau .
```

反之，若某个目标 response 不在特征张成的空间中，就存在特征匹配、却被它区分的分布（[Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(a)），只在训练族上做 audit 无法认证目标族。

*两种迁移.* 方法迁移评测 $`L_{E'}(S)`$：把冻结的流程 $`S`$ 用到新任务族 $`E'`$、用它自己的 $`K`$ 条示范；它的检验类是 $`E'`$ 经由更新定义的 response 类。策略迁移评测 $`J_{E'}(\pi_z)`$：蒸馏好的学生 $`z`$ 不再更新；它的检验函数是 $`E'`$ 的部署得分。两者是检验类不同的两个泛函，一个的界不能推出另一个的界，所以分开报告。 $`\square`$

## 相关工作

- **数据与 teacher 选择.** LESS（[Xia et al., 2024](../references_ZH.md#xia2024less)）与 GRACE（[Panigrahi et al., 2025](../references_ZH.md#panigrahi2025good)）按静态或基于梯度的分数选择数据或 teacher；Rule 2 说明这类分数何时可能与学习价值的排序一致。
- **影响函数与 metagradient.** TracIn（[Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating)）、影响函数（[Koh & Liang, 2017](../references_ZH.md#koh2017understanding)）、metagradient（[Engstrom et al., 2025](../references_ZH.md#engstrom2025optimizing)）与 In-Run Data Shapley（[Wang et al., 2025](../references_ZH.md#wang2025data)）都是一阶或 fixed-state 的归因；按 Rule 5，它们只在 [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) 的 margin 之上才能认证。

## 参考文献

- [Xia et al., 2024](../references_ZH.md#xia2024less) · [Panigrahi et al., 2025](../references_ZH.md#panigrahi2025good) · [Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating) · [Koh & Liang, 2017](../references_ZH.md#koh2017understanding) · [Engstrom et al., 2025](../references_ZH.md#engstrom2025optimizing) · [Wang et al., 2025](../references_ZH.md#wang2025data)
