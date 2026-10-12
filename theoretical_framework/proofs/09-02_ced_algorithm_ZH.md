# Algorithm 9.2 · Consequence-Calibrated Experience Distillation，CED

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/09-02_ced_algorithm_ZH.md) · [证明](09-02_ced_algorithm_ZH.md) · [实验](../experiments/09-02_ced_algorithm_ZH.md) · [English](09-02_ced_algorithm.md)

**本页证明什么.** 算法不是定理，所以本页确立两件事。第一，对 CED 的六个步骤中的每一步，指出是哪些先前的陈述迫使这一步：跳过它的方法会失去这些陈述提供的保证。第二，(G1)–(G3) 中的每一项保证，都是把一个所引定理用于某一步的输出得到的；(G4) 记录这些定理的假设不满足时还剩下什么。

**本页用到的设定与记号.**

- *学生与配方.* 完整学习器状态 $`z=(\theta,o)`$，即参数加 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)）；声明的配方（更新 $`U_t`$、optimizer、horizon $`R`$、参考尾段）；声明的部署任务，得分为 $`F_E`$；$`L_E(S)`$ 是 provider 策略 $`S`$ 的学习价值，$`L_E(0)`$ 是始终用参考时的学习价值。
- *证据.* 每个任务族 $`K`$ 条可信示范，以及 $`n`$ 次允许的实验预算（在学生访问的状态上查询 teacher、环境分支、reset）（[Definition 1.2](../README_ZH.md#s1-2)）。$`D`$ 是全部证据的 transcript；$`\mathcal R_{\rm info}(D)`$ 是它的信息 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2)）。
- *候选 feedback law.* $`P^0,\dots,P^M`$ 发出消息 $`Y\sim P^j_t(\cdot\mid z)`$；$`P^0`$ 是参考。一条消息产生完整状态的增量 $`\Delta^j`$。消息的一轮边界 response 为 $`g_{t,z}(Y)`$（[Definition 1.4](../README_ZH.md#s1-4)）；$`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$；相对参考的优势为 $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$：在 $`z`$ 上把一轮参考换成候选 $`j`$、之后走参考尾段时任务价值的变化。
- *学习器相对权重与覆盖.* 在学生访问的 agent 状态 $`s`$ 上，$`W_s`$ 是学习器相对权重，$`k_s=KH\,d_T(s)`$ 是 teacher 覆盖；$`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ 是 water-filling 分配，乘子 $`\lambda>0`$ 由 $`\sum_sn_s^\star=n`$ 确定（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）。
- *幅度相关的量.* 逐坐标地，$`\lambda`$（water-filling 公式之外）是消息相对 running RMS $`\sqrt v`$ 的幅度，$`\rho`$ 是从 optimizer state 读出的信噪比，$`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ 是价值峰值（[Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)）。
- *配对替换分数.* 在被 audit 的状态上，$`\hat S^{jk}`$ 是 $`r`$ 个独立副本上 surrogate 结果之差的均值：分别用 $`j`$ 与 $`k`$ 走一轮，之后用 common random numbers 走共享的 surrogate 尾段，按 $`\tilde F`$ 打分（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）。$`e^{jk}_{\rm stat}`$ 是它的 empirical Bernstein 半径，$`(\alpha,\varepsilon_{\rm sur})`$ 是 [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) 的 surrogate 校准。
- *置信水平* $`\delta`$；*检查点* $`t_1<\dots<t_A`$。

**陈述（[Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)）.**

*输入：* 带声明配方与部署任务的学生 $`z_0`$；每个任务族 $`K`$ 条可信示范，以及 $`n`$ 次允许的实验预算；$`P^1,\dots,P^M`$ 的候选生成器与参考 $`P^0`$；audit 预算（检查点、$`r`$ 个副本、surrogate $`\tilde F`$、少量全尾段校准 audit、置信水平 $`\delta`$）。*输出：* 蒸馏后的学生 $`z_R`$；证据台账；每个检查点上的分数 $`\hat S^{jk}`$ 及半径、被选中的律以及这次选择是否被认证；冻结的选择器，以及分开报告的方法迁移与策略迁移结果。

*步骤.*

1. **记录完整学习器状态** $`z_t=(\theta_t,o_t)`$：每个检查点都保存，包括 $`m`$、$`v`$、步数计数器、采样器与随机状态，并记录 warmth（[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11)）。
2. **先获取，再编译.** 用 pilot 估计 $`W_s`$ 与 $`k_s`$（坐标按状态分开时用 [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) 的闭式，否则用分组 CRN 分数）；按 $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ 分配剩余查询，一般情形下即目标为 $`c=\sum_tJ_t^*g_t`$ 的 c-optimal design；维护只让新证据计入的台账。
3. **编译候选** $`P^0,\dots,P^M`$，每个候选由它实际发出的 packet 确定。
4. **记录每个候选的学习器统计量并控制幅度：** 平均 optimizer 增量、裁剪激活率、逐坐标的 $`(\lambda,\rho)`$；标记在高 $`\rho`$ 坐标上超过 $`\lambda^\star`$ 或落在大 $`\lambda`$ 区间的质量；缩放或裁剪后的变体是新候选。
5. **在实际状态上测量后果：** 在 $`z_t`$ 上对所有候选对计算 $`\hat S^{jk}`$ 与 $`e^{jk}_{\rm stat}`$；用少量全尾段配对 audit 校准 $`(\alpha,\varepsilon_{\rm sur})`$；一阶分数只作为筛选，且只用于未被标记、margin 超过二阶余项的候选对；若拟合 response 模型，用有界系数目标。
6. **选择、训练、重新 audit、冻结、检验迁移：** 选择 $`j_t\in\arg\max_j[\hat S^{j0}-e^{j0}_{\rm stat}-\varepsilon_{\rm sur}]`$，没有正下界时保留参考；用 $`P^{j_t}`$ 训练到下一个检查点，只借助 transport 项或重新 audit 把选择推广到未 audit 的状态；在评测数据之前冻结一切，分开报告方法迁移与策略迁移。

*保证.* (G1) 在 Assumption 8.7 下，以至少 $`1-\delta`$ 的概率、对 $`z_t`$ 上所有候选对同时成立：下界为正的选择 $`j_t\ne0`$ 不会降低"一轮之后走参考尾段"的任务价值。(G2) 用实际状态上的全尾段配对 audit 并按下置信界选择时，$`L_E({\rm adaptive})-L_E(0)\ge-\delta`$。(G3) 后处理步骤不会降低信息 regret；在 [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5) 的假设下，第 2 步在其中所述常数内达到学习器相对的信息界。(G4) 其余情形为经验结论。*已知局限：* 用 Assumption 8.7 的先验区间时，半径可能超过候选间的典型差距（[Remark 8.12](../statements/08-12_open_problems_ZH.md#s8-12)），此时第 6 步属于 (G4)。

**证明.** 论证分两部分：每一步为什么必需，以及每项保证如何得出。

*第一部分：每一步为什么必需.*

- **第 1 步（完整状态）.** feedback law 必须携带的统计量取决于 optimizer state，而不只取决于 $`\theta`$：
  - 冷启动的 Adam 步是一个符号商，幅度只保留在 $`m`$ 中（[Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)）；
  - warm Adam 关于消耗的梯度一阶仿射，系数由 $`(m,v)`$ 决定，而冷步会把均值相同的反馈分开（[Propositions 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6) 与 [5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7)）；
  - 一阶统计量因学习器而异（[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)）。
  
  coverage 与 transport 都是关于完整状态的陈述（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)：transport 界以完整状态 $`z\in\mathcal C_t`$ 为下标）。只保存 $`\theta`$ 的检查点无法被 audit。
- **第 2 步（先获取，再编译）.**
  - 后处理不能降低信息 regret（[Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)：对每个随机映射 $`C`$，$`\mathcal R_{\rm info}(C(D))\ge\mathcal R_{\rm info}(D)`$）；在 aliasing 下 replay 的价值为零（[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)：示范律相同、学习器相对泛函不同的两个环境，任何蒸馏器都分不开）。
  - 被动数据可能留下只有查询才能补上的缺口（[Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)：不查询时最坏 regret 至少为 $`\Delta`$，查询后可以达到）。
  - 示范与查询共用一份预算，兑换率是学习器相对的（[Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11)、[Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8)）。
  - 最小误差与 $`\mathfrak n_{K,n}`$ 同阶（[Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)），由 c-optimal design 达到（[Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)），在状态覆盖模型中就是 water-filling（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）。
  
  所以必须在任何编译之前获取查询，并按学习器的权重分配：transcript 里缺的东西，后面的步骤都补不回来。
- **第 3 步（packet）.** 分组恰好通过联合 packet response 的不可加部分改变结果（[Proposition 9.1](../statements/09-01_coupling_width_ZH.md#s9-1)：耦合宽度为零当且仅当 packet response 可加）。所以候选是它的 packet 律，而不是一袋单元。
- **第 4 步（学习器统计量与幅度）.**
  - 平均梯度相同的候选在一阶上就可以不同（[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)）。
  - warm Adam 下，同方向、幅度更大的消息价值可能更低（[Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)：价值关于 $`\lambda`$ 是凹的，峰值在 $`\lambda^\star`$ 附近）。
  - 只看参数的分数和余弦分数在那一区域会排错或打平（[Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)），二阶矩通道占主导处一阶分数不可靠（[Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5)）。
  - 全局范数裁剪给 $`\lambda`$ 设了上限（[Remark 7.4](../statements/07-04_relation_prior_work_role_clipping_ZH.md#s7-4)），所以裁剪后的变体确实是另一个候选。
  
  因此被标记的候选只能用第 5 步的有限测量来排序。
- **第 5 步（在实际状态上测量）.**
  - 配对替换分数在期望上精确，不涉及一阶近似，也不需要曲率常数（[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)(a)）；在 common random numbers 下，它的方差由两个候选增量之差决定（[Proposition 8.10](../statements/08-10_cost_and_crn_ZH.md#s8-10)(c)）。
  - surrogate 与任务目标不同，所以必须用全尾段任务 audit 校准 $`(\alpha,\varepsilon_{\rm sur})`$（[Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7)(A)、[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)、[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）。
  - 一阶分数只有在一阶 margin 超过二阶余项时才能认证一对候选（[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)），这就是对筛选的限制。
  - 对特征上的 response 模型，有界系数目标就是精确的被认证的未分辨差距（[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)），所以残差更小但系数更大的模型给出更差的证书。
- **第 6 步（选择、重新 audit、冻结、迁移）.**
  - 按下界选择就是 [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)(c) 的判定规则；按点估计选择没有这样的保证。
  - 去掉 transport 项是不成立的：存在 audit 报告零差距、学习差距却为 $`1`$ 的实例（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(f)）。所以在未 audit 的状态上要用 transport 项或重新 audit。
  - [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(a) 的 audit 事件只对 audit 之前固定的候选集同时成立，所以一切都必须在评测它的数据之前冻结。
  - 方法迁移与策略迁移是不同的泛函，有不同的检验类（[Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(c)：只有通过对目标 response 的已核实分解才有迁移），所以分开报告。

*第二部分：各项保证.*

**(G1).** 在检查点 $`z_t`$ 上，第 5 步在 Assumption 8.7 的假设下对所有有序对计算 $`\hat S^{jk}`$ 与 $`e^{jk}_{\rm stat}`$。由 [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)(b)，以至少 $`1-\delta`$ 的概率，对所有候选对同时有

```math
L_F(P^j)-L_F(P^0)\ \ge\ \alpha\big(\hat S^{j0}-e^{j0}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

第 6 步只在方括号为正时选择 $`j_t\ne0`$，且 $`\alpha>0`$，所以在这个事件上 $`L_F(P^{j_t})-L_F(P^0)>0`$。由 [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)，$`L_F(P^j)-L_F(P^0)=A_t(z_t,j)`$，即在 $`z_t`$ 上把一轮参考换成 $`P^j`$、之后走参考尾段时任务价值的变化。这就是 (G1)。由于该事件对所有候选对同时成立，它对依赖数据的选择 $`j_t`$ 也成立。

**(G2).** 若第 5 步的 audit 是在训练实际到达的状态上做的全尾段配对 audit，第 6 步按其下置信界选择（否则保留参考），得到的 provider 策略就是 [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)(c) 的自适应策略。该定理 (c)(ii) 给出：对 audit 与训练取无条件期望，$`L_E({\rm adaptive})-L_E(0)\ge-\delta`$。其机制是 Theorem 8.3(b) 的带号恒等式 $`L_E(S)-L_E(0)=\sum_t\mathbb E_{d^S_t}A_t(Z_t,J_t)`$：整次训练的收益是实际到达状态上各轮优势之和，而 audit 控制了其中每一项。

**(G3).** 第 3–6 步拿第 2 步产生的 transcript $`D`$ 来变换（replay、改写、打包、选择），所以它们产生的学生是一个随机后处理 $`C(D)`$。由 [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)，$`\mathcal R_{\rm info}(C(D))\ge\mathcal R_{\rm info}(D)`$：这些步骤都不会降低信息 regret。第 2 步的获取是目标为 $`c=\sum_tJ_t^*g_t`$ 的 c-optimal design，在 [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5) 的假设下，它在其中所述的常数因子内达到学习器相对的信息界。

**(G4).** (G4) 列出的三种情形各自去掉了上面用到的一个假设：按点估计选择去掉了 (G1)–(G2) 所用的下界条件；未核实的校准去掉了 Assumption 8.7(A)；在未 audit 的状态上不带 transport 项使用，去掉了 Theorem 8.2 的 transport 假设，而 Theorem 8.2(f) 表明此时结论可能不成立。于是不剩任何证书，结论只关于冻结方法在声明总体上的平均结果，由整次训练的随机对照支持。

**已知局限.** 在 Assumption 8.7(B) 下区间 $`B`$ 是先验的（$`\tilde F\in[0,1]`$ 时 $`B\le2`$），$`e^{jk}_{\rm stat}`$ 中的区间项 $`7B\log(4M^2/\delta)/(3(r-1))`$ 对任何负担得起的 $`r`$ 都可能超过相近候选之间的差距（[Remark 8.12](../statements/08-12_open_problems_ZH.md#s8-12)）。此时没有正的下界，第 6 步保留参考或报告一个未认证的选择，这次运行属于 (G4)。$`\square`$

## 相关工作

- **数据与 teacher 选择.** LESS（[Xia et al., 2024](../references_ZH.md#xia2024less)）按与目标的梯度余弦给训练数据排序；GRACE（[Panigrahi et al., 2025](../references_ZH.md#panigrahi2025good)）给蒸馏用的 teacher 打分。Algorithm 9.2 改为按候选 feedback law 在实际 optimizer state 上的实测后果排序，[Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 指出余弦类分数在哪里不一致。
- **主动获取.** DAgger（[Ross et al., 2011](../references_ZH.md#ross2011reduction)）在学生的状态上查询 teacher。第 2 步按学习器相对权重、而不是均匀地分配这类查询（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）。
- **影响函数与 metagradient.** TracIn（[Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating)）、影响函数（[Koh & Liang, 2017](../references_ZH.md#koh2017understanding)）、metagradient（[Engstrom et al., 2025](../references_ZH.md#engstrom2025optimizing)）与 In-Run Data Shapley（[Wang et al., 2025](../references_ZH.md#wang2025data)）都是一阶或 fixed-state 的归因。在 Algorithm 9.2 中，它们只能在 [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) 的条件下作为第 5 步的筛选。

## 参考文献

- [Xia et al., 2024](../references_ZH.md#xia2024less) · [Panigrahi et al., 2025](../references_ZH.md#panigrahi2025good) · [Ross et al., 2011](../references_ZH.md#ross2011reduction) · [Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating) · [Koh & Liang, 2017](../references_ZH.md#koh2017understanding) · [Engstrom et al., 2025](../references_ZH.md#engstrom2025optimizing) · [Wang et al., 2025](../references_ZH.md#wang2025data)
