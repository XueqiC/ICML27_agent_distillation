# Algorithm 9.2 · Consequence-Calibrated Experience Distillation，CED

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/09-02_ced_algorithm_ZH.md) · [证明](09-02_ced_algorithm_ZH.md) · [实验](../experiments/09-02_ced_algorithm_ZH.md) · [English](09-02_ced_algorithm.md)

**本页证明什么.** 算法不是定理，所以本页确立两件事。第一，对 CED 的六个步骤中的每一步，指出是哪些先前的陈述迫使这一步：跳过它的方法会失去这些陈述提供的保证。第二，(G1)–(G3) 中的每一项保证，都是把一个所引定理用于某一步的输出得到的；(G4) 记录这些定理的假设不满足时还剩下什么；并说明 [Corollary 8.16](../statements/08-16_composed_guarantee_ZH.md#s8-16) 的组合保证与它们的关系。

**本页用到的设定与记号.**

- *学生与配方.* 完整学习器状态 $`z=(\theta,o)`$，即参数加 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)）；声明的配方（更新 $`U_t`$、optimizer、horizon $`R`$、参考尾段）；声明的部署任务，得分为 $`F_E`$；$`L_E(S)`$ 是 provider 策略 $`S`$ 的学习价值，$`L_E(0)`$ 是始终用参考时的学习价值。
- *证据.* 每个任务族 $`K`$ 条可信示范，以及 $`n`$ 次允许的实验预算（在学生访问的状态上查询 teacher、环境分支、reset）（[Definition 1.2](../README_ZH.md#s1-2)）。$`D`$ 是全部证据的 transcript；$`\mathcal R_{\rm info}(D)`$ 是它的信息 regret（[Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2)）。
- *候选 feedback law.* $`P^0,\dots,P^M`$ 发出消息 $`Y\sim P^j_t(\cdot\mid z)`$；$`P^0`$ 是参考。一条消息产生完整状态的增量 $`\Delta^j`$。消息的单边界 response 为 $`g_{t,z}(Y)`$（[Definition 1.4](../README_ZH.md#s1-4)）；$`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$，相对参考的优势为 $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$：在 $`z`$ 上把一轮参考换成候选 $`j`$、之后走参考尾段时任务价值的变化。
- *合法池与接口.* $`\lbrace y_1,\dots,y_m\rbrace`$ 是有限的已核验消息池；$`\phi(y)\in\mathbb R^r`$ 是 response 接口，系数矩阵为 $`C`$（[Definition 4.8](../statements/04-08_response_operator_ZH.md#s4-8)）；$`\mathcal M`$ 是目标特征均值的置信集合，$`\mathcal K`$ 是合法混合能达到的 response 输出，$`r(\mathcal M,\mathcal K)`$ 是信息—编译半径（[Proposition 6.16](../statements/06-16_information_compilation_radius_ZH.md#s6-16)）；$`B`$ 是混合的密度界。
- *学习器相对权重与覆盖.* 在学生访问的 agent 状态 $`s`$ 上，$`W_s`$ 是学习器相对权重，$`k_s=KH\,d_T(s)`$ 是 teacher 覆盖；$`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ 是 water-filling 分配，乘子 $`\lambda>0`$ 由 $`\sum_sn_s^\star=n`$ 确定（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）。
- *幅度相关的量.* 逐坐标地，$`\lambda`$（water-filling 公式之外）是消息相对 running RMS $`\sqrt v`$ 的幅度，$`\rho`$ 是从 optimizer state 读出的信噪比，$`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ 是 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2) 这个受限 mean-field 模型中的价值峰值。
- *配对替换分数.* 在一个被 audit 的状态上，$`\hat S^{jk}`$ 是 $`r`$ 个独立副本上的平均值：分别用 $`j`$ 与 $`k`$ 执行一轮，之后用 common random numbers 走共享的 surrogate 尾段，取 surrogate 结果 $`\tilde F`$ 之差（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）。$`e^{jk}_{\rm stat}`$ 是它的 empirical-Bernstein 半径，$`(\alpha,\varepsilon_{\rm sur})`$ 是 [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) 的 surrogate 校准。半径的区间项若用数据之前已知的区间，称为*严格*半径；若用同一批样本中观测到的区间，称为*诊断*半径。
- *置信水平* $`\delta`$；*检查点* $`t_1<\dots<t_A`$。

**陈述（[Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)）.**

*输入：* 学生 $`z_0`$ 及其声明的配方与部署任务；每个任务族 $`K`$ 条可信示范与 $`n`$ 次允许实验的预算；有限合法池上 $`P^1,\dots,P^M`$ 的候选生成器与参考 $`P^0`$；audit 预算（检查点、$`r`$ 个副本、surrogate $`\tilde F`$、少量全尾段校准 audit、置信水平 $`\delta`$）。*输出：* 蒸馏后的学生 $`z_R`$；证据台账；缺口报告；每个检查点上的分数 $`\hat S^{jk}`$ 及严格与诊断半径、被选中的律、选择是否被认证；成本台账；冻结的选择器，以及分开报告的方法迁移与策略迁移结果。

*步骤.*

1. **记录完整学习器状态** $`z_t=(\theta_t,o_t)`$，每个检查点都包括 $`m`$、$`v`$、步数计数器、采样器与随机状态，并记录 warmth（[Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11)）。
2. **先获取，再编译；把每个缺口分流.** 用 pilot 估计 $`W_s`$ 与 $`k_s`$。若已有校准过的接口，计算 $`r(\mathcal M,\mathcal K)`$：$`\ker C`$ 中的不确定性不予理会；$`C\mathcal M`$ 直径大是信息缺口（有针对性的查询）；目标在 $`\mathcal K`$ 之外是可实现性缺口（扩充合法池）；摘要在声明的更新下不闭合是闭合缺口（扩充测试或状态）。信息缺口的查询按 $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ 分配，一般情形下是目标为 $`c=\sum_tJ_t^*g_t`$ 的 c-optimal design；维护台账，只有新证据计数。
3. **把候选编译成合法池上的合法混合** $`P^0,\dots,P^M`$，每个候选由它实际发出的 packet 确定，并报告其密度界 $`B`$ 与可实现性残差。
4. **记录每个候选的学习器统计量；幅度只做筛查：** 平均 optimizer 增量、裁剪激活率、逐坐标的 $`(\lambda,\rho)`$；即时参数变化相同不算等价；过大的消息生成缩放或裁剪变体，作为新候选；不按振幅律做任何选择。
5. **在实际状态上测量后果：** 在 $`z_t`$ 上对所有候选对计算 $`\hat S^{jk}`$ 及严格与诊断半径；用少量全尾段配对 audit 校准 $`(\alpha,\varepsilon_{\rm sur})`$；只有当跨检查点复用的全部成本（含漂移检查）低于重新拟合时才复用接口；只有在独立 pilot 支持其成本条件后才用两保真度估计；一阶分数只作为筛选，用于 margin 超过二阶余项、未被标记的候选对；response 模型按有界系数目标拟合。
6. **选择、训练、重新 audit、冻结、检验迁移：** 用严格半径选择 $`j_t\in\arg\max_j[\hat S^{j0}-e^{j0}_{\rm stat}-\varepsilon_{\rm sur}]`$，若没有正的严格下界则保留参考；用 $`P^{j_t}`$ 训练到下一个检查点，只有带 transport 项或重新 audit 才把选择推广到未 audit 的状态；在评测数据之前冻结一切，分开报告方法迁移与策略迁移。

*保证.* (G1) 在 Assumption 8.7 下，以至少 $`1-\delta`$ 的概率、对 $`z_t`$ 上所有候选对同时成立：严格下界为正的选择 $`j_t\ne0`$，不会降低"一轮之后接参考尾段"的任务价值。(G2) 用实际状态上的全尾段配对 audit 并按下置信界选择时，$`L_E({\rm adaptive})-L_E(0)\ge-\delta`$。(G3) 后处理步骤都不会降低信息 regret；在 [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5) 的假设下，第 2 步在其中所述常数内达到学习器相对的信息界，只有有限合法池时再加上第 3 步的可实现性残差。(G4) 其余情形为经验结论。*组合保证：* [Corollary 8.16](../statements/08-16_composed_guarantee_ZH.md#s8-16) 有效，但在实测量级下是空的。*已知局限：* 用 Assumption 8.7 的先验区间时，严格半径可能超过候选之间的典型差距（[Remark 8.17](../statements/08-17_open_problems_ZH.md#s8-17)），此时第 6 步属于 (G4)。

**证明.** 论证分两部分：每一步为什么必需，以及每项保证如何得出。

*第一部分：每一步为什么必需.*

- **第 1 步（完整状态）.** 一个 feedback law 必须携带的统计量取决于 optimizer state，而不只是 $`\theta`$：
  - 冷启动的 Adam 一步是符号商，幅度只保留在 $`m`$ 中（[Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)）；
  - warm Adam 对消耗的梯度一阶仿射，系数由 $`(m,v)`$ 决定，而冷步骤能把均值相同的反馈区分开（[Propositions 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6) 与 [5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7)）；
  - 一阶统计量随学习器而定（[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)）；
  - 参数相同、矩不同的两个可达 Adam 状态，下一步不同（[Proposition 7.6](../statements/07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)）。
  
  覆盖与 transport 都是关于完整状态的陈述（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)：transport 界以完整状态 $`z\in\mathcal C_t`$ 为下标）。只保存 $`\theta`$ 的检查点无法 audit。
- **第 2 步（先获取，再编译；缺口分流）.**
  - 后处理不能降低信息 regret（[Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)：对任意随机映射 $`C`$ 有 $`\mathcal R_{\rm info}(C(D))\ge\mathcal R_{\rm info}(D)`$）；在混叠下，replay 的价值为零（[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)：示范分布相同而学习器相对泛函不同的两个环境，任何蒸馏器都分不开）。
  - 被动数据可能留下只有查询才能填上的缺口（[Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)：没有查询时最坏 regret 至少为 $`\Delta`$，有查询则可达到）。
  - 示范与查询共用一份预算，兑换率依学习器而定（[Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11)、[Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8)）。
  - 最小误差的量级为 $`\mathfrak n_{K,n}`$（[Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)），由 c-optimal design 达到（[Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)）；即便单个设计留下无法辨认的冗余方向，这样的设计也存在（[Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)），在状态覆盖模型中它就是 water-filling（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）。
  - *分流.* 对固定的接口，半径 $`r(\mathcal M,\mathcal K)`$ 至少是 $`C\mathcal M`$ 直径的一半，也至少是 $`C\mathcal M`$ 到 $`\mathcal K`$ 的最大距离，并且只有两者都为零时它才为零（[Proposition 6.16](../statements/06-16_information_compilation_radius_ZH.md#s6-16)）。因此：沿 $`\ker C`$ 缩小 $`\mathcal M`$ 不改变任一下界，花在那里的查询对声明的测试毫无作用；距离项占主导时，关于同一目标的信息再多，也不能让半径低于它，只有更大的合法池（更大的 $`\mathcal K`$）才有用；直径项占主导时，缺的才是信息。一个摘要可以准确预测当前测试，却在声明的更新之后留下大小为 1 的误差、而当前误差为零（[Proposition 4.11](../statements/04-11_static_not_dynamic_closure_error_ZH.md#s4-11)），关于当前测试的查询发现不了它；补救办法是扩充测试或记录的状态。
  
  所以必须在编译之前获取查询、按学习器的权重分配，并且只用于信息能填上的缺口：之后的任何步骤都补不回 transcript 缺少的东西，任何查询也提供不了合法池表达不了的东西。
- **第 3 步（合法混合与 packet）.**
  - 目标特征均值能由合法混合实现，当且仅当它落在合法池特征向量的凸包里，此时支撑至多为 $`r+1`$；稀疏的实现可能抬高密度界 $`B`$，而 $`B`$ 进入每一个谱误差界（[Proposition 6.14](../statements/06-14_convex_realizability_legal_pool_ZH.md#s6-14)、[Theorem 4.9](../statements/04-09_spectral_sufficiency_ZH.md#s4-9)）。所以要报告 $`B`$ 与可实现性残差；[Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5) 的 oracle 律只在可以生成时才发出。
  - 分组改变结果，恰好通过联合 packet response 的非加性部分（[Proposition 9.1](../statements/09-01_coupling_width_ZH.md#s9-1)：coupling width 为零当且仅当 packet response 可加）。所以候选是它的 packet 律，而不是一袋单元。
- **第 4 步（学习器统计量与幅度筛查）.**
  - 均值梯度相同的候选可以在一阶上不同（[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)），即时参数变化相同的候选可以在后续上不同（[Proposition 7.6](../statements/07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)）。
  - 在 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2) 的受限 mean-field 模型中，同方向更大的消息可能价值更低；在那个区间里，只看参数的分数与 cosine 分数会排错或打平（[Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)），二阶矩通道占主导时一阶分数不可靠（[Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5)）。这使幅度成为*生成*待测变体的理由。
  - 但它不是选择或淘汰的理由：该模型在真实模型上的预测与实测 response 符号相反（[Check 7.6](../experiments/07-02_stationary_amplitude_law_ZH.md#c7-6)），所以不能假定它的前提在实际状态上成立。
  - 全局范数裁剪给 $`\lambda`$ 封顶（[Remark 7.4](../statements/07-04_relation_prior_work_role_clipping_ZH.md#s7-4)），所以裁剪后的变体确实是另一个候选。
  
  因此被标记的候选及其变体，只按第 5 步的有限测量排序。
- **第 5 步（在实际状态上测量）.**
  - 配对替换分数在期望上精确，不需要一阶近似，也不需要曲率常数（[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)(a)）；用 common random numbers 时，其方差由两个候选的增量之差决定（[Proposition 8.10](../statements/08-10_cost_and_crn_ZH.md#s8-10)(c)）。
  - Theorem 8.8 的证书要求数据之前已知的区间（[Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7)）；用观测区间构造的半径没有这种保证。所以两个半径分开报告，只有严格半径参与选择。
  - surrogate 与任务目标不同，所以 $`(\alpha,\varepsilon_{\rm sur})`$ 必须用全尾段任务 audit 校准（[Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7)(A)、[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)、[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）。
  - *复用.* 共享消息特征的平均误差是奇异值的尾和，而不是算子范数（[Proposition 8.13](../statements/08-13_shared_features_across_checkpoints_ZH.md#s8-13)）；迁移锚点的接口，要付锚点的估计误差加上实测的算子漂移（[Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate_ZH.md#s8-14)）。两项都要付，而漂移检查本身需要配对续训；只有这些续训比重新拟合便宜时，复用才省计算。
  - *两保真度估计.* 两保真度估计量无偏，并且恰好在 [Proposition 8.15](../statements/08-15_two_fidelity_response_estimation_ZH.md#s8-15) 的成本条件成立时比完整续训便宜。这个条件涉及未知的方差和一个尚未在真实模型上测过的相关系数，所以必须由独立 pilot 支持，pilot 的成本也要计入。
  - 一阶分数只有在其一阶 margin 超过二阶余项时才认证一对候选（[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)），因此筛选受限。
  - 对特征上的 response 模型，有界系数目标就是精确的已认证未消除差异（[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)），所以残差更小、系数更大的模型给出更差的证书。
- **第 6 步（选择、重新 audit、冻结、迁移）.**
  - 按严格下界选择，就是 [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)(c) 的决策规则；按点估计或诊断半径选择没有这种保证。
  - 去掉 transport 项是无效的：存在这样的实例，audit 报告差异为零而学习差距为 $`1`$（[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(f)）。所以未 audit 的状态需要 transport 项或重新 audit。
  - [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)(a) 的 audit 事件只对 audit 之前固定的候选集合同时成立，所以一切都要在评测它的数据之前冻结。
  - 方法迁移与策略迁移是不同的泛函，测试类不同（[Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)(c)：迁移只能通过对目标 response 的已核实分解成立），所以分开报告。

*第二部分：各项保证.*

**(G1).** 在检查点 $`z_t`$ 上，第 5 步在 Assumption 8.7 的假设下对所有有序对计算 $`\hat S^{jk}`$ 与严格半径 $`e^{jk}_{\rm stat}`$。由 [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)(b)，以至少 $`1-\delta`$ 的概率，对所有候选对同时有

```math
L_F(P^j)-L_F(P^0)\ \ge\ \alpha\big(\hat S^{j0}-e^{j0}_{\rm stat}-\varepsilon_{\rm sur}\big).
```

第 6 步只在括号为正时选 $`j_t\ne0`$，而 $`\alpha>0`$，所以在这个事件上 $`L_F(P^{j_t})-L_F(P^0)>0`$。由 [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)，$`L_F(P^j)-L_F(P^0)=A_t(z_t,j)`$，即在 $`z_t`$ 上把一轮参考换成 $`P^j`$、之后走参考尾段时任务价值的变化。这就是 (G1)。由于事件对所有候选对同时成立，它对依赖数据的选择 $`j_t`$ 也成立。诊断半径不满足 Assumption 8.7 的区间假设，所以用它做出的选择不在 (G1) 之内。

**(G2).** 若第 5 步的 audit 是在训练实际到达的状态上做的全尾段配对 audit，第 6 步按其下置信界选择（否则保留参考），得到的 provider 策略就是 [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)(c) 的自适应策略。该定理 (c)(ii) 给出：对 audit 与训练取无条件期望，$`L_E({\rm adaptive})-L_E(0)\ge-\delta`$。其机制是 Theorem 8.3(b) 的带号恒等式 $`L_E(S)-L_E(0)=\sum_t\mathbb E_{d^S_t}A_t(Z_t,J_t)`$：整次训练的收益是实际到达状态上各轮优势之和，而 audit 控制了其中每一项。

**(G3).** 第 3–6 步拿第 2 步产生的 transcript $`D`$ 来变换（replay、改写、混合、打包、选择），所以它们产生的学生是一个随机后处理 $`C(D)`$。由 [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)，$`\mathcal R_{\rm info}(C(D))\ge\mathcal R_{\rm info}(D)`$：这些步骤都不会降低信息 regret。第 2 步的获取是目标为 $`c=\sum_tJ_t^*g_t`$ 的 c-optimal design，在 [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5) 的假设下，它在其中所述的常数因子内达到学习器相对的信息界。这些假设之一是发出估计点处的 oracle 律。第 3 步只能混合有限合法池时，发出的律与 oracle 律相差 [Proposition 6.14](../statements/06-14_convex_realizability_legal_pool_ZH.md#s6-14) 的可实现性残差；由三角不等式，这个残差经接口传递后加到界上。

**(G4).** (G4) 列出的各种情形各自去掉了上面用到的一个假设：按点估计或诊断半径选择，去掉了 (G1)–(G2) 所用的严格下界条件；未核实的校准去掉了 Assumption 8.7(A)；在未 audit 的状态上不带 transport 项使用，去掉了 Theorem 8.2 的 transport 假设，而 Theorem 8.2(f) 表明此时结论可能不成立。于是不剩任何证书，结论只关于冻结方法在声明总体上的平均结果，由整次训练的随机对照支持。

**组合保证.** [Corollary 8.16](../statements/08-16_composed_guarantee_ZH.md#s8-16) 通过带近似价值的停止分解（[Proposition 3.4](../statements/03-04_joint_decomposition_approximate_value_ZH.md#s3-4)），把锚点复用界、两保真度估计误差与可实现性半径合成一个关于整次训练差距的界。它的每步项是有界结果集中不等式给出的绝对精度：不随单步效应的大小缩小，而分解又把它们逐步相加。在实测的评测噪声和负担得起的复现数下，每步的下限已经超过得分的取值范围（数字见 Corollary 8.16 及其实验页）。所以这个界有效，但在这个量级下是空的，不能把第 5–6 步从 (G4) 中提出来。这是关于那里所用的估计量和不等式的陈述，不是"后续训练充分性无法认证"的证明。

**已知局限.** 在 Assumption 8.7(B) 下区间 $`B`$ 是先验的（$`\tilde F\in[0,1]`$ 时 $`B\le2`$），$`e^{jk}_{\rm stat}`$ 中的区间项 $`7B\log(4M^2/\delta)/(3(r-1))`$ 对任何负担得起的 $`r`$ 都可能超过相近候选之间的差距。在真实模型上用 $`r=64`$ 个副本，第一对候选的分数为 $`+0.0014`$，严格半径 $`0.325`$，诊断半径 $`0.0012`$（[Check 8.17](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-17)）。此时没有正的严格下界，第 6 步保留参考或报告一个未认证的选择，这次运行属于 (G4)，直到有了比先验区间小得多、又可证明的区间（[Remark 8.17](../statements/08-17_open_problems_ZH.md#s8-17)）。$`\square`$

## 相关工作

- **数据与 teacher 选择.** LESS（[Xia et al., 2024](#ref-xia2024less)）按与目标的梯度 cosine 相似度给训练数据排序；GRACE（[Panigrahi et al., 2025](#ref-panigrahi2025good)）为蒸馏给 teacher 打分。Algorithm 9.2 则按候选 feedback law 在实际 optimizer state 上的实测后果排序，[Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 指出了 cosine 类分数在一个受限模型中何处不一致。
- **主动获取.** DAgger（[Ross et al., 2011](#ref-ross2011reduction)）在学生的状态上查询 teacher。第 2 步按学习器相对权重而不是均匀地分配这类查询（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)），而且只在缺口属于信息时才查询。
- **影响函数与 metagradient.** TracIn（[Pruthi et al., 2020](#ref-pruthi2020estimating)）、影响函数（[Koh & Liang, 2017](#ref-koh2017understanding)）、metagradient（[Engstrom et al., 2025](#ref-engstrom2025optimizing)）与 In-Run Data Shapley（[Wang et al., 2025](#ref-wang2025data)）都是一阶或固定状态的归因。在 Algorithm 9.2 中它们只能作为第 5 步的筛选，并受 [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) 约束。
- **多保真度估计.** 第 5 步的两保真度选项，把多保真度 Monte Carlo 的经典控制变量思想（[Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal)）用于 continuation response。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-engstrom2025optimizing"></a>

**Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry (2025).** Optimizing ML training with metagradient descent. arXiv preprint. arXiv [2503.13751](https://arxiv.org/abs/2503.13751).

<a id="ref-koh2017understanding"></a>

**Pang Wei Koh, Percy Liang (2017).** Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1885–1894. [proceedings.mlr.press/v70/koh17a](https://proceedings.mlr.press/v70/koh17a.html).

<a id="ref-panigrahi2025good"></a>

**Abhishek Panigrahi, Bingbin Liu, Sadhika Malladi, Sham Kakade, Surbhi Goel (2025).** In good GRACEs: Principled teacher selection for knowledge distillation. arXiv preprint. arXiv [2511.02833](https://arxiv.org/abs/2511.02833).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).

<a id="ref-wang2025data"></a>

**Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia (2025).** Data Shapley in one training run. *13th International Conference on Learning Representations (ICLR)*. arXiv [2406.11011](https://arxiv.org/abs/2406.11011).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
