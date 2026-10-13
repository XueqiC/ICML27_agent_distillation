# Algorithm 9.2 · Consequence-Calibrated Experience Distillation，CED

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](09-02_ced_algorithm_ZH.md) · [证明](../proofs/09-02_ced_algorithm_ZH.md) · [实验](../experiments/09-02_ced_algorithm_ZH.md) · [English](09-02_ced_algorithm.md)

**一段话概括.** 这是框架推导出的蒸馏方法。CED 不按反馈看上去有多好（teacher 质量、与目标的相似度、某个梯度统计量）来评判反馈，而是按每个候选 feedback law 对这个学习器、在它当前的 optimizer state 下实际造成的后果来评判。它先在学习器最需要的地方获取缺失的信息，再编译候选 feedback law；记录每个候选让 optimizer 做了什么；在实际训练状态上用 common random numbers 的配对替换测量每个候选的效应；按下置信界选择；最后在检验迁移之前冻结一切。每一步都引用了要求它的陈述，每个输出要么带有明确的保证，要么标注为"经验结论"。

**本页用到的设定与记号.**

- *学生与配方.* 完整学习器状态 $`z=(\theta,o)`$，即参数加 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)）；声明的配方（更新 $`U_t`$、optimizer、horizon $`R`$、参考尾段）；声明的部署任务，得分为 $`F_E`$；协议 $`S`$ 的学习价值记为 $`L_E(S)`$。
- *证据.* 每个任务族 $`K`$ 条可信示范，以及 $`n`$ 次允许的实验预算（在学生访问的状态上查询 teacher、环境分支、reset）（[Definition 1.2](../README_ZH.md#s1-2)）。
- *候选 feedback law（provider）.* $`P^1,\dots,P^M`$ 发出消息 $`Y\sim P^j_t(\cdot\mid z)`$（训练单元、packet 或目标）；$`P^0`$ 是参考，例如原始示范。一条消息的一轮边界 response 为 $`g_{t,z}(Y)`$（[Definition 1.4](../README_ZH.md#s1-4)）；候选的平均 response 为 $`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$，它相对参考的带号优势为 $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$。
- *学习器相对权重与覆盖.* 在学生访问的 agent 状态 $`s`$ 上，$`W_s`$ 是 [Proposition 6.10](06-10_water_filling_allocation_ZH.md#s6-10) 的学习器相对权重，$`k_s=KH\,d_T(s)`$ 是 teacher 覆盖；water-filling 分配 $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ 次查询，这个公式中的 $`\lambda>0`$ 是由预算 $`\sum_sn_s^\star=n`$ 确定的乘子。
- *幅度相关的量.* 逐坐标地，$`\lambda`$（water-filling 公式之外）是消息相对 running RMS $`\sqrt v`$ 的幅度，$`\rho`$ 是从 optimizer state 读出的信噪比，$`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ 是 [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 的价值峰值。
- *配对替换分数.* $`\hat S^{jk}`$ 是候选 $`j,k`$ 在一个被 audit 状态上的 CRN 配对替换分数，统计半径为 $`e^{jk}_{\rm stat}`$；surrogate 目标为 $`\tilde F`$，校准常数为 $`(\alpha,\varepsilon_{\rm sur})`$（[Definition 8.6](08-06_crn_paired_replacement_score_ZH.md#s8-6)、[Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)）。
- *置信水平* $`\delta`$；*检查点* $`t_1<\dots<t_A`$；每个候选、每个检查点 $`r`$ 个 CRN 副本。

<a id="s9-2"></a>

**Algorithm 9.2（Consequence-Calibrated Experience Distillation，CED）.**

*输入.*

- 学生：初始完整学习器状态 $`z_0=(\theta_0,o_0)`$；声明的训练配方（更新 $`U_t`$、optimizer、horizon $`R`$、参考尾段）；声明的部署任务及得分 $`F_E`$。
- 证据：每个任务族 $`K`$ 条可信示范（demonstration），以及 $`n`$ 次允许的额外实验预算（在学生访问的状态上查询 teacher、环境分支、reset）。
- 候选生成器：产生 feedback law $`P^1,\dots,P^M`$（例如示范的 replay、teacher 在学生状态上的修正、经环境奖励筛选的自生成轨迹、teacher 对示范的改写），以及参考律 $`P^0`$。
- audit 预算：检查点 $`t_1<\dots<t_A`$，每个候选、每个检查点 $`r`$ 个副本（replica），一个 surrogate 目标 $`\tilde F`$，少量用于校准的全尾段任务评测配对 audit，以及置信水平 $`\delta`$。

*输出.*

- 蒸馏后的学生 $`z_R`$。
- 证据台账：把新证据与 replay、改写、合成派生分开记录。
- 每个检查点上：配对替换分数 $`\hat S^{jk}`$ 及其半径、被选中的 feedback law，以及这次选择是否被认证。
- 冻结的选择器，以及分开报告的方法迁移与策略迁移结果。

*步骤.*

1. **记录完整学习器状态.** 每个检查点都保存可恢复的完整状态 $`z_t=(\theta_t,o_t)`$，包括 $`m`$、$`v`$、步数计数器、采样器与随机状态。按 [Remark 5.11](../statements/05-11_checking_warm_state_conditions_ZH.md#s5-11) 记录 warmth：敏感度加权的 warmth 及其最大项。
2. **先获取，再编译.**
   - 先做一个 pilot，在学生访问的状态上估计学习器相对权重（learner-relative weight）$`W_s`$ 与 teacher 覆盖 $`k_s=KH\,d_T(s)`$。参数坐标按状态分开（state-owned）时用 [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) 的闭式；否则在状态簇上用分组 CRN 替换分数（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）测量 $`W_s`$。
   - 剩余的查询按 water-filling 分配：$`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)）；一般模型中这就是以 $`c=\sum_tJ_t^*g_t`$ 为目标向量的 c-optimal design（[Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)）。
   - 维护证据台账。只有新证据计入信息预算。
3. **编译候选.** 把证据整理成有限的候选 feedback law 菜单 $`P^0,\dots,P^M`$。每个候选由它实际发出的 packet 确定：单元内容与分组方式。
4. **记录每个候选的学习器统计量，并控制幅度.**
   - 在 $`z_t`$ 上，为每个候选记录在当前 optimizer 下决定其一阶效应的量：平均 optimizer 增量 $`\mathbb E\,\Delta`$（[Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)；AdamW 下是 [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5) 的平均 Adam 增量）、裁剪的激活率（[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)），以及逐坐标的幅度比 $`\lambda`$（相对 running RMS）和从 optimizer state 读出的信噪比 $`\rho`$（[Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)）。
   - 若一个候选在高 $`\rho`$ 坐标上的质量落在价值峰值 $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ 之外，或落在 Corollary 7.2(c) 的大 $`\lambda`$ 区间，就标记它。候选经缩放或裁剪得到的变体是新的候选，进入第 5 步的菜单。
5. **在实际状态上测量后果.**
   - 在 $`z_t`$ 上用 common random numbers（CRN）抽 $`r`$ 个副本，对所有候选对（包括每个候选对参考）计算配对替换分数 $`\hat S^{jk}`$（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）及半径 $`e_{\rm stat}^{jk}`$（[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)）。
   - 在少数候选对上，用全尾段任务评测的配对 audit 校准 $`(\alpha,\varepsilon_{\rm sur})`$（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)、[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）。
   - 一阶分数只能作为筛选，且只用于一阶 margin 超过二阶余项（[Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9)）、第 4 步未标记的候选对。
   - 若在特征上拟合 response 模型，用 [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1) 的有界系数目标。
6. **选择、训练、重新 audit、冻结、检验迁移.**
   - 选择 $`j_t\in\arg\max_j\big[\hat S^{j0}-e_{\rm stat}^{j0}-\varepsilon_{\rm sur}\big]`$。若没有候选相对参考的下界为正，就保留参考；按点估计做出的选择报告为未认证。
   - 用 $`P^{j_t}`$ 训练到下一个检查点。把选择用到未经 audit 的状态上，需要 [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 的 transport 项与退出项，或者在下一个检查点重新 audit、核对排序是否仍然成立（[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）。
   - 在评测数据之前冻结候选菜单、特征、cell、surrogate、校准与选择器。在留出的任务族、配方与状态上，分别评测方法迁移（在新任务族上、用它自己的 $`K`$ 条示范重跑冻结的流程）与策略迁移（把蒸馏好的学生直接部署到新任务族、不再更新）。

*保证.* 每项保证都是把所引定理用于相应步骤的输出。

- **(G1) 单个检查点.** 在 [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) 下，以至少 $`1-\delta`$ 的概率、对 $`z_t`$ 上所有候选对同时成立：若第 6 步因下界为正而选中 $`j_t\ne0`$，则在 $`z_t`$ 上把一轮参考换成 $`P^{j_t}`$、之后走参考尾段，不会降低任务价值（[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)）。
- **(G2) 整次训练.** 若第 5 步的 audit 是实际状态上的全尾段配对 audit，第 6 步按其下置信界选择，则 $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)）。
- **(G3) 信息.** 只对 transcript 做后处理的步骤都不会降低信息 regret（[Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)）。在 [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5) 的假设下，第 2 步的获取在其中所述的常数因子内达到学习器相对的信息界。
- **(G4) 其余情形为经验结论.** 按点估计做出的选择、迁移未经核实的校准、或在未 audit 的状态上不带 transport 项使用的选择，都不带证书。此时结论是关于冻结方法在声明的任务与 seed 总体上的平均结果，由整次训练的随机对照支持，并标注为经验结论。
- **已知局限.** 用 [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) 的先验区间时，Theorem 8.8 的半径可能超过候选之间的典型差距，这是一个未解决的问题（[Remark 8.12](../statements/08-12_open_problems_ZH.md#s8-12)）。在这种情况下第 6 步属于 (G4)。

**说了什么.** 这样做蒸馏：保留完整的 optimizer state；在学习器相对权重高、teacher 覆盖低的地方获取信息；把候选编译成 packet 律；记录每个候选的 optimizer 统计量并标记过大的消息；在实际状态上用 CRN 配对替换测量每个候选的后果，并用少量全尾段 audit 校准；按下置信界选择；在检验迁移之前冻结。

**直觉.** 选择反馈的依据，是这个学习器在当前状态下消耗它之后发生了什么，而不是反馈本身看上去有多好。

**在框架中的作用.**

- 它是框架的落点：每一步都由 §2–§8 的某个陈述所要求，[Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3) 推导了这些步骤背后的六条设计规则。
- 计划中的真实模型检验 R0–R3 检验它的各个步骤（[Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4)）。

**推导要点.** [证明页](../proofs/09-02_ced_algorithm_ZH.md)对每一步给出要求它的陈述，并把 Theorem 8.8、Theorem 8.3、Proposition 2.4 与 Theorem 6.5 分别用于第 5、6、2 步的输出，推出 (G1)–(G3)。

**示例.** 设参考是 $`K`$ 条示范的 replay，第二个候选在长尾的学生状态上加入 teacher 修正。对访问足够频繁、处于 [Proposition 6.11](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) 频繁访问区间的状态，warm Adam 下的学习器相对权重与访问率 $`d(s)`$ 成正比，而不像 SGD 那样与 $`d(s)^2`$ 成正比。所以相对于常见状态，长尾状态的权重更大，第 2 步在那里花的查询比 SGD 式分配更多。然后第 5 步按实测后果判断：编译成 packet 的修正，在当前状态上是否真的优于 replay。

**用到.** 每一步都引用了它所用的陈述；主要有

- [Proposition 2.4](02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)、[Theorem 6.5](06-05_achievability_ZH.md#s6-5)、[Propositions 6.10–6.11](06-10_water_filling_allocation_ZH.md#s6-10)：获取（第 2 步）；
- [Proposition 9.1](09-01_coupling_width_ZH.md#s9-1)：候选是 packet 律（第 3 步）；
- [Theorem 5.10](05-10_first_order_statistic_by_learner_ZH.md#s5-10)、[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)、[Remark 7.5](07-05_where_first_order_rankers_fail_ZH.md#s7-5)：学习器统计量与幅度控制（第 4 步）；
- [Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)、[Theorem 8.3](08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)、[Corollary 8.9](08-09_first_order_version_margin_criterion_ZH.md#s8-9)：测量与认证（第 5–6 步）；
- [Theorem 8.2](08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)：未 audit 状态上的 transport（第 6 步）。

**被用于.**

- [Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3)：各步骤实现的设计规则。
- [Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4)：计划中的真实模型检验。

## Related work

- 第 2 步在学生自己的状态上查询 teacher，与 DAgger 相同（[Ross et al., 2011](#ref-ross2011reduction)）；这里的查询按学习器相对权重分配。
- LESS（[Xia et al., 2024](#ref-xia2024less)）与 GRACE（[Panigrahi et al., 2025](#ref-panigrahi2025good)）这类数据选择与 teacher 选择分数，用梯度的某个统计量给候选排序；CED 则按候选对学习器的实测后果排序。

## 参考文献

- [Ross et al., 2011](../references_ZH.md#ross2011reduction)：DAgger，在学生的状态上查询（第 2 步按学习器相对权重分配这类查询）。
- [Xia et al., 2024](../references_ZH.md#xia2024less) 与 [Panigrahi et al., 2025](../references_ZH.md#panigrahi2025good)：数据与 teacher 选择分数（CED 改为按实测后果排序）。
