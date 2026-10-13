# Remark 9.3 · 设计规则的推导

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](09-03_derivation_design_rules_ZH.md) · [证明](../proofs/09-03_derivation_design_rules_ZH.md) · [实验](../experiments/09-03_derivation_design_rules_ZH.md) · [English](09-03_derivation_design_rules.md)

**一段话概括.** [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)（CED）是框架的蒸馏方法。本 remark 提炼出它背后的六条设计规则，并对每一条说明是哪些先前的陈述迫使它成立。这里的规则是必要条件：声称具有相应保证的方法必须遵守它。本 remark 不声称遵守这些规则就能让某个 CED 实例优于给定的 baseline；那是一个经验问题。

**本页用到的设定与记号.**

- *学习器与候选.* 完整学习器状态 $`z=(\theta,o)`$，即参数加 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)）；候选 feedback law $`P^0,\dots,P^M`$ 在第 $`t`$ 轮发出消息 $`Y\sim P^j_t(\cdot\mid z)`$，$`P^0`$ 是参考。
- *response.* 消息的一轮边界 response $`g_{t,z}(Y)`$：在 $`z`$ 上用 $`Y`$ 更新、再走声明尾段之后的期望最终任务价值（[Definition 1.4](../README_ZH.md#s1-4)）。它在候选 $`j`$ 下的均值为 $`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$，$`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$ 是把一轮参考换成 $`j`$ 的价值。
- *transcript 与信息 regret.* $`D`$ 是全部证据（示范与允许的实验）的 transcript；它的信息 regret $`\mathcal R_{\rm info}(D)`$ 是任何使用 $`D`$ 的算法都无法消除的那部分 regret（[Theorem 2.2](02-02_regret_decomposition_decision_ambiguity_ZH.md#s2-2)）。
- *平均增量.* $`\mu_t^P(z)=\mathbb E_P\Delta`$ 是候选 $`P`$ 下完整状态的平均增量；warmth 度量一个 Adam 状态离它的冻结分母描述有多远（[Remark 5.11](05-11_checking_warm_state_conditions_ZH.md#s5-11)）。
- *response 模型.* 消息的特征 $`\phi_1,\dots,\phi_d`$，矩阵为 $`\Phi`$；response 模型为 $`c+b^\top\phi`$；特征均值的测量容差为 $`\epsilon_j`$；$`\widehat g`$ 是 response 估计；$`\mathrm{osc}(h)=\max h-\min h`$。
- *迁移.* $`L_E(S)`$ 是协议 $`S`$ 在任务族 $`E`$ 上的学习价值；$`J_E(\pi_z)`$ 是学生 $`z`$ 的部署价值。*方法迁移*在新任务族 $`E'`$ 上、用它自己的 $`K`$ 条示范重跑冻结的协议，评测 $`L_{E'}(S)`$；*策略迁移*把蒸馏好的学生直接部署到 $`E'`$、不再更新，评测 $`J_{E'}(\pi_z)`$。

<a id="s9-3"></a>

**Remark 9.3（设计规则的推导）.** Algorithm 9.2 实现了六条规则。每条规则说明一个声称具有相应保证的方法必须做什么；[证明页](../proofs/09-03_derivation_design_rules_ZH.md)给出迫使它成立的陈述。这些规则是获得这些保证的必要条件；它们本身并不说明某个 CED 实例优于某个给定的 baseline。

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

**说了什么.**

- 用新的实验获取缺失的信息，并针对这个学习器来规划；改写旧数据替代不了它。
- 按候选对这个学习器的实测后果来评判，包括单元怎样打包。
- 要匹配的是 optimizer 在完整状态上的统计量（warm Adam 下包括消息相对 running RMS 的幅度），而不是原始梯度。
- 在有噪声的特征上拟合系数小的 response 模型；用配对替换或全尾段 audit 认证；推广到新状态时计入覆盖。
- 在评测数据之前冻结一切，并分开报告两种迁移。

**直觉.** 一个方法可能走的每条捷径（用改写代替获取、按静态分数排序、匹配梯度、相信 audit 在没见过的状态上也成立、在评测数据上调参），都对应一个先前的陈述，它给出这条捷径失败的情形。

**在框架中的作用.**

- 它是从 §2–§8 的理论到方法的桥梁：[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 的每一步都实现一条或几条规则。
- 它的必要性已被证明；它在实践中是否有用，由受控学习器（[实验页](../experiments/09-03_derivation_design_rules_ZH.md)）与计划中的真实模型检验（[Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4)）检验。

**推导要点.**

- **Rule 1.** 后处理不能降低信息 regret；即使 teacher 最优，被动数据也可能留下缺口；获取多少、在哪里获取由学习器决定（[Proposition 2.4](02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)、[Theorem 2.6](02-06_passive_impossibility_active_attainability_ZH.md#s2-6)、[Theorems 6.3](06-03_learner_relative_lower_bound_ZH.md#s6-3) 与 [6.5](06-05_achievability_ZH.md#s6-5)、[Proposition 6.10](06-10_water_filling_allocation_ZH.md#s6-10)）。
- **Rule 2.** response 是不变量（[Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)）；静态价值与平均梯度确定不了排序（[Proposition 3.1](03-01_static_value_not_learning_value_ZH.md#s3-1)）；task-blind 统计量以及只看参数或余弦的分数不一致（[Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)）；packet 的组织是候选的一部分（[Proposition 9.1](09-01_coupling_width_ZH.md#s9-1)）；不存在通用的小统计量（[Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2)(d)）。
- **Rule 3.** 一阶统计量是完整状态的平均增量（[Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1)、[Theorem 5.10](05-10_first_order_statistic_by_learner_ZH.md#s5-10)）；warm Adam 下，相对 running RMS 的幅度是它的一部分（[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)、[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)）。
- **Rule 4.** 特征上的测量误差，代价是它系数的大小（[Theorem 8.1](08-01_stability_approximation_tradeoff_ZH.md#s8-1)）。
- **Rule 5.** 去掉 transport 不成立，全尾段 audit 代价高，配对替换在期望下精确，一阶分数只在 margin 之上才能认证（[Theorems 8.2](08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)、[8.3](08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)、[8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)、[Corollary 8.9](08-09_first_order_version_margin_criterion_ZH.md#s8-9)）。
- **Rule 6.** audit 事件只对事先固定的候选同时成立；迁移只能通过目标 response 的已核实分解来保证（[Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2)(c)）；方法迁移与策略迁移是不同的泛函。

**用到.**

- [Proposition 2.4](02-04_post_processing_cannot_reduce_information_ZH.md#s2-4)、[Theorem 2.6](02-06_passive_impossibility_active_attainability_ZH.md#s2-6)、[Theorems 6.3](06-03_learner_relative_lower_bound_ZH.md#s6-3)、[6.5](06-05_achievability_ZH.md#s6-5)：获取是必需的，且由学习器决定（Rule 1）。
- [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)、[Proposition 3.1](03-01_static_value_not_learning_value_ZH.md#s3-1)、[Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)、[Proposition 9.1](09-01_coupling_width_ZH.md#s9-1)：按 response 选择（Rule 2）。
- [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1)、[Theorem 5.10](05-10_first_order_statistic_by_learner_ZH.md#s5-10)、[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)：学习器的一阶统计量（Rule 3）。
- [Theorem 8.1](08-01_stability_approximation_tradeoff_ZH.md#s8-1)：稳定的 response 模型（Rule 4）。
- [Theorems 8.2](08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)、[8.3](08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)、[8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)、[Corollary 8.9](08-09_first_order_version_margin_criterion_ZH.md#s8-9)：四种保证形式（Rule 5）。
- [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2)：迁移需要已核实的分解（Rule 6）。

**被用于.**

- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：每一步都实现这些规则。
- [Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4)：计划中的真实模型检验。

## Related work

- Rule 1 依据的是 Blackwell 实验比较理论中容易的那个方向：对 transcript 做 garbling 不能增加信息（[Blackwell, 1953](#ref-blackwell1953equivalent)）。在学生自己的状态上获取标签，是 DAgger 的做法（[Ross et al., 2011](#ref-ross2011reduction)）。
- Rule 2 与 Rule 3 针对的是一阶数据分数：influence functions（[Koh & Liang, 2017](#ref-koh2017understanding)）、TracIn（[Pruthi et al., 2020](#ref-pruthi2020estimating)）与 LESS（[Xia et al., 2024](#ref-xia2024less)）都按梯度统计量给训练数据排序，而不是按在 complete learner state 上实测的后果。
- Rule 6 把调方法的数据与检验方法的数据分开，与 high-confidence policy improvement 的留出安全检验相同（[Thomas et al., 2015](#ref-thomas2015high)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).

<a id="ref-koh2017understanding"></a>

**Pang Wei Koh, Percy Liang (2017).** Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1885–1894. [proceedings.mlr.press/v70/koh17a](https://proceedings.mlr.press/v70/koh17a.html).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).

<a id="ref-thomas2015high"></a>

**Philip S. Thomas, Georgios Theocharous, Mohammad Ghavamzadeh (2015).** High confidence policy improvement. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37:2380–2388. [proceedings.mlr.press/v37/thomas15](https://proceedings.mlr.press/v37/thomas15.html).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
