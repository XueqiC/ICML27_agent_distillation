# Remark 9.5 · 假设审计

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](09-05_assumption_audit_ZH.md) · [English](09-05_assumption_audit.md)

**一段话概括.** 一个对任意可测学习器都成立的框架，若说不清每个有用的结论还需要哪些额外条件，对具体的学习器就几乎说不出什么。本 remark 把三件事分开。第一，主线需要的条件：很少，而且本来就是问题的一部分。第二，部分结果用到的更强条件：每一条都列出用到它的结果，以及它不成立时框架怎么做。第三，无论假设什么都无法同时得到的东西。本 remark 没有自己的证明页和实验页：每一行都指向证明或使用它的陈述。

**本页用到的设定与记号.**

- *主线.* [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2) 的精确后果恒等式，它的近似价值版本（[Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4)），response 算子及其谱（[Definition 4.8](04-08_response_operator_ZH.md#s4-8)、[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)），学习器相对的信息界（Theorem A），以及 §8 的有限证书。
- *受限模型.* 在主线之外附加条件的模型，陈述在其中精确成立或在主阶成立；它的结论对真实系统是有待检验的机制假说。

<a id="s9-5"></a>

**Remark 9.5（假设审计）.**

**(a) 主线需要什么.** 有限的训练预算；每个声明的协议都有可测的真实条件 kernel（非线性更新、裁剪、冷启动 Adam、非平稳学习率与依赖状态的随机性都包括在内）；有界的任务得分；声明的比较范围（环境、初始状态分布、评测器与训练机制相同，或者把差异显式写出）。token 独立、梯度平稳、小步长、低秩、全局收缩、Fisher 正则性与已知的隐藏状态都不在其中。

**(b) 更强的条件、用到它们的结果，以及条件不成立时怎么办.**

| 条件 | 判断 | 框架的处理 |
|---|---|---|
| 只用参数向量就是学习器状态 | 对带记忆的 optimizer 不成立 | 完整状态包括 $`m`$、$`v`$、时钟、缓冲区以及更新用到的历史（[Definition 1.3](../README_ZH.md#s1-3)）；参数相同的状态后续可以不同（[Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)） |
| 原始观察是 Markov 的 | 不必要 | 用完整历史或预测状态（[Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)） |
| 梯度、token 或重复数据相互独立 | 实践中通常过强 | 主线不用它；只在实验复现之间假设独立（[Assumption 8.7](08-07_certificate_assumptions_ZH.md#s8-7)、[Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)） |
| 仿射平均学习器、独立的经验块、有限的决策次数 | 受限模型的条件 | 它们给出 [Theorems 4.4–4.5](04-04_affine_learners_moment_order_ZH.md#s4-4) 的精确矩阶与 cumulant 速率，作为可计算的例子；不推广到任意 optimizer |
| 小学习率使完整状态的 Taylor 余项为二阶 | 一般不成立：optimizer state 可以按别的尺度变化 | 主线用精确 kernel；局部展开各自报告余项（[Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1)） |
| continuation 梯度冻结且与环境无关 | 对一阶目标成立，对真实收益的导数可能不够 | Theorem A 的目标是 [Definition 6.2](06-02_learner_relative_functional_information_norm_ZH.md#s6-2) 的一阶泛函；真实收益的导数还要对 response 机制本身求导 |
| 每个设计的信息矩阵都正定 | 对线性目标过强 | 允许奇异设计，只要求目标方向可辨认（[Lemma 6.4](06-04_design_minimax_equality_ZH.md#s6-4)） |
| 存在高效的两阶段估计量 | 实质性条件，c-optimal design 推不出它 | 线性高斯情形已证明（[Proposition 6.15](06-15_linear_gaussian_target_ZH.md#s6-15)）；一般情形是 [Theorem 6.5](06-05_achievability_ZH.md#s6-5) 的假设 |
| 估计点处的 oracle 律可以生成 | 对真实蒸馏很强 | 作为 Theorem 6.5 的假设写明；只有有限合法池时，用凸可实现性及其残差代替（[Proposition 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14)） |
| 访问独立、坐标独立、频繁访问区间、mean field | 受限模型的条件 | [Proposition 6.11](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) 的学习器权重在这些区间里主阶成立，被当作可计算的分配规则使用，其用处由 R2 检验（[Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4)） |
| Adam 处于平稳点 $`m=\mathbb Eg`$、$`v=\mathbb Eg^2`$；尾段梯度与参数无关；终点得分仿射；$`\mathbb E[m/\sqrt v]`$ 用 $`\mathbb Em/\sqrt{\mathbb Ev}`$ 代替；没有 bias correction、$`\epsilon=0`$、没有 decay | 受限的 mean-field 模型；比值期望的替换不是恒等式 | [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 在该模型中证明；它在真实模型上的预测与实测符号相反（[Check 7.6](../experiments/07-02_stationary_amplitude_law_ZH.md#c7-6)）；一般的机制是 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 的精确分解 |
| warmth 的分位数能控制误差 | 对任务加权的误差不够 | 报告敏感度加权的误差及其最大项（[Remark 5.11](05-11_checking_warm_state_conditions_ZH.md#s5-11)、[Check 5.19](../experiments/05-11_checking_warm_state_conditions_ZH.md#c5-19)） |
| 低秩、谱快速衰减、有限维的预测状态 | 不是普遍事实 | 作为实测性质或有条件的压缩；不可压缩的情形照样保留（[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)、[Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)） |
| 全局有界的密度比 | 可能过强 | 对有限目录，用它的混合作参考测度（[Definition 4.8](04-08_response_operator_ZH.md#s4-8)）；目录之外要重新支付覆盖代价 |
| 平均误差小就意味着每个任务、每个状态都准确 | 不成立 | 逐测试的残差及其权重，以及对实际占据分布的覆盖（[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)、[Proposition 8.13](08-13_shared_features_across_checkpoints_ZH.md#s8-13)） |
| 所有动态都稳定或收缩 | 不必要，也可能不成立 | 用有限时域增益或 Bellman 残差；不引入无限时域线性时不变系统的界（[Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11)、[Proposition 4.13](04-13_finite_horizon_gramians_ZH.md#s4-13)） |
| 每个观察的概率都有一致的正下界 | 对丰富的观察空间过强 | 只在正概率事件上做条件化；稀有分支单独计费（[Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)） |
| 谱或预测状态的输出可以直接作为合法消息发出 | 不成立 | 在合法池上做正权重编译，并检查可实现性（[Proposition 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14)） |
| 冻结的候选、独立的复现、已知的总体区间 | 有限样本证书真正需要的条件 | 保留，并在实验协议中落实（[Assumption 8.7](08-07_certificate_assumptions_ZH.md#s8-7)、[Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)）；自适应生成候选时，需要新的确认数据或序贯有效的分析 |
| 样本的观测极差就是总体区间 | 不成立 | 由它构造的半径只是诊断性的，从不报告为严格（[Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)、[Check 8.17](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-17)） |
| surrogate 校准在每个新状态上都成立 | 很强；有限的校准残差推不出它 | 作为可交换性假设（[Remark 8.11](08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）；可检验的迁移条件仍是未解决的问题（[Remark 8.17](08-17_open_problems_ZH.md#s8-17)） |
| 检查点可以免费恢复、未来可以免费分叉、状态可以免费查询 | 对真实系统不成立 | 只用声明过的干预，每一次分叉、pilot、拟合、漂移检查与评测都计费（[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)、[Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)） |

**(c) 无法同时得到的东西.** 这些障碍不会因为换一套说法而消失。

- *对任意反馈分布、要求逐任务保真时，不存在普遍的强压缩.* 用 $`m`$ 条消息和 $`m`$ 个指示函数 response，中心化后的 response 算子秩为 $`m-1`$，每一个反馈差异都对某个任务有影响，任何少于 $`m-1`$ 个统计量的固定族都不能确定全部 response 均值（[Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2)(d)）。这与固定密度预算、平均误差下的近似接口界并不矛盾。
- *当前充分不等于未来充分.* 一个摘要可以预测每个当前测试，却在一个未来测试上差满整个得分范围（[Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11)）。
- *没有信息就无法辨认.* 两个环境产生相同的 transcript 分布时，它们之差的一半这样的误差，任何蒸馏器都消除不了，生成也不行（[Theorem 6.3](06-03_learner_relative_lower_bound_ZH.md#s6-3)、[Corollary 6.7](06-07_aliasing_zero_value_replay_ZH.md#s6-7)）。
- *完全一般的 surrogate 不保持排序.* 若 surrogate 等于 1 减去真实得分，每一个排序都会颠倒；有限校准集没有覆盖的区域也可能这样，所以校准的迁移是一个假设（[Remark 8.11](08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）。
- *测到的 response 为零，不说明这个方向不存在.* 没有激励与覆盖时，没测到的方向看起来就像零模式；观测到的子矩阵低秩，只给出秩的下界（[Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)、[Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)）。

所以框架不主张"无需任何结构，就能得到统一低维、低成本、对所有任务最优的蒸馏"。它给出的是这样一种理论：结构可以测量，失配的代价可以计算，失败可以被识别。

**(d) 关于 response 算子、闭合、复用、估计成本、编译与记忆的结果需要什么.**

| 结果 | 需要什么 | 是否属于主线 | 不成立时 |
|---|---|---|---|
| 有限维分解与线性预测实现（[Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12)、[Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)） | 声明的未来测试空间有限维，并由所选测试张成 | 否；是精确有限表示的条件，可能很强 | 用残差版本；扩充测试或状态；不作一般的低维主张 |
| 对自适应未来协议的充分性 | 条件化算子在观察分支上闭合 | 只在支持自适应协议时需要 | 缩小协议类，或为遗漏的分支付误差 |
| 跨检查点共享消息特征（[Proposition 8.13](08-13_shared_features_across_checkpoints_ZH.md#s8-13)） | 共同的消息测度、含义相同的测试、每个状态上有界的密度差 | 并非所有环境天然具备 | 在共同的合法池上工作，或按状态簇分别建立接口 |
| 锚点迁移（[Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14)） | 可控制的 response 漂移；可恢复的分叉；独立的配对复现 | 是该证书的操作条件 | 建新锚点，或把迁移报告为未认证；不默认全局平滑 |
| 记忆构造（[Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)） | 指定的可达历史、可实现的梯度、非零的分母 | 构造实例的条件 | 只说明存在性与机制；真实模型上效应的大小另行测量（[Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4) 中的 T1） |
| 完整的一步伴随（[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)） | 更新与 continuation value 局部可微 | 否；是解释性的导数 | 不可微处用精确 response；有限变化保留余项 |
| 无偏的两保真度估计（[Proposition 8.15](08-15_two_fidelity_response_estimation_ZH.md#s8-15)） | 两组相互独立；便宜输出的边缘分布相同；估计量与分配已冻结 | 对该估计量是；条件寻常，但必须落实 | 在独立数据上确认，或另行证明自适应估计量的性质 |
| 两保真度估计更便宜 | 它的成本条件、足够的复用，pilot 与拟合都计费 | 否；是有待检验的条件 | 直接用完整续训，不主张便宜 |
| 信息—编译半径（[Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16)） | 有效的置信集合、紧的合法可行集、固定或已校准的 response 接口 | 是该条件性模型的适用范围 | 对接口与目标用联合不确定集合，并支付校准误差 |
| 推广到新任务、新候选或新检查点 | 覆盖，或有效的迁移控制 | 不能从通过旧目录自动得到 | 增加确认数据，或缩小保证的范围 |

这些结果都没有引入全局梯度平稳、Hessian 为零、所有动态收缩、默认低秩、免费的精确 response、普遍的校准迁移，或任意 oracle 律可以生成这样的假设。受限模型中的较强条件保留为受限条件，没有通过改名变成一般事实。

**说了什么.** 主线只需要有限预算、可测的真实 kernel、有界得分与声明的比较范围。每一条更强的条件都挂在用到它的结果上，并写明它不成立时框架怎么做；有些愿望的组合则根本无法同时满足。

**在框架中的作用.**

- 它告诉读者每个结论要付出哪些假设，防止把主线的一般性误读成"方法不需要任何结构"。
- 它与 [Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4) 对应：那里说证据支持哪些预测，这里说这些预测依赖哪些假设。

**用到.** 每一行都引用证明或使用它的陈述；主要有 [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)、[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)、[Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11)、[Theorem 6.5](06-05_achievability_ZH.md#s6-5)、[Proposition 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14)、[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)、[Assumption 8.7](08-07_certificate_assumptions_ZH.md#s8-7) 与 [Proposition 8.15](08-15_two_fidelity_response_estimation_ZH.md#s8-15)。

**被用于.**

- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：每一步背后的假设。
- [Remark 9.6](09-06_contribution_units_prior_work_ZH.md#s9-6)：每个贡献主张的适用范围。

## 相关工作

- 主线的恒等式属于 performance-difference 一族（[Kakade & Langford, 2002](#ref-kakade2002approximately)）；它们组织误差，但本身不让误差变小。
- 预测状态表示（[Littman, Sutton & Singh, 2001](#ref-littman2001predictive)）把有限维测试空间作为建模条件；这里同样的条件保留为条件，而不作为默认。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).
