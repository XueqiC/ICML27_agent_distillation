# Remark 9.4 · 证据覆盖了什么：受控检验与真实模型检验

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](09-04_scope_of_evidence_ZH.md) · [English](09-04_scope_of_evidence.md)

**一段话概括.** 框架的证据分两类。大多数检验是精确的有限计算，或者在受控的小网络里运行，它们检验的是每个陈述在其假设下是否正确。另一小部分是真实的 agent 蒸馏配方上的真实模型检验，每一项都在它所预测的运行之前预注册了预测与判据，检验的是陈述的假设与预测对这个配方是否成立。本 remark 列出每一项真实模型检验：它检验什么、预注册了什么、结果如何，并直接说明证据支持哪些主张、不支持哪些。到目前为止，没有一项真实模型检验支持框架的某个预测：一个预注册判据因设定有误而失败，一个受限模型的预测符号相反，一个排序只在诊断半径下成立，其余检验尚未完成。本 remark 没有自己的证明页和实验页；数字都在它所链接的实验页上。

**本页用到的设定与记号.**

- *配方.* ALFWorld（[Shridhar et al., 2021](#ref-shridhar2021alfworld)）；在 $`K=20`$ 条可信示范上做监督微调，学生是 LoRA（[Hu et al., 2022](#ref-hu2022lora)；Gemma-4-12B-it，rank 16），用 AdamW（[Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)）、常数学习率、全局梯度范数裁剪 1.0 训练。参考协议把这次训练继续到一个固定的 token 预算，下文称为 p40；p03 与 p10 是同一次运行中更早的检查点。
- *保存的状态.* $`z_{10}`$：p10 处的完整学习器状态（update 115；adapter、AdamW 的矩 $`m,v`$、步数计数器与随机状态），所有续训检验都从它重新开始。
- *surrogate 与评测.* $`\tilde F`$：40 条成功 teacher 轨迹上的留出 teacher-forcing 似然分数（几何平均 token 概率，per-token log-likelihood 作为第二个指标），只用于打分。任务评测在 300 个任务的留出开发集上进行；封存的测试集从不使用。
- *噪声参照.* 同一个 adapter 在两台配置相同的 server 上评测，分别做对 241 与 242 个任务，其中 15 个任务在两者之间翻转；这种大小的差异视为 server 噪声（[Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate_ZH.md#c8-21)）。一对 server、一个检查点，不足以说明这种噪声不可约或在不同运行之间独立。
- *CRN 配对替换分数.* $`\hat S^{jk}`$：从保存的状态出发，把下一次参考更新换成候选 $`j`$（或 $`k`$）的一个 batch，再按共享 seed 决定的顺序跑 $`T`$ 次参考更新，用 $`\tilde F`$ 打分，在 $`r`$ 个副本上平均配对差（[Definition 8.6](08-06_crn_paired_replacement_score_ZH.md#s8-6)）。它的*严格*半径用分数的先验区间；*诊断*半径用样本中观测到的区间。

<a id="s9-4"></a>

**Remark 9.4（证据覆盖了什么：受控检验与真实模型检验）.**

- **受控证据.** [Proposition 9.1](09-01_coupling_width_ZH.md#s9-1) 由精确的有限计算和一个受控小网络检验；[Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3) 的选择规则在该网络冻结的留出面板上检验（[Check 9.4](../experiments/09-03_derivation_design_rules_ZH.md#c9-4)、[Check 9.5](../experiments/09-03_derivation_design_rules_ZH.md#c9-5)）。Theorem A、B、C 以及其余每个编号结果的检验，都在被检验陈述的实验页上；它们是精确有限计算、受控小网络或合成构造。关于 response 算子、预测闭合、复用、估计成本、编译与 optimizer 记忆的结果（[Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4)、[Definition 4.8](04-08_response_operator_ZH.md#s4-8) 到 [Proposition 4.13](04-13_finite_horizon_gramians_ZH.md#s4-13)、[Propositions 6.14–6.16](06-14_convex_realizability_legal_pool_ZH.md#s6-14)、[Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)、[Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) 与 [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16)）只有合成检验支持。
- **真实模型检验.** 每一项都在上面的配方上运行，每个配置一个 seed；预测、判据以及计算与 teacher 预算，都在它们所预测的运行之前写定。
  - *R0a：warm 状态条件（[Remark 5.11](05-11_checking_warm_state_conditions_ZH.md#s5-11)）.* 预注册：在从 update 1 到 p40 的六个状态上，敏感度加权的 frozen-denominator 误差不超过一阶项的十分之一。结果：六个状态全部不满足，但原因是判据设定有误：它的分母把绝对敏感度乘上带号的参数变化，大部分相互抵消。改与带号的一阶项比较（这是看到数据之后才选的比较，标注为事后），比值在 update 1、4、8 处约为 30、2、1，在 p03、p10、p40 处约为 0.1（[Check 5.19](../experiments/05-11_checking_warm_state_conditions_ZH.md#c5-19)）。
  - *R0b：response 的非加性.* 预注册：response 非加性的中位数大于 0.2。结果：update 8 处为 0.19（未达到，处在边界上），p10 处为 0.86（达到）；每对只有 4 个副本，而且 p10 处 union 效应的均值低于单个副本的散布，所以这个比值噪声很大（[Check 5.20](../experiments/05-11_checking_warm_state_conditions_ZH.md#c5-20)）。
  - *R1：用 CRN 证书在同一状态上排序（[Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)）.* 从 $`z_{10}`$ 出发的四个候选：$`K`$ 条示范的 replay（P1，即参考）、teacher 在学生状态上的修正（P2）、经环境奖励筛选的自生成轨迹（P3）、teacher 对示范的改写（P4）。排序在任何续训之前记录，然后评测每个候选的续训结果。目前的结果：P1 对 P3，$`r=64`$ 个副本、$`T=10`$ 次尾段更新，分数为 $`+0.0014`$（64 个副本中 57 个为正）。它超过诊断半径 $`0.0012`$，但严格半径是 $`0.325`$，所以"P1 优于 P3"只在诊断半径下成立，严格证书有效但为空（[Check 8.17](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-17)）。surrogate 校准尚未估计。P2、P4 以及各候选的续训臂尚未完成。
  - *R2：学习器相对的查询分配（Theorem A）.* teacher 标签预算 $`n=100`$ 与 $`n=400`$，比较：不查询（现有的参考运行）、在学生状态上均匀分配、按 SGD 权重 $`W_s\propto d(s)^2\sigma_s^2`$ 做 water-filling、按 warm Adam 权重 $`W_s\propto d(s)`$ 做学习器相对的 water-filling（[Propositions 6.10](06-10_water_filling_allocation_ZH.md#s6-10) 与 [6.11](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)）。Adam 权重的 scale-free 形式是在任何查询之前确定的：当时发现预注册的原始方差形式与 Proposition 6.11(b) 矛盾，这次更正如实记录。预注册的主判据：$`n=100`$ 时，学习器相对分配减去均匀分配的差超过噪声参照，且配对 McNemar 检验 $`p<0.05`$。同样预注册的还有：$`n=400`$ 时 warm Adam 权重减 SGD 权重之差的符号与预测一致；在各臂上，预测误差与做对的任务数的 Spearman 相关不超过 $`-0.6`$。结果：尚未完成。
  - *R3：受限的振幅模型（[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)）.* 在 $`z_{10}`$ 上，把一个参考 batch 乘以 $`c=0.5,1,2,4,8`$，该步关闭裁剪，另外再在开启裁剪时做一次。预注册的预测：用保存的 optimizer state 代入 Corollary 7.2(a) 得到的 response 曲线。结果：实测的 per-token response 在 $`+0.0002`$ 与 $`+0.0012`$ 之间，全部落在各自的诊断半径内；预测则在 $`-0.019`$ 与 $`-0.080`$ 之间：符号相反，量级大 28 到 339 倍（[Check 7.6](../experiments/07-02_stationary_amplitude_law_ZH.md#c7-6)）。开启裁剪时，裁剪后梯度相同的两个乘子给出相同的均值，与 [Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4) 的裁剪上限一致。最可能失效的前提是线性 response（尾段梯度与参数无关）；这是假说，不是结论。
  - *T1：optimizer state 移植（[Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)）.* 在 $`z_{10}`$ 上，分别跑参考与一种替代反馈的前 100 次更新；在相同的参数上，把替代分支的一阶矩、二阶矩或两者移植到参考状态里，所有臂之后都跑同样的参考更新。一个重复的参考臂测数值不确定性 $`\delta_{\rm num}`$；surrogate 上的判定阈值为 $`\tau=\max(3\delta_{\rm num},0.0012)`$。已预注册，尚未运行。阴性结果将写成：在该检查点、该反馈块和固定的参考尾段下，两个矩合并移植的净效应未超过阈值。净效应等于动量效应、二阶矩效应与二者交互项之和，所以净效应小可能来自抵消，不能说明每个通道都小；超出预注册范围的通道解读都标为探索性。该检验测的是有限干预效应，不是 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 的伴随预测；阳性结果本身也不构成部署收益的证据。
  - baseline 只用完整的官方代码运行。
- **证据支持什么、不支持什么.**

| 证据 | 支持 | 不支持 |
|---|---|---|
| 每个编号结果的受控检验 | 每个陈述在其假设下、在精确或受控的设定中正确 | 任何关于真实模型的主张 |
| R0a | warm 状态的测量必须用带号的一阶项；预注册判据设定有误，事后比值从 p03 起约为 0.1 | 预注册判据通过了，或者用事后指标替代它 |
| R0b | response 非加性可以测出，并且在 p10 存在 | 精确的估计（只有 4 个副本），或 update 8 处达标 |
| R1（目前） | surrogate 上"P1 优于 P3"的诊断性排序 | 已认证的排序、任务层面的收益，或校准过的 surrogate |
| R3 | 受限振幅模型的预测在 p10 的真实模型上不成立 | 真实模型上的二次 response 律、最优振幅，或在两倍最优振幅附近变号；已经确定的唯一失效原因 |
| R2、T1 | 目前什么也不支持 | 在它们出结果之前，不支持任何主张 |

**说了什么.** 框架各陈述的证据都是精确的、受控的或合成的。真实模型检验中，R0a、R0b、R1（部分）与 R3 已有结果：没有一项支持框架的预测；R0a 的判据设定有误，按失败报告；R3 的预测符号相反；R1 的排序只在诊断半径下成立。R2 与 T1 尚未完成。

**在框架中的作用.**

- 它划定主张的边界：真实模型上，框架不主张超出这些检验所报告的任何东西；负面结果保留，不用事后指标替换。
- 它把每一项真实模型检验与它检验的陈述连起来，使每个结果都能直接对照事先写定的预测来读。

**用到.**

- [Remark 5.11](05-11_checking_warm_state_conditions_ZH.md#s5-11)：真实模型上的 warm 状态检验（R0a、R0b）。
- [Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)、[Remark 8.11](08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)、[Remark 8.17](08-17_open_problems_ZH.md#s8-17)：证书、它的校准与未解决的问题（R1）。
- [Propositions 6.10](06-10_water_filling_allocation_ZH.md#s6-10) 与 [6.11](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)、[Corollary 6.12](06-12_learner_separations_ZH.md#s6-12)：分配与学习器权重（R2）。
- [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)、[Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)、[Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4)：受限振幅模型、排序器的不一致、裁剪（R3）。
- [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)：optimizer 记忆（T1）。

**被用于.**

- [Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4)：真实模型上的裁剪检验。
- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 与 [Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3)：它们的真实模型检验。
- [Remark 9.5](09-05_assumption_audit_ZH.md#s9-5)：真实模型检验涉及哪些假设。

**预注册.** R0–R3 的配置、预测、判据与预算见 [R0–R3 预注册](../../docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md)，其中附录 B 记录了 R2 的权重更正；T1 的见 [T1 预注册](../../docs/direction/2026-10-14_T1_optimizer_state_transplant_prereg_ZH.md)。

## 相关工作

- R1 把 CRN 排序与模仿学习和蒸馏中的候选生成方式相比较；在学生自己的状态上做 teacher 修正，就是 DAgger 的查询（[Ross et al., 2011](#ref-ross2011reduction)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-hu2022lora"></a>

**Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022).** LoRA: Low-rank adaptation of large language models. *The Tenth International Conference on Learning Representations (ICLR)*. [openreview.net/forum?id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9); arXiv [2106.09685](https://arxiv.org/abs/2106.09685).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).
