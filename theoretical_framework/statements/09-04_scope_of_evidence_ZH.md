# Remark 9.4 · 受控证据覆盖了什么，以及计划中的真实模型检验

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](09-04_scope_of_evidence_ZH.md) · [English](09-04_scope_of_evidence.md)

**一段话概括.** 本框架中的每一项检验都是精确有限计算，或在受控小网络中运行。本 remark 直白地说明这个范围，并列出为在一个实际的 agent 蒸馏配方上检验三个主要结果而设计的真实模型检验：Theorem A（学习器需要多少信息、在哪里获取：[Theorems 6.3](06-03_learner_relative_lower_bound_ZH.md#s6-3) 与 [6.5](06-05_achievability_ZH.md#s6-5)）、Theorem B（Adam 的二阶矩通道：[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 及其推论）与 Theorem C（CRN 配对替换证书：[Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)）。它没有自己的证明页与实验页。

**本页用到的设定与记号.**

- *配方.* ALFWorld（[Shridhar et al., 2021](#ref-shridhar2021alfworld)）；主表配方：在 $`K=20`$ 条可信示范上做监督微调，LoRA 学生（[Hu et al., 2022](#ref-hu2022lora)；Gemma-4-12B-it，rank 16）用常数学习率的 AdamW（[Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)）训练，全局梯度范数裁剪为 1.0。
- *保存的状态.* $`z_{10}`$：训练中途保存的完整学习器状态（adapter、AdamW 的矩 $`m,v`$、步数计数器与随机状态），每项检验都从它重新开始。
- *surrogate 与评测.* $`\tilde F`$：留出的 teacher 成功轨迹上的平均 teacher-forced log-likelihood，只用于打分。任务评测在留出的开发集上进行；sealed 的测试划分永不使用。噪声参照（同一 adapter 在第二个 server 上再评一次）确定多大的差别才算数。
- *CRN 配对替换分数.* $`\hat S^{jk}`$：从 $`z_{10}`$ 出发，把下一次参考更新换成候选 $`j`$（或 $`k`$）的一个 batch，按共享种子确定的顺序走 $`T`$ 次参考更新，用 $`\tilde F`$ 打分，再把配对差在 $`r`$ 个副本上平均（[Definition 8.6](08-06_crn_paired_replacement_score_ZH.md#s8-6)）。
- *逐坐标的量.* $`\lambda`$：消息相对 running RMS $`\sqrt v`$ 的幅度；$`\rho`$：从 optimizer state 读出的信噪比；$`\lambda^\star`$：[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 的价值峰值。
- *学习器权重.* $`W_s`$：状态簇 $`s`$ 的学习器相对权重；$`d(s)`$ 是它的学生访问率，$`k_s`$ 是 teacher 覆盖，$`\sigma_s^2`$ 是簇内单样本梯度的二阶矩；water-filling 规则 $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ 中的 $`\lambda>0`$ 是由查询预算确定的乘子，不是幅度比（[Propositions 6.10](06-10_water_filling_allocation_ZH.md#s6-10)–[6.11](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)）。

<a id="s9-4"></a>

**Remark 9.4（受控证据覆盖了什么，以及计划中的真实模型检验）.**

- **已覆盖.** [Proposition 9.1](09-01_coupling_width_ZH.md#s9-1) 由精确有限计算和一个受控小网络检验；[Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3) 的选择规则在该网络的冻结留出面板上检验（[Check 9.4](../experiments/09-03_derivation_design_rules_ZH.md#c9-4)、[Check 9.5](../experiments/09-03_derivation_design_rules_ZH.md#c9-5)）。Theorem A、B、C 的检验在它们所检验的陈述的实验页上。这些都是精确有限计算或受控小网络。
- **未覆盖.** 框架中的任何陈述都没有真实模型或 benchmark 结果的支持。下面的真实模型检验已经设计好并预先注册；这些页面不报告它们的任何结果。
- **计划中的真实模型检验**（ALFWorld，上述配方，从保存的状态 $`z_{10}`$ 出发）。探索阶段每个配置单 seed；预测、判据以及算力与 teacher 预算都在它们所预测的运行之前写定。
  - *R0（已有 pilot 的再分析）.* 在 pilot 的 probe 状态上，报告敏感度加权的冻结分母误差及其最大项，并与一阶项对比（[Remark 5.11](05-11_checking_warm_state_conditions_ZH.md#s5-11)）；只有低于预先注册的阈值时，才说 warm Adam 的描述成立。用不相交 batch 对上的 CRN 尾段测量 **response** 的不可加性，与已测的 update map 不可加性并列。
  - *R1（Theorem C：同状态、同预算的排序）.* 从 $`z_{10}`$ 出发比较四个候选 feedback law：$`K`$ 条示范的 replay（参考）、teacher 在学生状态上的修正（查询，与 DAgger 相同：[Ross et al., 2011](#ref-ross2011reduction)）、经环境奖励筛选的自生成轨迹、teacher 对示范的改写。按 CRN 配对替换分数排序（[Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)），在任何续训之前记下排序，再用每个候选继续训练并评测。
    - 判据：被认证的排序与实际排序的秩一致性（Kendall $`\tau`$），以及实际差距超过噪声参照的每个被认证候选对是否同号。
    - 校准 $`(\alpha,\varepsilon_{\rm sur})`$ 由全尾段结果估计（[Remark 8.11](08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）；只有三个候选对，它的迁移在这里无法检验，这仍是 [Remark 8.12](08-12_open_problems_ZH.md#s8-12) 的开放问题。
    - 对照排序器按定义实现，不作为 baseline 报告：LESS 式余弦分数（[Xia et al., 2024](#ref-xia2024less)）、梯度范数、只看参数的一阶分数，以及静态 proxy。
  - *R2（Theorem A：查询分配）.* 查询预算固定，比较不查询、在学生状态上均匀分配、按 SGD 权重 $`W_s\propto d(s)^2\sigma_s^2`$ 做 water-filling，以及按 warm Adam 权重 $`W_s\propto d(s)\sigma_s^2`$ 做学习器相对的 water-filling，$`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$（[Proposition 6.10](06-10_water_filling_allocation_ZH.md#s6-10)）。[Proposition 6.11](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) 的闭式权重作为可计算的分配规则；真实模型的参数在状态之间共享，它的 state-owned 假设不成立，所以检验的是这条规则是否有用，而不是闭式是否精确。各臂的预测顺序与幅度事先由 Proposition 6.10 的方差公式算出；两个 water-filling 臂的比较检验 [Corollary 6.12](06-12_learner_separations_ZH.md#s6-12) 的学习器分离。
  - *R3（Theorem B：二阶矩通道）.* 在 $`z_{10}`$ 上把一个参考 batch 乘以倍数 $`c`$，使余弦分数对所有 $`c`$ 相同、只有幅度 $`\lambda`$ 变化；这一步关闭裁剪，另外也在保持裁剪时运行。预测的 response 曲线事先用保存的 optimizer state、由 [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)(a) 的 mean-field 公式算出；保持裁剪时，[Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4) 的裁剪上限预言：裁剪后梯度相同的倍数之间没有差别。凡实测价值随 $`c`$ 下降之处，余弦分数与只看参数的分数会把相应的倍数对排错（[Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)）；记录排错的对数。
  - baseline 只用完整的官方代码运行。

**说了什么.** 目前所有证据都是精确的或受控的；真实模型检验 R0–R3 已预先注册，依次针对 warm 状态条件与 response 的不可加性、证书（Theorem C）、查询分配（Theorem A）与二阶矩通道（Theorem B）。

**在框架中的作用.**

- 它标出结论的边界：在 R0–R3 报告之前，框架不对真实模型声称任何结论。
- 它把每项计划中的检验与它所检验的陈述联系起来，使真实模型的结果可以直接与事先写下的预测对照。

**用到.**

- [Remark 5.11](05-11_checking_warm_state_conditions_ZH.md#s5-11)：真实模型上的 warm 状态检查（R0）。
- [Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)、[Remarks 8.11](08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)–[8.12](08-12_open_problems_ZH.md#s8-12)：证书、它的校准与开放问题（R1）。
- [Propositions 6.10](06-10_water_filling_allocation_ZH.md#s6-10)–[6.11](06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)、[Corollary 6.12](06-12_learner_separations_ZH.md#s6-12)：分配与学习器权重（R2）。
- [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)、[Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)、[Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4)：幅度定律、排序器不一致性、裁剪（R3）。

**被用于.**

- [Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4)：真实模型上的裁剪检验。
- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 与 [Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3)：它们的计划中的真实模型检验。

**预先注册.** 完整的配置、预测、判据与预算见[预注册文档](../../docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md)。

## 参考文献

- [Xia et al., 2024](../references_ZH.md#xia2024less)：LESS，余弦分数对照排序器的来源。
