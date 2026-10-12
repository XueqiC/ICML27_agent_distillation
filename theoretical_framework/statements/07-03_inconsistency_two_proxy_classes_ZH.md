# Proposition 7.3 · 两类 proxy 的不一致性

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](07-03_inconsistency_two_proxy_classes_ZH.md) · [证明](../proofs/07-03_inconsistency_two_proxy_classes_ZH.md) · [实验](../experiments/07-03_inconsistency_two_proxy_classes_ZH.md) · [English](07-03_inconsistency_two_proxy_classes.md)

**一段话概括.** 数据选择与 data attribution 方法常用一个在当前状态上算出的便宜分数给训练样本排序：只依赖数据的统计量、经由参数步预测的损失变化，或训练梯度与目标梯度的余弦。本命题证明其中两类不是一致的排序器。不看部署任务的统计量，必然在某个任务上把某对反馈律排错。而在 warm Adam 下、[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 幅度定律的区间里，只看参数的分数会把两条同向消息排反，余弦分数则区分不了它们。

**本页用到的设定与记号.**

- *反馈律与学习价值.* 反馈律 $`P`$（消息律）由实际学习器在固定状态上消耗；$`L_F(P)`$ 是得分为 $`F\in[0,1]`$ 的部署任务上由此得到的期望最终得分（[Definitions 1.3–1.4](../README_ZH.md#s1-3)）。"排对"指按 $`L_F`$ 给反馈律排序。
- *task-blind 统计量.* 一个实值统计量 $`\mathcal T(P)`$，只依赖于当前状态与反馈律、不依赖于任务，并按它的值给反馈律排序。
- *Corollary 7.2 的 Adam 区间.* warm、不做 bias correction 的 Adam 的单个坐标 $`i`$，$`\epsilon=0`$，无 decay；尾段梯度与 $`\theta`$ 无关，均值 $`\bar g_i\ne0`$、平方的均值 $`\bar v_i`$；得分关于 $`\theta_i`$ 仿射；消息之后有 $`T`$ 次尾段更新。
  - $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$；$`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$ 是以 running RMS 为单位的消息幅度；$`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}`$ 是信噪比。
  - $`\Delta F_i(\lambda)`$ 是 Corollary 7.2(a) 的 mean-field response；$`Q_i`$、$`S_i`$、$`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$ 与误差界取自 Corollary 7.2(b)，$`\omega_T=1-\beta_2^{\,T+1}`$。
- *只看参数的分数.* $`S_{\rm PO}(g)=\langle\chi^\theta,\Delta\theta(g)\rangle`$，即 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 中单独的参数通道：限制在参数上的 continuation 梯度与消息自己的 Adam 步的配对。
- *余弦分数.* 任何只依赖于消息梯度方向（在单个坐标上，也可以只依赖于参数步方向）的分数。

<a id="s7-3"></a>

**Proposition 7.3（两类 proxy 的不一致性）.**

- **(a) task-blind 统计量.** 设 $`\mathcal T`$ 是只依赖于当前状态与 feedback law、不依赖于部署任务的统计量，并按它的值给反馈律排序。则存在两个任务，以及同一学习器、同一状态下的两个消息律 $`P,Q`$：$`P,Q`$ 的真实顺序在两个任务上相反，而 $`\mathcal T(P),\mathcal T(Q)`$ 在两个任务上都相同。因此 $`\mathcal T`$ 至少在一个任务上排错，或者分不出一个非零的差。
- **(b) 只看参数的分数与余弦分数.** 在 [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 的设定下取 $`\alpha_i>0`$，考虑两条同向消息，幅度 $`0<\lambda_1<\lambda_2`$。
  - 只看参数的分数满足 $`S_{\rm PO}=\alpha_i\eta\,h(\lambda)`$，其中 $`h(\lambda)=(\beta_1\rho_i+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$。$`h`$ 在 $`\lambda<\lambda_{\rm PO}:=(1-\beta_1)\beta_2/(\beta_1\rho_i(1-\beta_2))`$ 上严格递增。这一类包括用 Adam 步长的 TracIn 式分数和 fixed-state 的 Adam data attribution。若 $`\lambda_2<\lambda_{\rm PO}`$，它把较大的消息排在前面。
  - 余弦分数只依赖于消息梯度的方向；在单个坐标上，也可以只依赖于参数步的方向。这一类包括 LESS 式分数。任何这样的分数都给两条消息相同的值。
  - 只要 $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$，真实价值就偏好较小的消息。由 [Corollary 7.2(b)](07-02_stationary_amplitude_law_ZH.md#s7-2)，当 $`\lambda_1+\lambda_2>2\lambda^\star`$ 且 $`Q_i(\lambda_1)-Q_i(\lambda_2)>11(1-\beta_2)\max\lbrace1,\lambda_2^2\rbrace\big(S_i(\lambda_1)+S_i(\lambda_2)\big)`$ 时这一点成立；一般情形可由 [Corollary 7.2(a)](07-02_stationary_amplitude_law_ZH.md#s7-2) 直接读出。
  - 在这一区域，只看参数的分数把这对消息排反，余弦分数区分不了它们。

**说了什么.**

- 任何不看部署任务的排序规则，都不可能对所有任务都正确。
- 在 warm Adam 下有一个显式的消息幅度区间：两条同向消息中较大的那条更差，但只看参数的分数偏好它，余弦分数则给两者打平。

**直觉.**

- 只看参数的分数看到的是更大的即时一步，看不到大消息经由 $`v`$ 对此后每一步造成的损害。
- 余弦只看方向，而两条消息方向相同。

**在框架中的作用.**

- 它是一个证明，而不只是经验观察：Adam 下两类广泛使用的影响分数都不是一致的排序器。
- 这是方法按实测后果选择反馈（[Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3)，Rule 2）、并在排序前标记大消息（[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 第 4 步）的原因之一。
- 被证明的不一致性只限于这两类；对其他静态 proxy，框架只报告所测试的 proxy 没有可靠地选对。

**推导要点.**

- (a)：把任务反过来（$`F\mapsto1-F`$），所有真实顺序都反过来，而 task-blind 统计量一个也不变。[Proposition 3.1](03-01_static_value_not_learning_value_ZH.md#s3-1)(b) 的两个标签包给出一对更自然的例子。
- (b)：只看参数的分数就是消息自己这一步，它对 $`\lambda`$ 的导数与 $`(1-\beta_1)\beta_2-\beta_1\rho_i(1-\beta_2)\lambda`$ 同号。真实价值沿着凹函数 $`Q_i`$ 走，过了 $`\lambda^\star`$ 就下降；同一梯度的正倍数方向相同，所以余弦打平。

**示例（解析）.** 剩余步数长时 $`2\lambda^\star\approx2/\rho_i`$，而 $`\lambda_{\rm PO}\approx(1-\beta_1)/(\beta_1\rho_i(1-\beta_2))`$，在 $`\beta_1=0.9`$、$`\beta_2=0.99`$ 时约为 $`11/\rho_i`$。所以只看参数的分数偏好较大消息的区间（$`\lambda_2<\lambda_{\rm PO}`$），与真实价值偏好较小消息的区间（在 Corollary 7.2(b) 的误差内，$`\lambda_1+\lambda_2>2\lambda^\star`$）有重叠。重叠区间中的每一对，都被只看参数的分数排反，被所有余弦分数打平。

**用到.**

- [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)：精确幅度定律与二次律。
- [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)：定义 $`S_{\rm PO}`$ 的参数通道。
- [Proposition 3.1](03-01_static_value_not_learning_value_ZH.md#s3-1)：学习价值之差在两个任务间变号的两个标签包。

**被用于.**

- [Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4)：TracIn 式分数、fixed-state Shapley 与 LESS 分数的归类。
- [Remark 7.5](07-05_where_first_order_rankers_fail_ZH.md#s7-5)：一阶排序器在哪里失效。
- [Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3) 与 [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：按实测后果选择。

## 参考文献

- [Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating)：TracIn。
- [Ding et al., 2026](../references_ZH.md#ding2026inrun)：Adam 版 In-Run Data Shapley（fixed-state）。
- [Xia et al., 2024](../references_ZH.md#xia2024less)：LESS。
