# Corollary 6.12 · 学习器之间的分离

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-12_learner_separations_ZH.md) · [证明](../proofs/06-12_learner_separations_ZH.md) · [实验](../experiments/06-12_learner_separations_ZH.md) · [English](06-12_learner_separations.md)

**一段话概括.** 同一批示范可以对一个学习器充分、对另一个不充分。精确地说：当一个学习器忽略示范不揭示的尺度时；定量地说：当一条很少被覆盖的长尾状态对 warm Adam 比对 SGD 更重要时。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。
- **agent 状态** $`s\in\mathcal S`$，有限集；本页中 $`s`$ 表示 agent 的状态（[Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)）。
- **逐状态的 oracle 分布.** 在状态 $`s`$ 上 oracle 消息 $`Y\sim P^*_s`$（例如 teacher 在该状态的动作或修正），取值于有限字母表；各状态的分布互不相关，不假设状态之间有泛化。
- **influence function** $`f_s(y)`$：$`\Psi`$ 在 $`s`$ 处、把 $`P^*_s`$ 向 $`y`$ 处点质量移动时的导数；在 $`P^*_s`$ 下均值为零。
- **学习器相对权重** $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$：在 $`s`$ 上多一个 oracle 样本，对估计 $`\Psi`$ 值多少。
- **样本数.** 状态 $`s`$ 共得到 $`N_s=k_s+n_s`$ 个 oracle 样本：$`k_s`$ 来自示范（期望为 $`KH\,d_T(s)`$，$`H`$ 为轨迹长度，$`d_T`$ 为 teacher 的状态访问分布），$`n_s`$ 来自查询，$`\sum_sn_s=n`$。

<a id="s6-12"></a>

**Corollary 6.12（学习器之间的分离）.**

- **(a) 尺度不变性带来的精确分离.** 设 $`\theta=(\alpha,b)`$，$`b>0`$，oracle 的 consumed gradient 为 $`b\,G_\alpha(Y)`$（例如反馈的权重或 advantage 的尺度），示范的分布只依赖于 $`\alpha`$。
  - 对始终激活的范数裁剪 SGD（裁剪阈值 $`c_{\rm clip}`$，几乎处处 $`\lVert bG_\alpha(Y)\rVert>c_{\rm clip}`$）以及 normalized SGD，oracle 平均增量与 $`b`$ 无关，所以 $`\Psi`$ 只依赖于 $`\alpha`$，$`c(\theta)`$ 在 $`b`$ 方向上没有分量。只要示范关于 $`\alpha`$ 的 Fisher information 非退化，误差随 $`K\to\infty`$ 趋于零（对关于 $`\alpha`$ 的模型用 [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)，$`n=0`$；可实现性对任何使裁剪保持激活的尺度都成立）。
  - 对普通 SGD，$`c(\theta)`$ 的 $`b`$ 分量为 $`-\eta\sum_t\langle g_t,\mathbb E\,G_\alpha\rangle`$；当它非零时，示范不含关于 $`b`$ 的任何信息，由 [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)，任何 $`K`$ 都不够。
  - 所以同一批示范对一个学习器充分，对另一个不充分。若裁剪只以概率 $`p_{\rm act}<1`$ 激活，[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9) 保证每个裁剪后的平均 consumed gradient 与不含 $`b`$ 的向量 $`c_{\rm clip}\,\mathbb E[G_\alpha/\lVert G_\alpha\rVert]`$ 相差不超过 $`c_{\rm clip}(1-p_{\rm act})`$，分离随之平滑地减弱。
- **(b) 长尾覆盖带来的定量分离.** 由 [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10) 与 [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)(d)：当学生的状态分布有大量低频状态（都处在 warm Adam 的频繁访问区间）、而 teacher 很少到达这些状态时，这些状态承担的信息需求份额在 warm Adam 下比在 SGD 下更大；要达到同样的相对精度，warm Adam 在这些状态上需要相应更多的示范或查询（[Example 6.13](../statements/06-13_long_tail_example_ZH.md#s6-13)）。

**说了什么.**

- 揭示了反馈方向、却不揭示其尺度的示范，对始终裁剪的学习器充分，对 SGD 则对每个 $`K`$ 都不充分。定量地看，一条很少被覆盖的长尾状态，让 warm Adam 付出的信息代价远高于 SGD。

**直觉.**

- 学习器忽略的东西不必告诉它；学习器放大的东西必须告诉得很准。

**在框架中的作用.**

- 它把"同一批示范可以对一个学习器充分、对另一个不充分"变成了形式化的结论。

**推导要点.**

- (a) 始终激活的裁剪去掉了尺度，所以 $`c`$ 在 $`b`$ 方向上没有分量；对 SGD，尺度改变 $`\Psi`$ 而示范分布不变，这正是 [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)。若裁剪只以概率 $`p_{\rm act}`$ 激活，[Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9) 把剩下的尺度依赖控制在 $`c_{\rm clip}(1-p_{\rm act})`$ 以内。(b) 把 [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10) 与 [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) 的权重结合。

**用到.**

- [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)：裁剪学习器的可达性。
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)：SGD 的混叠。
- [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)：显式信息公式。
- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)：学习器权重。
- [Proposition 5.9](../statements/05-09_partially_active_clipping_ZH.md#s5-9)：部分激活的裁剪。

**被用于.**

- [Example 6.13](../statements/06-13_long_tail_example_ZH.md#s6-13)：(b) 的算例。
- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3)：设计规则：匹配学习器。

**状态.** 已证明；(b) 在所述区间内首项意义下成立。

## 相关工作

- 范数裁剪就是 [Pascanu et al., 2013](#ref-pascanu2013difficulty) 的梯度裁剪规则，normalized SGD 是 [Hazan et al., 2015](#ref-hazan2015beyond) 研究的方法。两者都让步长与梯度的尺度无关，这正是 (a) 中裁剪学习器的泛函里没有 $`b`$ 的原因。
- (b) 中的 warm Adam 是矩估计已经热身之后的 Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-hazan2015beyond"></a>

**Elad Hazan, Kfir Y. Levy, Shai Shalev-Shwartz (2015).** Beyond convexity: Stochastic quasi-convex optimization. *Advances in Neural Information Processing Systems 28 (NIPS)*. [proceedings.neurips.cc/paper/2015/hash/934815ad542a4a7c5e8a2dfa04fea9f5](https://proceedings.neurips.cc/paper/2015/hash/934815ad542a4a7c5e8a2dfa04fea9f5-Abstract.html); arXiv [1507.02030](https://arxiv.org/abs/1507.02030).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-pascanu2013difficulty"></a>

**Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013).** On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, PMLR 28, no. 3, 1310–1318. [proceedings.mlr.press/v28/pascanu13](https://proceedings.mlr.press/v28/pascanu13.html).
