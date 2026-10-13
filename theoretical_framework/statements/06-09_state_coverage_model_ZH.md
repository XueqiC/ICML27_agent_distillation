# Assumption 6.9 · 状态覆盖模型

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-09_state_coverage_model_ZH.md) · [English](06-09_state_coverage_model.md)

**一段话概括.** 这个假设把 Theorem A 落到 agent 蒸馏中最常见的结构上：示范覆盖 teacher 访问的状态，训练与部署却发生在学生访问的状态上。每个 agent 状态有自己的 oracle 消息分布，关于它的信息只来自 teacher 在该状态的访问与在该状态的查询。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。

<a id="s6-9"></a>

**Assumption 6.9（状态覆盖模型）.**

- **状态与分布.** 有限的 agent 状态集 $`\mathcal S`$。在状态 $`s`$ 上 oracle 消息 $`Y\sim P^*_s`$（例如 teacher 在 $`s`$ 的动作或修正），消息字母表有限。各状态的分布 $`(P^*_s)_{s\in\mathcal S}`$ 互相独立地变化：不假设跨状态的泛化。
- **泛函.** 学习器相对泛函 $`\Psi`$ 是 $`(P^*_s)_{s\in\mathcal S}`$ 的 Hadamard 可微泛函，它在 $`s`$ 处的 influence function 为 $`f_s(y)=\frac{\mathrm d}{\mathrm d\epsilon}\Psi\big(P^*_s+\epsilon(\delta_y-P^*_s)\big)\big\vert_{\epsilon=0}`$，因此 $`\mathbb E_{P^*_s}f_s=0`$。
- **样本数.** 状态 $`s`$ 得到 $`N_s=k_s+n_s`$ 个 oracle 样本：$`k_s`$ 个来自示范（期望为 $`KH\,d_T(s)`$，$`H`$ 是轨迹长度，$`d_T`$ 是 teacher 的状态访问分布），$`n_s`$ 个来自查询，$`\sum_sn_s=n`$。下面的界对给定的样本数陈述；样本数随机时，它们在给定样本数的条件下成立。

**是什么.**

- 不假设状态之间有泛化：一个状态上的样本不提供关于另一个状态的信息。
- 学习器相对泛函光滑地依赖于各状态的分布；influence function $`f_s`$ 度量在 $`s`$ 上多一个样本会怎样改变它。
- 示范样本按 teacher 的访问频率到来，查询样本落在蒸馏器选择的地方。

**在框架中的作用.**

- 它让信息范数有显式形式（[Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)），并让学习器权重可以计算（[Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)）。

**用到.**

- [Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)：环境与数据。
- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：学习器相对泛函。

**被用于.**

- [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)：显式界与 water-filling。
- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)：学习器权重。
- [Example 6.13](../statements/06-13_long_tail_example_ZH.md#s6-13)：长尾例子。

**状态.** 假设。

## 相关工作

- Hadamard 可微泛函的 influence function，以及把它变成渐近方差的 delta method，都是标准工具（[van der Vaart, 1998](#ref-vandervaart1998asymptotic)）；这里逐个状态使用它们。
- 示范所覆盖的 teacher 状态分布，与训练和部署所在的学生状态分布之间的差别，正是推动交互式模仿学习的分布偏移（[Ross et al., 2011](#ref-ross2011reduction)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-ross2011reduction"></a>

**Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell (2011).** A reduction of imitation learning and structured prediction to no-regret online learning. *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 15:627–635. [proceedings.mlr.press/v15/ross11a](https://proceedings.mlr.press/v15/ross11a.html).

<a id="ref-vandervaart1998asymptotic"></a>

**A. W. van der Vaart (1998).** Asymptotic Statistics. Cambridge University Press, Cambridge (Cambridge Series in Statistical and Probabilistic Mathematics 3). DOI [10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
