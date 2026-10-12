# Corollary 6.7 · 混叠与 replay 的零价值

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-07_aliasing_zero_value_replay_ZH.md) · [证明](../proofs/06-07_aliasing_zero_value_replay_ZH.md) · [实验](../experiments/06-07_aliasing_zero_value_replay_ZH.md) · [English](06-07_aliasing_zero_value_replay.md)

**本页证明什么.** 两个环境的 transcript 分布相同时，对每个 $`K`$，最坏误差至少是泛函差距的一半。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。

**陈述（[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)）.**

- 若 $`\theta_0\neq\theta_1`$ 满足 $`P^D_{\theta_0}=P^D_{\theta_1}`$、$`\Psi(\theta_0)\neq\Psi(\theta_1)`$，并且 $`n=0`$ 或对每个 $`x`$ 都有 $`P^Q_{\theta_0}(\cdot\mid x)=P^Q_{\theta_1}(\cdot\mid x)`$，则对每个 $`K`$，任何蒸馏器都有 $`\max_i\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$。
- replay、改写、合成都是 transcript 与私有随机性的函数，因此包含在"任何蒸馏器"之中，不能降低这个下界。
- 这个下界是学习器相对的：示范是否充分，取决于示范看不见的方向在**这个学习器**的 $`c`$ 上有没有分量。同一批示范可以对一个学习器充分、对另一个不充分（[Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)(a)）。
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability_ZH.md#s2-6)(a) 是其中一个实例：两个世界的示范分布相同，学生的最优策略不同；[Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information_ZH.md#s2-4) 说的是后处理帮不上忙。

**证明用到的结果.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：(a)，两点界，此时 TV 为零。

**证明.** 两条 transcript 的分布相同，所以 $`\mathrm{TV}=0`$，[Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)(a) 中平方根项为零。$`\square`$
