# Corollary 6.8 · 示范与查询的兑换率

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md) · [证明](../proofs/06-08_exchange_rate_between_demonstrations_queries_ZH.md) · [English](06-08_exchange_rate_between_demonstrations_queries.md)

**一段话概括.** 示范与查询逐方向地共用一份信息预算。在起约束作用的方向上，一条示范值确定数量的查询，而这个数量由学习器决定。

**本页用到的设定与记号.**

- **环境.** 环境族 $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知（[Definition 6.1](../statements/06-01_environment_family_and_distiller_ZH.md#s6-1)）。
- **oracle 反馈** $`P^*_\theta`$：若已知 $`\theta`$ 就会使用的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据.** $`K`$ 条独立同分布、分布为 $`P^D_\theta`$ 的示范，以及 $`n`$ 次可以自适应的查询：查询点 $`X_i\in\mathcal X_Q`$ 的结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。整条 transcript 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 从 transcript 与私有随机性（其分布与 $`\theta`$ 无关）到 feedback law $`Q`$ 的任何可测映射。replay、改写、合成都是蒸馏器（[Definition 1.2](../README_ZH.md#s1-2)）。
- **被审计的状态** $`z_1,\dots,z_T`$：固定的 complete learner state，即参数连同 optimizer state（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在各 checkpoint 到达的状态。
- **平均增量** $`\mu^P_t(z)`$：在 $`z`$ 上消费一条来自 $`P`$ 的消息时，complete learner state 的期望变化（[Definition 1.3](../README_ZH.md#s1-3)）。对 oracle 记 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$，**学习器 Jacobian** 为 $`J_t=D_\theta\mu^*_t`$。
- **任务方向** $`g_t=\nabla V_{t+1}(z_t)`$：声明的参考尾段下 continuation value 的梯度（[Definition 1.4](../README_ZH.md#s1-4)），与 $`\theta`$ 无关。
- **学习器相对泛函** $`\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle`$，梯度 $`c=\nabla_\theta\Psi=\sum_tJ_t^*g_t`$；**一阶 fidelity 误差** $`\mathrm{Err}_\theta(Q)=\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\rvert`$（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)，它是被审计状态上 oracle 反馈与 $`Q`$ 之间最终任务得分差距的一阶部分。
- **Fisher information**：一条示范的 $`I_D(\theta)`$，在 $`x\in\mathcal X_Q`$ 处一次查询的 $`I_Q(\theta;x)`$。对方向 $`h\in\mathbb R^p`$，$`\mathbb I_{K,n}(\theta;h)=K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h`$。
- **learner-relative information norm** $`\mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\lvert c(\theta)^\top h\rvert/\sqrt{\mathbb I_{K,n}(\theta;h)}`$，约定 $`0/0=0`$，$`a>0`$ 时 $`a/0=\infty`$。
- **增长条件 (G)**：存在 $`N\to\infty`$ 与 $`0<a\le b<\infty`$，使对每个单位向量 $`h`$ 有 $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$。

<a id="s6-8"></a>

**Corollary 6.8（示范与查询的兑换率）.** 对方向 $`h`$ 记 $`e_D(h)=h^\top I_D(\theta_0)h`$，$`e_Q(h)=\sup_xh^\top I_Q(\theta_0;x)h`$。在 [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)(b) 的条件下，若一个蒸馏器的局部最坏期望一阶误差不超过 $`\varepsilon`$，则对每个 $`h`$ 必须有

```math
K\,e_D(h)+n\,e_Q(h)\ge\frac{(c(\theta_0)^\top h)^2}{16\,\varepsilon^2}\,\big(1-o(1)\big).
```

- $`(K,n)`$ 的可行集是一族半平面的交。在可行集边界上，起约束作用的方向 $`h^\star`$ 决定了局部兑换率：一条示范值 $`e_D(h^\star)/e_Q(h^\star)`$ 次查询。
- $`h^\star`$ 由学习器的 $`c`$ 决定，所以兑换率是学习器相对的：同一批示范对不同的学习器值不同数量的查询。
- [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11) 是两世界检验情形下类似的线性预算；这里的预算逐方向成立，并按学习器加权。

**说了什么.**

- 示范与查询逐方向地进入同一个线性信息预算；在起约束作用的方向上，一条示范值确定数量的查询，而这个数量取决于学习器。

**直觉.**

- 示范与查询都是证据，它们的相对价值在于各自在这个学习器关心的方向上揭示了多少。

**在框架中的作用.**

- 它给出"多收集示范"与"花查询"之间的定量取舍；是 [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11) 的学习器相对形式。

**推导要点.**

- 对 [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)(b) 逐方向移项。

**用到.**

- [Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)：逐方向的局部下界。
- [Proposition 2.11](../statements/02-11_shared_information_budget_ZH.md#s2-11)：两世界的信息预算。

**被用于.**

- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3)：设计规则：先获取、再编译。

**状态.** 已证明。
