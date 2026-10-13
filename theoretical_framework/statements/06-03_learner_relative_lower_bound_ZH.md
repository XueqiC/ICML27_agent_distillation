# Theorem 6.3 · 学习器相对的下界

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-03_learner_relative_lower_bound_ZH.md) · [证明](../proofs/06-03_learner_relative_lower_bound_ZH.md) · [实验](../experiments/06-03_learner_relative_lower_bound_ZH.md) · [English](06-03_learner_relative_lower_bound.md)

**一段话概括.** 任何蒸馏器（包括 replay、改写、合成）在声明任务上复现 oracle 反馈的一阶效应时，误差都不低于 learner-relative information norm 的四分之一。这是 Theorem A 的下半部分。

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

<a id="s6-3"></a>

**Theorem 6.3（学习器相对的下界）.**

- **(a) 精确两点界.** 对任意 $`\theta_0,\theta_1\in\Theta`$ 与任意蒸馏器，

```math
\max_{i\in\lbrace0,1\rbrace}\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\big\lvert\Psi(\theta_1)-\Psi(\theta_0)\big\rvert\Big(1-\sqrt{\tfrac12\big(K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\big)}\Big),
```

  其中 $`\mathrm{KL}_D=\mathrm{KL}(P^D_{\theta_1}\Vert P^D_{\theta_0})`$，$`\mathrm{KL}_Q(x)=\mathrm{KL}(P^Q_{\theta_1}(\cdot\mid x)\Vert P^Q_{\theta_0}(\cdot\mid x))`$。

- **(b) 局部界.** 设每个 $`\mu^*_t`$ 在 $`\theta_0`$ 附近属于 $`C^1`$，两个实验在 $`\theta_0`$ 处 differentiable in quadratic mean，并且当 $`\tau\to0`$ 时

```math
\mathrm{KL}(P^D_{\theta_0+\tau h}\Vert P^D_{\theta_0})\le\tfrac12\tau^2h^\top I_D(\theta_0)h\,(1+o(1)),\qquad \mathrm{KL}_Q(x)\le\tfrac12\tau^2h^\top I_Q(\theta_0;x)h\,(1+o(1)),
```

  且对 $`x\in\mathcal X_Q`$ 与单位向量 $`h`$ 一致成立。取单位方向 $`h`$，满足 $`c(\theta_0)^\top h\neq0`$ 与 $`\mathbb I_{K,n}(\theta_0;h)\to\infty`$，令 $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}`$。则

```math
\inf_Q\max_{\theta\in\lbrace\theta_0,\,\theta_0+\tau h\rbrace}\mathbb E_\theta\mathrm{Err}_\theta(Q)\ge\frac{\lvert c(\theta_0)^\top h\rvert}{4\sqrt{\mathbb I_{K,n}(\theta_0;h)}}\,\big(1-o(1)\big).
```

- **(c) 范数就是局部下界.** 在 (G) 下，取 $`h`$ 为 [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2) 中比值的最大点，局部 minimax 下界为 $`\tfrac14\mathfrak n_{K,n}(\theta_0)\,(1-o(1))`$。

**说了什么.**

- 任何蒸馏器（包括 replay、改写、合成）在声明任务上复现 oracle 反馈的一阶效应时，误差都不低于 learner-relative information norm $`\mathfrak n_{K,n}`$ 的四分之一。这个范数度量的是：$`K`$ 条示范与 $`n`$ 次查询，在学习器与任务真正在意的那个方向 $`c`$ 上，揭示了多少关于环境的信息。

**直觉.**

- 数据分不开、而这个学习器会区别对待的两个环境，必然让蒸馏器在其中一个上出错；要紧的是学习器的敏感方向 $`c`$，而不是环境之间差得有多大。

**在框架中的作用.**

- 它是 Theorem A 的下半部分，也是"先获取、再编译"的定量核心；混叠（[Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)）与兑换率（[Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8)）都由它得到。

**推导要点.**

- 蒸馏器自身的贡献 $`\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$ 与 $`\theta`$ 无关，所以由三角不等式，两个世界中的误差之和至少是 $`\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$。
- Le Cam 两点论证：在两条 transcript 分布重叠的部分，没有蒸馏器能在两个世界里都对，所以期望误差之和至少是差距乘以 $`1-\mathrm{TV}`$。
- Pinsker 不等式加上沿自适应 transcript 的 KL 链式法则：示范贡献 $`K\,\mathrm{KL}_D`$，每次查询至多贡献 $`\sup_x\mathrm{KL}_Q(x)`$，查询的选择在两个世界中遵循同一规则，贡献为零。
- 局部地，KL 是位移的二次型，系数是 Fisher information；取 $`\tau=\mathbb I^{-1/2}`$ 使总变差不超过 $`1/2`$，由此得到常数 $`1/4`$。

**示例.** 一维参数、$`\Psi(\theta)=c\theta`$，每条示范的 Fisher information 为 $`i_D`$，每次查询为 $`i_Q`$，则 $`\mathfrak n_{K,n}=\lvert c\rvert/\sqrt{Ki_D+ni_Q}`$。敏感度 $`\lvert c\rvert`$ 越大的学习器，要达到同样的误差就需要成比例更多的信息。

**用到.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：泛函、误差与范数。
- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)：Err 是一阶的结果差距。

**被用于.**

- [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)：匹配的上界。
- [Remark 6.6](../statements/06-06_class_uniform_fidelity_ZH.md#s6-6)：对声明类一致的版本。
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay_ZH.md#s6-7)：混叠：TV 为零。
- [Corollary 6.8](../statements/06-08_exchange_rate_between_demonstrations_queries_ZH.md#s6-8)：兑换率。
- [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)：状态覆盖下的显式界。
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)：CED 的获取步骤。

**状态.** 已证明；(b)、(c) 为局部渐近 minimax 意义。

## 相关工作

- 两点法来自 Le Cam（[Le Cam, 1973](#ref-lecam1973convergence)）；Pinsker 不等式与 KL 散度的链式法则是标准工具（[Cover & Thomas, 2006](#ref-cover2006elements)）；differentiability in quadratic mean 下的局部形式遵循局部渐近 minimax 理论（[van der Vaart, 1998](#ref-vandervaart1998asymptotic)）。
- 这里特有的是泛函本身：它由学习器的增量映射与声明的任务决定，所以拿到同样数据的两个学习器可以面对不同的下界。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-cover2006elements"></a>

**Thomas M. Cover, Joy A. Thomas (2006).** Elements of Information Theory. 2nd edition, Wiley-Interscience, Hoboken, NJ. DOI [10.1002/047174882X](https://doi.org/10.1002/047174882X).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-vandervaart1998asymptotic"></a>

**A. W. van der Vaart (1998).** Asymptotic Statistics. Cambridge University Press, Cambridge (Cambridge Series in Statistical and Probabilistic Mathematics 3). DOI [10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
