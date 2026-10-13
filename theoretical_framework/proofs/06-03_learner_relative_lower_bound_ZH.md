# Theorem 6.3 · 学习器相对的下界

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-03_learner_relative_lower_bound_ZH.md) · [证明](../proofs/06-03_learner_relative_lower_bound_ZH.md) · [实验](../experiments/06-03_learner_relative_lower_bound_ZH.md) · [English](06-03_learner_relative_lower_bound.md)

**本页证明什么.** 任何蒸馏器在两个环境上的最坏期望一阶误差，至少是学习器相对泛函之差的一半乘以 1 减去两条 transcript 分布之间的 TV；在局部，这给出信息范数的四分之一。

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

**陈述（[Theorem 6.3](../statements/06-03_learner_relative_lower_bound_ZH.md#s6-3)）.**

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

**证明用到的结果.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：蒸馏器一项 $`\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$ 与 $`\theta`$ 无关；$`\mathbb I_{K,n}`$、$`\mathfrak n_{K,n}`$ 与 (G) 的定义。

**用到的经典工具.** Le Cam 两点法（[Le Cam, 1973](../references_ZH.md#lecam1973convergence)）；Pinsker 不等式与 KL 散度的链式法则（[Cover & Thomas, 2006](#ref-cover2006elements)）；differentiability in quadratic mean 下的局部渐近理论（[van der Vaart, 1998](../references_ZH.md#vandervaart1998asymptotic)）。

**证明.** (a) 记 $`s=\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$，$`a(Q)=\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$；后者与 $`\theta`$ 无关（[Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)）。由三角不等式，

```math
\mathrm{Err}_{\theta_0}(Q)+\mathrm{Err}_{\theta_1}(Q)=\lvert\Psi(\theta_0)-a(Q)\rvert+\lvert\Psi(\theta_1)-a(Q)\rvert\ge s .
```

$`Q`$ 是 transcript $`D`$ 与 $`U`$ 的函数，而 $`U`$ 在两个世界中分布相同；把 $`U`$ 并入 $`D`$，两条 transcript 分布之间的总变差距离不变。记 $`p_i`$ 为 $`\mathsf P^{K,n}_{\theta_i}`$ 关于 $`p_0+p_1`$ 的密度，则

```math
\mathbb E_{\theta_0}\mathrm{Err}_{\theta_0}+\mathbb E_{\theta_1}\mathrm{Err}_{\theta_1}\ge\int\min(p_0,p_1)\big(\mathrm{Err}_{\theta_0}+\mathrm{Err}_{\theta_1}\big)\ge s\big(1-\mathrm{TV}(\mathsf P^{K,n}_{\theta_0},\mathsf P^{K,n}_{\theta_1})\big).
```

由 Pinsker 不等式，$`\mathrm{TV}\le\sqrt{\mathrm{KL}(\mathsf P^{K,n}_{\theta_1}\Vert\mathsf P^{K,n}_{\theta_0})/2}`$。沿 transcript 对 KL 用链式法则：

- 示范贡献 $`K\,\mathrm{KL}_D`$；
- 第 $`i`$ 次查询的结果在给定历史时贡献 $`\mathbb E\,\mathrm{KL}_Q(X_i)\le\sup_x\mathrm{KL}_Q(x)`$；
- $`X_i`$ 的选择与私有随机性在两个世界中条件分布相同，贡献为零。

所以 KL 不超过 $`K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)`$。最大值不小于平均值，即得 (a)。

(b) 由 $`\mu^*_t\in C^1`$ 且 $`g_t`$ 固定，$`\Psi\in C^1`$，$`\lvert\Psi(\theta_0+\tau h)-\Psi(\theta_0)\rvert=\tau\lvert c(\theta_0)^\top h\rvert+o(\tau)`$。由 KL 的假设，

```math
K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\le\tfrac12\tau^2\,\mathbb I_{K,n}(\theta_0;h)\,(1+o(1)).
```

取 $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}\to0`$，(a) 中的平方根项不超过 $`\tfrac12(1+o(1))`$。于是 (a) 的下界至少为 $`\tfrac12\cdot\tau\lvert c^\top h\rvert(1+o(1))\cdot\tfrac12(1-o(1))`$，即所要的结论；写成相对形式用到 $`c(\theta_0)^\top h\neq0`$。

(c) 在 (G) 下，比值 $`\lvert c^\top h\rvert/\sqrt{\mathbb I_{K,n}(\theta_0;h)}`$ 在单位球面上连续，在某个 $`h^\star`$ 处取到上确界。若 $`c(\theta_0)\neq0`$，把 $`h^\star`$ 与 $`c/\lVert c\rVert`$ 比较，得 $`\lvert c^\top h^\star\rvert\ge\lVert c\rVert\sqrt{a/b}>0`$，所以 (b) 中的 $`o(1)`$ 项沿 $`h^\star=h^\star_{K,n}`$ 一致。对 $`h=h^\star`$ 用 (b) 即可。若 $`c(\theta_0)=0`$，结论平凡。$`\square`$

## 相关工作

- **下界技术.** 两点法来自 Le Cam（[Le Cam, 1973](../references_ZH.md#lecam1973convergence)）；带 Fisher information 与 differentiability in quadratic mean 的局部形式遵循标准的局部渐近 minimax 理论（[van der Vaart, 1998](../references_ZH.md#vandervaart1998asymptotic)）。这里特有的是泛函本身：它由学习器的增量映射与声明的任务决定。
- **数据与 teacher 的选择分数.** LESS（[Xia et al., 2024](../references_ZH.md#xia2024less)）与 GRACE（[Panigrahi et al., 2025](../references_ZH.md#panigrahi2025good)）给候选数据或 teacher 打分。Theorem A 不是又一个分数：它是任何蒸馏器都无法突破的下界，连同达到它的获取规则。

## 参考文献

- [Le Cam, 1973](../references_ZH.md#lecam1973convergence)
- [Panigrahi et al., 2025](../references_ZH.md#panigrahi2025good)
- [van der Vaart, 1998](../references_ZH.md#vandervaart1998asymptotic)
- [Xia et al., 2024](../references_ZH.md#xia2024less)
