# Lemma 6.4 · 设计的 minimax 等式

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-04_design_minimax_equality_ZH.md) · [证明](../proofs/06-04_design_minimax_equality_ZH.md) · [实验](../experiments/06-04_design_minimax_equality_ZH.md) · [English](06-04_design_minimax_equality.md)

**本页证明什么.** c-optimal design 的取值 $`\inf_\xi c^\top A(\xi)^{-1}c`$ 等于信息范数的平方，且下确界可以取到。

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
- **design.** approximate design $`\xi`$ 是查询空间上的概率测度，表示把 $`n`$ 次查询怎样分摊到各查询点上。$`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$ 是平均查询信息，$`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$ 是示范与查询合起来的总信息。

**陈述（[Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)）.** 设 $`\mathcal X_Q`$ 是紧度量空间，$`x\mapsto I_Q(\theta;x)`$ 连续。令 $`\xi`$ 取遍 $`\mathcal X_Q`$ 上的 Borel 概率测度（approximate design），$`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$，$`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$，并设对每个 $`\xi`$ 都有 $`A(\xi)\succ0`$。则

```math
\inf_\xi c^\top A(\xi)^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

并且下确界由某个 design $`\xi^*`$ 取到，它就是目标向量为 $`c=\sum_tJ_t^*g_t`$ 的 **c-optimal design**。当 $`n=0`$ 且 $`I_D`$ 奇异时：若 $`c\in\mathrm{range}(I_D)`$，等式以 $`c^\top(KI_D)^+c`$ 成立；否则两边都是 $`+\infty`$。

**证明用到的结果.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：信息范数 $`\mathfrak n_{K,n}`$ 的定义。

**用到的经典工具.** Cauchy–Schwarz 不等式、Prokhorov 定理与 Sion minimax 定理（[Sion, 1958](../references_ZH.md#sion1958general)）。

**证明.** 若 $`c=0`$，两边都为零；下设 $`c\neq0`$。对 $`A\succ0`$，Cauchy–Schwarz 给出 $`(c^\top h)^2\le(c^\top A^{-1}c)(h^\top Ah)`$，等号在 $`h=A^{-1}c`$ 处成立，所以

```math
c^\top A^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{h^\top Ah}=\frac{1}{\inf_{h\in H}h^\top Ah},\qquad H=\lbrace h:c^\top h=1\rbrace .
```

令 $`f(\xi,h)=h^\top A(\xi)h`$。引理左边等于 $`1/\sup_\xi\inf_{h\in H}f(\xi,h)`$。固定 $`h`$ 时，$`\xi\mapsto h^\top M(\xi)h=\int h^\top I_Q(x)h\,\xi(\mathrm dx)`$ 在连续函数 $`x\mapsto h^\top I_Q(x)h`$（定义在紧集 $`\mathcal X_Q`$ 上）的最大点处的点质量上取最大，所以 $`\sup_\xi f(\xi,h)=\mathbb I_{K,n}(h)`$，右边等于 $`1/\inf_{h\in H}\sup_\xi f(\xi,h)`$。剩下只需交换 $`\sup_\xi`$ 与 $`\inf_h`$：

- design 的集合是凸集，在弱拓扑下紧（Prokhorov 定理），位于有限符号测度这个局部凸空间中；
- $`f(\cdot,h)`$ 是仿射的，且弱连续，因为被积函数连续有界；
- $`f(\xi,\cdot)`$ 在凸集 $`H`$ 上凸且连续。

对 $`-f`$ 用 Sion minimax 定理（以紧凸的 design 集合为极小化变量），得 $`\max_\xi\inf_{h\in H}f=\inf_{h\in H}\sup_\xi f`$。函数 $`\xi\mapsto\inf_hf(\xi,h)`$ 是连续函数的下确界，因而上半连续，在紧的 design 集合上取到最大值，这就是 $`\xi^*`$。当 $`n=0`$ 时，Cauchy–Schwarz 的伪逆形式给出：若 $`c\perp\ker I_D`$，则 $`\sup_h(c^\top h)^2/(K\,h^\top I_Dh)=c^\top(KI_D)^+c`$（核中方向贡献 $`0/0=0`$）；否则某个 $`h\in\ker I_D`$ 满足 $`c^\top h\neq0`$，比值为 $`+\infty`$。$`\square`$

**附注.**

- 当查询信息是秩一的 $`I_Q(x)=f_xf_x^\top`$ 且 $`K=0`$ 时，Elfving 定理给出 $`\xi^*`$ 的几何刻画。$`K\,I_D`$ 这一项使它成为带先验信息的 c-optimal design。

## 相关工作

- **最优设计.** c-optimal design 及其几何刻画可以追溯到 [Elfving, 1952](../references_ZH.md#elfving1952optimum)；minimax 的交换是 Sion 定理（[Sion, 1958](../references_ZH.md#sion1958general)）；带先验信息的设计是标准内容（[Pukelsheim, 1993](../references_ZH.md#pukelsheim1993optimal)）。这里特有的是目标向量 $`c=\sum_tJ_t^*g_t`$：它由 optimizer 的增量映射与声明的任务决定。

## 参考文献

- [Elfving, 1952](../references_ZH.md#elfving1952optimum)
- [Pukelsheim, 1993](../references_ZH.md#pukelsheim1993optimal)
- [Sion, 1958](../references_ZH.md#sion1958general)
