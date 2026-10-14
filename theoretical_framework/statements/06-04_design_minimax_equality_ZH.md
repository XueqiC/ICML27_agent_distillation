# Lemma 6.4 · 设计的 minimax 等式

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-04_design_minimax_equality_ZH.md) · [证明](../proofs/06-04_design_minimax_equality_ZH.md) · [实验](../experiments/06-04_design_minimax_equality_ZH.md) · [English](06-04_design_minimax_equality.md)

**一段话概括.** 在所有查询 design 中，估计学习器相对泛函所能达到的最小方差恰好等于信息范数的平方，并且最优 design 存在：它就是经典的 c-optimal（Elfving）design，目标向量由学习器与任务决定。design 不必把每个参数方向都辨识出来。只要示范和查询都照不到的那些方向不影响目标，等式就以伪逆的形式成立；只有当其中某个方向影响目标时，取值才是无穷。

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
- **奇异信息.** $`I_D(\theta)`$ 与每个 $`I_Q(\theta;x)`$ 都半正定，所以 $`A(\xi)`$ 半正定，可能奇异。对半正定矩阵 $`A`$，$`A^+`$ 是它的 Moore–Penrose 伪逆，$`\mathrm{range}(A)=(\ker A)^\perp`$。

<a id="s6-4"></a>

**Lemma 6.4（设计的 minimax 等式）.** 设 $`\mathcal X_Q`$ 是紧度量空间，$`x\mapsto I_Q(\theta;x)`$ 连续。令 $`\xi`$ 取遍 $`\mathcal X_Q`$ 上的 Borel 概率测度（approximate design），$`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$，$`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$；不要求任何 design 使 $`A(\xi)`$ 可逆。对半正定矩阵 $`A`$ 定义

```math
\Phi_c(A)=\begin{cases}c^\top A^+c, & c\in\mathrm{range}(A),\\ +\infty, & c\notin\mathrm{range}(A).\end{cases}
```

则在约定 $`0/0=0`$、$`a>0`$ 时 $`a/0=+\infty`$ 之下，

```math
\inf_\xi\Phi_c\big(A(\xi)\big)=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

并且下确界由某个 design $`\xi^*`$ 取到，它就是目标向量为 $`c=\sum_tJ_t^*g_t`$ 的 **c-optimal design**。此外：

- **(a) 可逆的 design.** 若对每个 $`\xi`$ 都有 $`A(\xi)\succ0`$，则 $`\Phi_c(A(\xi))=c^\top A(\xi)^{-1}c`$，等式就是经典的 c-optimal design 等式。
- **(b) 取值何时有限.** 称 $`h`$ 为*未被照到的方向*，若 $`K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h=0`$。共同取值有限，当且仅当每个未被照到的方向都满足 $`c^\top h=0`$。此时 $`c\in\mathrm{range}\,A(\xi^*)`$，且 $`\xi^*`$ 可以让所有未被照到的方向都保持不可辨识。
- **(c) 没有查询.** 当 $`n=0`$、$`K>0`$ 时：若 $`c\in\mathrm{range}(I_D)`$，取值为 $`c^\top(KI_D)^+c`$；否则为 $`+\infty`$。

**说了什么.**

- 在所有查询 design 中，估计学习器相对泛函所能达到的最小方差，恰好等于信息范数的平方，并且最优 design 存在。它就是以"学习器 Jacobian 作用在 continuation value 梯度上"为目标向量的 c-optimal（Elfving）design。
- design 可以奇异。学习器目标不关心的 $`\theta`$ 方向（$`c^\top h=0`$）永远不必辨识；只有当数据照不到目标确实用到的某个方向时，取值才变成无穷。

**直觉.**

- 查询最好花在对学习器唯一用到的那个方向最有信息量的地方；这是一个经典的设计问题，只是目标由学习器决定。
- 对线性目标而言，花钱去辨识无关方向是浪费：伪逆直接忽略它们。

**在框架中的作用.**

- 它把下界变成一条获取规则，并让上界可以计算（[Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)）。
- 它是 [Proposition 6.15](../statements/06-15_linear_gaussian_target_ZH.md#s6-15) 中线性高斯设计计算在标量目标下的特例。
- 这里的放宽只针对线性泛函。[Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5) 仍然要把 design 正则化，使每个方向最终都被辨识，因为它的 plug-in 误差在所有方向上都有二阶余项；这个条件并没有在这里取消。

**推导要点.**

- 对每个半正定 $`A`$，$`\Phi_c(A)=\sup_h\big(2c^\top h-h^\top Ah\big)`$；再对 $`h`$ 的尺度取最优，就化为 $`(c^\top h)^2/h^\top Ah`$ 在上述约定下的上确界。整个过程不需要求逆。
- $`2c^\top h-h^\top A(\xi)h`$ 关于 design 仿射且弱连续、关于方向是凹的，design 构成弱紧的凸集，所以 Sion minimax 定理可以交换两个优化。
- 固定方向时，最好的 design 把全部质量放在信息量最大的查询点上，这就是 $`\sup_x`$ 项的来源。$`\xi\mapsto\Phi_c(A(\xi))`$ 是一族连续函数的上确界，因而下半连续，在紧的 design 集合上取到下确界。

**用到.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：范数的定义。

**被用于.**

- [Theorem 6.5](../statements/06-05_achievability_ZH.md#s6-5)：达到范数的 design。
- [Remark 6.6](../statements/06-06_class_uniform_fidelity_ZH.md#s6-6)：c-optimal design 的混合。
- [Proposition 6.15](../statements/06-15_linear_gaussian_target_ZH.md#s6-15)：线性高斯风险在标量目标下的情形。

**状态.** 已证明。

## 相关工作

- c-optimal design 及其几何刻画可追溯到 [Elfving, 1952](#ref-elfving1952optimum)；一般理论，包括带先验信息的 design（例如示范贡献的 $`K\,I_D`$ 项）以及带可估性条件 $`c\in\mathrm{range}\,A`$ 的奇异信息矩阵，见 [Pukelsheim, 1993](#ref-pukelsheim1993optimal)。两个优化的交换是 Sion 的 minimax 定理（[Sion, 1958](#ref-sion1958general)）。因此允许奇异 design 是经典设计对偶的直接推广，并不是新的设计理论。
- 这里特有的是目标向量 $`c=\sum_tJ_t^*g_t`$：它由学习器与任务决定。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).

<a id="ref-sion1958general"></a>

**Maurice Sion (1958).** On general minimax theorems. *Pacific Journal of Mathematics* 8, no. 1, 171–176. DOI [10.2140/pjm.1958.8.171](https://doi.org/10.2140/pjm.1958.8.171).
