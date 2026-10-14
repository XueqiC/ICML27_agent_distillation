# Lemma 6.4 · 设计的 minimax 等式

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-04_design_minimax_equality_ZH.md) · [证明](../proofs/06-04_design_minimax_equality_ZH.md) · [实验](../experiments/06-04_design_minimax_equality_ZH.md) · [English](06-04_design_minimax_equality.md)

**本页证明什么.** c-optimal design 的取值 $`\inf_\xi\Phi_c(A(\xi))`$ 等于信息范数的平方，不需要假设任何 design 使 $`A(\xi)`$ 可逆，并且下确界可以取到。证明先用一个对所有半正定矩阵都成立的变分公式代替求逆，再用 Sion 定理交换两个优化，最后把内层问题显式算出。

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

**陈述（[Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)）.** 设 $`\mathcal X_Q`$ 是紧度量空间，$`x\mapsto I_Q(\theta;x)`$ 连续。令 $`\xi`$ 取遍 $`\mathcal X_Q`$ 上的 Borel 概率测度（approximate design），$`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$，$`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$；不要求任何 design 使 $`A(\xi)`$ 可逆。对半正定矩阵 $`A`$ 定义

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

**证明用到的结果.**

- [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)：信息范数 $`\mathfrak n_{K,n}`$ 的定义。

**用到的经典工具.** Prokhorov 定理与 Sion minimax 定理（[Sion, 1958](#ref-sion1958general)）；对称半正定矩阵的 Moore–Penrose 伪逆。

**证明.**

**第 1 步（对每个半正定矩阵都成立的变分公式）.** 设 $`A\succeq0`$，$`\ell_A(h)=2c^\top h-h^\top Ah`$。我们证明

```math
\Phi_c(A)=\sup_{h\in\mathbb R^p}\ell_A(h)=\sup_{h\neq0}\frac{(c^\top h)^2}{h^\top Ah}\qquad(0/0=0,\ a/0=+\infty\text{ for }a>0).
```

若 $`c\in\mathrm{range}(A)`$，记 $`c=Aw`$。则 $`c^\top A^+c=w^\top AA^+Aw=w^\top Aw`$，并且

```math
\ell_A(h)=2w^\top Ah-h^\top Ah=w^\top Aw-(h-w)^\top A(h-w)\le w^\top Aw ,
```

等号在 $`h=w`$ 处成立，所以 $`\sup_h\ell_A=c^\top A^+c`$。若 $`c\notin\mathrm{range}(A)=(\ker A)^\perp`$，则存在 $`v\in\ker A`$ 使 $`c^\top v>0`$，于是 $`s\to\infty`$ 时 $`\ell_A(sv)=2s\,c^\top v\to+\infty`$。这就证明了第一个等号。对第二个等号，固定 $`h\neq0`$ 并对尺度 $`s\in\mathbb R`$ 求最大：$`\sup_s\ell_A(sh)=\sup_s\big(2s\,c^\top h-s^2h^\top Ah\big)`$ 在 $`h^\top Ah>0`$ 时等于 $`(c^\top h)^2/h^\top Ah`$，在 $`h^\top Ah=0`$ 且 $`c^\top h\neq0`$ 时为 $`+\infty`$，两者都为零时为 $`0`$。再对方向取上确界，就得到带上述约定的比值形式。

**第 2 步（交换两个优化）.** 记 $`\mathcal P`$ 为 $`\mathcal X_Q`$ 上 Borel 概率测度的全体，$`L(\xi,h)=\ell_{A(\xi)}(h)=2c^\top h-K\,h^\top I_Dh-n\int h^\top I_Q(x)h\,\xi(\mathrm dx)`$。由第 1 步，引理左边等于 $`\inf_{\xi\in\mathcal P}\sup_hL(\xi,h)`$。

- $`\mathcal P`$ 是凸集；由于 $`\mathcal X_Q`$ 是紧度量空间，Prokhorov 定理给出它在弱拓扑下紧。它位于有限带号测度构成的局部凸空间中。
- 固定 $`h`$，$`\xi\mapsto L(\xi,h)`$ 仿射且弱连续，因为 $`x\mapsto h^\top I_Q(x)h`$ 在紧集 $`\mathcal X_Q`$ 上连续有界。
- 固定 $`\xi`$，$`h\mapsto L(\xi,h)`$ 在凸集 $`\mathbb R^p`$ 上是凹的连续函数。

以 $`\xi`$ 为紧凸集 $`\mathcal P`$ 上的极小化变量（关于 $`\xi`$ 下半连续、拟凸），以 $`h`$ 为极大化变量（关于 $`h`$ 上半连续、拟凹），Sion minimax 定理给出

```math
\inf_{\xi\in\mathcal P}\sup_hL(\xi,h)=\sup_h\inf_{\xi\in\mathcal P}L(\xi,h).
```

**第 3 步（可达性）.** 映射 $`\xi\mapsto\Phi_c(A(\xi))=\sup_hL(\xi,h)`$ 是一族弱连续函数的逐点上确界，所以弱下半连续，取值在 $`(-\infty,+\infty]`$。紧集上的下半连续函数取到下确界；取到下确界的 design 就是 $`\xi^*`$。（若取值为 $`+\infty`$，则每个 design 都取到它。）

**第 4 步（内层问题）.** 固定 $`h`$，$`\sup_{\xi\in\mathcal P}\int h^\top I_Q(x)h\,\xi(\mathrm dx)=\max_xh^\top I_Q(x)h`$：积分不会超过连续被积函数的最大值，而放在最大值点上的点质量恰好取到它。于是记 $`q(h)=K\,h^\top I_Dh+n\max_xh^\top I_Q(x)h`$，有

```math
\sup_h\inf_{\xi\in\mathcal P}L(\xi,h)=\sup_h\big(2c^\top h-q(h)\big).
```

$`q`$ 非负且是 2 次正齐次函数，所以第 1 步对尺度的优化原样适用，得到同样约定下的 $`\sup_{h\neq0}(c^\top h)^2/q(h)`$。按 [Definition 6.2](../statements/06-02_learner_relative_functional_information_norm_ZH.md#s6-2)，这就是 $`\mathfrak n_{K,n}^2`$。第 1–4 步证明了显示的等式与可达性。

**第 5 步（三个特例）.**

- (a) 若 $`A(\xi)\succ0`$，则 $`\mathrm{range}\,A(\xi)=\mathbb R^p`$，且 $`A(\xi)^+=A(\xi)^{-1}`$。
- (b) 记 $`\mathcal N=\lbrace h:q(h)=0\rbrace`$ 为未被照到的方向。由于每个信息矩阵都半正定，$`q(h)=0`$ 意味着 $`I_Dh=0`$（当 $`K>0`$）且对每个 $`x`$ 有 $`I_Q(x)h=0`$（当 $`n>0`$），所以 $`\mathcal N`$ 是线性子空间，并且对每个 $`u\in\mathcal N`$ 有 $`q(h+u)=q(h)`$。若某个 $`u\in\mathcal N`$ 满足 $`c^\top u\neq0`$，比值在 $`u`$ 处为 $`+\infty`$。否则 $`c\perp\mathcal N`$；把 $`h`$ 写成 $`h=h_\perp+u`$，$`h_\perp\in\mathcal N^\perp`$，$`u\in\mathcal N`$，比值等于 $`(c^\top h_\perp)^2/q(h_\perp)`$。$`q`$ 连续（联合连续函数在紧集上的最大值），且在 $`\mathcal N^\perp`$ 的单位球面上严格为正，所以在那里有正的最小值 $`q_{\min}`$，比值不超过 $`\lVert c\rVert^2/q_{\min}<\infty`$。取值有限时 $`\Phi_c(A(\xi^*))<\infty`$，按定义这正是 $`c\in\mathrm{range}\,A(\xi^*)`$；并没有任何东西迫使 $`A(\xi^*)`$ 可逆。
- (c) $`n=0`$ 时每个 design 都给出 $`A=K\,I_D`$；若 $`c\in\mathrm{range}(I_D)`$，$`\Phi_c(KI_D)=c^\top(KI_D)^+c`$，否则为 $`+\infty`$。$`\square`$

**附注.**

- 当查询信息是秩一的 $`I_Q(x)=f_xf_x^\top`$ 且 $`K=0`$ 时，Elfving 定理给出 $`\xi^*`$ 的几何刻画。$`K\,I_D`$ 这一项使它成为带先验信息的 c-optimal design。
- 也可以对 $`A(\xi)+\varepsilon I`$ 用可逆情形，再令 $`\varepsilon\downarrow0`$，得到同一个等式；第 1 步的变分公式避免了把这个极限与对 design 的下确界交换。

## 相关工作

- **最优设计.** c-optimal design 及其几何刻画可以追溯到 [Elfving, 1952](#ref-elfving1952optimum)；minimax 的交换是 Sion 定理（[Sion, 1958](#ref-sion1958general)）；带先验信息的设计与奇异信息矩阵（$`c^\top\theta`$ 可估意味着 $`c\in\mathrm{range}\,A`$）都是标准内容（[Pukelsheim, 1993](#ref-pukelsheim1993optimal)）。这里特有的是目标向量 $`c=\sum_tJ_t^*g_t`$：它由 optimizer 的增量映射与声明的任务决定。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).

<a id="ref-sion1958general"></a>

**Maurice Sion (1958).** On general minimax theorems. *Pacific Journal of Mathematics* 8, no. 1, 171–176. DOI [10.2140/pjm.1958.8.171](https://doi.org/10.2140/pjm.1958.8.171).
