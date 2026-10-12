# §6 学习器相对的信息界（Theorem A）— 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/06_learner_relative_information_ZH.md) · [English](06_learner_relative_information.md)

**概要.**

- 本节回答：从 $`K`$ 条示范（demonstration）和 $`n`$ 次允许的查询（query）出发，蒸馏器（distiller）最多能知道多少关于 oracle 反馈的信息。这里的"多少"是沿着实际学习器与声明任务真正在意的那个方向来度量的。
- **Theorem A** 由 [Theorem 6.3](#s6-3) 与 [Theorem 6.5](#s6-5) 组成：
  - 任何蒸馏器（包括 replay、改写、合成）的一阶 fidelity 误差都至少是 $`\tfrac14\mathfrak n_{K,n}`$，其中 $`\mathfrak n_{K,n}`$ 是 [Definition 6.2](#s6-2) 的 learner-relative information norm；
  - 用 c-optimal（Elfving）查询设计喂给 plug-in 蒸馏器，可以达到 $`\mathfrak n_{K,n}(1+o(1))`$。设计的目标向量是学习器 Jacobian 作用在 continuation value 梯度上的结果，所以信息该往哪里收集由 optimizer 决定（[Lemma 6.4](#s6-4)）。
- 推论：混叠与 replay 的零价值（[Corollary 6.7](#s6-7)）；示范与查询的兑换率（[Corollary 6.8](#s6-8)）；针对 agent 蒸馏的状态覆盖结构给出显式公式与 water-filling 查询分配（[Proposition 6.10](#s6-10)）；SGD 与 warm Adam 的学习器权重（[Proposition 6.11](#s6-11)）；学习器之间的分离（[Corollary 6.12](#s6-12)、[Example 6.13](#s6-13)）。
- 用到：complete learner state、增量映射与 continuation value（[Definitions 1.3–1.4](../README_ZH.md#s1-3)）；一阶表示（[Proposition 5.1](05_optimizer_state_ZH.md#s5-1)）；Adam 的增量（[Proposition 5.5](05_optimizer_state_ZH.md#s5-5)、[Theorem 5.10](05_optimizer_state_ZH.md#s5-10)）；部分激活的梯度裁剪（[Proposition 5.9](05_optimizer_state_ZH.md#s5-9)）。[Corollary 6.7](#s6-7) 与 [Corollary 6.8](#s6-8) 分别是 [Theorem 2.6](02_feasibility_and_acquisition_ZH.md#s2-6)(a)、[Proposition 2.4](02_feasibility_and_acquisition_ZH.md#s2-4) 与 [Proposition 2.11](02_feasibility_and_acquisition_ZH.md#s2-11) 的学习器相对版本。

## 设定

<a id="s6-1"></a>

**Definition 6.1（环境族、数据实验、蒸馏器、被审计状态）.**

- **环境族.** $`\lbrace E_\theta:\theta\in\Theta\rbrace`$，$`\Theta\subseteq\mathbb R^p`$ 为开集，真值 $`\theta`$ 未知。
- **oracle 反馈.** $`P^*_\theta=(P^*_{\theta,t})_t`$ 是知道 $`\theta`$ 时应给的 feedback law，例如 teacher 在学生所访问状态上的反馈。
- **数据实验.**
  - $`K`$ 条示范 $`O_1,\dots,O_K`$ 独立同分布于 $`P^D_\theta`$。
  - $`n`$ 次查询可以自适应：查询点 $`X_i\in\mathcal X_Q`$（查询空间）由历史和私有随机性决定，结果 $`Y_i\sim P^Q_\theta(\cdot\mid X_i)`$。
  - 整条 transcript $`D`$ 的分布记为 $`\mathsf P^{K,n}_\theta`$。
- **蒸馏器.** 任何从 $`(D,U)`$ 到 feedback law $`Q=(Q_t)_t`$ 的可测映射，其中私有随机性 $`U`$ 的分布与 $`\theta`$ 无关。
  - replay、改写、合成都是蒸馏器，只要它们不再获取依赖于 $`\theta`$ 的信息。
  - teacher 模型的预训练知识属于公共初始信息，对所有 $`\theta`$ 相同；新调用 teacher 去观察一个新状态，算作一次查询。
- **被审计状态.** 固定的 complete learner state $`z_1,\dots,z_T`$（[Definition 1.3](../README_ZH.md#s1-3)），例如参考训练在若干检查点到达的状态。它们不依赖于 $`\theta`$。

<a id="s6-2"></a>

**Definition 6.2（学习器相对泛函与信息范数）.**

- **学习器 Jacobian.** 被审计状态上的 oracle 平均增量为 $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$。学习器 Jacobian $`J_t(\theta)=D_\theta\mu^*_t(\theta)`$ 是从 $`\mathbb R^p`$ 到状态空间 $`\mathcal Z`$ 的线性映射。它同时依赖于环境（通过 $`P^*_\theta`$）和学习器（通过增量映射 $`\Delta_t`$）。
- **学习器相对泛函（learner-relative functional）.** 声明的任务给出方向 $`g_t=\nabla V_{t+1}(z_t)\in\mathcal Z`$，即参考尾段在被审计状态上的 continuation value 梯度，与 $`\theta`$ 无关。定义

```math
\Psi(\theta)=\sum_{t=1}^{T}\big\langle g_t,\,\mu^*_t(\theta)\big\rangle,\qquad c(\theta)=\nabla_\theta\Psi(\theta)=\sum_{t=1}^{T}J_t(\theta)^{*}g_t\in\mathbb R^p .
```

- **一阶 fidelity 误差.** 对蒸馏器的输出 $`Q`$，

```math
\mathrm{Err}_\theta(Q)=\Big\lvert\Psi(\theta)-\sum_{t=1}^{T}\big\langle g_t,\,\mu^Q_t(z_t)\big\rangle\Big\rvert .
```

- **关于定义的说明.**
  - consumed gradient $`G_t(z,y)`$ 与增量映射 $`\Delta_t`$ 只依赖于学习器状态和消息，所以 $`\mu^Q_t(z_t)`$ 与 $`\theta`$ 无关；环境只通过"oracle 律会发出哪些消息"进入。
  - 由 [Proposition 5.1](05_optimizer_state_ZH.md#s5-1)(a)（取 $`k=1`$），$`\langle g_t,\mu^*_t(\theta)-\mu^Q_t(z_t)\rangle`$ 是"在 $`z_t`$ 上把一轮 $`Q`$ 反馈换成 oracle 反馈、之后走参考尾段"这一效应的一阶部分。$`\mathrm{Err}_\theta`$ 把这些一阶效应在被审计状态上加起来。
- **信息形式与范数.** 记 $`I_D(\theta)`$ 为一条示范的 Fisher information，$`I_Q(\theta;x)`$ 为在 $`x\in\mathcal X_Q`$ 处一次查询的 Fisher information。对方向 $`h\in\mathbb R^p`$，

```math
\mathbb I_{K,n}(\theta;h)=K\,h^\top I_D(\theta)h+n\sup_{x\in\mathcal X_Q}h^\top I_Q(\theta;x)h,\qquad \mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\frac{\lvert c(\theta)^\top h\rvert}{\sqrt{\mathbb I_{K,n}(\theta;h)}},
```

  约定 $`0/0=0`$，$`a/0=\infty`$（$`a>0`$）。$`\mathfrak n_{K,n}`$ 称为 **learner-relative information norm**：它是学习器相对梯度 $`c`$ 相对于信息规范 $`\sqrt{\mathbb I_{K,n}}`$ 的对偶范数。

- **增长条件 (G).** 下面部分渐近结论假设信息非退化地增长：存在 $`N\to\infty`$ 与常数 $`0<a\le b<\infty`$，使所有单位向量 $`h`$ 满足 $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$。例如 $`K`$ 与 $`n`$ 按比例增长、且 $`I_D+\sup_xI_Q(x)`$ 在每个方向上正定时，(G) 成立。

## 下界与可达性

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

- **(c) 范数就是局部下界.** 在 (G) 下，取 $`h`$ 为 [Definition 6.2](#s6-2) 中比值的最大点，局部 minimax 下界为 $`\tfrac14\mathfrak n_{K,n}(\theta_0)\,(1-o(1))`$。

*Proof.* (a) 记 $`s=\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$，$`a(Q)=\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$；后者与 $`\theta`$ 无关（[Definition 6.2](#s6-2)）。由三角不等式，

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

<a id="s6-4"></a>

**Lemma 6.4（设计的 minimax 等式）.** 设 $`\mathcal X_Q`$ 是紧度量空间，$`x\mapsto I_Q(\theta;x)`$ 连续。令 $`\xi`$ 取遍 $`\mathcal X_Q`$ 上的 Borel 概率测度（approximate design），$`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$，$`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$，并设对每个 $`\xi`$ 都有 $`A(\xi)\succ0`$。则

```math
\inf_\xi c^\top A(\xi)^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

并且下确界由某个 design $`\xi^*`$ 取到，它就是目标向量为 $`c=\sum_tJ_t^*g_t`$ 的 **c-optimal design**。当 $`n=0`$ 且 $`I_D`$ 奇异时：若 $`c\in\mathrm{range}(I_D)`$，等式以 $`c^\top(KI_D)^+c`$ 成立；否则两边都是 $`+\infty`$。

*Proof.* 若 $`c=0`$，两边都为零；下设 $`c\neq0`$。对 $`A\succ0`$，Cauchy–Schwarz 给出 $`(c^\top h)^2\le(c^\top A^{-1}c)(h^\top Ah)`$，等号在 $`h=A^{-1}c`$ 处成立，所以

```math
c^\top A^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{h^\top Ah}=\frac{1}{\inf_{h\in H}h^\top Ah},\qquad H=\lbrace h:c^\top h=1\rbrace .
```

令 $`f(\xi,h)=h^\top A(\xi)h`$。引理左边等于 $`1/\sup_\xi\inf_{h\in H}f(\xi,h)`$。固定 $`h`$ 时，$`\xi\mapsto h^\top M(\xi)h=\int h^\top I_Q(x)h\,\xi(\mathrm dx)`$ 在连续函数 $`x\mapsto h^\top I_Q(x)h`$（定义在紧集 $`\mathcal X_Q`$ 上）的最大点处的点质量上取最大，所以 $`\sup_\xi f(\xi,h)=\mathbb I_{K,n}(h)`$，右边等于 $`1/\inf_{h\in H}\sup_\xi f(\xi,h)`$。剩下只需交换 $`\sup_\xi`$ 与 $`\inf_h`$：

- design 的集合是凸集，在弱拓扑下紧（Prokhorov 定理），位于有限符号测度这个局部凸空间中；
- $`f(\cdot,h)`$ 是仿射的，且弱连续，因为被积函数连续有界；
- $`f(\xi,\cdot)`$ 在凸集 $`H`$ 上凸且连续。

对 $`-f`$ 用 Sion minimax 定理（以紧凸的 design 集合为极小化变量），得 $`\max_\xi\inf_{h\in H}f=\inf_{h\in H}\sup_\xi f`$。函数 $`\xi\mapsto\inf_hf(\xi,h)`$ 是连续函数的下确界，因而上半连续，在紧的 design 集合上取到最大值，这就是 $`\xi^*`$。当 $`n=0`$ 时，Cauchy–Schwarz 的伪逆形式给出：若 $`c\perp\ker I_D`$，则 $`\sup_h(c^\top h)^2/(K\,h^\top I_Dh)=c^\top(KI_D)^+c`$（核中方向贡献 $`0/0=0`$）；否则某个 $`h\in\ker I_D`$ 满足 $`c^\top h\neq0`$，比值为 $`+\infty`$。$`\square`$

- 当查询信息是秩一的 $`I_Q(x)=f_xf_x^\top`$ 且 $`K=0`$ 时，Elfving 定理给出 $`\xi^*`$ 的几何刻画。$`K\,I_D`$ 这一项使它成为带先验信息的 c-optimal design。

<a id="s6-5"></a>

**Theorem 6.5（可达性；标量情形紧到常数）.** 设 (G) 与 [Theorem 6.3](#s6-3)(b) 的正则性成立，并且：

- **(R) 可实现性.** 对任意 $`\theta'\in\Theta`$，蒸馏器能在被审计状态上发出 oracle 律 $`P^*_{\theta'}`$，例如在 $`\theta'`$ 下运行模拟器或 teacher 模型，或对查询得到的消息重新加权。
- **(I) 可辨识性.** 存在某个 design $`\xi_0`$，使 $`\lambda_{\min}(M(\xi_0))=\lambda_0>0`$。
- **(S) 光滑性.** $`\Psi`$ 在 $`\theta_0`$ 附近属于 $`C^2`$，且 $`\lVert\nabla^2\Psi\rVert\le H_\Psi`$。
- **Design.** 使用正则化的 c-optimal design $`\xi_n=(1-\varepsilon_n)\xi^*+\varepsilon_n\xi_0`$，其中 $`\xi^*`$ 来自 [Lemma 6.4](#s6-4)，$`\varepsilon_n\to0`$ 且 $`\varepsilon_n\sqrt n\to\infty`$（例如 $`\varepsilon_n=n^{-1/4}`$）。
- **(E) 有效的两阶段估计.** $`\xi_n`$ 在一个用 $`o(n)`$ 次查询得到的相合 pilot 估计处计算，并舍入成整数次数。最终估计量 $`\hat\theta`$ 在 $`\theta_0`$ 的一个收缩邻域上对 $`\theta`$ 一致地满足（$`\xi_n`$ 在真值 $`\theta`$ 处取值）

```math
\mathbb E_\theta\big[(c(\theta)^\top(\hat\theta-\theta))^2\big]\le c(\theta)^\top A(\xi_n)^{-1}c(\theta)\,(1+o(1)),\qquad \mathbb E_\theta\lVert\hat\theta-\theta\rVert^2\le\frac{C}{\lambda_{\min}(A(\xi_n))} .
```

令蒸馏器输出 $`Q=P^*_{\hat\theta}`$。则

```math
\mathbb E_\theta\mathrm{Err}_\theta(Q)\le\mathfrak n_{K,n}(\theta)\,\big(1+o(1)\big)+o\big(N^{-1/2}\big),
```

并且只要 $`c(\theta)\neq0`$，最后一项就是 $`o(\mathfrak n_{K,n}(\theta))`$。与 [Theorem 6.3](#s6-3)(c) 合起来，在局部 minimax 意义下，最小可达一阶误差介于 $`\tfrac14\mathfrak n_{K,n}`$ 与 $`\mathfrak n_{K,n}`$ 之间（相差 $`1\pm o(1)`$ 的因子）。最优查询服从目标向量为 $`c=\sum_tJ_t^*g_t`$ 的 c-optimal design。

*Proof.* 由 (R)，输出 $`P^*_{\hat\theta}`$ 是可得的，且 $`\mathrm{Err}_\theta(P^*_{\hat\theta})=\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert`$。由 (S)，

```math
\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert\le\lvert c(\theta)^\top(\hat\theta-\theta)\rvert+\tfrac12H_\Psi\lVert\hat\theta-\theta\rVert^2 .
```

- *线性项.* $`A(\xi_n)=KI_D+n(1-\varepsilon_n)M(\xi^*)+n\varepsilon_nM(\xi_0)\succeq(1-\varepsilon_n)A(\xi^*)`$，因此由 [Lemma 6.4](#s6-4)，$`c^\top A(\xi_n)^{-1}c\le c^\top A(\xi^*)^{-1}c/(1-\varepsilon_n)=\mathfrak n^2_{K,n}(1+o(1))`$。由 Jensen 不等式与 (E)，$`\mathbb E\lvert c^\top(\hat\theta-\theta)\rvert\le\mathfrak n_{K,n}(1+o(1))`$。
- *二次项.* 由 (E)，它不超过 $`\tfrac12H_\Psi C/\lambda_{\min}(A(\xi_n))`$，而 $`\lambda_{\min}(A(\xi_n))\ge\max\lbrace K\lambda_{\min}(I_D),\,n\varepsilon_n\lambda_0\rbrace`$。这个量远大于 $`\sqrt N`$：
  - 若对某个 $`\delta>0`$ 有 $`n\ge\delta N`$，则 $`n\varepsilon_n\gg\sqrt n\ge\sqrt{\delta N}`$；
  - 否则沿某个子列 $`n=o(N)`$，于是对 $`h\in\ker I_D`$ 有 $`\mathbb I_{K,n}(h)\le n\sup_xh^\top I_Qh=o(N)`$，与 (G) 矛盾；所以 $`I_D\succ0`$，$`K`$ 与 $`N`$ 同阶，$`K\lambda_{\min}(I_D)\asymp N`$。

  所以二次项是 $`o(N^{-1/2})`$。
- *比较.* 若 $`c(\theta)\neq0`$，由 (G) 有 $`\mathfrak n_{K,n}\ge\lVert c\rVert/\sqrt{bN}`$，所以 $`o(N^{-1/2})=o(\mathfrak n_{K,n})`$。$`\square`$

- 正则化 $`\varepsilon_n`$ 是必要的：c-optimal design 往往只支撑在少数查询点上，$`A(\xi^*)`$ 在 $`c`$ 看不到的方向上可以奇异；而 $`\Psi`$ 的二阶项仍然要求 $`\hat\theta`$ 在每个方向上相合。
- 在高斯线性实验中，$`\Psi`$ 是线性的，有效估计量精确服从高斯分布，plug-in 误差恰为 $`\sqrt{2/\pi}\,\sqrt{c^\top A^{-1}c}`$；在最优 design 处约为 $`0.80\,\mathfrak n_{K,n}`$。

<a id="s6-6"></a>

**Remark 6.6（对声明类一致的 fidelity）.** 若要求 fidelity 对一个声明的 continuation value 类 $`\mathcal V`$ 一致成立，每个 $`V\in\mathcal V`$ 给出方向 $`g^V_t=\nabla V_{t+1}(z_t)`$、泛函 $`\Psi_V`$ 与梯度 $`c_V`$；误差为 $`\mathrm{Err}^{\mathcal V}_\theta(Q)=\sup_{V\in\mathcal V}\lvert\Psi_V(\theta)-\sum_t\langle g^V_t,\mu^Q_t(z_t)\rangle\rvert`$。

- **下界.** [Theorem 6.3](#s6-3) 的证明照样成立，只需把 $`\lvert c^\top h\rvert`$ 换成 $`\sigma(h)=\sup_V\lvert c_V^\top h\rvert`$，因为 $`\sup_V\lvert a_V\rvert+\sup_V\lvert b_V\rvert\ge\sup_V\lvert a_V-b_V\rvert`$。交换两个上确界，局部下界为 $`\tfrac14\sup_V\mathfrak n_{K,n}(c_V)`$。
- **上界.** 记 $`d`$ 为 $`\mathrm{span}\lbrace c_V\rbrace`$ 的维数（$`d\le\dim\mathrm{span}\,\mathcal G`$）。在 (E) 对 $`d`$ 个线性泛函的类似假设下，配合合适的混合 design，plug-in 蒸馏器的误差不超过 $`d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)\,(1+o(1))`$。
- 所以上下界之间至多差 $`4d^{3/2}`$ 倍，只依赖于声明类的维数。这里不主张更紧的常数。

*上界的证明.* 令 $`\mathcal K`$ 为 $`\lbrace\pm c_V\rbrace`$ 的闭凸包，它是某个 $`d`$ 维子空间中的对称凸体。由 John 定理，存在椭球 $`\mathcal E=\lbrace\sum_kw_k\ell_k:\lVert w\rVert\le1\rbrace\subseteq\mathcal K\subseteq\sqrt d\,\mathcal E`$，所以 $`\sigma(u)=\sup_{v\in\mathcal K}v^\top u\le\sqrt d\,\big(\sum_{k=1}^d(\ell_k^\top u)^2\big)^{1/2}`$。令 $`\xi_k`$ 为 $`\ell_k`$ 的 c-optimal design，$`\bar\xi=\tfrac1d\sum_k\xi_k`$。则对每个 $`k`$ 有 $`A(\bar\xi)\succeq\tfrac1dA(\xi_k)`$，从而 $`\ell_k^\top A(\bar\xi)^{-1}\ell_k\le d\,\mathfrak n_{K,n}(\ell_k)^2`$。由于 $`\ell_k\in\mathcal K`$，且 $`\mathfrak n_{K,n}(\cdot)`$ 关于其向量自变量是半范数，$`\mathfrak n_{K,n}(\ell_k)\le\sup_V\mathfrak n_{K,n}(c_V)`$。由 Jensen 不等式，

```math
\mathbb E\,\sigma(\hat\theta-\theta)\le\sqrt d\,\Big(\sum_{k=1}^d\ell_k^\top A(\bar\xi)^{-1}\ell_k\Big)^{1/2}(1+o(1))\le d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)\,(1+o(1)),
```

二阶项的处理与 [Theorem 6.5](#s6-5) 相同。$`\square`$

## 推论

<a id="s6-7"></a>

**Corollary 6.7（混叠与 replay 的零价值）.**

- 若 $`\theta_0\neq\theta_1`$ 满足 $`P^D_{\theta_0}=P^D_{\theta_1}`$、$`\Psi(\theta_0)\neq\Psi(\theta_1)`$，并且 $`n=0`$ 或对每个 $`x`$ 都有 $`P^Q_{\theta_0}(\cdot\mid x)=P^Q_{\theta_1}(\cdot\mid x)`$，则对每个 $`K`$，任何蒸馏器都有 $`\max_i\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$。
- replay、改写、合成都是 transcript 与私有随机性的函数，因此包含在"任何蒸馏器"之中，不能降低这个下界。
- 这个下界是学习器相对的：示范是否充分，取决于示范看不见的方向在**这个学习器**的 $`c`$ 上有没有分量。同一批示范可以对一个学习器充分、对另一个不充分（[Corollary 6.12](#s6-12)(a)）。
- [Theorem 2.6](02_feasibility_and_acquisition_ZH.md#s2-6)(a) 是其中一个实例：两个世界的示范分布相同，学生的最优策略不同；[Proposition 2.4](02_feasibility_and_acquisition_ZH.md#s2-4) 说的是后处理帮不上忙。

*Proof.* 两条 transcript 的分布相同，所以 $`\mathrm{TV}=0`$，[Theorem 6.3](#s6-3)(a) 中平方根项为零。$`\square`$

<a id="s6-8"></a>

**Corollary 6.8（示范与查询的兑换率）.** 对方向 $`h`$ 记 $`e_D(h)=h^\top I_D(\theta_0)h`$，$`e_Q(h)=\sup_xh^\top I_Q(\theta_0;x)h`$。在 [Theorem 6.3](#s6-3)(b) 的条件下，若一个蒸馏器的局部最坏期望一阶误差不超过 $`\varepsilon`$，则对每个 $`h`$ 必须有

```math
K\,e_D(h)+n\,e_Q(h)\ge\frac{(c(\theta_0)^\top h)^2}{16\,\varepsilon^2}\,\big(1-o(1)\big).
```

- $`(K,n)`$ 的可行集是一族半平面的交。在可行集边界上，起约束作用的方向 $`h^\star`$ 决定了局部兑换率：一条示范值 $`e_D(h^\star)/e_Q(h^\star)`$ 次查询。
- $`h^\star`$ 由学习器的 $`c`$ 决定，所以兑换率是学习器相对的：同一批示范对不同的学习器值不同数量的查询。
- [Proposition 2.11](02_feasibility_and_acquisition_ZH.md#s2-11) 是两世界检验情形下类似的线性预算；这里的预算逐方向成立，并按学习器加权。

*Proof.* 由 [Theorem 6.3](#s6-3)(b)，$`\varepsilon\ge\lvert c^\top h\rvert/(4\sqrt{\mathbb I_{K,n}(\theta_0;h)})\,(1-o(1))`$，其中 $`\mathbb I_{K,n}(\theta_0;h)=K\,e_D(h)+n\,e_Q(h)`$。移项即得；$`c^\top h=0`$ 的方向不产生约束。$`\square`$

## 状态覆盖模型：显式公式

这一部分把 Theorem A 落到 agent 蒸馏中最常见的结构上：示范覆盖 teacher 访问的状态，训练与部署却发生在学生访问的状态上。本部分中 $`s`$ 表示 agent 的状态。

<a id="s6-9"></a>

**Assumption 6.9（状态覆盖模型）.**

- **状态与分布.** 有限的 agent 状态集 $`\mathcal S`$。在状态 $`s`$ 上 oracle 消息 $`Y\sim P^*_s`$（例如 teacher 在 $`s`$ 的动作或修正），消息字母表有限。各状态的分布 $`(P^*_s)_{s\in\mathcal S}`$ 互相独立地变化：不假设跨状态的泛化。
- **泛函.** 学习器相对泛函 $`\Psi`$ 是 $`(P^*_s)_{s\in\mathcal S}`$ 的 Hadamard 可微泛函，它在 $`s`$ 处的 influence function 为 $`f_s(y)=\frac{\mathrm d}{\mathrm d\epsilon}\Psi\big(P^*_s+\epsilon(\delta_y-P^*_s)\big)\big\vert_{\epsilon=0}`$，因此 $`\mathbb E_{P^*_s}f_s=0`$。
- **样本数.** 状态 $`s`$ 得到 $`N_s=k_s+n_s`$ 个 oracle 样本：$`k_s`$ 个来自示范（期望为 $`KH\,d_T(s)`$，$`H`$ 是轨迹长度，$`d_T`$ 是 teacher 的状态访问分布），$`n_s`$ 个来自查询，$`\sum_sn_s=n`$。下面的界对给定的样本数陈述；样本数随机时，它们在给定样本数的条件下成立。

<a id="s6-10"></a>

**Proposition 6.10（显式信息界与 water-filling 最优查询分配）.** 在 [Assumption 6.9](#s6-9) 下，记 $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$，称为状态 $`s`$ 的**学习器相对权重**（learner-relative weight）。

- **(a) 下界.** 设在 $`W_s>0`$ 的状态上样本数按比例增长（$`N_s=a_sN`$，$`N\to\infty`$）。任何蒸馏器的局部 minimax 一阶误差至少为

```math
\tfrac14\Big(\sum_{s}\frac{W_s}{N_s}\Big)^{1/2}\big(1-o(1)\big).
```

- **(b) 上界.** 设 $`\Psi`$ 关于概率向量 $`(P_s)_s`$ 二阶连续可微。逐状态的经验分布，即在每个状态上 replay 获取到的消息，达到

```math
\mathbb E\lvert\hat\Psi-\Psi\rvert\le\Big(\sum_s\frac{W_s}{N_s}\Big)^{1/2}\big(1+o(1)\big).
```

- **(c) 最优查询（water-filling）.** 在 $`n_s\ge0`$、$`\sum_sn_s=n`$ 下最小化 $`\sum_sW_s/(k_s+n_s)`$（连续松弛），解为

```math
n_s^\star=\Big(\sqrt{W_s/\lambda}-k_s\Big)_+,\qquad \text{其中 }\lambda>0\text{ 由 }\textstyle\sum_sn_s^\star=n\text{ 确定}.
```

  查询流向学习器相对权重高、teacher 覆盖低的状态。

- **(d) 不可约部分.** 若某个状态 $`W_s>0`$ 而 $`k_s=n_s=0`$（学生会到、teacher 不到、也从不查询），它留下的误差下限不能被任何 $`K`$ 消除（[Corollary 6.7](#s6-7)）。

*Proof.* (a) 固定 $`h\in\mathbb R^{\mathcal S}`$，取 tilting 子模型 $`P_{s,\tau}=(1+\tau h_sf_s)\,P^*_s`$。$`f_s`$ 在有限字母表上有界且均值为零，所以 $`\lvert\tau\rvert`$ 足够小时它是概率分布。

- 由 influence function 的定义，沿子模型 $`\frac{\mathrm d}{\mathrm d\tau}\Psi=\sum_s\int f_s\,h_sf_s\,\mathrm dP^*_s=\sum_sh_sW_s`$。
- $`s`$ 上一个样本的 score 为 $`h_sf_s`$，Fisher information 为 $`h_s^2W_s`$；所以对标量参数 $`\tau`$，整条 transcript 的信息为 $`\sum_sN_sh_s^2W_s`$。score 有界，KL 展开因而一致。

[Theorem 6.3](#s6-3) 的证明只用到 KL 的链式法则，它对固定的逐状态样本数同样适用，给出下界 $`\tfrac14\lvert\sum_sh_sW_s\rvert/(\sum_sN_sh_s^2W_s)^{1/2}(1-o(1))`$。由 Cauchy–Schwarz，

```math
\Big(\sum_sh_sW_s\Big)^2=\Big(\sum_s h_s\sqrt{N_sW_s}\cdot\sqrt{W_s/N_s}\Big)^2\le\Big(\sum_sN_sh_s^2W_s\Big)\Big(\sum_sW_s/N_s\Big),
```

等号在 $`h_s=1/N_s`$ 处成立，即得 (a)。

(b) 记 $`\hat P_s`$ 为 $`s`$ 上 $`N_s`$ 个样本的经验分布。二阶展开给出 $`\Psi(\hat P)-\Psi(P^*)=\sum_s\frac1{N_s}\sum_{i\le N_s}f_s(Y_{s,i})+\mathrm{Rem}`$，其中 $`\mathbb E\lvert\mathrm{Rem}\rvert\le C\sum_s\mathbb E\lVert\hat P_s-P^*_s\rVert^2=O(\sum_s1/N_s)=O(1/N)`$。线性项均值为零、方差为 $`\sum_sW_s/N_s`$，由 Jensen 不等式其期望绝对值不超过方差的平方根；在按比例增长下 $`O(1/N)=o\big((\sum_sW_s/N_s)^{1/2}\big)`$。输出 $`Q=\hat P`$ 可以由 replay 实现。

(c) 目标函数在 $`n_s\ge0`$ 上是凸的，在 $`W_s>0`$ 的坐标上严格凸。KKT 条件为：$`n_s>0`$ 时 $`W_s/(k_s+n_s)^2=\lambda`$，$`n_s=0`$ 时 $`W_s/k_s^2\le\lambda`$，这正是所列的公式；预算方程左边在活跃范围内关于 $`\lambda`$ 连续且严格递减，所以 $`\lambda`$ 唯一。

(d) 没有任何观测依赖于 $`P^*_s`$，而 $`\Psi`$ 依赖于它（$`W_s>0`$）。只在 $`s`$ 上不同的两个 oracle 分布给出相同的 transcript 分布、不同的 $`\Psi`$，于是 [Corollary 6.7](#s6-7) 适用。$`\square`$

<a id="s6-11"></a>

**Proposition 6.11（SGD 与 warm Adam 的学习器权重）.** 在 [Assumption 6.9](#s6-9) 之外，再设：

- 每个 agent 状态拥有自己的参数坐标（state-owned coordinates）；
- 每个训练步独立抽取一个状态 $`s_k\sim d`$，并在那里给一条 oracle 消息；
- consumed gradient 与参数无关（线性 response 区间）：$`g=G(s,Y)`$；在状态 $`s`$ 上记 $`m_1=\mathbb Eg`$，$`m_2=\mathbb Eg^2`$；
- 结果为 $`F=\langle\zeta,\theta_R\rangle`$，$`\Psi`$ 取为期望结果变化 $`\mathbb E F(\theta_R)-F(\theta_0)`$，它是 $`(P^*_s)_s`$ 的泛函；
- Adam 的 $`\epsilon`$ 相对 $`\sqrt v`$ 可以忽略，且 $`\beta_1<\sqrt{\beta_2}`$。

则在各自的区间内、当 $`R\to\infty`$ 时，在首项意义下：

- **(a) SGD.** $`f_s^{\rm SGD}(y)=-\eta R\,d(s)\,\zeta_s\,(g-m_1)`$，所以 $`W_s^{\rm SGD}=\eta^2R^2\zeta_s^2\,d(s)^2\,\mathrm{Var}(g)`$。
- **(b) warm Adam，频繁访问区间（frequent regime）.** 设二阶矩的有效平均窗口内访问次数很多：有 bias correction 时 $`d(s)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$，无 bias correction 时还要求 $`t\gg1/(1-\beta_2)`$。则

```math
f_s^{\rm Adam}(y)=-\eta R\sqrt{d(s)}\,\zeta_s\Big[\frac{g-m_1}{\sqrt{m_2}}-\frac{m_1(g^2-m_2)}{2m_2^{3/2}}\Big],\qquad W_s^{\rm Adam}=\eta^2R^2\zeta_s^2\,d(s)\,\mathrm{Var}\Big(\frac{g}{\sqrt{m_2}}-\frac{m_1g^2}{2m_2^{3/2}}\Big).
```

  括号中的第二项是二阶矩通道：一条抬高 $`m_2`$ 的消息会压低此后的每一个步长。

- **(c) warm Adam，孤立访问区间（isolated regime）.** 设 $`d(s)\ll\min\lbrace1-\beta_1,1-\beta_2\rbrace`$，且无 bias correction 或 $`t\gg1/(1-\beta_2)`$。此时每次访问遇到的都是 $`m\approx0`$、$`v\approx0`$，坐标移动 $`-\eta\,\kappa_{\rm iso}\,\mathrm{sign}(g)`$，其中

```math
\kappa_{\rm iso}=\frac{1-\beta_1}{\sqrt{1-\beta_2}\,\big(1-\beta_1/\sqrt{\beta_2}\big)} .
```

  因此 $`f_s(y)=-\eta R\,d(s)\,\zeta_s\,\kappa_{\rm iso}\,(\mathrm{sign}\,g-\mathbb E\,\mathrm{sign}\,g)`$，$`W_s\propto d(s)^2\,\mathrm{Var}(\mathrm{sign}\,g)`$：关于 $`d`$ 的标度与 SGD 相同，但消息只通过符号进入。有 bias correction 时，若在第 $`t\gg1/(1-\beta_1)`$ 步访问一个 $`m`$、$`v`$ 仍为零的坐标（例如第一次访问），位移为 $`-\eta\,\kappa_{\rm iso}\sqrt{1-\beta_2^{\,t}}\,\mathrm{sign}(g)`$：常数改变，关于 $`d`$ 的标度不变。

- **(d) 推论.** 设 $`s`$ 是高频状态、$`s'`$ 是较稀有的状态，$`d(s)>d(s')`$，两者都处在频繁访问区间，方差因子相同。权重比 $`W_{s'}/W_s`$ 在 SGD 下是 $`(d(s')/d(s))^2`$，在 warm Adam 下是 $`d(s')/d(s)`$。所以相对于高频状态，warm Adam 对稀有状态的权重是 SGD 的 $`d(s)/d(s')`$ 倍。覆盖了高频状态、漏掉学生长尾状态的示范，可以对 SGD 充分，对 warm Adam 不充分。

*Proof.* 坐标之间没有相互作用，所以 $`\Psi=\sum_s\zeta_s\,\mathbb E[\theta_{s,R}-\theta_{s,0}]`$，只需对每个 $`s`$ 计算期望位移关于 $`P^*_s`$ 的 Gateaux 导数。记 $`\delta m_1(y)=g(y)-m_1`$，$`\delta m_2(y)=g(y)^2-m_2`$，它们是 $`m_1,m_2`$ 沿 $`\delta_y-P^*_s`$ 方向的导数。

(a) SGD 的位移为 $`-\eta\sum_k1\lbrace s_k=s\rbrace g_k`$，均值 $`-\eta Rd(s)m_1`$，关于 $`P^*_s`$ 线性，导数为 $`-\eta Rd(s)\,\delta m_1`$。

(b) 访问构成 Bernoulli($`d(s)`$) 序列。二阶矩是逐步平方梯度 $`g_k^21\lbrace s_k=s\rbrace`$ 在约 $`\min\lbrace t,1/(1-\beta_2)\rbrace`$ 步有效窗口上的加权平均（bias correction 之后）。在频繁访问区间，窗口内访问很多，由大数定律 $`v_k=d(s)m_2(1+o(1))`$；单次访问使 $`v`$ 相对变化 $`(1-\beta_2)g^2/(d(s)m_2)=o(1)`$，所以 $`m_k`$ 与 $`v_k`$ 之间的相关是低阶的。动量关于梯度线性，$`\sum_km_k=\sum_jg_j1\lbrace s_j=s\rbrace(1-\beta_1^{R-j})`$，均值为 $`Rd(s)m_1(1+o(1))`$。于是期望位移为

```math
\Phi(m_1,m_2)=-\eta R\sqrt{d(s)}\,\frac{m_1}{\sqrt{m_2}}\,\big(1+o(1)\big).
```

$`\Phi`$ 是 $`(m_1,m_2)`$ 的光滑函数，由链式法则（delta method），它的 influence function 是 $`\partial_{m_1}\Phi\,\delta m_1+\partial_{m_2}\Phi\,\delta m_2`$，其中 $`\partial_{m_1}\Phi=-\eta R\sqrt{d}/\sqrt{m_2}`$，$`\partial_{m_2}\Phi=\tfrac12\eta R\sqrt d\,m_1/m_2^{3/2}`$。乘以 $`\zeta_s`$ 即得所列的 $`f_s`$；来自 $`\partial_{m_2}\Phi`$ 的那一项就是二阶矩通道。

(c) 两次访问之间，$`m`$ 与 $`v`$ 每步分别乘以 $`\beta_1`$ 与 $`\beta_2`$；间隔约为 $`1/d(s)`$ 步，所以 $`\beta_1^{1/d}`$ 与 $`\beta_2^{1/d}`$ 可以忽略，每次访问遇到 $`m\approx0`$、$`v\approx0`$。一次梯度为 $`g`$ 的访问之后第 $`j`$ 步，$`m=(1-\beta_1)\beta_1^{\,j}g`$，$`v=(1-\beta_2)\beta_2^{\,j}g^2`$，位移为

```math
-\eta\,\frac{(1-\beta_1)\beta_1^{\,j}g}{\sqrt{(1-\beta_2)\beta_2^{\,j}}\,\lvert g\rvert}=-\frac{\eta(1-\beta_1)}{\sqrt{1-\beta_2}}\Big(\frac{\beta_1}{\sqrt{\beta_2}}\Big)^{j}\mathrm{sign}(g).
```

由于 $`\beta_1<\sqrt{\beta_2}`$，几何级数收敛，各项在 $`O(1/(1-\beta_1/\sqrt{\beta_2}))`$ 步内衰减，远小于访问间隔；对 $`j\ge0`$ 求和得 $`-\eta\kappa_{\rm iso}\mathrm{sign}(g)`$。总位移为 $`-\eta\kappa_{\rm iso}\sum_k1\lbrace s_k=s\rbrace\mathrm{sign}(g_k)`$，均值 $`-\eta Rd(s)\kappa_{\rm iso}\mathbb E\,\mathrm{sign}(g)`$，关于 $`P^*_s`$ 线性，由此得到 $`f_s`$。有 bias correction 时，第 $`t`$ 步访问之后第 $`j`$ 步的位移要除以 $`(1-\beta_1^{\,t+j})`$、乘以 $`\sqrt{1-\beta_2^{\,t+j}}`$；当 $`t\gg1/(1-\beta_1)`$ 且 $`j`$ 在衰减范围内时，这两个因子分别是 $`1+o(1)`$ 与 $`\sqrt{1-\beta_2^{\,t}}\,(1+o(1))`$。

(d) 把 (a) 与 (b) 的权重相除，再用 [Proposition 6.10](#s6-10)(c)、(d)。$`\square`$

- 两个区间在 $`d(s)\approx1-\beta_2`$ 附近相接：在那里一次访问对 $`v`$ 的影响既平均不掉，也衰减不完。两个闭式在那里都不适用。

<a id="s6-12"></a>

**Corollary 6.12（学习器之间的分离）.**

- **(a) 尺度不变性带来的精确分离.** 设 $`\theta=(\alpha,b)`$，$`b>0`$，oracle 的 consumed gradient 为 $`b\,G_\alpha(Y)`$（例如反馈的权重或 advantage 的尺度），示范的分布只依赖于 $`\alpha`$。
  - 对始终激活的范数裁剪 SGD（裁剪阈值 $`c_{\rm clip}`$，几乎处处 $`\lVert bG_\alpha(Y)\rVert>c_{\rm clip}`$）以及 normalized SGD，oracle 平均增量与 $`b`$ 无关，所以 $`\Psi`$ 只依赖于 $`\alpha`$，$`c(\theta)`$ 在 $`b`$ 方向上没有分量。只要示范关于 $`\alpha`$ 的 Fisher information 非退化，误差随 $`K\to\infty`$ 趋于零（对关于 $`\alpha`$ 的模型用 [Theorem 6.5](#s6-5)，$`n=0`$；可实现性对任何使裁剪保持激活的尺度都成立）。
  - 对普通 SGD，$`c(\theta)`$ 的 $`b`$ 分量为 $`-\eta\sum_t\langle g_t,\mathbb E\,G_\alpha\rangle`$；当它非零时，示范不含关于 $`b`$ 的任何信息，由 [Corollary 6.7](#s6-7)，任何 $`K`$ 都不够。
  - 所以同一批示范对一个学习器充分，对另一个不充分。若裁剪只以概率 $`p_{\rm act}<1`$ 激活，[Proposition 5.9](05_optimizer_state_ZH.md#s5-9) 保证每个裁剪后的平均 consumed gradient 与不含 $`b`$ 的向量 $`c_{\rm clip}\,\mathbb E[G_\alpha/\lVert G_\alpha\rVert]`$ 相差不超过 $`c_{\rm clip}(1-p_{\rm act})`$，分离随之平滑地减弱。
- **(b) 长尾覆盖带来的定量分离.** 由 [Proposition 6.10](#s6-10) 与 [Proposition 6.11](#s6-11)(d)：当学生的状态分布有大量低频状态（都处在 warm Adam 的频繁访问区间）、而 teacher 很少到达这些状态时，这些状态承担的信息需求份额在 warm Adam 下比在 SGD 下更大；要达到同样的相对精度，warm Adam 在这些状态上需要相应更多的示范或查询（[Example 6.13](#s6-13)）。

*Proof.* (a) 裁剪始终激活时，$`\mathrm{clip}_{c_{\rm clip}}(bG)=c_{\rm clip}\,G/\lVert G\rVert`$ 与 $`b>0`$ 无关，所以 $`\partial_b\mu^*_t=0`$；其余由 [Theorem 6.5](#s6-5) 与 [Corollary 6.7](#s6-7) 得出：$`\theta_0=(\alpha,b)`$ 与 $`\theta_1=(\alpha,b')`$ 的示范分布相同，而对 SGD，$`\Psi(\theta_1)-\Psi(\theta_0)=-\eta(b'-b)\sum_t\langle g_t,\mathbb E\,G_\alpha\rangle\neq0`$。(b) 把 [Proposition 6.10](#s6-10)(a)–(b) 与 [Proposition 6.11](#s6-11) 的权重结合。$`\square`$

<a id="s6-13"></a>

**Example 6.13（一个常见状态加 $`N`$ 个长尾状态）.** 学生以概率 $`1-q`$ 访问常见状态，以概率 $`q/N`$ 访问 $`N`$ 个长尾状态中的每一个。teacher 对常见状态的覆盖为 $`1-q_T`$，对每个长尾状态为 $`q_T/N`$，所以 $`N_s\propto KH\,d_T(s)`$。各状态的方差因子与部署权重相同，长尾状态处在 warm Adam 的频繁访问区间，即 $`(q/N)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$。

- **对 $`\sum_sW_s/N_s`$ 的贡献**（略去公共因子）：
  - 长尾状态：SGD 下为 $`N(q/N)^2/(q_T/N)=q^2/q_T`$，warm Adam 下为 $`N(q/N)/(q_T/N)=Nq/q_T`$；
  - 常见状态：SGD 下为 $`(1-q)^2/(1-q_T)`$，warm Adam 下为 $`(1-q)/(1-q_T)`$。
- **放大倍数.** 相对于常见状态，长尾项在 warm Adam 下比在 SGD 下大 $`N(1-q)/q`$ 倍。例如 $`N=20`$、$`q=0.2`$ 时放大 $`80`$ 倍；取 $`\beta_2=0.999`$，每个长尾状态在一个二阶矩窗口内的期望访问数为 $`(q/N)/(1-\beta_2)=10`$，处在频繁访问区间。
- **分配.** 因此 [Proposition 6.10](#s6-10)(c) 的 water-filling 规则在 warm Adam 下会比在 SGD 下把更多查询预算分给长尾状态。

## 相关工作

- **下界技术.** 两点法来自 Le Cam（Le Cam, 1973）；带 Fisher information 与 differentiability in quadratic mean 的局部形式沿用标准的局部渐近 minimax 理论（van der Vaart, 1998）。统计泛函的 influence function 是经典工具，这里取的是学习器相对泛函的 influence function。
- **最优设计.** c-optimal design 及其几何刻画可追溯到 Elfving（1952）；[Lemma 6.4](#s6-4) 中的 minimax 交换是 Sion 定理（Sion, 1958）；带先验信息的设计与正则化设计是标准内容（Pukelsheim, 1993）。这里特别的是目标向量 $`c=\sum_tJ_t^*g_t`$：它由 optimizer 的增量映射与声明的任务决定。
- **模仿学习中的主动获取.** DAgger（Ross, Gordon and Bagnell, 2011）在学生访问的状态上查询 teacher。用 [Proposition 6.10](#s6-10) 的语言，它按学生访问频率分配查询；信息意义下最优的分配是按学习器相对权重做 water-filling，而这些权重随 optimizer 改变（[Proposition 6.11](#s6-11)）。
- **数据与 teacher 的选择分数.** LESS（Xia et al., 2024）与 GRACE（Panigrahi et al., 2025）给候选数据或 teacher 打分。Theorem A 不是又一个分数：它是任何蒸馏器都绕不过的下界，以及达到这个下界的获取规则。

**参考文献.** Elfving, G. (1952), *Ann. Math. Statist.* 23, no. 2. Le Cam, L. (1973), *Ann. Statist.* 1, no. 1. Panigrahi, A., et al. (2025), arXiv:2511.02833. Pukelsheim, F. (1993), *Optimal Design of Experiments*, Wiley. Ross, S., Gordon, G., Bagnell, D. (2011), AISTATS. Sion, M. (1958), *Pacific J. Math.* 8, no. 1. van der Vaart, A. W. (1998), *Asymptotic Statistics*, Cambridge University Press. Xia, M., et al. (2024), ICML.
