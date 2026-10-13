# Lemma 6.4 · 设计的 minimax 等式

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-04_design_minimax_equality_ZH.md) · [证明](../proofs/06-04_design_minimax_equality_ZH.md) · [实验](../experiments/06-04_design_minimax_equality_ZH.md) · [English](06-04_design_minimax_equality.md)

**本页检验什么.** Lemma 6.4 由 [Check 6.1](../experiments/06-04_design_minimax_equality_ZH.md#c6-1) 检验：在 216 个实例上（包括奇异与不可辨识的情形），c-optimal design 的取值与对偶的信息范数一致，相对差至多 $`4.5\times10^{-13}`$。结论：PASS。

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

**被检验的陈述（[Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)）.** 设 $`\mathcal X_Q`$ 是紧度量空间，$`x\mapsto I_Q(\theta;x)`$ 连续。令 $`\xi`$ 取遍 $`\mathcal X_Q`$ 上的 Borel 概率测度（approximate design），$`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$，$`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$，并设对每个 $`\xi`$ 都有 $`A(\xi)\succ0`$。则

```math
\inf_\xi c^\top A(\xi)^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

并且下确界由某个 design $`\xi^*`$ 取到，它就是目标向量为 $`c=\sum_tJ_t^*g_t`$ 的 **c-optimal design**。当 $`n=0`$ 且 $`I_D`$ 奇异时：若 $`c\in\mathrm{range}(I_D)`$，等式以 $`c^\top(KI_D)^+c`$ 成立；否则两边都是 $`+\infty`$。

<a id="c6-1"></a>

## Check 6.1 · c-optimal design 等于对偶的信息范数（Lemma 6.4）

**结论.** PASS。216 个实例上原问题与对偶问题全部一致，最大相对差 $`4.5\times10^{-13}`$，容差 $`10^{-6}`$。

**为什么做.** [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4) 断言：最优 approximate design 的取值 $`\inf_\xi c^\top A(\xi)^{-1}c`$ 等于 $`\mathfrak n_{K,n}^2`$，后者是对方向取的上确界。两边用互不相关的凸规划分别计算；若差距超过求解精度，或一边有限另一边无穷，就推翻了引理或它对奇异数据的约定。

**设置.**

- 高斯线性模型：$`\theta\in\mathbb R^p`$，$`p\in\lbrace2,3,5,8\rbrace`$；一条示范观测 $`A_D\theta+N(0,I)`$，所以 $`I_D=A_D^\top A_D`$；在 $`x`$ 处的查询观测 $`f_x^\top\theta+N(0,1)`$，所以 $`I_Q(x)=f_xf_x^\top`$，$`\lvert\mathcal X_Q\rvert\in\lbrace4,16\rbrace`$。
- 每个 $`(p,\lvert\mathcal X_Q\rvert)`$ 有三类示范：满秩；含一个精确零列且 $`c`$ 在该列方向上有非零分量（混叠）；含精确零列但 $`c`$ 在可辨识的范围内（奇异但可估计）。
- $`K\in\lbrace1,10,100\rbrace`$，$`n\in\lbrace0,10,100\rbrace`$：共 216 个实例、864 条断言。
- 原问题：用带解析梯度的 SLSQP 在 design 权重 $`w`$ 的单纯形上最小化 $`c^\top M(w)^{-1}c`$，并检查单纯形上的 KKT（Elfving）条件（[Elfving, 1952](#ref-elfving1952optimum)）。
- 对偶问题：在 $`c^\top h=1`$、$`t\ge\pm f_x^\top h`$ 约束下最小化 $`K\,h^\top I_Dh+n\,t^2`$，这是一个凸二次规划。若 SLSQP 报告状态失败，就用带精确 Hessian 的 trust-constr 独立求解，再用 active-face KKT 线性方程组验证：平稳性、可行性与乘子非负都要达到 $`10^{-9}`$。
- $`n=0`$：用谱方法检查值域；不可辨识的 $`c`$ 两边都是 $`+\infty`$，可辨识的奇异 $`c`$ 用伪逆。

**结果.**

| 断言 | 实例数 | 最大残差 | 容差 |
|---|---|---|---|
| 原–对偶相对差 | 216 | $`4.5\times10^{-13}`$ | $`10^{-6}`$ |
| 单纯形 KKT（Elfving）相对差 | 216 | $`1.96\times10^{-7}`$ | $`10^{-6}`$ |
| 对偶可行性 | 216 | $`1.0\times10^{-13}`$ | $`10^{-8}`$ |
| 求解成功 | 216 | 0 次失败 | 0 |

- 136 个对偶问题直接由 SLSQP 解出。另有 8 个 SLSQP 报告状态失败（数值其实准确），随后 trust-constr 与 active-face KKT 证书都通过。72 个 $`n=0`$ 的情形走谱方法。
- 24 个 $`n=0`$ 的混叠情形两边都是 $`+\infty`$，符合引理的约定。

**为什么支持 Lemma 6.4.** 一个在 design 上优化、一个在方向上优化，两种独立的表述在满秩、奇异与不可辨识的实例上给出同一个数，精度接近机器精度。所以引理中 $`\sup`$ 与 $`\inf`$ 的交换不是某个求解器的偶然结果。本检验用的是秩一的查询信息，引理本身并不需要这一点。

![Figure 6.1](../figures/fig6_1_information_bound.png)

*Figure 6.1 中与本页有关的是 (b).* c-optimal design 的取值对照对偶的信息目标（Lemma 6.4），点落在对角线上；两侧都为无穷的 24 个设定只计数、不画出。(a)、(c) 分别对应 Theorems 6.3/6.5 与 Corollary 6.7。

## 复现

在仓库根目录（只用 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2 --output /tmp/v016_C2.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

第一条命令把本页的检验组单独重跑到一个文件；不带 `--only` 运行 `theory/v016_checks/check_v016.py` 会重新生成完整结果文件（约 104 秒）。所有检验都在 CPU 上用 float64 运行，随机种子为 `numpy.SeedSequence([160160, group, subcase, ...])`（PCG64）。检验用 NumPy（[Harris et al., 2020](#ref-harris2020array)）做线性代数和随机数，用 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）做优化和统计函数。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C2.design`，每项都含断言、最大残差、失败记录与数据行。
- `theory/v016_checks/results_v016_raw.json.gz`：完整的数据行；不纳入 git，其 SHA-256 记录在主文件的 `raw_artifact` 下。
- `theory/v016_checks/SPEC.md`（`v016:C2` 两节或一节）与 `theory/v016_checks/check_v016.py`。
- `figures/fig6_1_information_bound.png`（同目录下有 PDF 版本）

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
