# Lemma 6.4 · 设计的 minimax 等式

*实验*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-04_design_minimax_equality_ZH.md) · [证明](../proofs/06-04_design_minimax_equality_ZH.md) · [实验](../experiments/06-04_design_minimax_equality_ZH.md) · [English](06-04_design_minimax_equality.md)

**本页检验什么.** 两项检验。

- [Check 6.1](../experiments/06-04_design_minimax_equality_ZH.md#c6-1)：在 216 个实例上（包括奇异与不可辨识的情形），c-optimal design 的取值与对偶的信息范数一致，相对差至多 $`4.5\times10^{-13}`$。结论：PASS。
- [Check 6.7](../experiments/06-04_design_minimax_equality_ZH.md#c6-7)：一个**所有** design 都让某个参数方向不可辨识的实例，因此没有任何 $`A(\xi)`$ 可逆。伪逆形式的 design 取值与对偶都等于 $`2.25`$，并在预言的 design 处取到。结论：PASS。

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

**被检验的陈述（[Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4)）.** 设 $`\mathcal X_Q`$ 是紧度量空间，$`x\mapsto I_Q(\theta;x)`$ 连续。令 $`\xi`$ 取遍 $`\mathcal X_Q`$ 上的 Borel 概率测度（approximate design），$`M(\xi)=\int I_Q(\theta;x)\,\xi(\mathrm dx)`$，$`A(\xi)=K\,I_D(\theta)+n\,M(\xi)`$；不要求任何 design 使 $`A(\xi)`$ 可逆。对半正定矩阵 $`A`$ 定义

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

| Check | 检验内容 | Code ID | 结果 |
|---|---|---|---|
| [Check 6.1](#c6-1) | 满秩、奇异与不可辨识实例上的等式 | `v016:C2.design` | PASS |
| [Check 6.7](#c6-7) | 所有 design 都奇异时的等式与可达性，(b) 部分 | `v017:C12.singular` | PASS |

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

<a id="c6-7"></a>

## Check 6.7 · 所有 design 都奇异：伪逆取值等于对偶（Lemma 6.4(b)）

**结论.** PASS。没有任何 design 使 $`A(\xi)`$ 可逆，但取值 $`\inf_\xi\Phi_c(A(\xi))`$ 与对偶上确界完全一致（两边都是 $`2.25`$），最优 design 正是预言的那个，对所有 design 的网格搜索也从未低于它。

**为什么做.** [Lemma 6.4](../statements/06-04_design_minimax_equality_ZH.md#s6-4) 的 (b) 部分去掉了"每个 design 都可逆"的要求。检验它的最小情形是：存在一个任何 design 都照不到的无关方向。原来的表述对这种情形根本不适用；引理则预言取值有限，因为目标在这个方向上没有分量。

**设置.**

- 参数 $`\theta\in\mathbb R^3`$，没有示范（$`K=0`$），查询预算为一个单位（$`n=1`$）。
- 两个查询点：$`x_1`$ 的信息为 $`I_Q(x_1)=e_1e_1^\top`$，$`x_2`$ 的信息为 $`I_Q(x_2)=4\,e_2e_2^\top`$。design 在 $`x_1`$ 上放权重 $`p`$、在 $`x_2`$ 上放 $`1-p`$，于是 $`A(p)=\mathrm{diag}(p,\,4(1-p),\,0)`$；第三个坐标永远照不到。
- 目标 $`c=(1,1,0)`$，所以未被照到的方向 $`e_3`$ 满足 $`c^\top h=0`$，并且当 $`0<p<1`$ 时 $`c\in\mathrm{range}\,A(p)`$。于是 $`\Phi_c(A(p))=1/p+1/(4(1-p))`$，在 $`p^*=2/3`$ 取最小值 $`9/4`$。
- 对偶：$`\sup_h(c^\top h)^2/\max\lbrace h_1^2,4h_2^2\rbrace`$，在 $`h=(2,1,7)`$ 处求值；第三个坐标可以任取，因为它不进入任何一边。
- 独立确认：在 $`p\in[0.001,0.999]`$ 的 10,001 个网格点上计算 $`\Phi_c(A(p))`$。

**结果.**

| 量 | 取值 |
|---|---|
| 原问题 $`c^\top A(p^*)^+c`$，$`p^*=2/3`$ | 2.25 |
| 对偶，$`h=(2,1,7)`$ | 2.25 |
| 对 design 网格取的最小值 | 2.2500000000045 |
| $`A(p^*)`$ 的秩 | 2（共 3） |
| 未被照到的方向 | 1 |

**为什么支持 Lemma 6.4.** 原问题与对偶取值相等，同时证明了两者都是最优的；网格搜索确认没有 design 做得更好。最优 design 让 $`e_3`$ 保持不可辨识，却没有任何损失，这正是 (b) 部分的内容：目标不关心的方向永远不必辨识。

## 复现

在仓库根目录（只用 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2 --output /tmp/v016_C2.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

第一条命令把 [Check 6.1](#c6-1) 的 `v016` 检验组单独重跑到一个文件；不带 `--only` 运行 `theory/v016_checks/check_v016.py` 会重新生成完整结果文件（约 104 秒）。这些检验都在 CPU 上用 float64 运行，随机种子为 `numpy.SeedSequence([160160, group, subcase, ...])`（PCG64）。[Check 6.7](#c6-7) 属于 `v017` 检验集，几秒即可跑完：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

检验用 NumPy（[Harris et al., 2020](#ref-harris2020array)）做线性代数和随机数，用 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）做优化和统计函数。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C2.design`，每项都含断言、最大残差、失败记录与数据行。
- `theory/v016_checks/results_v016_raw.json.gz`：完整的数据行；不纳入 git，其 SHA-256 记录在主文件的 `raw_artifact` 下。
- `theory/v016_checks/SPEC.md`（`v016:C2` 两节或一节）与 `theory/v016_checks/check_v016.py`。
- `theory/v017_checks/results_v017.json`：条目 `v017:C12.singular`；`theory/v017_checks/check_v017.py` 中的函数 `c_singular`。
- `figures/fig6_1_information_bound.png`（同目录下有 PDF 版本）

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
