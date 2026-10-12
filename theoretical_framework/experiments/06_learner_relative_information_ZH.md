# §6 学习器相对的信息界（Theorem A）— 实验

[← 框架总览](../README_ZH.md) · [证明](../proofs/06_learner_relative_information_ZH.md) · [English](06_learner_relative_information.md)

**结论.** 六项检验全部通过。它们是精确的凸规划与合成高斯模型、表格模型上的 Monte Carlo 模拟，检验的是 Theorem A 的数学本身，不是真实的蒸馏运行。

- **设计对偶（[Lemma 6.4](../proofs/06_learner_relative_information_ZH.md#s6-4)）.** 在 216 个实例上，c-optimal design 的取值与对偶的信息范数一致，相对差至多 $`4.5\times10^{-13}`$，奇异与不可辨识的情形也包括在内。
- **下界与可达性（[Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)、[Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5)）.** 在 192 个可辨识实例上，整数舍入后的 c-optimal design 下 plug-in 风险介于 $`0.789\,\mathfrak n`$ 与 $`0.850\,\mathfrak n`$ 之间（中位数 $`0.798\approx\sqrt{2/\pi}`$），落在预测的区间 $`[\mathfrak n/4,\mathfrak n]`$ 之内。
- **混叠（[Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7)）.** 沿示范的零空间，数据分布对直到 $`10^6`$ 的每个 $`K`$ 都精确相同，最坏误差从不低于泛函差距的一半。
- **学习器权重（[Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11)）.** 敏感度关于 $`d`$ 的实测斜率：SGD 为 $`1.00`$，频繁访问区间的 Adam 为 $`0.489`$（权重斜率 $`0.98`$），孤立访问区间的 Adam 为 $`0.970`$。频繁访问区间的闭式（含二阶矩项）误差为 0.6–7.1%，去掉这一项误差变成 14–21%。孤立访问区间的闭式误差不超过 8.6%。在 $`d\approx1-\beta_2`$ 附近，两个闭式都不适用。
- **water-filling（[Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)）.** 在全部 16 个随机 profile 上，water-filling 的实际误差最低；其他分配方式的平均绝对误差高 3% 到 105%，每个预测比值与实际比值相差不超过 $`0.022`$。
- **分离（[Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)(a)）.** 用同一批示范，始终裁剪的学习器误差按 $`K^{-1/2}`$ 下降（从 $`K=1`$ 时的 $`0.091`$ 降到 $`K=10^5`$ 时的 $`0.00024`$），SGD 的最坏误差对每个 $`K`$ 都停在下限 $`0.45`$。

| 检验 | 检验对象 | Code ID | 结果 |
|---|---|---|---|
| [Check 6.1](#c6-1) | [Lemma 6.4](../proofs/06_learner_relative_information_ZH.md#s6-4) | `v016:C2.design` | PASS |
| [Check 6.2](#c6-2) | [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)、[Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5) | `v016:C2.risk` | PASS |
| [Check 6.3](#c6-3) | [Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7) | `v016:C2.aliasing` | PASS |
| [Check 6.4](#c6-4) | [Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11) | `v016:C3.sensitivity` | PASS |
| [Check 6.5](#c6-5) | [Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10) | `v016:C3.allocation` | PASS |
| [Check 6.6](#c6-6) | [Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)(a) | `v016:C3.separation` | PASS |

所有检验都在 CPU 上用 float64 运行，随机种子为 `numpy.SeedSequence([160160, group, subcase, ...])`（PCG64）。`v016:C2` 组耗时 2.3 秒，`v016:C3` 组耗时 27.3 秒。

<a id="c6-1"></a>

## Check 6.1 · c-optimal design 等于对偶的信息范数（Lemma 6.4）

**结论.** PASS。216 个实例上原问题与对偶问题全部一致，最大相对差 $`4.5\times10^{-13}`$，容差 $`10^{-6}`$。

**为什么做.** [Lemma 6.4](../proofs/06_learner_relative_information_ZH.md#s6-4) 断言：最优 approximate design 的取值 $`\inf_\xi c^\top A(\xi)^{-1}c`$ 等于 $`\mathfrak n_{K,n}^2`$，后者是对方向取的上确界。两边用互不相关的凸规划分别计算；若差距超过求解精度，或一边有限另一边无穷，就推翻了引理或它对奇异数据的约定。

**设置.**

- 高斯线性模型：$`\theta\in\mathbb R^p`$，$`p\in\lbrace2,3,5,8\rbrace`$；一条示范观测 $`A_D\theta+N(0,I)`$，所以 $`I_D=A_D^\top A_D`$；在 $`x`$ 处的查询观测 $`f_x^\top\theta+N(0,1)`$，所以 $`I_Q(x)=f_xf_x^\top`$，$`\lvert\mathcal X_Q\rvert\in\lbrace4,16\rbrace`$。
- 每个 $`(p,\lvert\mathcal X_Q\rvert)`$ 有三类示范：满秩；含一个精确零列且 $`c`$ 在该列方向上有非零分量（混叠）；含精确零列但 $`c`$ 在可辨识的范围内（奇异但可估计）。
- $`K\in\lbrace1,10,100\rbrace`$，$`n\in\lbrace0,10,100\rbrace`$：共 216 个实例、864 条断言。
- 原问题：用带解析梯度的 SLSQP 在 design 权重 $`w`$ 的单纯形上最小化 $`c^\top M(w)^{-1}c`$，并检查单纯形上的 KKT（Elfving）条件。
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

<a id="c6-2"></a>

## Check 6.2 · 实际风险介于 𝔫/4 与 𝔫 之间（Theorems 6.3 与 6.5）

**结论.** PASS。在全部 192 个可辨识实例上，整数舍入后的 c-optimal design 下，有效 plug-in 估计量的 Monte Carlo 平均绝对误差落在 $`[0.789,\,0.850]\cdot\mathfrak n_{K,n}`$ 内。没有任何两点下界超过实际风险。

**为什么做.** [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3) 说任何蒸馏器的误差都不低于 $`\mathfrak n/4`$，[Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5) 说 c-optimal 的 plug-in 能达到 $`\mathfrak n(1+o(1))`$。比值落在 $`[1/4,1]`$ 之外就推翻了这一对结论。design 的整数舍入是渐近定理没有覆盖的有限 $`n`$ 效应，所以这里实测它，而不是假设它可以忽略。

**设置.**

- Check 6.1 的 216 个实例去掉 24 个不可辨识的：192 个实例、960 条断言。
- 把 approximate c-optimal design 按最大余数法舍入成整数次数，不改变 $`n`$；rounding loss 定义为方差 $`c^\top A^{-1}c`$ 的相对增量。
- 为示范与查询的充分统计量独立生成高斯噪声，计算 $`c^\top\theta`$ 的有效线性估计量：每个实例 30,000 次重复。这是在模拟观测，而不是从预测的方差里直接抽误差。
- 同时记录：精确的高斯平均绝对误差 $`\sqrt{2/\pi}\sqrt{c^\top A_{\rm int}^{-1}c}`$；[Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3)(a) 在最不利位移 $`\tau=\mathbb I^{-1/2}`$ 处的 Pinsker 两点界；同一位移处精确的高斯总变差两点界。

**结果.**

| 量 | 值 |
|---|---|
| 实际风险 $`/\,\mathfrak n_{K,n}`$：最小 / 中位数 / 最大 | $`0.7885`$ / $`0.7982`$ / $`0.8504`$ |
| $`\sqrt{2/\pi}`$（精确 design，不舍入） | $`0.7979`$ |
| 最大 rounding loss（即比值为 $`0.850`$ 的实例：16 个查询点上只有 $`n=10`$ 次查询） | 方差增加 12.3% |
| 高于 $`\mathfrak n/4`$ / 低于 $`\mathfrak n`$ 的最小余量（绝对值） | $`0.0204`$ / $`0.0075`$ |
| Pinsker 两点界 $`/\,\mathfrak n`$ | 每个实例都是 $`0.2500`$ |
| 精确高斯 TV 两点界 $`/\,\mathfrak n`$（同一位移） | 每个实例都是 $`0.3085=1-\Phi(1/2)`$ |
| 实际风险 $`/`$ Pinsker 界，最小值 | $`3.15`$ |
| Monte Carlo 与精确高斯 MAE 之差，最大 | $`2.91`$ 个标准误（99.9% 逐点容差 $`3.29`$） |

![Figure 6.1：信息界、设计对偶与混叠](../figures/fig6_1_information_bound.png)

*Figure 6.1.* (a) 全部 192 个可辨识设定上，实际 plug-in 风险除以信息范数 $`\mathfrak n_{K,n}`$，并画出 Theorems 6.3 与 6.5 的因子 $`1/4`$、$`1`$ 以及高斯值 $`\sqrt{2/\pi}`$；少数高于 $`\sqrt{2/\pi}`$ 的点，是小查询预算的整数舍入放大了方差。(b) c-optimal design 的取值对对偶信息目标（Lemma 6.4），落在对角线上；两边都无穷的 24 个设定只计数、不画出。(c) 八个混叠实例的最坏误差下限 $`\lvert c^\top h\rvert/2`$ 随 $`K`$ 的变化，每个实例都是水平线（Corollary 6.7）。

**为什么支持 Theorems 6.3 与 6.5.**

- 实际风险落在 $`\sqrt{2/\pi}\,\mathfrak n`$ 附近，即精确的高斯值；即使做了整数舍入，[Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5) 的上界常数 $`1`$ 仍有余量。
- 在定理所取的位移处，Pinsker 两点界恰好等于 $`\mathfrak n/4`$，所以 [Theorem 6.3](../proofs/06_learner_relative_information_ZH.md#s6-3) 中的常数 $`1/4`$ 正是证明给出的；同一点上精确的 TV 界稍大（$`0.31\,\mathfrak n`$）。
- 局限：这个模型中 $`\Psi`$ 是线性的，实验是高斯的，所以 [Theorem 6.5](../proofs/06_learner_relative_information_ZH.md#s6-5) 的二阶条件 (S) 与有效性条件 (E) 精确成立。本检验没有覆盖非线性 $`\Psi`$ 或非高斯实验。

<a id="c6-3"></a>

## Check 6.3 · 混叠方向留下任何 K 都去不掉的下限（Corollary 6.7）

**结论.** PASS。对每个混叠模型与每个 $`K\in\lbrace1,10,100,10^6\rbrace`$，$`\theta`$ 与 $`\theta+h`$ 处的数据分布精确相同，最坏误差至少是泛函差距的一半。

**为什么做.** [Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7) 断言：若 $`h`$ 在示范的零空间中且 $`c^\top h\neq0`$，示范再多也没有用。数据分布出现非零差别，或某个估计量在两个世界中都好于 $`\lvert\Delta\Psi\rvert/2`$，都会推翻它。

**设置.**

- 8 个混叠模型（$`A_D`$ 有一个精确的零列，$`A_Dh=0`$，不涉及数值容差），$`n=0`$，$`K\in\lbrace1,10,100,10^6\rbrace`$：32 个实例、3,408 条断言。
- 比较 $`\theta`$ 与 $`\theta+h`$ 处示范充分统计量的均值与协方差；检查下界背后的确定性三角不等式，并记录取到下界的中点估计量。

**结果.**

- 均值与协方差之差：208 次比较全部精确为 $`0`$。
- 三角不等式：3,200 次检查，最大残差 $`5.6\times10^{-17}`$（容差 $`2\times10^{-14}`$）。
- 例：泛函差距 $`0.1866`$，minimax 下限 $`0.0933`$，中点估计量在每个 $`K`$ 上都取到它（Figure 6.1(c)）。

**为什么支持 Corollary 6.7.** 两个分布是完全相同，而不仅是接近，所以下限是精确的，且与 $`K`$ 无关；replay 或改写示范帮不上忙，正是这个机制。

<a id="c6-4"></a>

## Check 6.4 · 学习器权重：SGD 按 d² 标度，warm Adam 按 d 标度（Proposition 6.11）

**结论.** PASS。每个断言的斜率都落在其 bootstrap 95% 区间加上预先声明的 $`0.1`$ 之内。频繁访问区间的 Adam 敏感度关于 $`d`$ 的斜率为 $`0.489`$，即权重斜率 $`0.98`$；SGD 为 $`1.00`$（权重斜率 $`2.0`$）。闭式在各自区间内准确，并且如命题所说，在 $`d\approx1-\beta_2`$ 附近失效。

**为什么做.** [Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11) 对每个 optimizer 预测：在某个状态上多一个 oracle 样本，会让结果移动多少。这些权重决定查询该往哪里分配（[Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)），也决定了 SGD 与 Adam 之间的分离。实测敏感度的斜率不对，或闭式漏掉二阶矩项，都会推翻这个命题。

**设置.**

- 每个目标状态拥有一个标量坐标；每步访问一个状态；teacher 标签 $`y\sim\mathrm{Bernoulli}(\pi)`$，$`\pi=0.35`$；梯度 $`g=c_0-y`$，$`c_0=0.2`$；$`\zeta=1`$，$`\eta=0.01`$，$`\epsilon=10^{-12}`$。
- 学习器：SGD；始终裁剪到 $`0.1`$ 的 SGD；$`\beta_1=0.9`$ 的 Adam。三者共用访问序列与标签的随机数。
- 敏感度 $`\mathrm d\,\mathbb E\theta_{s,R}/\mathrm d\pi_s`$ 用 common random numbers 的中心差分测量（$`\delta=0.02`$；$`\delta=0.04`$ 作诊断），扰动在共同的 burn-in 之后才开始。

| 区间 | $`\beta_2`$ | Bias correction | $`d(s)`$（对数等距） | $`d/(1-\beta_2)`$ | Burn-in | $`R`$ | 重复次数 |
|---|---|---|---|---|---|---|---|
| frequent | 0.999 | 否 | 0.005–0.4，7 个值 | 5–400 | 20,000 | 30,000 | 384 |
| isolated | 0.9 | 否 | 0.0005–0.01，6 个值 | 0.005–0.1 | 2,000 | 60,000 | 384 |
| transition | 0.999 | 否 | 0.0003–0.02，7 个值 | 0.3–20 | 20,000 | 30,000 | 384 |
| 短 horizon | 0.999 | 是 | 0.0005–0.01，6 个值 | 0.5–10（$`R`$ 内 0.1–2 次访问） | 0 | 200 | 20,000 |

- 断言的斜率：SGD 与裁剪 SGD 在所有区间都是 $`1`$；频繁访问区间的 Adam 为 $`1/2`$；孤立访问区间与短 horizon 的 Adam 为 $`1`$。transition 区间的斜率只做描述。

**结果.**

| 区间 | 学习器 | 关于 $`d`$ 的斜率 [bootstrap 95% CI] | 闭式与实测之差 |
|---|---|---|---|
| frequent | Adam | $`0.4887`$ $`[0.4827,0.4948]`$ | 频繁访问闭式：0.6–7.1% |
| frequent | SGD / 裁剪 SGD | $`1.0029`$ $`[0.9967,1.0093]`$ | 精确的线性公式 |
| isolated | Adam | $`0.9699`$ $`[0.9443,0.9969]`$ | 孤立访问闭式，$`\kappa_{\rm iso}=6.16`$：+0.2% 到 −8.6% |
| isolated | SGD / 裁剪 SGD | $`0.9970`$ $`[0.9715,1.0234]`$ | 精确的线性公式 |
| transition | Adam | $`0.6252`$ $`[0.5952,0.6610]`$ | 见下 |
| transition | SGD / 裁剪 SGD | $`1.0185`$ $`[0.9896,1.0518]`$ | 精确的线性公式 |
| 短 horizon | Adam | $`0.9166`$ $`[0.8599,0.9786]`$ | 见下 |
| 短 horizon | SGD / 裁剪 SGD | $`0.9722`$ $`[0.9172,1.0328]`$ | 精确的线性公式 |

- **二阶矩项.** 在频繁访问区间，去掉 [Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11)(b) 的第二项，会把实测敏感度高估 14%（$`d=0.005`$）到 21%（$`d=0.4`$）。完整公式的最大误差 7.1% 出现在最小的 $`d`$ 上，那里一个二阶矩窗口内只有 5 次访问。
- **孤立访问区间.** 孤立闭式的误差从 $`d/(1-\beta_2)=0.005`$ 时的 0.2% 增长到 $`0.1`$ 时的 8.6%，因为访问开始相互重叠。
- **过渡区（$`d\approx1-\beta_2`$）.** 频繁访问闭式在 $`d/(1-\beta_2)=0.3`$ 时偏差 −46%，$`0.6`$ 时 −17%，从 $`1.2`$ 起在 10% 以内。孤立访问闭式在 $`0.3`$ 时偏差 −19%，$`0.6`$ 时 −12.5%，再往上差得多（−22% 到 −81%）。实测斜率 $`0.625`$ 介于两个区间之间。两个闭式的切换点在 $`d/(1-\beta_2)\approx1`$。
- **带 bias correction 的短 horizon.** 斜率 $`0.917`$ 在预先声明的容差内与 $`1`$ 一致，但它的 95% 区间离 $`1`$ 差 $`0.021`$，这是本检验中最大的斜率残差。量级只有未修正孤立闭式的 24–33%。这正是 [Proposition 6.11](../proofs/06_learner_relative_information_ZH.md#s6-11)(c) 对 bias correction 的预测：第 $`t`$ 步的第一次访问带有因子 $`\sqrt{1-\beta_2^{\,t}}`$，它在 $`t\le200`$ 上的平均为 $`0.29`$（粗略估算，不属于检验本身）。
- **线性学习器.** 52 次比较中，Monte Carlo 均值与精确公式之差都在 $`1.52`$ 个标准误以内（门限 $`4.5`$）。12 行的敏感度全部为正。
- **差分步长.** 把 $`\delta`$ 从 $`0.02`$ 换成 $`0.04`$，敏感度最多变化 $`1.24`$（频繁访问区间的 Adam，$`d=0.4`$，占数值的 0.4%）。

![Figure 6.2：学习器权重、查询分配与尺度不变性分离](../figures/fig6_2_learner_weights.png)

*Figure 6.2.* (a) 对数坐标下实测敏感度随访问频率 $`d(s)`$ 的变化，带逐点 95% 区间，图例给出拟合斜率与预测斜率：SGD 与裁剪 SGD 为 $`1`$，频繁访问区间的 Adam 为 $`1/2`$，孤立访问区间与短 horizon 带 bias correction 的 Adam 为 $`1`$。各区间的 horizon 与 burn-in 不同，不同曲线的绝对高度不可比。(b) 每个 profile 上 water-filling、均匀、正比于 $`d`$、SGD 权重四种分配的实际平均绝对误差；water-filling 在每个 profile 上都最低。(c) 预测误差比与实际误差比，带 familywise 区间，落在对角线上（Proposition 6.10）。(d) 误差随 $`K`$ 的变化：始终裁剪的 SGD 按 $`K^{-1/2}`$ 下降并低于其 RMS 界；SGD 在两个世界中的最坏误差停在混叠下限 $`0.45`$（Corollary 6.12(a)）。

**为什么支持 Proposition 6.11.**

- 斜率正好按预测把学习器分开：SGD 的权重按 $`d^2`$ 标度，频繁访问区间的 Adam 按 $`d`$ 标度，孤立访问区间的 Adam 回到 $`d^2`$，且只依赖符号。
- 含二阶矩项的闭式在其区间内准确，这一项也是必需的：去掉它，误差从 0.6–7.1% 升到 14–21%。
- 偏差出现在命题指明闭式失效的地方：$`d\approx1-\beta_2`$ 附近；以及短 horizon 加 bias correction 时的常数（而不是斜率）。

<a id="c6-5"></a>

## Check 6.5 · water-filling 使实际误差最小（Proposition 6.10）

**结论.** PASS。全部 16 个 profile 上 water-filling 的实际平均绝对误差最小，五种对照分配的预测误差比与实际误差比在 Monte Carlo 误差内一致。

**为什么做.** [Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10) 既给出最优分配，也通过 $`\sum_sW_s/N_s`$ 预测其他分配会差多少。若某个对照分配的实际误差更低，或某个预测比值落在联合置信区间之外，就推翻了它。

**设置.**

- 16 个随机 profile，每个 6–10 个状态：学生访问频率 $`d`$ 对数等距后打乱，teacher 访问频率 $`d_T`$ 取 Dirichlet，Bernoulli 分布的 $`\pi\in[0.15,0.85]`$，学习器相对权重 $`W_s=d(s)\,e^{N(0,1.44)}`$（关于 $`d`$ 线性，与频繁访问区间的 Adam 一样）。
- teacher 访问 800 次，查询 $`n=4{,}000`$ 次。plug-in 泛函关于各状态的概率精确线性，系数为 $`\sqrt{W_s/(\pi_s(1-\pi_s))}`$，所以它的方差公式是精确的。
- water-filling 先对 $`\lambda`$ 求标量根，再用边际增益交换做整数化。对照：均匀；正比于 $`d`$；用 SGD 型权重 $`W_s\,d(s)`$ 做 water-filling（相当于用了另一个学习器的权重）；按梯度范数（$`\propto d\sqrt{0.04+0.6\pi}`$）；按不确定性（$`\propto\pi(1-\pi)`$）。
- 用二项分布的充分统计量模拟整数次访问：每种分配 60,000 次重复。比值区间是 familywise 95%（对两种比值、所有 profile、五种对照做 Bonferroni 校正）。
- 另加一个情形：某个状态 $`W>0`$，却既没有 teacher 访问也没有查询。

**结果.**

| 对照分配 | 实际 MAE 与 water-filling 之比：最小 / 中位数 / 最大 | $`\lvert`$预测 − 实际$`\rvert`$ 的最大值 |
|---|---|---|
| 均匀 | $`1.136`$ / $`1.415`$ / $`1.858`$ | $`0.010`$ |
| 正比于 $`d`$ | $`1.033`$ / $`1.209`$ / $`1.695`$ | $`0.009`$ |
| 用 SGD 型权重做 water-filling | $`1.078`$ / $`1.165`$ / $`1.315`$ | $`0.011`$ |
| 梯度范数 | $`1.028`$ / $`1.188`$ / $`1.743`$ | $`0.011`$ |
| 不确定性 | $`1.146`$ / $`1.441`$ / $`2.054`$ | $`0.022`$ |

- 预算残差至多 $`5.9\times10^{-11}`$；KKT 残差 $`4.4\times10^{-16}`$；连续与整数分配在所比较的分配中都是最优的。
- 每个 profile 上，最好的对照分配的实际平均绝对误差比 water-filling 高 2.8% 到 21.9%。
- 预测比值与实际比值之间的最大标准化偏差为 $`3.23`$，低于 familywise 临界值 $`3.60`$。
- 不可约情形：权重 $`(1,2)`$，teacher 次数 $`(100,0)`$，不查询。预测方差为 $`+\infty`$；未覆盖状态上的两个候选分布 $`0.25`$ 与 $`0.75`$ 给出完全相同的观测；泛函差距 $`0.5`$，minimax 下限 $`0.25`$。

**为什么支持 Proposition 6.10.**

- 公式 $`\sum_sW_s/N_s`$ 对实际误差比的预测精确到约百分之一，由它导出的分配在每个 profile 上都胜出。
- 用另一个学习器的权重要多付出 8% 到 32% 的误差：分配是学习器相对的，不是数据本身的性质。
- 局限：profile 是合成的，plug-in 泛函是线性的；本检验没有从真实学习器上估计 $`W_s`$。

<a id="c6-6"></a>

## Check 6.6 · 同一批示范对裁剪 SGD 充分、对 SGD 不充分（Corollary 6.12(a)）

**结论.** PASS。裁剪学习器的增量精确地尺度不变，误差按 $`K^{-1/2}`$ 衰减；SGD 的最坏误差从不低于泛函差距的一半。

**为什么做.** [Corollary 6.12](../proofs/06_learner_relative_information_ZH.md#s6-12)(a) 断言一个精确的分离：揭示 $`\alpha`$ 却不揭示尺度 $`b`$ 的示范，对始终裁剪的学习器充分，对 SGD 毫无用处。若裁剪后的增量依赖于 $`b`$，或 SGD 的误差低于下限，就推翻了它。

**设置.**

- $`\alpha=0.65`$；梯度取值 $`\lbrace0.2,-0.8\rbrace`$，乘以 $`b\in\lbrace1,3\rbrace`$；裁剪阈值 $`0.1`$，所以裁剪始终激活；$`\eta R=1`$。
- 示范只观测标签，其分布是 $`\mathrm{Binomial}(K,\alpha)`$，与 $`b`$ 无关。
- $`K=1,10,\dots,10^5`$；60,000 次重复。在两个世界中分别计算裁剪学习器的泛函与一个取中点尺度的 SGD 估计量。

**结果.**

| $`K`$ | 裁剪学习器 MAE | RMS 界 | SGD 最坏误差 | 下限 $`\lvert\Delta\Psi\rvert/2`$ |
|---|---|---|---|---|
| 1 | $`0.0910`$ | $`0.0954`$ | $`1.0448`$ | $`0.45`$ |
| 10 | $`0.0247`$ | $`0.0302`$ | $`0.4714`$ | $`0.45`$ |
| 100 | $`0.00762`$ | $`0.00954`$ | $`0.4502`$ | $`0.45`$ |
| 1,000 | $`0.00240`$ | $`0.00302`$ | $`0.4501`$ | $`0.45`$ |
| 10,000 | $`0.000763`$ | $`0.000954`$ | $`0.4500`$ | $`0.45`$ |
| 100,000 | $`0.000241`$ | $`0.000302`$ | $`0.4500`$ | $`0.45`$ |

- 裁剪增量的尺度不变性：12 次比较全部精确成立。
- 这里用的 SGD 估计量逼近下限；而下限本身对任何估计量成立，因为两个世界的示范分布完全相同。

**为什么支持 Corollary 6.12(a).** 这个分离是精确的：用同一批示范，一个学习器的误差以参数速率消失，另一个对每个 $`K`$ 都离零有固定距离。哪些示范算充分，取决于学习器。

## 复现

在仓库根目录（只用 CPU）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C2,C3 --output /tmp/v016_C2_C3.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

第一条命令重新生成完整的结果文件（约 104 秒）；第二条只把本页的两组检验写到一个单独的文件。

## 产物

- `theory/v016_checks/results_v016.json`：claim id `v016:C2.design`、`v016:C2.risk`、`v016:C2.aliasing`、`v016:C3.sensitivity`、`v016:C3.allocation`、`v016:C3.separation`，每项都含断言、最大残差、失败记录与数据行。
- `theory/v016_checks/results_v016_raw.json.gz`：完整的数据行（design 输入、各分配的次数）；不纳入 git，其 SHA-256 记录在主文件的 `raw_artifact` 下。
- `theory/v016_checks/SPEC.md`（`v016:C2` 与 `v016:C3` 两节）与 `theory/v016_checks/check_v016.py`（函数 `c2`、`c3`）。
- `figures/fig6_1_information_bound.png`、`figures/fig6_2_learner_weights.png`（同目录下有 PDF 版本）。
