# Proposition 4.13 · 有限时域 Gramian，以及控制理论中不能照搬的部分

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-13_finite_horizon_gramians_ZH.md) · [证明](04-13_finite_horizon_gramians_ZH.md) · [实验](../experiments/04-13_finite_horizon_gramians_ZH.md) · [English](04-13_finite_horizon_gramians.md)

**本页证明什么.** (a) 利用 $`XY`$ 与 $`YX`$ 有相同非零特征值，挪动 $`H^\top H`$ 的因子。(b) 是线性方程组的最小范数解加一个 Rayleigh 商。除了时域有限，没有用到 $`A_t`$ 的任何性质。

**本页用到的设定.**

- 有限时域线性化：可达性矩阵 $`\mathscr C_\tau`$、可观性矩阵 $`\mathscr O_\tau`$、Hankel 算子 $`H_\tau=\mathscr O_\tau\mathscr C_\tau`$、Gramian $`W_c=\mathscr C_\tau\mathscr C_\tau^\top`$ 与 $`W_o=\mathscr O_\tau^\top\mathscr O_\tau`$，加权后都用欧氏范数（单步情形见 [Proposition 4.12](../statements/04-12_response_predictive_state_factorization_ZH.md#s4-12)）。下面省略下标 $`\tau`$。

**陈述（[Proposition 4.13](../statements/04-13_finite_horizon_gramians_ZH.md#s4-13)）.** (a) 对每个 $`j`$，$`\sigma_j(H)^2=\lambda_j(W_c^{1/2}W_oW_c^{1/2})`$。(b) 对 $`x\in\mathrm{range}\,W_c`$，到达 $`x`$ 的最小能量为 $`x^\top W_c^+x`$，每个到达 $`x`$ 的输入给出的输出能量都是 $`x^\top W_ox`$，$`\sigma_1(H)^2=\sup_xx^\top W_ox/x^\top W_c^+x`$（对非零可达的 $`x`$ 取），且 $`H=0`$ 当且仅当 $`W_oW_c=0`$。

**证明.**

**(a)** $`H`$ 的奇异值平方是 $`H^\top H=\mathscr C^\top\mathscr O^\top\mathscr O\mathscr C=\mathscr C^\top W_o\mathscr C`$ 的特征值。对矩阵 $`X\in\mathbb R^{a\times b}`$、$`Y\in\mathbb R^{b\times a}`$，$`XY`$ 与 $`YX`$ 有相同的非零特征值及代数重数。取 $`X=\mathscr C^\top`$、$`Y=W_o\mathscr C`$，$`\mathscr C^\top W_o\mathscr C`$ 的非零特征值就是 $`W_o\mathscr C\mathscr C^\top=W_oW_c`$ 的非零特征值。再取 $`X=W_c^{1/2}`$、$`Y=W_oW_c^{1/2}`$，它们又是对称半正定矩阵 $`W_c^{1/2}W_oW_c^{1/2}`$ 的非零特征值。$`H^\top H`$ 与 $`W_c^{1/2}W_oW_c^{1/2}`$ 的特征值都非负，所以排序并补零后两个序列相同。

**(b)** *最小能量.* 对 $`x\in\mathrm{range}\,W_c=\mathrm{range}\,\mathscr C`$，$`\mathscr Cu=x`$ 的解构成一个仿射空间，其中范数最小的元素是 $`u^\star=\mathscr C^+x`$。它的范数平方为

```math
\lVert u^\star\rVert^2=x^\top(\mathscr C^+)^\top\mathscr C^+x=x^\top(\mathscr C\mathscr C^\top)^+x=x^\top W_c^+x ,
```

这里用到 Moore–Penrose 逆的恒等式 $`(\mathscr C^+)^\top\mathscr C^+=(\mathscr C\mathscr C^\top)^+`$。

*输出能量.* 若 $`\mathscr Cu=x`$，则 $`Hu=\mathscr Ox`$，且无论 $`u`$ 是什么，$`\lVert Hu\rVert^2=x^\top W_ox`$。

*Rayleigh 商.* 把任意输入分解为 $`u=u_0+u_1`$，$`u_0\in\ker\mathscr C`$，$`u_1\perp\ker\mathscr C`$；于是 $`Hu=Hu_1`$，$`\lVert u\rVert\ge\lVert u_1\rVert`$，且对 $`x=\mathscr Cu`$ 有 $`u_1=\mathscr C^+x`$。因此

```math
\sigma_1(H)^2=\sup_{u\ne0}\frac{\lVert Hu\rVert^2}{\lVert u\rVert^2}=\sup_{x\in\mathrm{range}\,W_c,\ x\ne0}\frac{x^\top W_o\,x}{x^\top W_c^+\,x},
```

其中 $`\mathscr Cu=0`$ 的输入贡献为零，已略去。（作为一致性检验：代入 $`x=W_c^{1/2}y`$，$`y\in\mathrm{range}\,W_c`$，商变为 $`y^\top W_c^{1/2}W_oW_c^{1/2}y/\lVert y\rVert^2`$，其上确界正是 (a) 中最大的特征值。）

*零增益.* $`H=\mathscr O\mathscr C=0`$ 当且仅当 $`\mathrm{range}\,\mathscr C\subseteq\ker\mathscr O`$。由于 $`\mathrm{range}\,W_c=\mathrm{range}\,\mathscr C`$、$`\ker W_o=\ker\mathscr O`$，这等价于 $`\mathrm{range}\,W_c\subseteq\ker W_o`$，即 $`W_oW_c=0`$。$`\square`$

**关于证明的说明.** (a) 与 (b) 都没有用到 $`\lVert A_t\rVert\le1`$、时不变性或无限时域。这些只在经典稳定性理论与 balanced truncation 的先验误差界中才需要，而框架不用它们。

## 相关工作

- (a) 中的恒等式，是 balanced realization 所用的"Hankel 奇异值与 Gramian 乘积的关系" ([Moore, 1981](#ref-moore1981principal)) 在有限时域、时变情形下的形式。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-moore1981principal"></a>

**Bruce C. Moore (1981).** Principal component analysis in linear systems: Controllability, observability, and model reduction. *IEEE Transactions on Automatic Control* 26, no. 1, 17–32. DOI [10.1109/TAC.1981.1102568](https://doi.org/10.1109/TAC.1981.1102568).
