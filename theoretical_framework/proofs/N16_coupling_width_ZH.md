# N16 · coupling width — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N16_coupling_width_ZH.md)

## 设定与假设

$`\mathcal Y_1=\lbrace 1,\dots,m\rbrace`$ 和 $`\mathcal Y_2=\lbrace 1,\dots,n\rbrace`$ 是有效反馈单元的有限集合，例如填入一个训练 packet 两个槽位的单元。

$`\mu\in\Delta(\mathcal Y_1)`$ 和 $`\nu\in\Delta(\mathcal Y_2)`$ 是两个槽位的边际，假设严格为正。（若有零项，下面所有结论在正支撑的乘积上成立。）

每一对 $`(i,j)`$ 都是有效的 packet。

**coupling 多面体**为

```math
\mathcal C(\mu,\nu)=\lbrace \Gamma\in\mathbb R^{m\times n}_{\ge0}:\ \Gamma\mathbf 1=\mu,\ \Gamma^\top\mathbf 1=\nu\rbrace .
```

在学习者状态 $`z`$ 处，$`H_z(i,j)`$ 是对联合 packet $`(i,j)`$ 做**实际更新**、再接参照 suffix 之后的期望真实终端回报。它是 N05 意义下的联合 packet response。

**coupling width** 为

```math
w(H_z)=\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle.
```

它是在保持两个槽位边际不变、因而每个单元的内容也不变的前提下，仅仅重新安排哪些单元打包在一起，所能造成的期望学习价值的最大变化。

## 陈述

**命题.**

**1.** **(a)** $`w(H_z)=0`$ 当且仅当存在 $`a\in\mathbb R^m`$、$`b\in\mathbb R^n`$ 使 $`H_z(i,j)=a_i+b_j`$。

**2.** **(b)** 对所有 $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$，

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty. \qquad(46)
```

这个不等式可以是严格的。

**3.** **(b′)** **受限配对.** 如果禁止某些配对，可行 coupling 就构成 $`\mathcal C`$ 的一个真面，此时 width 为零并不能推出可加性。

**4.** **(c)** **严格的可加零效应.** 在公共参数上、以相同的逐样本权重对所有 packet 的并集做一步 SGD，结果与样本如何分组成 packet 无关。两步或更多步的顺序更新一般不具有这种不变性。对二次损失，差异是 $`\eta^2`$ 阶的，下面给出显式形式。

## 证明

**第 1 部分，"若".** 对 $`H=a\mathbf 1^\top+\mathbf 1b^\top`$，

```math
\langle\Gamma,H\rangle=\sum_ia_i\sum_j\Gamma_{ij}+\sum_jb_j\sum_i\Gamma_{ij}=\mu^\top a+\nu^\top b,
```

它对每个 $`\Gamma\in\mathcal C(\mu,\nu)`$ 都相同。

**第 1 部分，"仅当".**

$`\Gamma_0=\mu\nu^\top`$ 属于 $`\mathcal C(\mu,\nu)`$，并且严格为正。

固定行 $`i\ne i'`$ 和列 $`j\ne j'`$，令 $`E=e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$。$`E`$ 的行和与列和都为零，所以当 $`0<s\le\min_{k,l}(\Gamma_0)_{kl}`$ 时 $`\Gamma_0\pm sE\in\mathcal C(\mu,\nu)`$。

width 为零意味着 $`\langle\Gamma,H\rangle`$ 在 $`\mathcal C`$ 上是常数。所以 $`\langle\Gamma_0+sE,H\rangle=\langle\Gamma_0-sE,H\rangle`$，即 $`\langle E,H\rangle=0`$：

```math
H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}=0\quad\text{for all rectangles}.
```

令 $`a_i=H_{i1}`$，$`b_j=H_{1j}-H_{11}`$。矩形 $`(i,1,j,1)`$ 给出 $`H_{ij}=H_{i1}+H_{1j}-H_{11}=a_i+b_j`$（当 $`i=1`$ 或 $`j=1`$ 时显然成立）。

**第 2 部分.** $`\Delta=\Gamma-\Gamma'`$ 的行和与列和都为零，所以对所有 $`a,b`$ 有 $`\langle\Delta,a\mathbf 1^\top+\mathbf 1b^\top\rangle=0`$。又 $`\lVert\Delta\rVert_1\le\lVert\Gamma\rVert_1+\lVert\Gamma'\rVert_1=2`$。于是对每个 $`a,b`$，

```math
\lvert\langle\Delta,H\rangle\rvert=\lvert\langle\Delta,H-a\mathbf 1^\top-\mathbf 1b^\top\rangle\rvert\le\lVert\Delta\rVert_1\,\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty\le2\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty.
```

对 $`(a,b)`$ 取下确界即可。

*严格性.* 令 $`\mu=\nu=(0.9,0.1)`$，$`H=e_{11}`$。

- 每个 coupling 都形如 $`\Gamma_{11}=x`$，$`\Gamma_{12}=\Gamma_{21}=0.9-x`$，$`\Gamma_{22}=x-0.8`$，其中 $`x\in[0.8,0.9]`$。所以 $`\langle\Gamma,H\rangle=x`$，width 为 $`0.1`$。
- 每个可加表的矩形和为 $`0`$，而 $`H`$ 的矩形和为 $`1`$。所以残差 $`R=H-a\mathbf 1^\top-\mathbf 1b^\top`$ 的矩形和为 $`1`$，必有某项满足 $`\lvert R_{kl}\rvert\ge1/4`$。
- 棋盘残差 $`R=\tfrac14\begin{pmatrix}1&-1\cr -1&1\end{pmatrix}`$ 取到这个下界，对应 $`a=(\tfrac34,\tfrac14)`$、$`b=(0,-\tfrac12)`$。
- 因此 (46) 的右边是 $`2\cdot\tfrac14=0.5>0.1`$。

coupling 是一个受限的可行集（对照 N06 和 N13），所以 (46) 是上界，而不是对偶等式。

**第 3 部分.** 令 $`\mu=\nu=(\tfrac12,\tfrac12)`$，禁止配对 $`(2,2)`$，并令 $`H=e_{22}`$，即表 $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$。每个可行 coupling 都有 $`\Gamma_{22}=0`$，所以 $`\langle\Gamma,H\rangle=0`$，可行 width 为 $`0`$。但 $`H`$ 的矩形和为 $`1`$，不是可加的。

**第 4 部分.** 设样本为 $`\lbrace (x_k,w_k)\rbrace _{k\in I}`$，逐样本损失为 $`\ell_k`$；一种分组是把 $`I`$ 划分成若干 packet。

*在公共参数上做一步.* 更新为

```math
\theta^+=\theta-\eta\sum_{k\in I}w_k\nabla\ell_k(\theta).
```

它只依赖样本与权重构成的多重集，而与划分无关。重新分组不改变任何东西，因此这是一个严格的零效应：任何测得的差异都说明实现或随机性有差别。

*顺序做两步.* 先施加 packet $`P_1`$，再施加 $`P_2`$。取二次损失 $`\ell_k(\theta)=\tfrac12(x_k^\top\theta-y_k)^2`$，记 $`g_P(\theta)=\sum_{k\in P}w_k\nabla\ell_k(\theta)`$，$`H_P=\sum_{k\in P}w_kx_kx_k^\top`$。则

```math
\theta_1=\theta_0-\eta g_{P_1}(\theta_0),\qquad g_{P_2}(\theta_1)=g_{P_2}(\theta_0)-\eta H_{P_2}g_{P_1}(\theta_0),
```

于是

```math
\theta_2=\theta_0-\eta\big(g_{P_1}(\theta_0)+g_{P_2}(\theta_0)\big)+\eta^2H_{P_2}\,g_{P_1}(\theta_0).
```

一阶项只依赖 $`P_1\cup P_2`$。二阶项 $`H_{P_2}g_{P_1}(\theta_0)`$ 依赖于哪些样本在哪个 packet 里，所以一般会随分组而变。非线性网络和 Adam 还会带来更多交互。$`\square`$

## 注记

1. **width 非零意味着什么.** 在边际固定的前提下重组真实单元，只能通过联合 packet response $`H_z`$ 中不可加的部分改变学习结果。这是超出单元质量、梯度方差或动作层面代价之外的学习后果效应。N16 要问的是：这一部分是否存在。
2. **$`H_z`$ 必须是什么.** 它是在声明的学习实验下，对整个 packet 的完整尾部 response。一张 token 级损失的表，不是一张学习后果的表。
3. **范围.** 这个结果是 coupling 上的线性代数，不声称提出新的对偶。它把"packet 的组织方式对这个学习者是否重要"这个问题，变成一个可以测量、并带有精确零效应对照的量。

## 依赖与来源

- 依赖 [N05](N05_continuation_response_ZH.md)（联合 packet response 是一次单边界 response）。

**出处：** [v0.14.1 §12](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#12-one-derived-mechanism-coupling-valid-experience-units) — Prop. 21; eq. (46). [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §12，Prop. 21，式 (46)。复核：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) §C1。
