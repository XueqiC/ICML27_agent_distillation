# Proposition 9.1 · 耦合宽度

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/09-01_coupling_width_ZH.md) · [证明](09-01_coupling_width_ZH.md) · [实验](../experiments/09-01_coupling_width_ZH.md) · [English](09-01_coupling_width.md)

**本页证明什么.** 在两个槽位的边缘分布都固定时，重新安排哪些单元打包在一起，只能通过 packet response 的不可加部分改变期望学习价值：宽度为零恰好对应可加 response，一般情形下宽度至多是 response 到可加表距离的两倍。禁止配对会破坏这一刻画；在同一参数处走一步 SGD 与重组严格无关，连续两步则不然。证明是耦合多面体上的线性代数，加上二次损失下的两步展开。

**本页用到的设定与记号.**

- $`\mathcal Y_1=\lbrace1,\dots,m\rbrace`$、$`\mathcal Y_2=\lbrace1,\dots,n\rbrace`$ 是填入训练 packet 两个槽位的有效单元的有限集合；除非另有说明，每一对 $`(i,j)`$ 都是有效的 packet。
- $`\mu\in\Delta(\mathcal Y_1)`$、$`\nu\in\Delta(\mathcal Y_2)`$ 是严格为正的槽位边缘分布（否则在正支撑的乘积上讨论）。
- 耦合多面体为

```math
\mathcal C(\mu,\nu)=\lbrace \Gamma\in\mathbb R^{m\times n}_{\ge0}:\ \Gamma\mathbf 1=\mu,\ \Gamma^\top\mathbf 1=\nu\rbrace .
```

- $`H_z(i,j)`$ 是在完整状态 $`z`$ 上对 packet $`(i,j)`$ 执行实际更新、再走参考尾段之后最终任务回报的期望：以 packet 为消息的一轮边界 response（[Definition 1.4](../README_ZH.md#s1-4)、[Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)）。
- 耦合宽度为

```math
w(H_z)=\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle .
```

- $`e_{ij}`$ 是只在 $`(i,j)`$ 处为 1 的表；表 $`H`$ 在行 $`i\ne i'`$、列 $`j\ne j'`$ 上的*矩形和*为 $`H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}`$。

**陈述（[Proposition 9.1](../statements/09-01_coupling_width_ZH.md#s9-1)）.** (a) $`w(H_z)=0`$ 当且仅当存在 $`a,b`$ 使 $`H_z(i,j)=a_i+b_j`$。(b) 对所有 $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$，

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty ,
```

且不等式可以严格成立。(c) 有禁止配对时，可行宽度为零并不推出可加。(d) 在同一参数处、对所有 packet 的并集、以相同的逐样本权重做一步 SGD，与重组无关；连续两步及以上一般不然，对二次损失差别是 $`\eta^2`$ 阶。

**证明.**

*(a)，"充分性".* 对 $`H=a\mathbf 1^\top+\mathbf 1b^\top`$，

```math
\langle\Gamma,H\rangle=\sum_ia_i\sum_j\Gamma_{ij}+\sum_jb_j\sum_i\Gamma_{ij}=\mu^\top a+\nu^\top b,
```

对每个 $`\Gamma\in\mathcal C(\mu,\nu)`$ 都相同。

*(a)，"必要性".*

- $`\Gamma_0=\mu\nu^\top`$ 属于 $`\mathcal C(\mu,\nu)`$，且严格为正。
- 取行 $`i\ne i'`$、列 $`j\ne j'`$，令 $`E=e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$。$`E`$ 的行和与列和都为零，所以当 $`0<s\le\min_{k,l}(\Gamma_0)_{kl}`$ 时 $`\Gamma_0\pm sE\in\mathcal C(\mu,\nu)`$。这些矩形移动就是运输多面体的基本环路方向（[Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)）。
- 宽度为零意味着 $`\langle\Gamma,H\rangle`$ 在 $`\mathcal C(\mu,\nu)`$ 上为常数。于是 $`\langle\Gamma_0+sE,H\rangle=\langle\Gamma_0-sE,H\rangle`$，即 $`\langle E,H\rangle=0`$：

```math
H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}=0\quad\text{对所有矩形成立}.
```

- 令 $`a_i=H_{i1}`$，$`b_j=H_{1j}-H_{11}`$。矩形 $`(i,1,j,1)`$ 给出 $`H_{ij}=H_{i1}+H_{1j}-H_{11}=a_i+b_j`$；$`i=1`$ 或 $`j=1`$ 时这是平凡的。

*(b).* $`\Delta=\Gamma-\Gamma'`$ 的行和与列和都为零，所以对所有 $`a,b`$ 有 $`\langle\Delta,a\mathbf 1^\top+\mathbf 1b^\top\rangle=0`$。又 $`\lVert\Delta\rVert_1\le\lVert\Gamma\rVert_1+\lVert\Gamma'\rVert_1=2`$。因此对每个 $`a,b`$，

```math
\lvert\langle\Delta,H\rangle\rvert=\lvert\langle\Delta,H-a\mathbf 1^\top-\mathbf 1b^\top\rangle\rvert\le\lVert\Delta\rVert_1\,\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty\le2\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty .
```

对 $`(a,b)`$ 取下确界即得。

*严格性.* 令 $`\mu=\nu=(0.9,0.1)`$，$`H=e_{11}`$。

- 每个耦合都形如 $`\Gamma_{11}=x`$，$`\Gamma_{12}=\Gamma_{21}=0.9-x`$，$`\Gamma_{22}=x-0.8`$，其中 $`x\in[0.8,0.9]`$。所以 $`\langle\Gamma,H\rangle=x`$，宽度为 $`0.1`$。
- 每个可加表的矩形和为 $`0`$，而 $`H`$ 的矩形和为 $`1`$。所以残差 $`R=H-a\mathbf 1^\top-\mathbf 1b^\top`$ 的矩形和为 $`1`$，必有某个分量满足 $`\lvert R_{kl}\rvert\ge1/4`$。
- 棋盘形残差 $`R=\tfrac14\begin{pmatrix}1&-1\cr -1&1\end{pmatrix}`$ 取到这个值，对应 $`a=(\tfrac34,\tfrac14)`$，$`b=(0,-\tfrac12)`$。
- 因此 (b) 的右端是 $`2\cdot\tfrac14=0.5>0.1`$。

耦合是一个受限的可行集，不同于 [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 中不受限制的分布族（那里特征匹配下的最坏差距恰好等于最佳一致逼近误差的两倍），所以 (b) 只是上界，不是对偶等式。

*(c).* 令 $`\mu=\nu=(\tfrac12,\tfrac12)`$，禁止配对 $`(2,2)`$，并令 $`H=e_{22}`$，即表 $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$。每个可行耦合都有 $`\Gamma_{22}=0`$，所以 $`\langle\Gamma,H\rangle=0`$，可行宽度为 $`0`$。但 $`H`$ 的矩形和为 $`1`$，不可加。

*(d).* 设样本为 $`\lbrace (x_k,w_k)\rbrace _{k\in I}`$，逐样本损失为 $`\ell_k`$；一种分组就是把 $`I`$ 划分成若干 packet。

*在同一参数处走一步.* 更新为

```math
\theta^+=\theta-\eta\sum_{k\in I}w_k\nabla\ell_k(\theta).
```

它只依赖于样本与权重构成的多重集，与划分无关。所以重组不改变任何东西，这是严格的零假设：任何测到的差别都说明实现或随机性有差异。

*连续两步.* 先用 packet $`P_1`$，再用 $`P_2`$。取二次损失 $`\ell_k(\theta)=\tfrac12(x_k^\top\theta-y_k)^2`$，记 $`g_P(\theta)=\sum_{k\in P}w_k\nabla\ell_k(\theta)`$，$`H_P=\sum_{k\in P}w_kx_kx_k^\top`$。则

```math
\theta_1=\theta_0-\eta\,g_{P_1}(\theta_0),\qquad g_{P_2}(\theta_1)=g_{P_2}(\theta_0)-\eta H_{P_2}g_{P_1}(\theta_0),
```

因此

```math
\theta_2=\theta_0-\eta\big(g_{P_1}(\theta_0)+g_{P_2}(\theta_0)\big)+\eta^2H_{P_2}\,g_{P_1}(\theta_0).
```

- 一阶项只依赖于 $`P_1\cup P_2`$。
- 二阶项 $`H_{P_2}g_{P_1}(\theta_0)`$ 依赖于哪些样本落在哪个 packet 里，所以重组一般会改变它。
- 非线性网络和 Adam 会带来更多交互。$`\square`$

**如何理解这个命题.**

- **宽度不为零意味着什么.** 在边缘分布固定的前提下重组真实的单元，只能通过联合 packet response $`H_z`$ 的不可加部分改变学习结果。这是学习后果层面的效应，与单个单元的质量、梯度方差或动作层面的代价是两回事。
- **$`H_z`$ 必须是什么.** 它是声明的学习实验下、对整个 packet 的全尾段 response。token 级损失的表格不是学习后果的表格。
- **范围.** 这一结果是耦合上的线性代数，不主张新的对偶。它的作用是把"packet 的组织方式对这个学习器有没有影响"变成一个可测、并带有精确零假设的量。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-bertsimas1997introduction"></a>

**Dimitris Bertsimas, John N. Tsitsiklis (1997).** Introduction to Linear Optimization. Athena Scientific, Belmont, MA (Athena Scientific Series in Optimization and Neural Computation 6). ISBN 978-1-886529-19-9.
