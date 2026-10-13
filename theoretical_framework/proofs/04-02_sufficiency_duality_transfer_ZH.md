# Theorem 4.2 · 充分性对偶与迁移

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-02_sufficiency_duality_transfer_ZH.md) · [证明](04-02_sufficiency_duality_transfer_ZH.md) · [实验](../experiments/04-02_sufficiency_duality_transfer_ZH.md) · [English](04-02_sufficiency_duality_transfer.md)

**本页证明什么.** 若两个 feedback law 只通过某些特征的均值来比较，那么它们保证给出相同的学习结果，当且仅当特征张成每一个 response；否则最坏的隐藏差恰为一个最佳一致逼近误差的两倍。证明用的是线性代数加线性规划对偶：概率分布之差恰好是 $`\ell_1`$ 范数至多为 2 的零和向量，而到子空间的上确界范数距离有一个精确的对偶。

**本页用到的设定.**

- $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。$`\mathcal Y`$ 上的函数看作 $`\mathbb R^m`$ 中的向量；$`\mathbf 1`$ 是全一向量，$`e_y`$ 是 $`y`$ 的示性向量。分布是 $`\Delta(\mathcal Y)=\lbrace P\ge0:\mathbf 1^\top P=1\rbrace`$ 中的向量，$`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$。
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ 是 response 类：每个 $`g\in\mathcal G`$ 把一轮中收到的消息映为期望最终任务回报（[Definition 1.4](../README_ZH.md#s1-4)）。
- 特征接口传递 $`\phi_1,\dots,\phi_d`$ 的均值。$`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$，$`H_\phi^\perp`$ 是它的正交补，$`\mathcal M_\phi`$ 是特征匹配的分布对集合，$`\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\sup_{g\in\mathcal G}\lvert(P-Q)g\rvert`$ 是未分辨差距，$`d_0(\mathcal G)`$ 是使差距为零的最少特征个数（[Definition 4.1](../statements/04-01_feature_sufficiency_unresolved_discrepancy_ZH.md#s4-1)）。
- $`\mathrm{dist}_\infty(g,H)=\min_{h\in H}\lVert g-h\rVert_\infty`$；$`\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$。

**陈述（[Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)）.**

- **(a) 精确充分性.** "对所有 $`(P,Q)\in\mathcal M_\phi`$ 与所有 $`g\in\mathcal G`$ 都有 $`(P-Q)g=0`$"当且仅当 $`\mathrm{span}(\mathcal G)\subseteq H_\phi`$。因此

```math
d_0(\mathcal G)=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1. \qquad(4.1)
```

- **(b) 近似充分性.**

```math
\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\ \mathrm{dist}_\infty(g,H_\phi). \qquad(4.2)
```

- 因此用 $`d`$ 个特征能达到的最小未分辨差距是 $`2\inf_H\sup_{g\in\mathcal G}\mathrm{dist}_\infty(g,H)`$，下确界取遍满足 $`H\ni\mathbf 1`$、$`\dim H\le d+1`$ 的子空间。
- **(c) 迁移.** 若目标类中的每个 $`g`$ 都能写成 $`g=c+\sum_{\ell=1}^db_\ell\phi_\ell+r`$，且 $`\lVert b\rVert_1\le B`$、$`\lVert r\rVert_\infty\le\tau`$，则对所有 $`P,Q\in\Delta(\mathcal Y)`$，

```math
\sup_g\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau .
```

- **(d) 不受限的 learner.** 固定一个任务，它有两个 policy，价值分别为 $`a`$ 与 $`a+\gamma`$，$`\gamma>0`$。假设对每个事件 $`A\subseteq\mathcal Y`$，learner 族都含有一步规则"当且仅当 $`Y\in A`$ 时部署较好的 policy"。则

```math
\sup_{A\subseteq\mathcal Y}\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert=\gamma\,\mathrm{TV}(P,Q),\qquad d_0(\mathcal G)=m-1 .
```

**证明.** 先证两个断言。

*Claim 1（分布之差）.* $`\lbrace P-Q:P,Q\in\Delta(\mathcal Y)\rbrace=\lbrace v\in\mathbb R^m:\mathbf 1^\top v=0,\ \lVert v\rVert_1\le2\rbrace`$。

- （$`\subseteq`$）$`\mathbf 1^\top(P-Q)=1-1=0`$，且 $`\lVert P-Q\rVert_1\le\lVert P\rVert_1+\lVert Q\rVert_1=2`$。
- （$`\supseteq`$）设 $`\mathbf 1^\top v=0`$、$`\lVert v\rVert_1\le2`$。写 $`v=v_+-v_-`$，$`v_\pm\ge0`$ 且支撑不相交，则 $`s:=\mathbf 1^\top v_+=\mathbf 1^\top v_-=\tfrac12\lVert v\rVert_1\le1`$。任取 $`\rho\in\Delta(\mathcal Y)`$，令 $`P=v_++(1-s)\rho`$、$`Q=v_-+(1-s)\rho`$。两者非负、总质量为 1，且 $`P-Q=v`$。

*Claim 2（上确界范数下到子空间的距离）.* 对任意 $`g\in\mathbb R^m`$ 与子空间 $`H\subseteq\mathbb R^m`$，

```math
\mathrm{dist}_\infty(g,H)=\max\lbrace\sigma^\top g:\ \sigma\in H^\perp,\ \lVert\sigma\rVert_1\le1\rbrace ,
```

并且两个极值都能取到。令 $`A\in\mathbb R^{m\times k}`$ 的列张成 $`H`$。

- *弱对偶.* 对 $`\sigma\in H^\perp`$、$`\lVert\sigma\rVert_1\le1`$ 及任意 $`h\in H`$，$`\sigma^\top g=\sigma^\top(g-h)\le\lVert\sigma\rVert_1\lVert g-h\rVert_\infty\le\lVert g-h\rVert_\infty`$。
- *强对偶.* 距离是线性规划

```math
\min_{t\in\mathbb R,\,c\in\mathbb R^k}\ t\quad\text{s.t.}\quad g-Ac\le t\mathbf 1,\ \ -(g-Ac)\le t\mathbf 1
```

- 的值。它可行（$`c=0`$、$`t=\lVert g\rVert_\infty`$）且下有界（$`\ge0`$）。由 LP 强对偶（[Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)），它的值等于对偶问题

```math
\max_{y^+,y^-\ge0}\ (y^+-y^-)^\top g\quad\text{s.t.}\quad A^\top(y^+-y^-)=0,\ \ \mathbf 1^\top(y^++y^-)=1
```

- 的值，且两个最优值都能取到。令 $`\sigma=y^+-y^-`$，则 $`\sigma\in H^\perp`$，$`\lVert\sigma\rVert_1\le\mathbf 1^\top(y^++y^-)=1`$。反过来，对 $`\sigma\in H^\perp`$、$`\lVert\sigma\rVert_1\le1`$，取 $`y^\pm=\sigma_\pm+\tfrac{1-\lVert\sigma\rVert_1}{2}e_1`$，它是对偶可行解，目标值为 $`\sigma^\top g`$。所以对偶值恰为所述的最大值。

*(a).*（$`\Leftarrow`$）设 $`\mathrm{span}\mathcal G\subseteq H_\phi`$、$`g\in\mathcal G`$，于是 $`g=c\mathbf 1+\sum_jb_j\phi_j`$。对 $`(P,Q)\in\mathcal M_\phi`$，$`(P-Q)g=c\,\mathbf 1^\top(P-Q)+\sum_jb_j(P-Q)\phi_j=0`$。

（$`\Rightarrow`$）设某个 $`g\in\mathcal G`$ 满足 $`g\notin H_\phi`$。在正交分解 $`\mathbb R^m=H_\phi\oplus H_\phi^\perp`$ 下，令 $`v`$ 为 $`g`$ 在 $`H_\phi^\perp`$ 中的分量。

- 由 $`g\notin H_\phi`$ 得 $`v\ne0`$，且 $`v^\top g=\lVert v\rVert_2^2>0`$。
- 令 $`u=\mathbf 1/m`$、$`a=1/(2m\lVert v\rVert_\infty)`$、$`P=u`$、$`Q=u+av`$。则 $`Q\ge u-\tfrac1{2m}\mathbf 1\ge0`$；又由 $`\mathbf 1\in H_\phi`$ 得 $`\mathbf 1^\top v=0`$，所以 $`\mathbf 1^\top Q=1`$。
- $`(P-Q)\phi_j=-a\,v^\top\phi_j=0`$，所以 $`(P,Q)\in\mathcal M_\phi`$，但 $`(P-Q)g=-a\lVert v\rVert_2^2\ne0`$。

关于计数 (4.1)，令 $`s=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)`$。

- *上界.* 把 $`\lbrace\mathbf 1\rbrace`$ 扩充为 $`\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)`$ 的基 $`\lbrace\mathbf 1,\phi_1,\dots,\phi_{s-1}\rbrace`$，则 $`H_\phi\supseteq\mathrm{span}\mathcal G`$，由（$`\Leftarrow`$），$`s-1`$ 个特征就够了。
- *下界.* 若 $`d`$ 个特征已充分，由（$`\Rightarrow`$）得 $`\mathrm{span}\mathcal G\subseteq H_\phi`$。又 $`\mathbf 1\in H_\phi`$，所以 $`\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\subseteq H_\phi`$，从而 $`s\le\dim H_\phi\le d+1`$。

*(b).* 固定 $`g\in\mathcal G`$，记 $`e(g)=\mathrm{dist}_\infty(g,H_\phi)`$。

- *上界.* 设 $`(P,Q)\in\mathcal M_\phi`$。每个 $`h\in H_\phi`$ 都是 $`\mathbf 1`$ 与各 $`\phi_j`$ 的线性组合，故 $`(P-Q)h=0`$。于是 $`\lvert(P-Q)g\rvert=\lvert(P-Q)(g-h)\rvert\le\lVert P-Q\rVert_1\lVert g-h\rVert_\infty\le2\lVert g-h\rVert_\infty`$，对 $`h`$ 取最小得 $`\lvert(P-Q)g\rvert\le2e(g)`$。
- *可达.* 由 Claim 2，存在 $`\sigma\in H_\phi^\perp`$，$`\lVert\sigma\rVert_1\le1`$，$`\sigma^\top g=e(g)`$。向量 $`v=2\sigma`$ 满足 $`\mathbf 1^\top v=0`$、$`\lVert v\rVert_1\le2`$，由 Claim 1 存在分布 $`P,Q`$ 使 $`v=P-Q`$。由于 $`v\perp\phi_j`$，$`(P,Q)\in\mathcal M_\phi`$，并且 $`(P-Q)g=2e(g)`$。

所以对每个 $`g`$，$`\sup_{(P,Q)\in\mathcal M_\phi}\lvert(P-Q)g\rvert=2e(g)`$；交换 $`\mathfrak D_\phi`$ 定义中两个上确界的次序即得 (4.2)。关于 $`d`$ 个特征的结论，只需在所有包含 $`\mathbf 1`$、维数至多 $`d+1`$ 的子空间 $`H_\phi`$ 上对 (4.2) 取最小。

*(c).* 令 $`\Delta\phi=\mathbb E_P\phi-\mathbb E_Q\phi\in\mathbb R^d`$。对 $`g=c+b^\top\phi+r`$，

```math
(P-Q)g=\underbrace{c\,\mathbf 1^\top(P-Q)}_{=0}+\sum_\ell b_\ell\,\Delta\phi_\ell+(\mathbb E_Pr-\mathbb E_Qr).
```

由 Hölder 不等式，$`\lvert\sum_\ell b_\ell\Delta\phi_\ell\rvert\le\lVert b\rVert_1\lVert\Delta\phi\rVert_\infty\le B\lVert\Delta\phi\rVert_\infty`$；又 $`\lvert\mathbb E_Pr-\mathbb E_Qr\rvert\le2\tau`$。对 $`g`$ 取上确界即可。

*(d).* 恰在 $`A`$ 上部署较好 policy 的 learner，最终价值为 $`a+\gamma\mathbf 1_A(Y)`$，所以 $`g_A=a+\gamma\mathbf 1_A\in\mathcal G`$，且 $`(P-Q)g_A=\gamma(P(A)-Q(A))`$。

- $`\sup_A\lvert P(A)-Q(A)\rvert=\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$，在 $`A=\lbrace y:P(y)>Q(y)\rbrace`$ 处取到（全变差的事件形式）。
- 单点事件给出 $`g_{\lbrace y\rbrace}=a\mathbf 1+\gamma e_y`$。由于 $`\gamma>0`$，$`\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\supseteq\mathrm{span}\lbrace e_y:y\in\mathcal Y\rbrace=\mathbb R^m`$，由 (4.1) 得 $`d_0=m-1`$。$`\square`$

## 附注

- **(b) 中的量词.** (4.2) 的上确界取遍所有共享某组特征均值的分布对。如果参考分布事先固定，或要求两个分布落在同一个预先指定的特征均值纤维上，可行的差分集合会变小，对偶要在更小的集合上取，(4.2) 只是上界。
- **受限分布族.** 若只有分布族 $`\mathcal P\subsetneq\Delta(\mathcal Y)`$ 可实现，就把 Claim 1 中的集合换成 $`\mathcal P`$ 内的差分。此时 (a) 应读作：$`(P-Q)g`$ 在特征匹配对上恒为零，当且仅当每个特征匹配的差分都零化 $`g`$。
- **系数稳定性.** (c) 显式写出了系数大小 $`B`$。残差 $`\tau`$ 小而 $`B`$ 大时，特征估计误差会被放大。[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)（带测量容差时逼近残差与系数大小之间的精确权衡）把这一点精确化。
- **(d) 对丰富性的依赖.** $`d_0=m-1`$ 要求 learner 能实现每个单点示性函数。实际的 learner 与任务能实现的 response 少得多。(d) 只说明：不限制任务与 learner 类，就不存在一致小的接口。

## 相关工作

- LP 对偶与全变差的事件形式是经典工具（[Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)），在此注明，不作为新结果。
- 本页新增的是用它们计算：对给定的 response 类，一个 feedback law 必须携带多少信息——一个 span 维数或一个逼近宽度。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-bertsimas1997introduction"></a>

**Dimitris Bertsimas, John N. Tsitsiklis (1997).** Introduction to Linear Optimization. Athena Scientific, Belmont, MA (Athena Scientific Series in Optimization and Neural Computation 6). ISBN 978-1-886529-19-9.
