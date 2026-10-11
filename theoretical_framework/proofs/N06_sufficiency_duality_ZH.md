# N06 · 充分性对偶与迁移 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N06_sufficiency_duality_ZH.md) · [English](N06_sufficiency_duality.md)

## 设定与假设

- $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。$`\mathcal Y`$ 上的函数看作 $`\mathbb R^m`$ 中的向量。概率分布（law）是 $`\Delta(\mathcal Y)=\lbrace P\in\mathbb R^m_{\ge0}:\mathbf 1^\top P=1\rbrace`$ 中的向量，记 $`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$。
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ 是 response 类。在框架中，$`\mathcal G`$ 收集 N05 的单边界 response $`g_{E,f,t,z}`$；这里只用到它们在 $`\mathcal Y`$ 上的取值。
- 接口传递固定特征 $`\phi_1,\dots,\phi_d:\mathcal Y\to\mathbb R`$ 的均值，记 $`H_\phi=\mathrm{span}\lbrace \mathbf 1,\phi_1,\dots,\phi_d\rbrace \subseteq\mathbb R^m`$。
- $`H_\phi^\perp=\lbrace v\in\mathbb R^m:v^\top h=0\ \forall h\in H_\phi\rbrace`$ 是零化子。任意 $`v\in H_\phi^\perp`$ 都满足 $`\mathbf 1^\top v=0`$。
- 允许 $`\mathcal Y`$ 上的所有概率分布。受限分布族的情形见"附注"。

**定义 1（特征匹配的分布对；未分辨差异）。** 令 $`\mathcal M_\phi=\lbrace (P,Q)\in\Delta(\mathcal Y)^2:\ (P-Q)\phi_j=0,\ j=1,\dots,d\rbrace`$，定义

```math
\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\ \sup_{g\in\mathcal G}\ \lvert(P-Q)g\rvert .
```

**定义 2（精确维数）。** $`d_0(\mathcal G)`$ 是满足下述条件的最小 $`d`$：存在特征 $`\phi_1,\dots,\phi_d`$，使得对所有 $`g\in\mathcal G`$ 都有 $`(P,Q)\in\mathcal M_\phi\Rightarrow (P-Q)g=0`$。

## 陈述

**定理。**

**(a) 精确充分性。** "对所有 $`(P,Q)\in\mathcal M_\phi`$ 和所有 $`g\in\mathcal G`$ 都有 $`(P-Q)g=0`$"当且仅当 $`\mathrm{span}(\mathcal G)\subseteq H_\phi`$。因此

```math
d_0(\mathcal G)=\dim\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)-1. \qquad(19)
```

**(b) 近似充分性。**

```math
\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\ \inf_{h\in H_\phi}\lVert g-h\rVert_\infty. \qquad(20)
```

因此用 $`d`$ 个特征所能达到的最小未分辨差异等于 $`2\inf_{H}\sup_{g}\mathrm{dist}_\infty(g,H)`$，其中下确界取遍满足 $`H\ni\mathbf 1`$、$`\dim H\le d+1`$ 的子空间。

**(c) 迁移。** 若目标类中的每个 $`g`$ 都可写成 $`g=c+\sum_{\ell=1}^db_\ell\phi_\ell+r`$，且 $`\lVert b\rVert_1\le B`$、$`\lVert r\rVert_\infty\le\tau`$，则对所有 $`P,Q\in\Delta(\mathcal Y)`$，

```math
\sup_{g}\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau.
```

**(d)** **推论。** 固定一个任务，它有两个 policy，价值分别为 $`a`$ 和 $`a+\gamma`$，其中 $`\gamma>0`$。假设对每个事件 $`A\subseteq\mathcal Y`$，learner 族中都有一步规则"当且仅当 $`Y\in A`$ 时部署较好的 policy"。则

```math
\sup_{A\subseteq\mathcal Y}\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert=\gamma\,\mathrm{TV}(P,Q),\qquad d_0(\mathcal G)=m-1.
```

## 证明

### 引理 1（分布之差）

```math
\lbrace P-Q:\ P,Q\in\Delta(\mathcal Y)\rbrace =\lbrace v\in\mathbb R^m:\ \mathbf 1^\top v=0,\ \lVert v\rVert_1\le2\rbrace .
```

*证明。*（$`\subseteq`$）$`\mathbf 1^\top(P-Q)=1-1=0`$，且 $`\lVert P-Q\rVert_1\le\lVert P\rVert_1+\lVert Q\rVert_1=2`$。

（$`\supseteq`$）设 $`\mathbf 1^\top v=0`$、$`\lVert v\rVert_1\le2`$。写 $`v=v_+-v_-`$，其中 $`v_\pm\ge0`$ 且支撑不相交。于是 $`s:=\mathbf 1^\top v_+=\mathbf 1^\top v_-=\tfrac12\lVert v\rVert_1\le1`$。任取 $`\rho\in\Delta(\mathcal Y)`$，令 $`P=v_++(1-s)\rho`$、$`Q=v_-+(1-s)\rho`$。两者都非负且总质量为 1，并且 $`P-Q=v`$。$`\square`$

### 引理 2（上确界范数下到子空间的距离）

范数为 $`\ell_\infty`$。
对任意 $`g\in\mathbb R^m`$ 和任意子空间 $`H\subseteq\mathbb R^m`$，

```math
\mathrm{dist}_\infty(g,H):=\min_{h\in H}\lVert g-h\rVert_\infty=\max\lbrace \sigma^\top g:\ \sigma\in H^\perp,\ \lVert\sigma\rVert_1\le1\rbrace ,
```

并且两个极值都能取到。

*证明。* 令 $`A\in\mathbb R^{m\times k}`$ 的列张成 $`H`$。

*弱对偶。* 对 $`\sigma\in H^\perp`$、$`\lVert\sigma\rVert_1\le1`$ 以及任意 $`h\in H`$，有 $`\sigma^\top g=\sigma^\top(g-h)\le\lVert\sigma\rVert_1\lVert g-h\rVert_\infty\le\lVert g-h\rVert_\infty`$。

*强对偶。* 原问题是线性规划

```math
\min_{t\in\mathbb R,\,c\in\mathbb R^k}\ t\quad\text{s.t.}\quad g-Ac\le t\mathbf 1,\ \ -(g-Ac)\le t\mathbf 1.
```

它可行（取 $`c=0`$、$`t=\lVert g\rVert_\infty`$），且下有界（$`\ge0`$）。由 LP 强对偶，其最优值等于对偶问题

```math
\max_{y^+,y^-\ge0}\ (y^+-y^-)^\top g\quad\text{s.t.}\quad A^\top(y^+-y^-)=0,\ \ \mathbf 1^\top(y^++y^-)=1
```

的最优值，且两者都能取到。令 $`\sigma=y^+-y^-`$，则 $`\sigma\in H^\perp`$，且 $`\lVert\sigma\rVert_1\le\mathbf 1^\top(y^++y^-)=1`$。反过来，设 $`\sigma\in H^\perp`$、$`\lVert\sigma\rVert_1\le1`$，取 $`y^\pm=\sigma_\pm+\tfrac{1-\lVert\sigma\rVert_1}{2}e_1`$，它是对偶可行解，目标值同为 $`\sigma^\top g`$。所以对偶最优值恰为 $`\max\lbrace \sigma^\top g:\sigma\in H^\perp,\lVert\sigma\rVert_1\le1\rbrace`$。$`\square`$

### (a) 的证明

（$`\Leftarrow`$）设 $`\mathrm{span}\mathcal G\subseteq H_\phi`$，$`g\in\mathcal G`$。写 $`g=c\mathbf 1+\sum_jb_j\phi_j`$。对 $`(P,Q)\in\mathcal M_\phi`$：

```math
(P-Q)g=c\,\mathbf 1^\top(P-Q)+\sum_jb_j(P-Q)\phi_j=0.
```

（$`\Rightarrow`$）设某个 $`g\in\mathcal G`$ 满足 $`g\notin H_\phi`$。作正交分解 $`\mathbb R^m=H_\phi\oplus H_\phi^\perp`$，令 $`v`$ 为 $`g`$ 在 $`H_\phi^\perp`$ 中的分量。由 $`g\notin H_\phi`$ 得 $`v\ne0`$ 且 $`v^\top g=\lVert v\rVert_2^2>0`$。令 $`u=\mathbf 1/m`$、$`a=\tfrac1{2m\lVert v\rVert_\infty}`$，取 $`P=u`$、$`Q=u+av`$。则 $`Q\ge u-\tfrac1{2m}\mathbf 1\ge0`$；又因 $`\mathbf 1\in H_\phi`$，有 $`\mathbf 1^\top v=0`$，所以 $`\mathbf 1^\top Q=1`$。另有 $`(P-Q)\phi_j=-a\,v^\top\phi_j=0`$，即 $`(P,Q)\in\mathcal M_\phi`$。但 $`(P-Q)g=-a\lVert v\rVert_2^2\ne0`$。

（计数 (19)）令 $`s=\dim\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)`$。

- *上界。* 把 $`\lbrace \mathbf 1\rbrace`$ 扩充为 $`\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)`$ 的一组基 $`\lbrace \mathbf 1,\phi_1,\dots,\phi_{s-1}\rbrace`$。此时 $`H_\phi=\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)\supseteq\mathrm{span}\mathcal G`$，由（$`\Leftarrow`$），$`s-1`$ 个特征就够了。
- *下界。* 若 $`d`$ 个特征已充分，由（$`\Rightarrow`$）得 $`\mathrm{span}\mathcal G\subseteq H_\phi`$。又 $`\mathbf 1\in H_\phi`$，所以 $`\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)\subseteq H_\phi`$，从而 $`s\le\dim H_\phi\le d+1`$。

因此 $`d_0=s-1`$。$`\square`$

### (b) 的证明

固定 $`g\in\mathcal G`$，记 $`e(g)=\mathrm{dist}_\infty(g,H_\phi)`$。

*上界。* 设 $`(P,Q)\in\mathcal M_\phi`$。每个 $`h\in H_\phi`$ 都是 $`\mathbf 1`$ 与各 $`\phi_j`$ 的线性组合，故 $`(P-Q)h=0`$。于是

```math
\lvert(P-Q)g\rvert=\lvert(P-Q)(g-h)\rvert\le\lVert P-Q\rVert_1\,\lVert g-h\rVert_\infty\le2\lVert g-h\rVert_\infty .
```

对 $`h`$ 取最小，得 $`\lvert(P-Q)g\rvert\le2e(g)`$。

*下界（可达）。* 由引理 2，存在 $`\sigma\in H_\phi^\perp`$，$`\lVert\sigma\rVert_1\le1`$，$`\sigma^\top g=e(g)`$。令 $`v=2\sigma`$，则 $`\mathbf 1^\top v=0`$（因 $`\mathbf 1\in H_\phi`$）且 $`\lVert v\rVert_1\le2`$。由引理 1，存在 $`P,Q\in\Delta(\mathcal Y)`$ 使 $`v=P-Q`$。因为 $`v\perp\phi_j`$，这一对分布属于 $`\mathcal M_\phi`$，并且 $`(P-Q)g=2e(g)`$。

所以对每个 $`g`$ 都有 $`\sup_{(P,Q)\in\mathcal M_\phi}\lvert(P-Q)g\rvert=2e(g)`$。交换定义 1 中两个上确界的次序即得 (20)。关于 $`d`$ 个特征的结论，只需在允许的特征集上（即所有包含 $`\mathbf 1`$、维数至多 $`d+1`$ 的子空间 $`H_\phi`$）对 (20) 取最小。$`\square`$

### (c) 的证明

令 $`\Delta\phi=\mathbb E_P\phi-\mathbb E_Q\phi\in\mathbb R^d`$。对 $`g=c+b^\top\phi+r`$：

```math
(P-Q)g=\underbrace{c\,\mathbf 1^\top(P-Q)}_{=0}+\sum_\ell b_\ell\,\Delta\phi_\ell+(\mathbb E_Pr-\mathbb E_Qr).
```

由 Hölder 不等式，$`\lvert\sum_\ell b_\ell\Delta\phi_\ell\rvert\le\lVert b\rVert_1\lVert\Delta\phi\rVert_\infty\le B\lVert\Delta\phi\rVert_\infty`$；又 $`\lvert\mathbb E_Pr-\mathbb E_Qr\rvert\le\lvert\mathbb E_Pr\rvert+\lvert\mathbb E_Qr\rvert\le2\tau`$。对 $`g`$ 取上确界即可。$`\square`$

### 推论的证明

恰在 $`A`$ 上部署较好 policy 的 learner，最终价值为 $`a+\gamma\mathbf 1_A(Y)`$，所以 $`g_A=a+\gamma\mathbf 1_A\in\mathcal G`$。于是

```math
(P-Q)g_A=\gamma\,(P(A)-Q(A)),
```

而 $`\sup_A\lvert P(A)-Q(A)\rvert=\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$。这是全变差的事件刻画，在 $`A=\lbrace y:P(y)>Q(y)\rbrace`$ 处取到。

维数方面，单点事件给出 $`g_{\lbrace y\rbrace }=a\mathbf 1+\gamma e_y`$。由于 $`\gamma>0`$，

```math
\mathrm{span}(\lbrace \mathbf 1\rbrace \cup\mathcal G)\supseteq\mathrm{span}\lbrace e_y:y\in\mathcal Y\rbrace =\mathbb R^m.
```

由 (19)，$`d_0=m-1`$。$`\square`$

## 附注

1. **(b) 中的量词。** (20) 的上确界取遍*所有*共享某组特征均值的分布对。如果事先固定参考分布 $`P`$，或者要求两个分布落在同一个预先指定的特征均值纤维上，可行的差分集合会变小。此时对偶要在更小的集合上取，(20) 只是上界。
2. **受限分布族。** 若只有分布族 $`\mathcal P\subsetneq\Delta(\mathcal Y)`$ 可实现，就把引理 1 中的 $`\lbrace P-Q\rbrace`$ 换成 $`\mathcal P`$ 内的差分。此时 (a) 应读作：$`(P-Q)g`$ 在特征匹配对上恒为零，当且仅当每个特征匹配的差分都零化 $`g`$。
3. **计的是什么。** $`d_0`$ 和 $`d_\varepsilon`$ 计的是固定的线性期望统计量个数，不是比特复杂度、样本量或查询次数。估计这些统计量是另一个问题，见 N13–N14。
4. **系数稳定性。** (c) 显式写出了系数大小 $`B`$。若残差 $`\tau`$ 小而 $`B`$ 大，特征估计误差会被放大。这正是稳定性–近似权衡（N13）存在的原因。
5. **推论对 learner 族丰富性的依赖。** $`d_0=m-1`$ 要求 learner 能实现所有单点示性函数。实际的 learner 和任务能实现的 response 要少得多。推论只说明：如果不限制任务和 learner 类，就不存在一致小的接口。

## 依赖与来源

- **用到：** N05 的 response 类 $`\mathcal G`$（continuation-response 刻画）。
- **被用于：** [N07](N07_affine_moment_order_ZH.md)（其中 $`\mathcal G`$ 是多项式空间）、[N09](N09_approximate_complexity_ZH.md)（$`d_\varepsilon`$ 与次数之别）、[N13](N13_stability_tradeoff_ZH.md)（带噪特征）。

**出处：** [v0.14.1 §6](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#6-exact-and-approximate-information-complexity) — Thm. 8; Cor. 9; eqs. (18)–(22). v0.14.1 主框架 Thm. 8 与 Cor. 9，式 (18)–(22)，[Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §6。独立审阅：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) A24、A25、B1。LP 对偶与全变差的事件形式是经典工具，在此注明出处，不作为新结果。
