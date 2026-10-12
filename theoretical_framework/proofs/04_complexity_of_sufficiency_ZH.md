# §4 充分性的复杂度 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/04_complexity_of_sufficiency_ZH.md) · [English](04_complexity_of_sufficiency.md)

本页保留 Definition 4.1 至 Theorem 4.4；速率与复合见[后续证明](04_rates_and_composition_ZH.md)。

**概要.**

- §3 确定了充分性的对象：两个 feedback law 可以互换，当且仅当它们在 response 类 $`\mathcal G`$ 中的每个 response 上一致（[Definition 1.4](../README_ZH.md#s1-4)、[Theorem 3.2](03_learning_consequence_ZH.md#s3-2)）。本节度量这种一致需要多少信息。
- [Theorem 4.2](#s4-2) 是一般的回答。特征接口的精确充分性是一个 span 条件，所需个数为 $`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1`$。近似充分性等于最佳一致逼近误差的两倍；系数有界的分解把特征均值的误差传递为 response 的误差。
- [Theorem 4.4](#s4-4) 与 [4.5](04_rates_and_composition_ZH.md#s4-5) 对二元 context 学生上的凸仿射 learner 把一切算清楚：精确所需的信息是阶数不超过部署决策数 $`N`$ 的联合矩，每一阶都必要；差距的界与训练轮数无关；差距随独立块数衰减，速率由第一个不匹配的 cumulant 决定。replay 与确定性派生不会让差距衰减。
- [Proposition 4.6](04_rates_and_composition_ZH.md#s4-6) 把精确阶数与近似复杂度分开：约 $`\sqrt{N\log(1/\varepsilon)}`$ 阶矩就能以精度 $`\varepsilon`$ 分辨 $`p^N`$，而 parity 用任何低于 $`N`$ 的次数都逼近不到 $`1/2`$ 以内。
- [Theorem 4.7](04_rates_and_composition_ZH.md#s4-7) 说明阶数由 learner 决定。复合的多项式更新使次数相乘；阈值 learner 只用一次决策就破坏了一阶矩充分性；光滑 SGD 的一步以梯度均值为一阶统计量、以梯度二阶矩为二阶统计量。[Proposition 5.1](05_optimizer_state_ZH.md#s5-1) 把最后一点推广到任意带状态的 optimizer。

## 特征充分性

本节中 $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。

- $`\mathcal Y`$ 上的函数看作 $`\mathbb R^m`$ 中的向量。概率分布是 $`\Delta(\mathcal Y)=\lbrace P\in\mathbb R^m_{\ge0}:\mathbf 1^\top P=1\rbrace`$ 中的向量，记 $`(P-Q)g=\sum_y(P(y)-Q(y))g(y)`$。
- $`\mathcal G\subseteq[0,1]^{\mathcal Y}`$ 是一个 response 类。在框架中，它收集 [Definition 1.4](../README_ZH.md#s1-4) 的单边界 response；这里只用到它们在 $`\mathcal Y`$ 上的取值。
- 除非另有说明，$`\mathcal Y`$ 上的所有概率分布都允许。

<a id="s4-1"></a>

**Definition 4.1（特征充分性与未分辨差距）.**

- *特征接口*传递固定特征 $`\phi_1,\dots,\phi_d:\mathcal Y\to\mathbb R`$ 的均值。记 $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace\subseteq\mathbb R^m`$，$`H_\phi^\perp=\lbrace v\in\mathbb R^m:v^\top h=0\ \forall h\in H_\phi\rbrace`$。任意 $`v\in H_\phi^\perp`$ 都满足 $`\mathbf 1^\top v=0`$。
- *特征匹配的分布对*为 $`\mathcal M_\phi=\lbrace(P,Q)\in\Delta(\mathcal Y)^2:(P-Q)\phi_j=0,\ j=1,\dots,d\rbrace`$。
- 接口在 $`\mathcal G`$ 上的*未分辨差距*为

```math
\mathfrak D_\phi(\mathcal G)=\sup_{(P,Q)\in\mathcal M_\phi}\ \sup_{g\in\mathcal G}\ \lvert(P-Q)g\rvert .
```

- *精确维数* $`d_0(\mathcal G)`$ 是满足下述条件的最小 $`d`$：存在特征 $`\phi_1,\dots,\phi_d`$，使得对所有 $`(P,Q)\in\mathcal M_\phi`$ 与所有 $`g\in\mathcal G`$ 都有 $`(P-Q)g=0`$。*近似维数* $`d_\varepsilon(\mathcal G)`$ 是使某 $`d`$ 个特征满足 $`\mathfrak D_\phi(\mathcal G)\le\varepsilon`$ 的最小 $`d`$。
- 对 $`g\in\mathbb R^m`$ 与子空间 $`H`$，记 $`\mathrm{dist}_\infty(g,H)=\min_{h\in H}\lVert g-h\rVert_\infty`$。

这些维数计的是固定的线性期望统计量个数，不是比特复杂度、样本量或查询次数；从有限数据估计这些统计量是 §8 的内容。

<a id="s4-2"></a>

**Theorem 4.2（充分性对偶与迁移）.**

- **(a) 精确充分性.** "对所有 $`(P,Q)\in\mathcal M_\phi`$ 与所有 $`g\in\mathcal G`$ 都有 $`(P-Q)g=0`$"当且仅当 $`\mathrm{span}(\mathcal G)\subseteq H_\phi`$。因此

```math
d_0(\mathcal G)=\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1. \qquad(4.1)
```

- **(b) 近似充分性.**

```math
\mathfrak D_\phi(\mathcal G)=2\sup_{g\in\mathcal G}\ \mathrm{dist}_\infty(g,H_\phi). \qquad(4.2)
```

  因此用 $`d`$ 个特征能达到的最小未分辨差距是 $`2\inf_H\sup_{g\in\mathcal G}\mathrm{dist}_\infty(g,H)`$，下确界取遍满足 $`H\ni\mathbf 1`$、$`\dim H\le d+1`$ 的子空间。
- **(c) 迁移.** 若目标类中的每个 $`g`$ 都能写成 $`g=c+\sum_{\ell=1}^db_\ell\phi_\ell+r`$，且 $`\lVert b\rVert_1\le B`$、$`\lVert r\rVert_\infty\le\tau`$，则对所有 $`P,Q\in\Delta(\mathcal Y)`$，

```math
\sup_g\lvert(P-Q)g\rvert\le B\,\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau .
```

- **(d) 不受限的 learner.** 固定一个任务，它有两个 policy，价值分别为 $`a`$ 与 $`a+\gamma`$，$`\gamma>0`$。假设对每个事件 $`A\subseteq\mathcal Y`$，learner 族都含有一步规则"当且仅当 $`Y\in A`$ 时部署较好的 policy"。则

```math
\sup_{A\subseteq\mathcal Y}\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert=\gamma\,\mathrm{TV}(P,Q),\qquad d_0(\mathcal G)=m-1 .
```

*Proof.* 先证两个断言。

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

- 的值。它可行（$`c=0`$、$`t=\lVert g\rVert_\infty`$）且下有界（$`\ge0`$）。由 LP 强对偶，它的值等于对偶问题

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

所以对每个 $`g`$，$`\sup_{(P,Q)\in\mathcal M_\phi}\lvert(P-Q)g\rvert=2e(g)`$；交换 Definition 4.1 中两个上确界的次序即得 (4.2)。关于 $`d`$ 个特征的结论，只需在所有包含 $`\mathbf 1`$、维数至多 $`d+1`$ 的子空间 $`H_\phi`$ 上对 (4.2) 取最小。

*(c).* 令 $`\Delta\phi=\mathbb E_P\phi-\mathbb E_Q\phi\in\mathbb R^d`$。对 $`g=c+b^\top\phi+r`$，

```math
(P-Q)g=\underbrace{c\,\mathbf 1^\top(P-Q)}_{=0}+\sum_\ell b_\ell\,\Delta\phi_\ell+(\mathbb E_Pr-\mathbb E_Qr).
```

由 Hölder 不等式，$`\lvert\sum_\ell b_\ell\Delta\phi_\ell\rvert\le\lVert b\rVert_1\lVert\Delta\phi\rVert_\infty\le B\lVert\Delta\phi\rVert_\infty`$；又 $`\lvert\mathbb E_Pr-\mathbb E_Qr\rvert\le2\tau`$。对 $`g`$ 取上确界即可。

*(d).* 恰在 $`A`$ 上部署较好 policy 的 learner，最终价值为 $`a+\gamma\mathbf 1_A(Y)`$，所以 $`g_A=a+\gamma\mathbf 1_A\in\mathcal G`$，且 $`(P-Q)g_A=\gamma(P(A)-Q(A))`$。

- $`\sup_A\lvert P(A)-Q(A)\rvert=\mathrm{TV}(P,Q)=\tfrac12\lVert P-Q\rVert_1`$，在 $`A=\lbrace y:P(y)>Q(y)\rbrace`$ 处取到（全变差的事件形式）。
- 单点事件给出 $`g_{\lbrace y\rbrace}=a\mathbf 1+\gamma e_y`$。由于 $`\gamma>0`$，$`\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\supseteq\mathrm{span}\lbrace e_y:y\in\mathcal Y\rbrace=\mathbb R^m`$，由 (4.1) 得 $`d_0=m-1`$。$`\square`$

*关于 Theorem 4.2 的附注.*

- **(b) 中的量词.** (4.2) 的上确界取遍所有共享某组特征均值的分布对。如果参考分布事先固定，或要求两个分布落在同一个预先指定的特征均值纤维上，可行的差分集合会变小，对偶要在更小的集合上取，(4.2) 只是上界。
- **受限分布族.** 若只有分布族 $`\mathcal P\subsetneq\Delta(\mathcal Y)`$ 可实现，就把 Claim 1 中的集合换成 $`\mathcal P`$ 内的差分。此时 (a) 应读作：$`(P-Q)g`$ 在特征匹配对上恒为零，当且仅当每个特征匹配的差分都零化 $`g`$。
- **系数稳定性.** (c) 显式写出了系数大小 $`B`$。残差 $`\tau`$ 小而 $`B`$ 大时，特征估计误差会被放大。[Theorem 8.1](08_verification_ZH.md#s8-1) 把这一权衡精确化。
- **(d) 对丰富性的依赖.** $`d_0=m-1`$ 要求 learner 能实现每个单点示性函数。实际的 learner 与任务能实现的 response 少得多。(d) 只说明：不限制任务与 learner 类，就不存在一致小的接口。

## 仿射 learner

<a id="s4-3"></a>

**Definition 4.3（二元 context 学生与凸仿射 learner）.**

- **学生.** 学生有 $`k`$ 个二元动作 context，参数为 $`p\in[0,1]^k`$。在 context 为 $`j`$ 的决策点，给定全部历史，它以条件概率 $`p_j`$ 选择动作 1。
- **部署.** 部署过程是一棵有限树，含三类节点：
  - *机会节点*：转移概率与 $`p`$ 无关；
  - *决策节点*：各带一个 context 标签 $`j`$，标签可以依赖历史，但不能直接依赖 $`p`$；
  - *终止节点*：奖励在 $`[0,1]`$ 中，且与 $`p`$ 无关。
- 每条根到叶的路径至多含 $`N`$ 个决策节点，同一 context 可在一条路径上重复出现。$`J_E(p)`$ 是期望终止奖励。
- **经验.** 一个经验块为 $`Z\in[0,1]^k`$，块内各坐标可任意相关。块 $`Z_1,\dots,Z_R`$ 独立同分布于 $`P`$（或 $`Q`$）。
- **凸仿射 learner.** 设 $`p_0\in[0,1]^k`$ 公共且固定，权重事先固定、与数据无关：

```math
p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i,\qquad w_i\ge0,\quad\sum_{i=0}^Rw_i=1 .
```

- **记号.** 对 $`\alpha\in\mathbb N^k`$，记 $`p^\alpha=\prod_jp_j^{\alpha_j}`$、$`\lvert\alpha\rvert=\sum_j\alpha_j`$、$`m_P(\alpha)=\mathbb E_PZ^\alpha`$。对多项式 $`f=\sum_\alpha c_\alpha p^\alpha`$，记 $`\lVert f\rVert_{\rm coef}=\sum_\alpha\lvert c_\alpha\rvert`$。

于是 $`p_R\in[0,1]^k`$，一个数据源的学习价值为 $`\mathbb E\,J_E(p_R)`$。

<a id="s4-4"></a>

**Theorem 4.4（仿射 learner：矩阶充分且必要）.** 在 Definition 4.3 的设定下：

- **(a) 部署价值是多项式.** $`J_E`$ 有唯一表示

```math
J_E(p)=\sum_{\lvert\alpha\rvert\le N}c_\alpha p^\alpha,\qquad\sum_\alpha\lvert c_\alpha\rvert\le3^N, \qquad(4.3)
```

  且在 $`[0,1]^k`$ 上 $`J_E(p)\in[0,1]`$。
- **(b) 联合阶数的充要条件.** 以下两条等价：
  - 对所有 $`\lvert\alpha\rvert\le N`$，$`m_P(\alpha)=m_Q(\alpha)`$；
  - 对每个至多 $`N`$ 次决策的部署和每个凸仿射 learner，$`\mathbb E_{P^R}J_E(p_R)=\mathbb E_{Q^R}J_E(p_R)`$。
- **(c) 每一阶都必要.**
  - *标量见证.* 设 $`N\ge1`$，$`z_i=i/N`$，$`i=0,\dots,N`$。令 $`P_N(z_i)=\binom Ni/2^{N-1}`$（$`i`$ 为偶数；奇数处为 $`0`$），$`Q_N(z_i)=\binom Ni/2^{N-1}`$（$`i`$ 为奇数；偶数处为 $`0`$）。则 $`0\le\ell<N`$ 时 $`m_{P_N}(\ell)=m_{Q_N}(\ell)`$，而

```math
\Delta_N:=m_{P_N}(N)-m_{Q_N}(N)=\frac{(-1)^NN!}{2^{N-1}N^N},
```

  - 并且任务 $`J(p)=p^N`$ 配合一步 learner $`p=Z`$ 实现了差距 $`\Delta_N`$。
  - *Parity 见证.* 取 $`k=N`$，$`P`$ 为 $`\lbrace0,1\rbrace^N`$ 中偶 parity 向量上的均匀分布，$`Q`$ 为奇 parity 向量上的均匀分布。两者所有总次数低于 $`N`$ 的联合矩都相等。取 $`J_{\rm even}(p)=\tfrac12\big(1+\prod_{j=1}^N(1-2p_j)\big)`$，一步 learner $`p=Z`$ 给出 $`\mathbb E_PJ_{\rm even}(Z)=1`$、$`\mathbb E_QJ_{\rm even}(Z)=0`$。
- **(d) 精确统计量个数与二元壁垒.**
  - 在网格 $`\mathcal Y=\lbrace0,\tfrac1N,\dots,1\rbrace^k`$ 上，要对 (b) 中所有任务与 learner 都充分，单个块的线性期望统计量最少需要 $`\binom{k+N}{N}-1`$ 个。
  - 在 $`\mathcal Y=\lbrace0,1\rbrace^k`$ 上，设 $`N\ge k`$，$`k`$ 维动作向量上的终止奖励任意。任何少于 $`2^k-1`$ 个统计量的特征接口，未分辨差距都恰好是 $`1`$。所以容忍任何 $`\varepsilon<1`$ 的差距都不会降低这一最坏情形维数；$`2^k-1`$ 个仿射无关的特征给出差距 $`0`$。

*Proof.* 用到三个断言。

*Claim 3（基本运算下的系数范数）.* 设 $`f,g`$ 是 $`p`$ 的多项式，$`j\in\lbrace1,\dots,k\rbrace`$。

- $`\lVert p_jf\rVert_{\rm coef}=\lVert f\rVert_{\rm coef}`$，因为 $`p^\beta\mapsto p^{\beta+e_j}`$ 把不同单项式映到不同单项式。
- $`\lVert f+g\rVert_{\rm coef}\le\lVert f\rVert_{\rm coef}+\lVert g\rVert_{\rm coef}`$。
- 对权重 $`\pi_e\ge0`$、$`\sum_e\pi_e=1`$，$`\lVert\sum_e\pi_ef_e\rVert_{\rm coef}\le\max_e\lVert f_e\rVert_{\rm coef}`$。

*Claim 4（仿射 learner 的块矩展开）.* 设 $`\lvert\alpha\rvert=a`$，把 $`p_R^\alpha`$ 的坐标因子列为 $`j(1),\dots,j(a)`$，其中坐标 $`j`$ 出现 $`\alpha_j`$ 次。则

```math
p_R^\alpha=\sum_{s:\lbrace1,\dots,a\rbrace\to\lbrace0,\dots,R\rbrace}\Big(\prod_{l=1}^aw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^RZ_i^{\beta_i(s)},\qquad\beta_i(s)=\sum_{l:s(l)=i}e_{j(l)} .
```

- 系数 $`\prod_lw_{s(l)}`$ 非负，总和为 $`\prod_l\sum_iw_i=1`$。
- 对每个 $`s`$，$`\sum_i\lvert\beta_i(s)\rvert\le a`$，所以至多 $`a`$ 个块满足 $`\beta_i(s)\ne0`$。
- 块独立同分布时，

```math
\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)). \qquad(4.4)
```

展开式就是对 $`a`$ 个线性形式 $`w_0p_{0,j(l)}+\sum_{i\ge1}w_iZ_{i,j(l)}`$ 的乘积用分配律，每个因子选定一个来源 $`s(l)`$；块间独立使每个块单项式乘积的期望可以分解。

*Claim 5（网格上的线性无关）.* 在 $`\mathcal Y=\lbrace0,\tfrac1N,\dots,1\rbrace^k`$ 上，单项式 $`\lbrace p^\alpha:\max_j\alpha_j\le N\rbrace`$ 作为 $`\mathcal Y`$ 上的函数线性无关；特别地，$`\lvert\alpha\rvert\le N`$ 的 $`\binom{k+N}N`$ 个单项式线性无关。

- $`k=1`$：次数不超过 $`N`$ 的多项式若在 $`N+1`$ 个不同点为零，则恒为零。
- $`k>1`$：写 $`f=\sum_{a=0}^Np_1^af_a(p_2,\dots,p_k)`$，设 $`f`$ 在 $`\mathcal Y`$ 上为零。对子网格中每个固定的 $`(p_2,\dots,p_k)`$，$`p_1\mapsto f`$ 次数不超过 $`N`$、有 $`N+1`$ 个零点，所以每个 $`f_a`$ 在 $`k-1`$ 维子网格上为零，由归纳假设是零多项式。
- 总次数单项式是其子集，个数为 $`\binom{k+N}N`$（隔板法：$`\lvert\alpha\rvert\le N`$ 的 $`\alpha\in\mathbb N^k`$ 与 $`\lvert\alpha\rvert=N`$ 的 $`\alpha\in\mathbb N^{k+1}`$ 一一对应）。

*(a).* 对部署树的节点 $`u`$，令 $`V_u(p)`$ 为从 $`u`$ 出发的期望终止奖励，$`n(u)`$ 为从 $`u`$ 到叶的路径上决策节点数的最大值。对 $`u`$ 的高度归纳，证明 $`\deg V_u\le n(u)`$、$`\lVert V_u\rVert_{\rm coef}\le3^{n(u)}`$。

- *叶节点.* $`V_u=r\in[0,1]`$ 为常数，范数至多 $`1=3^0`$。
- *机会节点.* $`V_u=\sum_e\pi_eV_{u_e}`$，概率与 $`p`$ 无关，且 $`n(u_e)\le n(u)`$。次数至多 $`n(u)`$；由 Claim 3，范数至多 $`\max_e3^{n(u_e)}\le3^{n(u)}`$。
- *context 为 $`j`$ 的决策节点.* 子节点为 $`u_1`$（动作 1）与 $`u_0`$（动作 0），$`V_u=p_jV_{u_1}+(1-p_j)V_{u_0}=V_{u_0}+p_jV_{u_1}-p_jV_{u_0}`$。次数至多 $`1+(n(u)-1)=n(u)`$；由 Claim 3，范数至多 $`3\cdot3^{n(u)-1}`$。

根节点处 $`n\le N`$。$`V_u`$ 是 $`[0,1]`$ 中奖励的平均，所以 $`J_E\in[0,1]`$。唯一性：若两个多项式在 $`[0,1]^k`$ 上相等，它们的差在一个内部非空的集合上为零，于是在内点处所有偏导数为零，所有系数相同。

*(b).*（第一条 ⇒ 第二条）由 (a)，$`\mathbb EJ_E(p_R)=\sum_{\lvert\alpha\rvert\le N}c_\alpha\,\mathbb Ep_R^\alpha`$。(4.4) 中出现的每个块矩都满足 $`\lvert\beta_i\rvert\le\lvert\alpha\rvert\le N`$，所以在 $`P`$ 与 $`Q`$ 下相同。

（第二条 ⇒ 第一条）固定 $`1\le\lvert\alpha\rvert\le N`$ 的 $`\alpha`$（$`\alpha=0`$ 平凡）。按固定顺序访问 context $`j`$ 恰好 $`\alpha_j`$ 次、当且仅当所有动作都为 1 时回报 1 的部署，含 $`\lvert\alpha\rvert\le N`$ 个决策，价值为 $`p^\alpha`$。取 $`R=1`$、$`w_0=0`$、$`w_1=1`$，学习价值在 $`P`$ 下为 $`m_P(\alpha)`$，在 $`Q`$ 下为 $`m_Q(\alpha)`$，所以第二条迫使两者相等。

*边缘分布不够.* $`P=\tfrac12\delta_{(0,0)}+\tfrac12\delta_{(1,1)}`$ 与 $`Q=\tfrac12\delta_{(0,1)}+\tfrac12\delta_{(1,0)}`$ 下两个坐标都是 $`\mathrm{Bernoulli}(\tfrac12)`$，但 $`m_P(1,1)=\tfrac12`$、$`m_Q(1,1)=0`$。任务"两个 context 各访问一次，两个动作都为 1 时回报 1"使训练出的学生相差 $`\tfrac12`$。

*(c).* 令 $`(\mathsf Ef)(x)=f(x+1)`$、$`\Delta=\mathsf E-I`$。由可交换算子 $`\mathsf E`$、$`I`$ 的二项式定理，$`(\Delta^Nf)(0)=\sum_{i=0}^N(-1)^{N-i}\binom Nif(i)`$。若 $`f`$ 次数为 $`d\ge1`$、首项系数为 $`c`$，则 $`\Delta f`$ 次数为 $`d-1`$、首项系数为 $`cd`$；所以 $`\deg f<N`$ 时 $`\Delta^Nf=0`$，且 $`\Delta^Nx^N=N!`$。

- *标量见证，总质量.* $`N\ge1`$ 时 $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$，因为两者之差为 $`(1-1)^N=0`$，之和为 $`2^N`$。
- *标量见证，矩.* 对 $`\ell\ge0`$，

```math
m_{P_N}(\ell)-m_{Q_N}(\ell)=\frac{1}{2^{N-1}N^\ell}\sum_{i=0}^N(-1)^i\binom Ni\,i^\ell=\frac{(-1)^N}{2^{N-1}N^\ell}\big(\Delta^Nx^\ell\big)(0),
```

- 它在 $`\ell<N`$ 时为零，在 $`\ell=N`$ 时等于 $`(-1)^NN!/(2^{N-1}N^N)`$。
- *标量见证，实现.* 访问同一 context $`N`$ 次、仅当全部动作为 1 时回报 1，得到 $`J(p)=p^N`$；配合 $`p=Z`$，学习差距为 $`\Delta_N`$。
- *Parity 见证，真子集上均匀.* 设 $`S\subsetneq\lbrace1,\dots,N\rbrace`$、$`x_S\in\lbrace0,1\rbrace^S`$。补集 $`S^c\ne\varnothing`$ 有 $`2^{N-\lvert S\rvert}`$ 种补全，奇偶各占一半，因为翻转 $`S^c`$ 中某个固定坐标是改变 parity 的双射。所以 $`P(Z_S=x_S)=Q(Z_S=x_S)=2^{-\lvert S\rvert}`$。
- *Parity 见证，矩.* $`\lvert\alpha\rvert<N`$ 时，支撑 $`S=\lbrace j:\alpha_j>0\rbrace`$ 满足 $`\lvert S\rvert<N`$。在二元向量上，$`\alpha_j\ge1`$ 时 $`Z_j^{\alpha_j}=Z_j`$，所以 $`Z^\alpha=\prod_{j\in S}Z_j`$ 只依赖于 $`Z_S`$，两种分布下期望相同。
- *Parity 见证，价值.* 对每个 context 访问一次，动作 $`a_j\sim\mathrm{Bernoulli}(p_j)`$ 相互独立，$`\sum_ja_j`$ 为偶数时回报 1。则 $`\Pr(\text{even})=\tfrac12\big(1+\mathbb E(-1)^{\sum_ja_j}\big)=\tfrac12\big(1+\prod_j(1-2p_j)\big)=J_{\rm even}(p)`$。当 $`p=Z`$ 为二元向量时动作就等于 $`Z`$，所以 $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\ \text{even}\rbrace`$，它在 $`P`$ 下期望为 $`1`$，在 $`Q`$ 下为 $`0`$。

*(d)，网格计数.*

- *上界.* 由 (b)，$`\binom{k+N}N-1`$ 个非常数矩 $`\lbrace m(\alpha):1\le\lvert\alpha\rvert\le N\rbrace`$ 已经充分。
- *下界.* 一步 learner $`p=Z`$ 配合 (b) 中的单项式任务，把所有 $`\lvert\alpha\rvert\le N`$ 的单项式 $`p^\alpha`$（限制在 $`\mathcal Y`$ 上）放进 $`\mathcal G`$。有限网格上的所有分布都允许，所以由 [Theorem 4.2(a)](#s4-2)，充分的接口必须满足 $`\mathrm{span}\mathcal G\subseteq H_\phi`$。由 Claim 5，$`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)\ge\binom{k+N}N`$，因此 $`d\ge\binom{k+N}N-1`$。

*(d)，二元壁垒.*

- *每张奖励表都是 response.* 取 $`p=Z`$，部署对 $`k\le N`$ 个 context 各访问一次，动作向量就等于 $`Z\in\lbrace0,1\rbrace^k`$，所以任意终止奖励 $`f:\lbrace0,1\rbrace^k\to[0,1]`$ 都实现为 $`g=f`$。因此 $`\mathcal G=[0,1]^{\mathcal Y}`$。
- *少于 $`2^k-1`$ 个特征时差距为 1.* 若 $`d<2^k-1`$，则 $`\dim H_\phi\le d+1<2^k=\lvert\mathcal Y\rvert`$，存在 $`v\in H_\phi^\perp\setminus\lbrace0\rbrace`$。由于 $`\mathbf 1\in H_\phi`$，$`v_\pm`$ 质量相同，记为 $`s>0`$。分布 $`P=v_+/s`$、$`Q=v_-/s`$ 支撑不相交，且 $`(P-Q)\phi_j=v^\top\phi_j/s=0`$。取 $`f=\mathbf 1_{\mathrm{supp}\,v_+}\in\mathcal G`$，得 $`(P-Q)f=1`$。response 都在 $`[0,1]`$ 中，差距不会超过 1。
- *用 $`2^k-1`$ 个特征时差距为 0.* 若 $`\lbrace\mathbf 1,\phi_1,\dots,\phi_{2^k-1}\rbrace`$ 线性无关，则 $`H_\phi=\mathbb R^{\mathcal Y}`$，特征均值相等就推出 $`P=Q`$。$`\square`$

*关于 Theorem 4.4 的附注.*

- **全称量词.** (b) 的必要性是对整个任务与 learner 族而言的。固定的任务或固定的更新配方可能只需要少得多的矩组合。
- **保留的是什么.** (b) 保留的是期望最终任务回报，不是训练所得参数的分布，也不是任务类中没有出现的尾部事件。
- **见证的大小.** $`\lvert\Delta_N\rvert=N!/(2^{N-1}N^N)\sim2\sqrt{2\pi N}(2e)^{-N}`$ 指数衰减，所以标量见证证明的是每一阶的精确必要性，而不是固定容差下的困难性。parity 见证在每一阶都给出常数差距，代价是需要 $`k=N`$ 个坐标。[Proposition 4.6](04_rates_and_composition_ZH.md#s4-6) 展开这一对比。
- **(4.3) 中的 $`3^N`$ 是保守的.** 对抗构造的部署离它有多近，见实验页。
- **假设在哪里起作用.**
  - 分解 (4.4) 需要独立同分布、与状态无关的块；依赖当前学生的 on-policy 反馈会破坏它。
  - 展开需要事先固定的非负凸权重；依赖数据的权重会破坏它。
  - 次数是在概率坐标 $`p`$ 中度量的。$`p=\sigma(\theta)`$ 这样的重参数化一般会破坏有限次数。
- **(d) 的二元壁垒有意取得很宽.** 任意终止奖励使每个函数都成为 response。结构化的任务族维数可以小得多。
