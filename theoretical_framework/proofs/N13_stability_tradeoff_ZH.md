# N13 · 稳定性–近似权衡 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N13_stability_tradeoff_ZH.md)

## 设定与假设

- $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。$`\mathcal Y`$ 上的函数看作 $`\mathbb R^m`$ 中的向量，概率律看作单纯形 $`\Delta(\mathcal Y)`$ 中的向量。
- $`g\in\mathbb R^m`$ 是一个 response（N05）。$`\phi_1,\dots,\phi_d\in\mathbb R^m`$ 是固定特征，$`\Phi\in\mathbb R^{m\times d}`$ 是以 $`\phi_j`$ 为列的矩阵，记 $`\Phi b=\sum_jb_j\phi_j`$。
- $`\mathbf{\epsilon}=(\epsilon_1,\dots,\epsilon_d)`$ 是容差，$`\epsilon_j\ge0`$。允许容差为零，也允许特征线性相关。
- $`h\in\mathbb R^m`$ 的振幅定义为 $`\mathrm{osc}(h)=\max_yh(y)-\min_yh(y)`$。

$`g`$ 的**容差差异**（tolerant discrepancy）定义为

```math
W_{\mathbf{\epsilon}}(g)=\sup\Big\lbrace \lvert(P-Q)g\rvert:\ P,Q\in\Delta(\mathcal Y),\ \lvert(P-Q)\phi_j\rvert\le\epsilon_j\ \ (j=1,\dots,d)\Big\rbrace ,
```

其中 $`(P-Q)h=\sum_y(P(y)-Q(y))h(y)`$。它是"特征均值之差不超过声明容差"的两个消息律之间，response 能相差的最大值。有限 audit 把每个特征均值测到精度 $`\epsilon_j`$ 之后，面对的正是这个情形。

## 陈述

**定理.**

```math
W_{\mathbf{\epsilon}}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]. \qquad(36)
```

两个最小值都可以取到。特别地，$`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$；当 $`g\in[0,1]^m`$ 时它不超过 $`1`$。

**(b)** **推论（精确情形）.** 当 $`\mathbf{\epsilon}=0`$ 时，$`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$，其中 $`H_\phi=\mathrm{span}\lbrace \mathbf 1,\phi_1,\dots,\phi_d\rbrace`$。这就是 [N06(b)](N06_sufficiency_duality_ZH.md)（N06）在单个 response 上的形式。

**(c)** **推论（稳定性见证）.** 设 $`g\in\lbrace 0,1\rbrace ^m`$ 不是常数，$`\phi_1=\alpha g`$，$`\alpha>0`$，$`d=1`$。则 $`W_\epsilon(g)=\min\lbrace 1,\epsilon/\alpha\rbrace`$。

**(d)** **推论（有限测量的桥接）.** 固定一个同时成立的事件，在该事件上每个被 audit 的特征均值满足 $`\lvert\widehat\mu_{P,j}-\mu_{P,j}\rvert\le e_{P,j}`$ 和 $`\lvert\widehat\mu_{Q,j}-\mu_{Q,j}\rvert\le e_{Q,j}`$。令 $`\epsilon_j=\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}`$。则在该事件上 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(g)`$。若另有 $`\lVert\widehat g-g\rVert_\infty\le\eta`$，则 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$。

## 证明

**引理 1（差集）.** $`\lbrace P-Q:P,Q\in\Delta(\mathcal Y)\rbrace =V:=\lbrace v\in\mathbb R^m:\mathbf 1^\top v=0,\ \lVert v\rVert_1\le2\rbrace`$。

*证明.* 若 $`v=P-Q`$，则 $`\mathbf 1^\top v=1-1=0`$，且 $`\lVert v\rVert_1\le\lVert P\rVert_1+\lVert Q\rVert_1=2`$。反过来，取 $`v\in V`$，写成 $`v=v^+-v^-`$，其中 $`v^\pm\ge0`$ 且支撑不相交。由 $`\mathbf 1^\top v=0`$，两部分质量相同，记为 $`s=\mathbf 1^\top v^+=\mathbf 1^\top v^-`$；又 $`2s=\lVert v\rVert_1\le2`$，故 $`s\le1`$。任取 $`r\in\Delta(\mathcal Y)`$，令 $`P=v^++(1-s)r`$，$`Q=v^-+(1-s)r`$。两者非负、总质量为 1，且 $`P-Q=v`$。$`\square`$

**引理 2（$`V`$ 的支撑函数）.** 对任意 $`h\in\mathbb R^m`$，$`\max_{v\in V}h^\top v=\mathrm{osc}(h)`$。

*证明.* 取 $`v\in V`$，正负部分如上，公共质量 $`s\le1`$。由于 $`\mathrm{osc}(h)\ge0`$，有 $`h^\top v=h^\top v^+-h^\top v^-\le s\max h-s\min h=s\mathrm{osc}(h)\le\mathrm{osc}(h)`$。取 $`v=e_{y^+}-e_{y^-}`$，其中 $`y^\pm`$ 分别是 $`h`$ 的最大点和最小点，等号成立。$`\square`$

**引理 3（振幅与中心化 sup 范数）.** $`\mathrm{osc}(h)=2\min_{c\in\mathbb R}\lVert h-c\mathbf 1\rVert_\infty`$，在 $`c^\star=(\max h+\min h)/2`$ 处取到。

*证明.* 对任意 $`c`$，$`\max h-\min h=(\max h-c)+(c-\min h)\le2\lVert h-c\mathbf1\rVert_\infty`$。在 $`c^\star`$ 处两项都等于 $`\mathrm{osc}(h)/2`$，所以 $`\lVert h-c^\star\mathbf 1\rVert_\infty=\mathrm{osc}(h)/2`$。$`\square`$

**第 1 步（化为线性规划）.** 由引理 1，

```math
W_{\mathbf{\epsilon}}(g)=\sup\lbrace \lvert g^\top v\rvert:\ v\in V,\ -\mathbf{\epsilon}\le\Phi^\top v\le\mathbf{\epsilon}\rbrace .
```

可行集 $`F`$ 在 $`v\mapsto-v`$ 下不变，因此可以去掉绝对值：$`W_{\mathbf{\epsilon}}(g)=\sup_{v\in F}g^\top v`$。$`F`$ 含 $`v=0`$，又受 $`\lVert v\rVert_1\le2`$ 约束，是非空多面体，所以上确界是一个有限的最大值。

**第 2 步（弱对偶）.** 固定 $`b\in\mathbb R^d`$。对每个 $`v\in F`$，

```math
g^\top v=(g-\Phi b)^\top v+b^\top\Phi^\top v\le\mathrm{osc}(g-\Phi b)+\sum_j\lvert b_j\rvert\,\lvert\phi_j^\top v\rvert\le\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert,
```

第一项用引理 2，第二项用特征约束。对 $`v`$ 取最大、对 $`b`$ 取下确界，得

```math
W_{\mathbf{\epsilon}}(g)\le\inf_b\Big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\Big].
```

**第 3 步（强对偶）.** 给约束 $`\Phi^\top v\le\mathbf{\epsilon}`$ 和 $`-\Phi^\top v\le\mathbf{\epsilon}`$ 分别引入乘子 $`u,w\in\mathbb R^d_{\ge0}`$，把 $`v\in V`$ 留作定义域。Lagrangian 为

```math
\mathcal L(v;u,w)=g^\top v-u^\top(\Phi^\top v-\mathbf{\epsilon})-w^\top(-\Phi^\top v-\mathbf{\epsilon})=(g-\Phi(u-w))^\top v+\mathbf{\epsilon}^\top(u+w).
```

$`\mathcal L`$ 是双线性的，$`V`$ 是紧凸集，$`\mathbb R^{2d}_{\ge0}`$ 是凸集，由 Sion 极小极大定理，

```math
\max_{v\in V}\ \inf_{u,w\ge0}\mathcal L=\inf_{u,w\ge0}\ \max_{v\in V}\mathcal L.
```

左边等于 $`\max_{v\in F}g^\top v`$：若 $`v\in F`$，内层下确界为 $`g^\top v`$，否则为 $`-\infty`$。由引理 2，右边等于 $`\inf_{u,w\ge0}[\mathrm{osc}(g-\Phi(u-w))+\mathbf{\epsilon}^\top(u+w)]`$。固定 $`b=u-w`$ 时，因为 $`u_j+w_j\ge\lvert u_j-w_j\rvert`$，表示代价满足 $`\mathbf{\epsilon}^\top(u+w)\ge\sum_j\epsilon_j\lvert b_j\rvert`$，并在 $`u=b^+`$、$`w=b^-`$ 处取等。因此

```math
W_{\mathbf{\epsilon}}(g)=\inf_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\Big].
```

**第 4 步（可达性）.** 目标函数 $`f(b)=\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$ 是有限个 $`b`$ 的仿射函数的最大值，因而是凸的多面体函数。由第 2 步，它以 $`W_{\mathbf{\epsilon}}(g)\ge0`$ 为下界。有下界的多面体凸函数能取到下确界：对它做最小化，等价于关于 $`(b,t,s)`$ 的一个可行线性规划，约束为 $`t\ge(g-\Phi b)(y)-(g-\Phi b)(y')`$（对所有 $`y,y'`$）和 $`s_j\ge\pm b_j`$。这个线性规划有有限最优值，所以最小值可以取到。

**第 5 步（第二种形式）.** 由引理 3，$`\mathrm{osc}(g-\Phi b)=2\min_c\lVert g-c\mathbf 1-\Phi b\rVert_\infty`$。代入第 4 步，即得 (36) 的第一种表达式。取 $`b=0`$ 得 $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$。$`\square`$

**(b)** **推论 的证明.** $`\mathbf{\epsilon}=0`$ 时罚项消失。由于 $`c\mathbf 1+\Phi b`$ 恰好遍历 $`H_\phi`$，有 $`\min_{b,c}2\lVert g-c\mathbf 1-\Phi b\rVert_\infty=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$。$`\square`$

**(c)** **推论 的证明.** 此时 $`g-b\alpha g=(1-\alpha b)g`$。因为 $`g`$ 取到 $`0`$ 和 $`1`$ 两个值，$`\mathrm{osc}((1-\alpha b)g)=\lvert1-\alpha b\rvert`$，所以 $`f(b)=\lvert1-\alpha b\rvert+\epsilon\lvert b\rvert`$。它是凸的分段线性函数，拐点在 $`b=0`$ 和 $`b=1/\alpha`$：在 $`b<0`$ 上随 $`b`$ 减小而增大，在 $`b>1/\alpha`$ 上随 $`b`$ 增大而增大，在 $`[0,1/\alpha]`$ 上是仿射的。因此 $`\min_bf(b)=\min\lbrace f(0),f(1/\alpha)\rbrace =\min\lbrace 1,\epsilon/\alpha\rbrace`$。$`\square`$

**(d)** **推论 的证明.** 在该事件上，由三角不等式，$`\lvert(P-Q)\phi_j\rvert\le\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}=\epsilon_j`$，所以 $`(P,Q)`$ 满足 $`W_{\mathbf{\epsilon}}(g)`$ 定义中的约束。至于第二个结论，$`\lvert P(g-\widehat g)\rvert`$ 和 $`\lvert Q(g-\widehat g)\rvert`$ 都不超过 $`\eta`$，所以 $`\lvert(P-Q)g\rvert\le\lvert(P-Q)\widehat g\rvert+\lvert(P-Q)(g-\widehat g)\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$。系数 $`b`$ 可以在看到特征界之后再优化，因为这个事件同时控制了固定特征的所有线性组合。$`\square`$

## 注记

1. **含义.** $`\mathrm{osc}(g-\Phi b)`$ 是近似误差，$`\sum_j\epsilon_j\lvert b_j\rvert`$ 是测量误差的代价。一组特征即使能精确表示 $`g`$，只要需要很大的系数，就仍然是很差的 certificate。(c) 给出 $`W=\epsilon/\alpha`$，特征尺度 $`\alpha`$ 越小，它越大。(36) 给出的是两者之间精确的最优权衡。
2. **紧性.** (36) 在不受限的律对上是等式。如果固定参照律 $`P`$、限定可实现律的范围，或者固定边际的 coupling（N16），右边仍是上界，但未必能取到。
3. **下游用途.** N14 用到线性项 $`B_t(\cdot)`$，其中 $`B_t\ge\lVert b\rVert_1`$。对 (36) 做优化，就是 CED 规则 4（N17）的有限 LP 拟合目标。

## 依赖与来源

- 依赖 [N06](N06_sufficiency_duality_ZH.md)（作为(b) 得到的近似对偶）和 [N09](N09_approximate_complexity_ZH.md)（看近似宽度，而不是次数）。
- 这是把经典的 LP/范数对偶用在操作性差异上，不声称提出新的对偶原理。

**出处：** [v0.14.1 Theorem 18](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-18-an-exact-stabilityapproximation-tradeoff) — Thm. 18; eq. (36). [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §10，Thm. 18，式 (36)。独立复核：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) §D1。
