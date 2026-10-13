# Theorem 8.1 · 稳定性–近似的权衡

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-01_stability_approximation_tradeoff_ZH.md) · [证明](../proofs/08-01_stability_approximation_tradeoff_ZH.md) · [实验](../experiments/08-01_stability_approximation_tradeoff_ZH.md) · [English](08-01_stability_approximation_tradeoff.md)

**本页证明什么.** 把容差差异精确地写成线性规划的对偶并证明最小值可以取到；由此得到零容差情形、一个稳定性见证，以及与实测特征均值之间的衔接。

**本页用到的设定与记号.**

设 $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。

- $`\mathcal Y`$ 上的函数是 $`\mathbb R^m`$ 中的向量，概率律是单纯形 $`\Delta(\mathcal Y)`$ 中的向量。对 $`h\in\mathbb R^m`$ 记 $`(P-Q)h=\sum_y(P(y)-Q(y))h(y)`$。
- $`g\in\mathbb R^m`$ 是一个 response。$`\phi_1,\dots,\phi_d\in\mathbb R^m`$ 是固定的特征，$`\Phi\in\mathbb R^{m\times d}`$ 以 $`\phi_j`$ 为列，$`\Phi b=\sum_jb_j\phi_j`$。
- $`\mathbf{\epsilon}=(\epsilon_1,\dots,\epsilon_d)`$ 是容差，$`\epsilon_j\ge0`$。允许零容差，也允许特征线性相关。
- $`h`$ 的振幅（oscillation）为 $`\mathrm{osc}(h)=\max_yh(y)-\min_yh(y)`$。

$`g`$ 的**容差差距**（tolerant discrepancy）定义为

```math
W_{\mathbf{\epsilon}}(g)=\sup\Big\lbrace\lvert(P-Q)g\rvert:\ P,Q\in\Delta(\mathcal Y),\ \lvert(P-Q)\phi_j\rvert\le\epsilon_j\ \ (j=1,\dots,d)\Big\rbrace .
```

它是特征均值之差不超过所声明容差的两个消息律之间，response 差距的最大值。一次有限 audit 把每个特征均值测到精度 $`\epsilon_j`$ 之后，面对的正是这种情形。

**陈述（[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)）.**

**(a) 精确权衡.** 下面两个最小值都能取到，并且

```math
W_{\mathbf{\epsilon}}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big].
```

特别地 $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$；当 $`g\in[0,1]^m`$ 时它不超过 $`1`$。

- **(b) 精确情形.** $`\mathbf{\epsilon}=0`$ 时，$`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$，其中 $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$。这是 [Theorem 4.2(b)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 的单个 response 版本。
- **(c) 稳定性反例.** 设 $`g\in\lbrace0,1\rbrace^m`$ 不是常数，$`d=1`$，$`\phi_1=\alpha g`$，$`\alpha>0`$。则 $`W_\epsilon(g)=\min\lbrace1,\epsilon/\alpha\rbrace`$。
- **(d) 有限测量的桥梁.** 固定一个同时成立的事件，在其上每个被 audit 的特征均值满足 $`\lvert\widehat\mu_{P,j}-\mu_{P,j}\rvert\le e_{P,j}`$ 与 $`\lvert\widehat\mu_{Q,j}-\mu_{Q,j}\rvert\le e_{Q,j}`$。令 $`\epsilon_j=\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}`$。则在该事件上 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(g)`$。若还有 $`\lVert\widehat g-g\rVert_\infty\le\eta`$，则 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$。

*Proof.* 先证三个基本事实。

*Claim 1（差集）.* $`\lbrace P-Q:P,Q\in\Delta(\mathcal Y)\rbrace=\mathcal D:=\lbrace v\in\mathbb R^m:\mathbf 1^\top v=0,\ \lVert v\rVert_1\le2\rbrace`$。

- 若 $`v=P-Q`$，则 $`\mathbf 1^\top v=1-1=0`$，且 $`\lVert v\rVert_1\le\lVert P\rVert_1+\lVert Q\rVert_1=2`$。
- 反之，设 $`v\in\mathcal D`$，写成 $`v=v^+-v^-`$，$`v^\pm\ge0`$ 且支撑不相交。由 $`\mathbf 1^\top v=0`$，两部分质量相同，记为 $`s=\mathbf 1^\top v^+=\mathbf 1^\top v^-`$，并且 $`2s=\lVert v\rVert_1\le2`$。任取 $`q\in\Delta(\mathcal Y)`$，令 $`P=v^++(1-s)q`$，$`Q=v^-+(1-s)q`$。两者都是概率向量，且 $`P-Q=v`$。

*Claim 2（支撑函数）.* 对每个 $`h\in\mathbb R^m`$，$`\max_{v\in\mathcal D}h^\top v=\mathrm{osc}(h)`$。

- 对 $`v\in\mathcal D`$，两部分的共同质量 $`s\le1`$，于是 $`h^\top v=h^\top v^+-h^\top v^-\le s\max h-s\min h\le\mathrm{osc}(h)`$，这里用到 $`\mathrm{osc}(h)\ge0`$。
- 在 $`v=e_{y^+}-e_{y^-}`$ 处取等号，其中 $`y^\pm`$ 分别是 $`h`$ 的最大点与最小点。

*Claim 3（振幅与中心化 sup 范数）.* $`\mathrm{osc}(h)=2\min_{c\in\mathbb R}\lVert h-c\mathbf 1\rVert_\infty`$，在 $`c^\star=(\max h+\min h)/2`$ 处取到。事实上，对每个 $`c`$ 都有 $`\max h-\min h=(\max h-c)+(c-\min h)\le2\lVert h-c\mathbf 1\rVert_\infty`$，而在 $`c^\star`$ 处两项都等于 $`\mathrm{osc}(h)/2`$。

*Step 1（化为线性规划）.* 由 Claim 1，

```math
W_{\mathbf{\epsilon}}(g)=\sup\big\lbrace\lvert g^\top v\rvert:\ v\in\mathcal D,\ -\mathbf{\epsilon}\le\Phi^\top v\le\mathbf{\epsilon}\big\rbrace .
```

可行集 $`\mathcal F`$ 在 $`v\mapsto-v`$ 下不变，所以可以去掉绝对值。$`\mathcal F`$ 是非空多面体（含 $`0`$ 且有界），上确界是最大值。Step 2–3 对这个规划建立弱对偶与强对偶；对线性规划而言，这就是经典的对偶定理（[Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)）。

*Step 2（弱对偶）.* 固定 $`b\in\mathbb R^d`$。对每个 $`v\in\mathcal F`$，由 Claim 2 与特征约束，

```math
g^\top v=(g-\Phi b)^\top v+b^\top\Phi^\top v\le\mathrm{osc}(g-\Phi b)+\sum_j\lvert b_j\rvert\,\lvert\phi_j^\top v\rvert\le\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert .
```

对 $`v`$ 取最大、对 $`b`$ 取下确界，得 $`W_{\mathbf{\epsilon}}(g)\le\inf_b\big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\big]`$。

*Step 3（强对偶）.* 对约束 $`\Phi^\top v\le\mathbf{\epsilon}`$ 与 $`-\Phi^\top v\le\mathbf{\epsilon}`$ 引入乘子 $`u,w\in\mathbb R^d_{\ge0}`$，保留 $`v\in\mathcal D`$ 作为定义域。Lagrangian 为

```math
\mathcal L(v;u,w)=g^\top v-u^\top(\Phi^\top v-\mathbf{\epsilon})-w^\top(-\Phi^\top v-\mathbf{\epsilon})=\big(g-\Phi(u-w)\big)^\top v+\mathbf{\epsilon}^\top(u+w).
```

- $`\mathcal L`$ 是双线性的，$`\mathcal D`$ 紧且凸，$`\mathbb R^{2d}_{\ge0}`$ 凸。由 Sion minimax 定理 ([Sion, 1958](../references_ZH.md#sion1958general))，$`\max_{v\in\mathcal D}\inf_{u,w\ge0}\mathcal L=\inf_{u,w\ge0}\max_{v\in\mathcal D}\mathcal L`$。
- 左端等于 $`\max_{v\in\mathcal F}g^\top v`$，因为内层下确界在 $`\mathcal F`$ 上是 $`g^\top v`$，在 $`\mathcal F`$ 外是 $`-\infty`$。
- 由 Claim 2，右端为 $`\inf_{u,w\ge0}\big[\mathrm{osc}(g-\Phi(u-w))+\mathbf{\epsilon}^\top(u+w)\big]`$。固定 $`b=u-w`$ 时，由 $`u_j+w_j\ge\lvert u_j-w_j\rvert`$ 有 $`\mathbf{\epsilon}^\top(u+w)\ge\sum_j\epsilon_j\lvert b_j\rvert`$，在 $`u=b^+`$、$`w=b^-`$ 处取等号。

所以 $`W_{\mathbf{\epsilon}}(g)=\inf_b\big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\big]`$。

*Step 4（可达性）.* 目标函数 $`f(b)=\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$ 是有限个仿射函数的最大值，因而是凸多面体函数，并且以 $`0`$ 为下界。最小化它等价于关于 $`(b,t,s)`$ 的可行线性规划，约束为对所有 $`y,y'`$ 有 $`t\ge(g-\Phi b)(y)-(g-\Phi b)(y')`$，以及 $`s_j\ge\pm b_j`$。目标有下界的可行线性规划有有限最优值，所以最小值能取到。

*Step 5（第一种形式）.* 由 Claim 3，$`\mathrm{osc}(g-\Phi b)=2\min_c\lVert g-c\mathbf 1-\Phi b\rVert_\infty`$，代入即得 (a) 的第一个表达式。取 $`b=0`$ 得 $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$。

*(b) 的证明.* $`\mathbf{\epsilon}=0`$ 时罚项为零，而 $`c\mathbf 1+\Phi b`$ 取遍 $`H_\phi`$。

*(c) 的证明.* 此时 $`g-b\alpha g=(1-\alpha b)g`$；由于 $`g`$ 同时取到 $`0`$ 与 $`1`$，$`\mathrm{osc}((1-\alpha b)g)=\lvert1-\alpha b\rvert`$。于是 $`f(b)=\lvert1-\alpha b\rvert+\epsilon\lvert b\rvert`$。它是凸的分段线性函数，拐点在 $`0`$ 与 $`1/\alpha`$，在 $`[0,1/\alpha]`$ 上仿射，离开这个区间后递增。所以 $`\min_bf(b)=\min\lbrace f(0),f(1/\alpha)\rbrace=\min\lbrace1,\epsilon/\alpha\rbrace`$。

*(d) 的证明.* 在该事件上，由三角不等式 $`\lvert(P-Q)\phi_j\rvert\le\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}=\epsilon_j`$，所以 $`(P,Q)`$ 在 $`W_{\mathbf{\epsilon}}(g)`$ 的定义中可行。第二个结论：$`\lvert(P-Q)g\rvert\le\lvert(P-Q)\widehat g\rvert+\lvert P(g-\widehat g)\rvert+\lvert Q(g-\widehat g)\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$。系数 $`b`$ 可以在看到测量结果之后再优化，因为该事件同时控制了固定特征的所有线性组合。$`\square`$

## 参考文献

- [Sion, 1958](../references_ZH.md#sion1958general). Sion, M. (1958). On general minimax theorems. Pacific Journal of Mathematics 8, no. 1, 171–176.
