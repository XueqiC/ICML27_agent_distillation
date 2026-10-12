# §8 验证（Theorem C）— 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/08_verification_ZH.md) · [English](08_verification.md)

本页保留 Theorem 8.1 至 Theorem 8.3 及设定；平均增量测量与 Theorem C 见[证书页面](08_certificate_ZH.md)。

**本节概要.**

- 本节回答：只用有限次测量，怎样认证一个 feedback law 足够好，或者在几个候选 law 之间排序。各个结果的区别在于测的是什么。
  - [Theorem 8.1](#s8-1)：特征均值带容差测量时，能保证的 response 差距是近似误差与系数大小之间的精确权衡。
  - [Theorem 8.2](#s8-2)：在事先冻结的 cell 上做 audit，只有带上显式的 transport（coverage）项才能认证学习价值，而且这一项去不掉。
  - [Theorem 8.3](#s8-3)：在训练实际到达的状态上做配对的完整尾段实验，按置信下界选择，可以保证相对参照的损失不超过 $`\delta`$，但代价很高。
  - [Proposition 8.4](08_certificate_ZH.md#s8-4) 与 [Remark 8.5](08_certificate_ZH.md#s8-5)：平均增量只需梯度计算就能测量，但要把它变成一个界，还需要 continuation value 的常数，而样本本身不估计这些常数。
  - **Theorem C**（[Theorem 8.8](08_certificate_ZH.md#s8-8)）：在 surrogate 尾段上，用 common random numbers（CRN）配对的替换分数，可以无偏地估计一轮替换的**有限**效应。半径只含统计项与 surrogate 校准项，不需要 continuation 梯度、曲率常数，也没有二阶余项。
  - [Corollary 8.9](08_certificate_ZH.md#s8-9)：同一测量的一阶版本、它的 margin 判据与上限。一阶项不超过二阶余项界的候选对，在一阶意义下永远不会被认证。
  - [Proposition 8.10](08_certificate_ZH.md#s8-10)、[Remark 8.11](08_certificate_ZH.md#s8-11)、[Remark 8.12](08_certificate_ZH.md#s8-12)：代价与 CRN 的作用；校准与沿训练的迁移；两个未解决的问题。
- 用到的结果：[Theorem 3.2](03_learning_consequence_ZH.md#s3-2) 的 telescoping 恒等式；[Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2) 的近似对偶；[Theorem 4.5](04_rates_and_composition_ZH.md#s4-5) 的仿射速率；[Proposition 5.1](05_optimizer_state_ZH.md#s5-1) 的一阶展开；[Theorem 5.10](05_learner_statistics_ZH.md#s5-10) 的学习器统计量。

## 设定

全节沿用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 轮次 $`t=0,\dots,R-1`$，$`z`$ 是 complete learner state。对一阶 optimizer，实际更新为 $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$。
- 在第 $`t`$ 轮状态 $`z`$ 处，参照发出 $`Y\sim P^0_{t,z}`$，候选 $`j\in\lbrace1,\dots,M\rbrace`$ 发出 $`Y\sim P^j_{t,z}`$。候选 $`j`$ 的 kernel 记为 $`K^j_t`$。
- 参照 continuation 为 $`V^0_{t+1}`$，$`V^0_R=F_E\in[0,1]`$。response 为 $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$。对消息积分得到 kernel 恒等式

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- 对由候选 $`j`$ 驱动的过程，$`L_E(j)=\mathbb E\,F_E(Z^j_R)`$。所有过程从相同的 $`Z_0`$ 初始律出发。
- 候选 $`j`$ 在 $`(t,z)`$ 处的**局部优势**为

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- 它表示：把一轮参照替换成一轮候选 $`j`$，之后走参照尾段，对最终得分的影响。由 [Theorem 3.2](03_learning_consequence_ZH.md#s3-2)，整次训练的差距就是这些项在替代过程到达的状态上的和。

## 测量误差与系数稳定性

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

<a id="s8-1"></a>

**Theorem 8.1（稳定性–近似的权衡）.**

**(a) 精确权衡.** 下面两个最小值都能取到，并且

```math
W_{\mathbf{\epsilon}}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big].
```

特别地 $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$；当 $`g\in[0,1]^m`$ 时它不超过 $`1`$。

- **(b) 精确情形.** $`\mathbf{\epsilon}=0`$ 时，$`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$，其中 $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$。这是 [Theorem 4.2(b)](04_complexity_of_sufficiency_ZH.md#s4-2) 的单个 response 版本。
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

可行集 $`\mathcal F`$ 在 $`v\mapsto-v`$ 下不变，所以可以去掉绝对值。$`\mathcal F`$ 是非空多面体（含 $`0`$ 且有界），上确界是最大值。

*Step 2（弱对偶）.* 固定 $`b\in\mathbb R^d`$。对每个 $`v\in\mathcal F`$，由 Claim 2 与特征约束，

```math
g^\top v=(g-\Phi b)^\top v+b^\top\Phi^\top v\le\mathrm{osc}(g-\Phi b)+\sum_j\lvert b_j\rvert\,\lvert\phi_j^\top v\rvert\le\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert .
```

对 $`v`$ 取最大、对 $`b`$ 取下确界，得 $`W_{\mathbf{\epsilon}}(g)\le\inf_b\big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\big]`$。

*Step 3（强对偶）.* 对约束 $`\Phi^\top v\le\mathbf{\epsilon}`$ 与 $`-\Phi^\top v\le\mathbf{\epsilon}`$ 引入乘子 $`u,w\in\mathbb R^d_{\ge0}`$，保留 $`v\in\mathcal D`$ 作为定义域。Lagrangian 为

```math
\mathcal L(v;u,w)=g^\top v-u^\top(\Phi^\top v-\mathbf{\epsilon})-w^\top(-\Phi^\top v-\mathbf{\epsilon})=\big(g-\Phi(u-w)\big)^\top v+\mathbf{\epsilon}^\top(u+w).
```

- $`\mathcal L`$ 是双线性的，$`\mathcal D`$ 紧且凸，$`\mathbb R^{2d}_{\ge0}`$ 凸。由 Sion minimax 定理，$`\max_{v\in\mathcal D}\inf_{u,w\ge0}\mathcal L=\inf_{u,w\ge0}\max_{v\in\mathcal D}\mathcal L`$。
- 左端等于 $`\max_{v\in\mathcal F}g^\top v`$，因为内层下确界在 $`\mathcal F`$ 上是 $`g^\top v`$，在 $`\mathcal F`$ 外是 $`-\infty`$。
- 由 Claim 2，右端为 $`\inf_{u,w\ge0}\big[\mathrm{osc}(g-\Phi(u-w))+\mathbf{\epsilon}^\top(u+w)\big]`$。固定 $`b=u-w`$ 时，由 $`u_j+w_j\ge\lvert u_j-w_j\rvert`$ 有 $`\mathbf{\epsilon}^\top(u+w)\ge\sum_j\epsilon_j\lvert b_j\rvert`$，在 $`u=b^+`$、$`w=b^-`$ 处取等号。

所以 $`W_{\mathbf{\epsilon}}(g)=\inf_b\big[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert\big]`$。

*Step 4（可达性）.* 目标函数 $`f(b)=\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$ 是有限个仿射函数的最大值，因而是凸多面体函数，并且以 $`0`$ 为下界。最小化它等价于关于 $`(b,t,s)`$ 的可行线性规划，约束为对所有 $`y,y'`$ 有 $`t\ge(g-\Phi b)(y)-(g-\Phi b)(y')`$，以及 $`s_j\ge\pm b_j`$。目标有下界的可行线性规划有有限最优值，所以最小值能取到。

*Step 5（第一种形式）.* 由 Claim 3，$`\mathrm{osc}(g-\Phi b)=2\min_c\lVert g-c\mathbf 1-\Phi b\rVert_\infty`$，代入即得 (a) 的第一个表达式。取 $`b=0`$ 得 $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$。

*(b) 的证明.* $`\mathbf{\epsilon}=0`$ 时罚项为零，而 $`c\mathbf 1+\Phi b`$ 取遍 $`H_\phi`$。

*(c) 的证明.* 此时 $`g-b\alpha g=(1-\alpha b)g`$；由于 $`g`$ 同时取到 $`0`$ 与 $`1`$，$`\mathrm{osc}((1-\alpha b)g)=\lvert1-\alpha b\rvert`$。于是 $`f(b)=\lvert1-\alpha b\rvert+\epsilon\lvert b\rvert`$。它是凸的分段线性函数，拐点在 $`0`$ 与 $`1/\alpha`$，在 $`[0,1/\alpha]`$ 上仿射，离开这个区间后递增。所以 $`\min_bf(b)=\min\lbrace f(0),f(1/\alpha)\rbrace=\min\lbrace1,\epsilon/\alpha\rbrace`$。

*(d) 的证明.* 在该事件上，由三角不等式 $`\lvert(P-Q)\phi_j\rvert\le\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}=\epsilon_j`$，所以 $`(P,Q)`$ 在 $`W_{\mathbf{\epsilon}}(g)`$ 的定义中可行。第二个结论：$`\lvert(P-Q)g\rvert\le\lvert(P-Q)\widehat g\rvert+\lvert P(g-\widehat g)\rvert+\lvert Q(g-\widehat g)\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$。系数 $`b`$ 可以在看到测量结果之后再优化，因为该事件同时控制了固定特征的所有线性组合。$`\square`$

*关于 Theorem 8.1 的说明.*

- **含义.** $`\mathrm{osc}(g-\Phi b)`$ 是近似误差，$`\sum_j\epsilon_j\lvert b_j\rvert`$ 是测量误差的代价。一组特征即使能精确表示 $`g`$，若需要很大的系数，也不是好的证书：(c) 中 $`W=\epsilon/\alpha`$，随特征尺度 $`\alpha`$ 变小而变大。
- **紧性.** (a) 是对不受限制的消息律对的等式。若固定参照、只允许一族可实现的律，或者只允许固定边缘分布的 coupling（[Proposition 9.1](09_mechanism_and_method_ZH.md#s9-1)），右端仍是上界，但不一定取到。
- **用途.** [Theorem 8.2](#s8-2) 用到线性项，系数界 $`B_t\ge\lVert b\rVert_1`$。最小化 (a) 的右端，就是方法中拟合 response 模型的目标（[Remark 9.3](09_mechanism_and_method_ZH.md#s9-3)）。

## 冻结的 audit 需要 transport 项

**冻结的 audit.** 只用独立的 pilot 数据，在 audit 之前冻结：

- $`M`$ 个可执行的候选 feedback law；
- 一个有限的 audit cell 集合 $`\mathcal U`$，每个 cell $`u`$ 指定一个 complete learner state、一个任务、一个查询、更新方式和参照尾段（字母 $`u`$ 只在本小节表示 cell）；
- 每个 cell 上的 $`d`$ 个特征 $`\phi_{u,\ell}:\mathcal Y\to[0,1]`$。

$`P^0_u`$ 与 $`P^j_u`$ 分别是 cell $`u`$ 中参照与候选 $`j`$ 的消息律。audit 在每个 cell 抽取 $`n_0`$ 条独立的参照消息，在每个（候选，cell）上抽取 $`n_S`$ 条独立消息。记 $`\mu_{j,u,\ell}=\mathbb E_{P^j_u}\phi_{u,\ell}`$，经验均值为 $`\widehat\mu_{j,u,\ell}`$，并令

```math
D_j=\max_{u,\ell}\lvert\mu_{0,u,\ell}-\mu_{j,u,\ell}\rvert,\qquad\widehat D_j=\max_{u,\ell}\lvert\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell}\rvert,
```

```math
e_0=\sqrt{\frac{\log(4\lvert\mathcal U\rvert d/\delta)}{2n_0}},\qquad e_S=\sqrt{\frac{\log(4M\lvert\mathcal U\rvert d/\delta)}{2n_S}}.
```

被选中的候选 $`\widehat j`$ 在已知的优化误差 $`\eta_{\rm opt}`$ 之内最小化 $`\widehat D_j`$。

**被认证的状态.** 每一轮 $`t`$ 有一个由 complete state 组成的认证集 $`\mathcal C_t`$。每个 $`z\in\mathcal C_t`$ 附带：

- 一个被 audit 的 cell $`u=u(t,z)`$，其特征用 $`z`$ 处的消息坐标表示；
- 一个 **transport 界**：对 audit 事件上每个可能被选中的候选，

```math
\Big\lVert(P^0_{t,z}-P^j_{t,z})\phi_u-(P^0_u-P^j_u)\phi_u\Big\rVert_\infty\le a_{t,j}(z);\qquad\text{(T)}
```

- 一个 **response 分解**

```math
g_{t,z}=c_{t,z}+b_{t,z}^\top\phi_u+r_{t,z},\qquad\lVert b_{t,z}\rVert_1\le B_t,\quad\lVert r_{t,z}\rVert_\infty\le\tau_t.\qquad\text{(D)}
```

令 $`\sigma=\inf\lbrace t<R:Z_t\notin\mathcal C_t\rbrace`$（若不存在则 $`\sigma=\infty`$），它是训练过程的停时。**新鲜训练**：在 audit 结果的条件下，被选中的候选用与 audit 样本独立的随机性训练，并且 $`\Pr_{\widehat j}(\sigma<R\mid\text{audit result})\le\rho`$。

<a id="s8-2"></a>

**Theorem 8.2（带 transport 的冻结有限接口选择）.** 下面所有的界都可以截断到 $`1`$。

**(a) audit 的集中性.** 以至少 $`1-\delta`$ 的概率，对所有 $`j`$ 同时有

```math
\lvert D_j-\widehat D_j\rvert\le e_0+e_S,\qquad\Longrightarrow\qquad D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}.
```

**(b) 局部 transport 界.** 对 $`z\in\mathcal C_t`$ 与 $`u=u(t,z)`$，

```math
\big\lvert(K^0_t-K^j_t)V^0_{t+1}(z)\big\rvert\le\min\Big\lbrace1,\ B_t\big(D_j+a_{t,j}(z)\big)+2\tau_t\Big\rbrace .
```

**(c) 学习价值的界.** 在 (a) 的事件上，

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace\Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace\Big]+\rho .
```

- **(d) 特例.** transport 精确（$`a\equiv0`$）时，$`\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_tB_t(\widehat D_{\widehat j}+e_0+e_S)+2\sum_t\tau_t+\rho`$。在 [Theorem 4.5(a)](04_rates_and_composition_ZH.md#s4-5) 的与状态无关的 IID block 与凸仿射更新假设下，若被 audit 的特征就是矩 $`Z^\alpha`$（$`1\le\lvert\alpha\rvert\le N`$），则 $`\lvert L_E(0)-L_E(\widehat j)\rvert\le C_J(\widehat D_{\widehat j}+e_0+e_S)`$。
- **(e) 特征漂移.** 若依赖状态的特征 $`\phi_z`$ 满足 $`\sup_y\lVert\phi_z(y)-\phi_u(y)\rVert_\infty\le\alpha_{zu}`$，则 $`\lVert(P^0_{t,z}-P^j_{t,z})\phi_z\rVert_\infty\le D_j+a_{t,j}(z)+2\alpha_{zu}`$；把 (D) 中的特征换成 $`\phi_z`$、把 $`a`$ 换成 $`a+2\alpha_{zu}`$，(b)–(c) 仍成立。
- **(f) transport 项的必要性.** 对任意 audit 规模，都存在一个实例：$`D_j=0`$，response 分解精确（$`B=1`$，$`\tau=0`$），而学习差距为 $`1`$。在这个实例中，不含 transport 的表达式 $`B(\widehat D+e_0+e_S)`$ 严格小于差距，(c) 的界仍然成立而且是紧的。

*Proof.*

**(a)** 每个 $`\phi_{u,\ell}\in[0,1]`$，每个经验均值背后的样本相互独立。由双侧 Hoeffding 不等式，

```math
\Pr\big(\lvert\widehat\mu_{0,u,\ell}-\mu_{0,u,\ell}\rvert>e_0\big)\le2e^{-2n_0e_0^2}=\frac{\delta}{2\lvert\mathcal U\rvert d},\qquad\Pr\big(\lvert\widehat\mu_{j,u,\ell}-\mu_{j,u,\ell}\rvert>e_S\big)\le\frac{\delta}{2M\lvert\mathcal U\rvert d}.
```

- 对 $`\lvert\mathcal U\rvert d`$ 个参照均值与 $`M\lvert\mathcal U\rvert d`$ 个候选均值取 union bound，得到事件 $`\mathcal A`$，$`\Pr(\mathcal A^c)\le\delta`$。同一条消息上不同特征之间的相关性不影响 union bound。
- 在 $`\mathcal A`$ 上，对每个 $`j,u,\ell`$ 有 $`\big\lvert(\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell})-(\mu_{0,u,\ell}-\mu_{j,u,\ell})\big\rvert\le e_0+e_S`$。映射 $`w\mapsto\max_k\lvert w_k\rvert`$ 关于 sup 范数是 $`1`$-Lipschitz 的，所以对所有 $`j`$，$`\lvert\widehat D_j-D_j\rvert\le e_0+e_S`$。
- 取 $`j^\star\in\arg\min_jD_j`$。在 $`\mathcal A`$ 上，$`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S\le\widehat D_{j^\star}+\eta_{\rm opt}+e_0+e_S\le D_{j^\star}+2(e_0+e_S)+\eta_{\rm opt}`$。在 audit 数据上做选择是合法的，因为 $`\mathcal A`$ 对冻结的候选集合同时成立。

**(b)** 由 (K)，$`(K^0_t-K^j_t)V^0_{t+1}(z)=(P^0_{t,z}-P^j_{t,z})g_{t,z}`$。把 (D) 逐项代入。

- 常数 $`c_{t,z}`$ 对两个概率律之差积分为零。
- 线性项：令 $`w=(P^0_{t,z}-P^j_{t,z})\phi_u\in\mathbb R^d`$，则 $`\lvert b^\top w\rvert\le\lVert b\rVert_1\lVert w\rVert_\infty`$；由 (T) 与三角不等式，$`\lVert w\rVert_\infty\le\lVert(P^0_u-P^j_u)\phi_u\rVert_\infty+a_{t,j}(z)\le D_j+a_{t,j}(z)`$。
- 残差：$`\lvert(P^0_{t,z}-P^j_{t,z})r\rvert\le\lvert P^0_{t,z}r\rvert+\lvert P^j_{t,z}r\rvert\le2\tau_t`$。
- 由于 $`g\in[0,1]`$，左端也不超过 $`1`$。

**(c)** 在 $`\mathcal A`$ 上、固定一个 audit 结果来论证，于是 $`\widehat j`$ 固定。由新鲜训练，训练过程的条件分布就是由 $`\widehat j`$ 驱动的过程的分布。

- *切换过程.* 令 $`S'`$ 在第 $`t`$ 轮当 $`t<\sigma`$ 时用 $`K^{\widehat j}_t`$，否则用 $`K^0_t`$。由于 $`\lbrace\sigma\le t\rbrace`$ 由 $`Z_0,\dots,Z_t`$ 决定，$`S'`$ 是良定义的依赖历史的过程；$`V^0`$ 不依赖切换标记。用相同的随机性把 $`S'`$ 与 $`\widehat j`$ 过程耦合，两条路径在 $`Z_\sigma`$ 之前（含）一致，在 $`\lbrace\sigma\ge R\rbrace`$ 上直到 $`Z_R`$ 都一致。终值在 $`[0,1]`$ 中，所以 $`\lvert L_E(\widehat j)-L_E(S')\rvert\le\Pr_{\widehat j}(\sigma<R)\le\rho`$。
- *Telescoping.* 由塔性质，$`\mathbb E[V^0_{t+1}(Z_{t+1})\mid Z_0,\dots,Z_t]=(K^{S'}_tV^0_{t+1})(Z_t)`$。对 $`t`$ 求和 $`\mathbb E_{S'}[V^0_t(Z_t)-V^0_{t+1}(Z_{t+1})]`$，利用相同的初始律与 $`V^0_R=F_E`$，得

```math
L_E(0)-L_E(S')=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[(K^0_t-K^{S'}_t)V^0_{t+1}(Z_t)\big]=\sum_{t=0}^{R-1}\mathbb E_{S'}\big[1\lbrace t<\sigma\rbrace(K^0_t-K^{\widehat j}_t)V^0_{t+1}(Z_t)\big],
```

第二个等号是因为 $`t\ge\sigma`$ 时两个 kernel 相同。这是 [Theorem 3.2(a)](03_learning_consequence_ZH.md#s3-2) 的停时形式。

- *退出前的占据分布.* 在 $`\sigma`$ 之前，$`S'`$ 与 $`\widehat j`$ 过程用相同的 kernel，所以对有界的 $`h`$ 有 $`\mathbb E_{S'}[1\lbrace t<\sigma\rbrace h(Z_t)]=\mathbb E_{\widehat j}[1\lbrace t<\sigma\rbrace h(Z_t)]`$。在 $`\lbrace t<\sigma\rbrace`$ 上 $`Z_t\in\mathcal C_t`$，(b) 适用；在 $`\mathcal A`$ 上 $`D_{\widehat j}\le\widehat D_{\widehat j}+e_0+e_S`$。这就把 $`\lvert L_E(0)-L_E(S')\rvert`$ 界为 (c) 中的和。
- 与切换过程的界合起来，用三角不等式即得 (c)。这个界对 $`\mathcal A`$ 中的每个 audit 结果都成立，所以 (T)、(D) 与 $`\rho`$ 的界必须对 $`\mathcal A`$ 上每个可能被选中的候选都成立。

**(d)** $`a\equiv0`$ 时，(c) 的每一项至多为 $`B_t(\widehat D_{\widehat j}+e_0+e_S)+2\tau_t`$，因为 $`1\lbrace t<\sigma\rbrace\le1`$ 且括号非负。在 IID 仿射假设下，候选 $`j`$ 的每一轮训练都从一个固定的律 $`P^j`$ 中抽一个新的 block，与状态无关。取单个 cell，特征为单项式 $`Z^\alpha`$（$`1\le\lvert\alpha\rvert\le N`$），它们取值于 $`[0,1]`$。于是 $`D_j=\delta_N(P^0,P^j)`$；[Theorem 4.5(a)](04_rates_and_composition_ZH.md#s4-5) 给出 $`\lvert L_E(0)-L_E(j)\rvert\le C_J\delta_N(P^0,P^j)`$；在 $`\mathcal A`$ 上 $`\delta_N(P^0,P^{\widehat j})\le\widehat D_{\widehat j}+e_0+e_S`$。这里不需要 transport，因为被 audit 的律就是每个状态上实际使用的律；训练 block 必须是从这个律中新抽的。

**(e)** 写 $`\phi_z=\phi_u+(\phi_z-\phi_u)`$。第一部分如 (b) 至多贡献 $`D_j+a_{t,j}(z)`$。对每个坐标 $`\ell`$，$`\lvert(P^0_{t,z}-P^j_{t,z})(\phi_z-\phi_u)_\ell\rvert\le2\alpha_{zu}`$。用对 $`\phi_z`$ 写出的 (D) 重复 (b)–(c) 即得。只在抽到的消息上成立的界，不是关于 $`\sup_y`$ 的界。

**(f)** 取 $`\mathcal Y=\lbrace0,1\rbrace`$，$`\phi(y)=y`$，$`R=1`$；更新把最终策略参数设为 $`p=y`$，$`F_E(p)=p`$。于是在每个状态上 $`g_{0,z}(y)=y`$，(D) 以 $`c=0`$、$`b=1`$、$`r=0`$ 成立。

- 在被 audit 的状态 $`z_a`$ 上，两个律都发出 $`0`$，所以 $`D_j=0`$。
- 训练从 $`z_b`$ 开始，参照发出 $`0`$，候选发出 $`1`$，所以 $`\lvert L_E(0)-L_E(j)\rvert=1`$。
- 若把 $`z_b`$ 认证为 cell $`u(z_b)=z_a`$ 而不加 transport 项，得到的表达式为 $`B(\widehat D+e_0+e_S)=e_0+e_S`$。取 $`n_0=n_S=10^4`$、$`\delta=0.05`$、$`\lvert\mathcal U\rvert=M=d=1`$，它等于 $`2\sqrt{\log80/20000}\approx0.0296<1`$。
- 在 (c) 中，要认证 $`z_b`$ 就必须 $`a_{0,j}(z_b)\ge\lvert(0-1)-(0-0)\rvert=1`$，界至少为 $`1`$；或者 $`z_b\notin\mathcal C_0`$，于是 $`\sigma=0`$、$`\rho=1`$，界为 $`1`$。

两种情形下，截断后的 (c) 都等于差距，所以 transport 项既不能去掉，也不能按约定设为零。同样的构造中，若一个被覆盖的状态以概率 $`r`$ 到达一个未被覆盖的状态，则说明：每个访问到的退出前状态都必须贡献自己的项，或者首次访问未覆盖状态必须计入 $`\rho\ge r`$；否则界会差 $`r`$。$`\square`$

*关于 Theorem 8.2 的说明.*

- **complete state.** cell、transport 与认证都针对 complete learner state。参数相同、optimizer moment 不同的两个状态，会产生不同的下一状态，response 也可以不同（[Proposition 5.3](05_optimizer_state_ZH.md#s5-3)）。只凭参数接近不能推出 (T)。
- **精确 transport 弱于分布相等.** $`a=0`$ 只要求两个特征差向量一致，不要求四个分布都相同。
- **认证的是什么.** (c) 界定的是被选中的冻结流程在新鲜训练下的总体期望值，而不是某一个训练 seed 的分数。用 audit 样本构造新候选，或者在这些样本上训练，就改变了实验，需要新的独立 audit。
- **其他置信失败.** 若 transport 界、系数或残差本身是统计估计的，它们的失败概率要加到 $`\delta`$ 上。

## 在实际状态上做 audit

**自适应 audit.** **provider 策略** $`S`$ 在每一轮根据历史选择一个 kernel 下标 $`J_t`$。

- 记 $`\mathcal H_t`$ 为第 $`t`$ 轮 audit 之前的历史。在 $`\mathcal H_t`$ 条件下，冻结实际状态 $`Z_t`$、$`M_t`$ 个非参照候选 kernel 组成的有限集合、副本数 $`n_t`$ 与置信分配 $`\delta_t`$，几乎处处 $`\sum_t\delta_t\le\delta`$。
- 对每个候选，从 $`Z_t`$ 出发做 $`n_t`$ 次独立的配对实验。每次实验执行候选更新或参照更新，再执行**完整的**参照尾段与真实部署。每一臂都有正确的边缘分布，配对对比落在 $`[-1,1]`$ 中。
- 记 $`\widehat A_t(j)`$ 为平均对比，并令

```math
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},\qquad\ell_t(j)=\widehat A_t(j)-e_t,\qquad\ell_t(0)=0 .
```

- 选择 $`J_t\in\arg\max_j\ell_t(j)`$，令 $`\Gamma_t=\ell_t(J_t)\in[0,1]`$。然后从 $`K^{J_t}_t`$ 做一次**新鲜**更新，丢弃 audit 副本。未做 audit 的轮次使用参照，令 $`\Gamma_t=0`$。
- audit 不改变实际的学习器状态与环境状态，参照也不读取 audit 记录。$`\mathcal F_t\supseteq\mathcal H_t`$ 是 audit 与选择之后、新鲜更新之前的历史。

<a id="s8-3"></a>

**Theorem 8.3（实际状态上的自适应 audit；保真与改进）.**

- **(a) 保真.** 若 $`L_E(0)-J_E(\pi_0)\ge\gamma`$ 且 $`L_E(0)-L_E(S)\le\varepsilon`$，则 $`L_E(S)-J_E(\pi_0)\ge\gamma-\varepsilon`$。这里 $`J_E(\pi_0)`$ 是初始学生的价值；$`\varepsilon`$ 可以取 [Theorem 8.2(c)](#s8-2) 的证书，$`\gamma`$ 可以取用 $`K`$ 条示范做参照训练时测得的提升。
- **(b) 带符号的恒等式.** 对任何依赖历史、初始律相同的 provider 策略 $`S`$，

```math
L_E(S)-L_E(0)=\sum_{t=0}^{R-1}\mathbb E_{d^S_t}\,A_t(Z_t,J_t).
```

- **(c) 自适应 audit.** (i) 以至少 $`1-\delta`$ 的概率，$`\sum_tA_t(Z_t,J_t)\ge\sum_t\Gamma_t`$。(ii) 对 audit 与训练无条件地，

```math
L_E(\mathrm{adaptive})-L_E(0)\ge-\delta,\qquad L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_t\Gamma_t-2\delta .
```

- **(d) 代价.** 要使半径 $`e_t\le e`$，每个候选需要 $`n_t\ge\lceil2\log(2M_t/\delta_t)/e^2\rceil`$ 个配对副本。$`M=4`$、$`\delta=0.05`$、$`e=0.1`$ 时为 $`\lceil2\log160/0.01\rceil=1016`$。这个数是充分的，不是必要的。

*Proof.*

**(a)** $`L_E(S)-J_E(\pi_0)=[L_E(0)-J_E(\pi_0)]-[L_E(0)-L_E(S)]\ge\gamma-\varepsilon`$。这里只用了单侧的保真界，所以允许候选优于参照。

**(b)** 沿 $`S`$ 过程，由塔性质 $`\mathbb E[V^0_{t+1}(Z_{t+1})\mid\text{history to }t]=(K^{J_t}_tV^0_{t+1})(Z_t)`$，所以 $`\mathbb E[V^0_{t+1}(Z_{t+1})-V^0_t(Z_t)]=\mathbb E\,A_t(Z_t,J_t)`$。对 $`t`$ 求和即 telescoping；由相同初始律 $`\mathbb EV^0_0(Z_0)=L_E(0)`$，由 $`V^0_R=F_E`$ 有 $`\mathbb EV^0_R(Z_R)=L_E(S)`$。

**(c)** *audit 事件.* 在 $`\mathcal H_t`$ 条件下，候选 $`j`$ 的 $`n_t`$ 个对比独立同分布，取值于 $`[-1,1]`$，均值为 $`A_t(Z_t,j)`$；这里用到了正确的边缘分布、完整尾段与精确恢复。对取值范围为 $`2`$ 的变量用 Hoeffding 不等式，

```math
\Pr\big(\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert>e_t\ \big\vert\ \mathcal H_t\big)\le2\exp(-n_te_t^2/2)=\delta_t/M_t .
```

- 对 $`M_t`$ 个候选取 union bound，得到事件 $`\mathcal A_t`$，$`\Pr(\mathcal A_t^c\mid\mathcal H_t)\le\delta_t`$，其上对所有 $`j`$ 有 $`\lvert\widehat A_t(j)-A_t(Z_t,j)\rvert\le e_t`$。这个半径要求 $`n_t`$ 在看到本次 audit 的结果之前就固定。
- *被选中候选的漂移.* 记 $`A_t:=A_t(Z_t,J_t)`$。若 $`J_t=0`$，则 $`A_t=0=\Gamma_t`$。否则 $`\Gamma_t=\widehat A_t(J_t)-e_t\ge0`$，而在 $`\mathcal A_t`$ 上 $`A_t\ge\widehat A_t(J_t)-e_t=\Gamma_t`$。未做 audit 的轮次 $`A_t=\Gamma_t=0`$。
- *结论 (i).* 逐次对 $`\mathcal H_t`$ 取条件，$`\Pr(\bigcup_t\mathcal A_t^c)\le\mathbb E\sum_t\Pr(\mathcal A_t^c\mid\mathcal H_t)\le\mathbb E\sum_t\delta_t\le\delta`$。在 $`\bigcap_t\mathcal A_t`$ 上 $`\sum_tA_t\ge\sum_t\Gamma_t`$。这对实际到达的随机状态成立，因为每次 audit 的陈述都以它自己的 audit 前历史为条件。
- *漂移恒等式.* 实际更新是从 $`K^{J_t}_t`$ 新抽的，在 $`\mathcal F_t`$ 条件下与 audit 副本独立，而且 audit 不改变状态。所以 $`A_t=\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]-V^0_t(Z_t)`$。差 $`D_{t+1}=V^0_{t+1}(Z_{t+1})-\mathbb E[V^0_{t+1}(Z_{t+1})\mid\mathcal F_t]`$ 是交替的 audit/更新 filtration 下的鞅差，telescoping 给出

```math
F_E(Z_R)-V^0_0(Z_0)=\sum_{t=0}^{R-1}A_t+\sum_{t=0}^{R-1}D_{t+1}.
```

- *结论 (ii).* 总有 $`A_t\in[-1,1]`$、$`\Gamma_t\in[0,1]`$，所以 $`A_t\ge-\mathbf 1_{\mathcal A_t^c}`$，且 $`A_t\ge\Gamma_t-2\cdot\mathbf 1_{\mathcal A_t^c}`$（在 $`\mathcal A_t^c`$ 上 $`A_t\ge-1\ge\Gamma_t-2`$）。对 telescoping 恒等式取期望，并用 $`\mathbb EV^0_0(Z_0)=L_E(0)`$ 与 $`\mathbb E\sum_tD_{t+1}=0`$，

```math
L_E(\mathrm{adaptive})-L_E(0)=\mathbb E\sum_tA_t\ge-\mathbb E\sum_t\Pr(\mathcal A_t^c\mid\mathcal H_t)\ge-\delta ,
```

同理 $`\mathbb E\sum_tA_t\ge\mathbb E\sum_t\Gamma_t-2\delta`$。

**(d)** $`e_t\le e`$ 等价于 $`n_t\ge2\log(2M_t/\delta_t)/e^2`$。$`M=4`$、$`\delta=0.05`$ 时 $`2\log160\approx10.150`$，除以 $`0.01`$ 再向上取整得 $`1016`$。Hoeffding 不等式只是一种充分的选择；任何半径更小、对固定 $`n`$ 成立的置信界都可以替代它。$`\square`$

*关于 Theorem 8.3 的说明.*

- **置信下界是必需的.** 即使 audit 用的是精确的完整尾段，按点估计 $`\widehat A_t(j)`$ 选择的期望增益也可以严格为负；一个略差的候选也可能以正概率通过置信下界的门槛。所以 (c)(ii) 中的 $`-\delta`$ 项去不掉：精确的"不退化"（$`\ge0`$）一般不成立。第一个事实的精确有限实例见[实验页](../experiments/08_verification_ZH.md)。
- **无条件，而非条件.** (c)(ii) 是对 audit 结果取平均。把终值条件在"所有 audit 都成功"上，一般是不对的，因为后面的 audit 事件依赖于到达的状态；精确实例见[实验页](../experiments/08_verification_ZH.md)。
- **停时版本.** 若 $`\tau`$ 是第一次失败的 audit（截断到 $`R`$），则 $`L_E(\mathrm{adaptive})-L_E(0)\ge\mathbb E\sum_{t<\tau}\Gamma_t-\delta`$。
- **去掉了什么，没去掉什么.** 在实际状态上做 audit，去掉了 [Theorem 8.2](#s8-2) 的 transport 项，但没有去掉恢复状态、完整尾段的代价与独立性要求。把胜出的 audit 副本直接当作更新，或者用短尾段做 audit，都会改变被估计的量；短尾段需要一个显式的 surrogate 余量，这正是 [Theorem 8.8](08_certificate_ZH.md#s8-8) 提供的。
- **保真不等于改进.** (a) 只继承参照测得的提升 $`\gamma`$。对一个无效参照的高保真复制，并不能改进初始学生。
