# N12 · 优化器的 warmth 决定一阶反馈统计量 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/N12_optimizer_warmth_ZH.md)

## 设定与假设

**状态与更新。** 带 bias correction 的 Adam/AdamW 按 [N11(a)](N11_adam_cold_start_ZH.md) 更新（节点 [N11](N11_adam_cold_start_ZH.md)），超参数满足 $`\beta_1\in[0,1)`$、$`\beta_2\in(0,1)`$、$`\eta>0`$、$`\lambda\ge0`$、$`\epsilon\ge0`$。考虑从完整学习器状态 $`z=(\theta,m,v,t-1)`$ 出发的第 $`t\ge1`$ 轮。有限集 $`Y`$ 中的反馈消息 $`y`$ 产生消耗梯度 $`G(y)\in\mathbb R^p`$（累积、归一化和 clipping 之后的梯度）。逐坐标地，

```math
n_i(y)=\frac{\beta_1m_i+(1-\beta_1)G_i(y)}{1-\beta_1^t},\qquad d_i(y)=\sqrt{\frac{\beta_2v_i+(1-\beta_2)G_i(y)^2}{1-\beta_2^t}},
```

```math
\theta_i^+(y)=(1-\eta\lambda)\theta_i-\eta\frac{n_i(y)}{d_i(y)+\epsilon}.
```

下一个完整状态为 $`z^+(y)=(\theta^+(y),m^+(y),v^+(y),t)`$，其中 $`m^+=\beta_1m+(1-\beta_1)G`$，$`v^+=\beta_2v+(1-\beta_2)G^{\odot2}`$。

**定义（warmth）。** 假设 $`v_i>0`$；若某个 $`v_i=0`$，则假设 $`\epsilon>0`$。令

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^t}},\qquad\kappa_i(y)=\frac{(1-\beta_2)G_i(y)^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(y)=\infty\ \ (v_i=0),
```

```math
u_i(y)=\frac{\lvert n_i(y)\rvert}{a_i+\epsilon}.
```

当 $`v_i>0`$ 时，$`d_i(y)=a_i\sqrt{1+\kappa_i(y)}`$，且 $`v_i^+(y)=\beta_2v_i\,(1+\kappa_i(y))`$。也就是说，$`\kappa_i`$ 是新梯度对已有二阶矩造成的相对扰动。

**定义（冻结分母的替代更新）。** 令 $`P=\mathrm{diag}(1/(a_i+\epsilon))`$，

```math
\tilde\theta^+(y)=(1-\eta\lambda)\theta-\eta P\,n(y),\qquad\tilde z^+(y)=(\tilde\theta^+(y),\,m^+(y),\,\beta_2v,\,t).
```

$`\tilde z^+`$ 是 $`G(y)`$ 的仿射函数：它相当于一步带固定对角 preconditioner $`P`$ 的 momentum SGD，外加一个忽略新梯度的二阶矩。对 $`g\in\mathbb R^p`$，记 $`\tilde z^+(g)`$ 为同一映射在消耗梯度 $`g`$ 处的取值。

**Continuation 与 response。** $`V`$ 把完整状态映到 $`[0,1]`$，代表声明的参考学习后缀加上部署，后缀的随机性已经积分进去（节点 [N05](N05_continuation_response_ZH.md)）。消息 $`y`$ 的单边界 response 为 $`\psi(y)=V(z^+(y))`$，即更新随机性包含在 $`V`$ 内的 [N05](N05_continuation_response_ZH.md)。定义 $`F(g)=V(\tilde z^+(g))`$。

**(W1) 的假设。** 下面所说的状态，都指由 $`z`$ 经 $`Y`$ 中消息生成的状态；步计数固定为 $`t`$。

- **(W)** 对每个坐标 $`v_i>0`$（warm 状态）。
- **(L)** 对所有这样的 $`m,v`$：在 $`(m,v)=(m^+(y),v^+(y))`$ 处、对点对 $`\big(\theta^+(y),\tilde\theta^+(y)\big)`$，有 $`\lvert V(\vartheta,m,v)-V(\vartheta',m,v)\rvert\le\sum_iL_i\lvert\vartheta_i-\vartheta_i'\rvert`$。
- **(S)** 存在 $`L_v\ge0`$ 和上限 $`\bar k\in(0,\infty]`$，使得在 $`(\vartheta,m)=(\tilde\theta^+(y),m^+(y))`$ 处、对 $`k=\kappa(y)`$，有 $`\lvert V(\vartheta,m,\beta_2v\odot(1+k))-V(\vartheta,m,\beta_2v)\rvert\le L_v\max_i\min\lbrace k_i,\bar k\rbrace`$。由于 $`V\in[0,1]`$，只要 $`L_v\bar k\ge1`$，(S) 总成立；这个假设的实质在于 $`L_v`$ 很小。
- **(H)** $`F`$ 在 $`\lbrace 0\rbrace \cup\lbrace G(y):y\in Y\rbrace`$ 的凸包 $`\mathcal C`$ 上二阶连续可微，并且对所有 $`g\in\mathcal C`$、$`s\in\mathbb R^p`$ 有 $`\lvert s^\top\nabla^2F(g)\,s\rvert\le H_G\lVert s\rVert_\infty^2`$。

**Warm-up。** 长度为 $`W`$ 的 *moment warm-up* 是指：用声明数据的梯度，以严格为 $`0`$ 的学习率做 $`W`$ 步优化器更新。它不改变 $`\theta`$，只推进 $`m`$、$`v`$ 和步计数。凡是被吸收进 $`v`$ 的梯度，无论来自 warm-up 还是真实更新，都算一次*梯度评估*。

## 命题

**(W0)** **引理。** 对每个坐标，

```math
\lvert\theta_i^+(y)-\tilde\theta_i^+(y)\rvert\le\eta\,u_i(y)\,\min\lbrace \kappa_i(y)/2,\;1\rbrace .
```

**(W1)** **Proposition（warm Adam 在消耗梯度上一阶仿射）。** 在 (W)、(L)、(S)、(H) 下，令 $`c=F(0)`$、$`b=\nabla F(0)`$，两者都与 $`y`$ 无关。则对每个 $`y\in Y`$，

```math
\big\lvert\psi(y)-c-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i(y)\min\lbrace \tfrac{\kappa_i(y)}2,1\rbrace +L_v\max_i\min\lbrace \kappa_i(y),\bar k\rbrace +\tfrac12H_G\lVert G(y)\rVert_\infty^2=:r(y).
```

因此，对 $`Y`$ 上任意两个消息分布 $`P_0,P_1`$，

```math
\big\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\big\rvert\le\min\Big\lbrace 1,\;\big\lvert b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)\big\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r\Big\rbrace .
```

特别地，消耗梯度均值相同的两个分布，response 之差不超过 $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$。若 $`V`$ 对 $`(\theta,m)`$ 可微，则

```math
b=(1-\beta_1)\Big[-\frac{\eta}{1-\beta_1^t}P\,\nabla_\theta V+\nabla_mV\Big]\Big|_{\tilde z^+(0)}.
```

所以一阶统计量就是经过 preconditioner 的消耗梯度均值。

**(W2)** **Proposition（冷启动在一阶上区分分布）。** 取冷状态（$`m=v=0`$，$`t=1`$）、理想符号映射（$`\epsilon\to0`$，节点 [N11](N11_adam_cold_start_ZH.md)）、标量参数，以及即时效用 $`V(\theta)=\tfrac12+w\theta`$（限制在 $`V\in[0,1]`$ 的邻域内）。

- (a) *均值相同。* 令 $`P_0(G=1)=\tfrac14`$、$`P_0(G=-\tfrac13)=\tfrac34`$，$`P_1(G=0)=1`$。两者均值都是 $`0`$，而 $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$。
- (b) *任意固定阶以下的矩都相同。* 对每个 $`N\ge1`$，取 [N07(c)](N07_affine_moment_order_ZH.md) 中 $`\lbrace i/N\rbrace`$ 上的分布 $`P_N,Q_N`$（节点 [N07](N07_affine_moment_order_ZH.md)），并令 $`G=q-Z`$，$`q\in(0,1/N)`$。这两个梯度分布在所有低于 $`N`$ 阶的矩上一致，而 $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$。

两种情形下的分离都是 $`\eta`$ 的一阶量。在同样效用的 warm 状态下，W1 把均值相同分布之间的差距控制在 $`\eta\lvert w\rvert(\mathbb E_{P_0}u\min\lbrace \kappa/2,1\rbrace +\mathbb E_{P_1}u\min\lbrace \kappa/2,1\rbrace )`$ 以内。

**(W3)** **Corollary（冷启动窗口）。**

- (a) *平均场 warmth。* 设 $`v`$ 从 $`0`$ 开始，已吸收 $`s\ge1`$ 次梯度评估，且这些梯度各坐标的二阶矩都是 $`\sigma_i^2>0`$。则 $`\mathbb E v_i=(1-\beta_2^s)\sigma_i^2`$，平均场 warmth $`\bar\kappa_i(s):=(1-\beta_2)\sigma_i^2/(\beta_2\,\mathbb E v_i)`$ 满足

  ```math
  \frac1{\beta_2\,s}\le\bar\kappa_i(s)\le\frac1{\beta_2^{\,s}\,s}.
  ```

  特别地，当 $`s\le(1-\beta_2)^{-1}`$ 时 $`\bar\kappa_i(s)\le e^{1/\beta_2}/s`$。做了 $`W`$ 次 warm-up 评估后，第 $`s`$ 次真实更新之前共有 $`W+s-1`$ 次评估。
- (b) *累积。* 通过 [N05(a)–(c)](N05_continuation_response_ZH.md) 的 telescoping 恒等式（节点 [N05](N05_continuation_response_ZH.md)），在 $`R`$ 次真实更新上比较参考数据源与备选数据源。设 $`\mathcal S_{\rm warm}\subseteq\lbrace 1,\dots,R\rbrace`$ 为满足以下条件的更新：在每个到达的状态上，(W)、(L)、(S)、(H) 成立（常数以 $`s`$ 为下标），两个分布的消耗梯度均值相同，并且对某个常数 $`c>0`$（与 W1 中的 $`c=F(0)`$ 无关）几乎必然有 $`\max_i\kappa_i\le c/(W+s)`$。则

  ```math
  \lvert L(0)-L(S)\rvert\le\lvert\lbrace s\notin\mathcal S_{\rm warm}\rbrace\rvert+\bar A\sum_{s\in\mathcal S_{\rm warm}}\min\Big\lbrace 1,\frac{c}{W+s}\Big\rbrace +\sum_{s\in\mathcal S_{\rm warm}}\bar B_s,
  ```

  其中 $`\bar A=\sup_s2\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)`$，$`\bar B_s=\sup H_{G,s}\lVert G\rVert_\infty^2`$，$`\bar u_{i,s}`$ 是 $`u_i`$ 的上界。
  - 有 $`W\ge1`$ 次 warm-up 评估时，通常 $`\mathcal S_{\rm warm}=\lbrace 1,\dots,R\rbrace`$，受 warmth 控制的和不超过 $`c\log(1+R/W)`$。
  - $`W=0`$ 时，第一次真实更新从 $`v=0`$ 出发，违反 (W)。它就是节点 [N11](N11_adam_cold_start_ZH.md) 中的冷启动一步，贡献至多为 $`1`$；由 W2，这一贡献可以是 $`\eta`$ 的一阶量。若 $`\mathcal S_{\rm warm}=\lbrace 2,\dots,R\rbrace`$，受 warmth 控制的和不超过 $`c\log R`$。

*哪些是精确的，哪些是近似的。* (a) 对 $`v`$ 的*均值*是精确的。(b) 在其假设（$`\mathcal S_{\rm warm}`$ 上 $`\max_i\kappa_i\le c/(W+s)`$）下是精确的。从 (a) 走到这个假设，需要把实际的 $`v_i`$ 换成它的均值，并把 $`G_i^2/\sigma_i^2`$ 当作有界常数。这一步是近似，依赖于指数滑动平均的集中性和梯度尺度的平稳性。检验 E8-W3 在实际训练轨迹上检验了这一步。

## 证明

**(W0)**

- *情形 $`v_i>0`$。* $`\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\lvert n_i\rvert\,\lvert\tfrac1{a_i\sqrt{1+\kappa_i}+\epsilon}-\tfrac1{a_i+\epsilon}\rvert`$。通分得

  ```math
  \lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\,u_i\cdot\frac{a_i(\sqrt{1+\kappa_i}-1)}{a_i\sqrt{1+\kappa_i}+\epsilon}.
  ```

  由于 $`\epsilon\ge0`$，最后一个因子不超过 $`(\sqrt{1+\kappa_i}-1)/\sqrt{1+\kappa_i}=1-(1+\kappa_i)^{-1/2}`$。它小于 $`1`$；又因为 $`\sqrt{1+\kappa}-1\le\kappa/2`$ 且 $`\sqrt{1+\kappa}\ge1`$，它也小于 $`\kappa_i/2`$。所以该因子不超过 $`\min\lbrace \kappa_i/2,1\rbrace`$。
- *情形 $`v_i=0`$ 且 $`\epsilon>0`$。* 此时 $`a_i=0`$、$`\kappa_i=\infty`$、$`u_i=\lvert n_i\rvert/\epsilon`$。两个倒数 $`1/(d_i+\epsilon)`$ 和 $`1/\epsilon`$ 都在 $`[0,1/\epsilon]`$ 内，差不超过 $`1/\epsilon`$，所以左边不超过 $`\eta u_i=\eta u_i\min\lbrace \infty,1\rbrace`$。$`\square`$

**(W1) 的证明。** 固定 $`y`$ 并在记号中略去。作分解

```math
\psi-c-b^\top G=\underbrace{\big[V(z^+)-V(\tilde\theta^+,m^+,v^+)\big]}_{(i)}+\underbrace{\big[V(\tilde\theta^+,m^+,v^+)-V(\tilde\theta^+,m^+,\beta_2v)\big]}_{(ii)}+\underbrace{\big[F(G)-F(0)-\nabla F(0)^\top G\big]}_{(iii)}.
```

这个恒等式成立，是因为 $`V(\tilde\theta^+,m^+,\beta_2v)=V(\tilde z^+(G))=F(G)`$。

- *(i)* 由 (L) 和 (W0)，$`\lvert(i)\rvert\le\sum_iL_i\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\sum_iL_iu_i\min\lbrace \kappa_i/2,1\rbrace`$。
- *(ii)* 由 (W)，$`v^+=\beta_2v\odot(1+\kappa)`$，其中 $`\kappa\ge0`$ 有限。在 $`(\tilde\theta^+,m^+)`$ 处应用 (S)，得 $`\lvert(ii)\rvert\le L_v\max_i\min\lbrace \kappa_i,\bar k\rbrace`$。
- *(iii)* 线段 $`\lbrace \tau G:\tau\in[0,1]\rbrace`$ 落在 $`\mathcal C`$ 内。带积分余项的 Taylor 定理给出

  ```math
  F(G)-F(0)-\nabla F(0)^\top G=\int_0^1(1-\tau)\,G^\top\nabla^2F(\tau G)\,G\,d\tau.
  ```

  由 (H)，被积函数的绝对值不超过 $`(1-\tau)H_G\lVert G\rVert_\infty^2`$，因此 $`\lvert(iii)\rvert\le\tfrac12H_G\lVert G\rVert_\infty^2`$。

三项相加即得逐点结论。

两个分布的情形：写 $`\psi=c+b^\top G+e`$，$`\lvert e\rvert\le r`$，则

```math
\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi=b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)+\mathbb E_{P_0}e-\mathbb E_{P_1}e.
```

常数 $`c`$ 抵消，因为两者都是概率测度；$`Y`$ 有限，所有期望都存在。由三角不等式得到界，再由 $`\psi\in[0,1]`$ 得到上限 $`1`$。

关于 $`b`$ 的公式：用链式法则，其中 $`\partial\tilde\theta^+/\partial g=-\eta(1-\beta_1)P/(1-\beta_1^t)`$，$`\partial m^+/\partial g=(1-\beta_1)I`$，$`v`$ 分量与 $`g`$ 无关。$`\square`$

**(W2)**

*(a)* 由 [N11(a)](N11_adam_cold_start_ZH.md) 及 $`h_0=\mathrm{sign}`$，$`\theta_1=\theta_0-\eta\mathrm{sign}(G)`$。在 $`P_0`$ 下 $`\mathbb E\mathrm{sign}G=\tfrac14-\tfrac34=-\tfrac12`$，故 $`\mathbb E\theta_1=\theta_0+\eta/2`$；在 $`P_1`$ 下 $`\mathrm{sign}0=0`$，故 $`\mathbb E\theta_1=\theta_0`$。效用是仿射的，response 之差为 $`\lvert w\rvert\eta/2`$。两者均值相同：$`\tfrac14\cdot1+\tfrac34\cdot(-\tfrac13)=0`$。

*(b)* $`P_N`$ 支撑在偶数 $`i`$ 上，$`Q_N`$ 支撑在奇数 $`i`$ 上，二者所有 $`\ell<N`$ 阶矩相同（[N07(c)](N07_affine_moment_order_ZH.md)），所以仿射像 $`G=q-Z`$ 的低于 $`N`$ 阶的矩也相同。取 $`q\in(0,1/N)`$：

- 在 $`P_N`$ 下，$`G>0`$ 当且仅当 $`i=0`$，概率为 $`\binom N0/2^{N-1}=2^{1-N}`$；其余情形 $`Z\ge2/N>q`$，$`G<0`$。故 $`\mathbb E_{P_N}\mathrm{sign}G=2^{2-N}-1`$。
- 在 $`Q_N`$ 下总有 $`Z\ge1/N>q`$，故 $`\mathbb E_{Q_N}\mathrm{sign}G=-1`$。

两者第一步的期望相差 $`\eta\,2^{2-N}`$，仿射效用给出差距 $`\lvert w\rvert\eta2^{2-N}`$。

*warm 状态下的对应结论。* 对没有后缀的 $`V(\theta)=\tfrac12+w\theta`$，$`F`$ 关于 $`g`$ 是仿射的，所以 $`H_G=0`$；$`V`$ 不依赖于 $`(m,v)`$，所以 $`L_v=0`$；又 $`L=\lvert w\rvert`$。均值相同时，由 W1 即得所示的界。$`\square`$

**(W3)**

*(a)* $`v_i=(1-\beta_2)\sum_{r=1}^s\beta_2^{s-r}(G_i^{(r)})^2`$，故 $`\mathbb E v_i=(1-\beta_2)\sigma_i^2\sum_{r=0}^{s-1}\beta_2^r=(1-\beta_2^s)\sigma_i^2`$，从而 $`\bar\kappa_i(s)=(1-\beta_2)/(\beta_2(1-\beta_2^s))`$。由 $`1-\beta^s=(1-\beta)\sum_{r=0}^{s-1}\beta^r`$ 和 $`\beta^{s-1}\le\beta^r\le1`$，有 $`(1-\beta)s\beta^{s-1}\le1-\beta^s\le(1-\beta)s`$。代入即得 $`1/(\beta_2s)\le\bar\kappa_i\le1/(\beta_2^ss)`$。最后一点：由 $`\log\beta\ge-(1-\beta)/\beta`$，当 $`s\le1/(1-\beta)`$ 时 $`\beta^{s}\ge\beta^{1/(1-\beta)}\ge e^{-1/\beta}`$。

*(b)* [N05(a)–(c)](N05_continuation_response_ZH.md) 给出

```math
L(0)-L(S)=\sum_{s=1}^R\mathbb E_{d_s^S}\big[(K_s^0-K_s^S)V_{s+1}^0(Z_s)\big],
```

并且在每个到达的状态上，$`(K_s^0-K_s^S)V_{s+1}^0=\mathbb E_{P_0}\psi_s-\mathbb E_{P_1}\psi_s`$，是两个取值在 $`[0,1]`$ 的期望之差。

- 对 $`s\notin\mathcal S_{\rm warm}`$，其绝对值以 $`1`$ 为界。
- 对 $`s\in\mathcal S_{\rm warm}`$，均值相同时由 W1，它不超过 $`\mathbb E_{P_0}r_s+\mathbb E_{P_1}r_s`$。利用 $`\min\lbrace \kappa_i/2,1\rbrace \le\min\lbrace \kappa_{\max},1\rbrace`$，以及 $`\min\lbrace \kappa,\bar k\rbrace \le\max\lbrace 1,\bar k\rbrace \min\lbrace 1,\kappa\rbrace`$（$`\kappa\le1`$ 与 $`\kappa>1`$ 时都成立），$`r_s`$ 的前两项不超过 $`\big(\eta\sum_iL_{i,s}\bar u_{i,s}+L_{v,s}\max\lbrace 1,\bar k\rbrace \big)\min\lbrace 1,c/(W+s)\rbrace`$，曲率项不超过 $`\tfrac12\bar B_s`$。对两个分布求和，得到因子 $`2`$。

再对 $`s`$ 求和即得结论。

求和的估计：若 $`W\ge1`$，则 $`\sum_{s=1}^Rc/(W+s)\le c\int_W^{W+R}dx/x=c\log(1+R/W)`$。若 $`W=0`$ 且 $`s`$ 从 $`2`$ 开始，则 $`\sum_{s=2}^Rc/s\le c\int_1^Rdx/x=c\log R`$。$`\square`$

## 注记

1. **切换。** 反馈源需要匹配的统计量取决于优化器状态：
   - 冷启动的 Adam 状态下，要匹配消耗梯度的联合符号分布，以及保存在 $`m`$ 中的幅度（节点 [N11](N11_adam_cold_start_ZH.md)）；
   - warm 的 Adam 状态（$`\kappa\ll1`$）下，要匹配经 preconditioner 的消耗梯度均值，误差为 $`O(\eta\,u\,\kappa+L_v\kappa+H_G\lVert G\rVert^2)`$；
   - SGD 下，要匹配消耗梯度均值，误差为 $`O(\eta^2)`$（[N10(c)](N10_update_composition_ZH.md)）。
2. **残差的阶。** 对光滑的后缀，$`F`$ 通过 $`\tilde\theta^+`$（被 $`\eta`$ 缩放）和 $`m^+`$（在后续更新中被 $`\eta`$ 缩放）依赖于 $`g`$，因此预期 $`H_G=O(\eta^2)`$。这是对后缀的预期，不是 (H) 的一部分。检验 E8-W1 中测得的 $`H_G`$ 在 $`\eta=0.01`$ 到 $`0.1`$ 之间大致按 $`\eta^2`$ 增长。
3. **warmth 消除不了的部分。** 曲率项 $`\tfrac12H_G\lVert G\rVert_\infty^2`$ 与 $`\kappa`$ 无关。在 (W3)(b) 中，它随 $`R`$ 线性累积，而受 warmth 控制的部分只按对数累积。所以在长视野下，均值相同的数据源之间剩下的差异由曲率（轨迹的非线性）决定，SGD 也有这部分。warm-up 只能控制冷启动窗口。
4. **消息与梯度。** W1 是关于*消耗*梯度的仿射性。当 $`G(y)`$ 本身对 $`y`$ 非线性时，response 对消息就不是仿射的。例子有：序列级 teacher forcing（前缀会改变 Jacobian）和处于激活状态的全局范数 clipping。
5. **局部性。** W1 是单边界的结论，它的常数是 continuation value 的性质。要在整个训练过程中给出证书，还需要 [N14(b)](N14_frozen_audit_transport_ZH.md) 中的 transport 和 coverage 项（节点 [N14](N14_frozen_audit_transport_ZH.md)）。
6. **前人工作。** bias-corrected 的第一步和把 Adam 看作符号下降都是已知的：Kingma 和 Ba（2015），Balles 和 Hennig（2018），以及 Liu 等人关于早期步方差的工作（RAdam，ICLR 2020）。这里的贡献是一个信息层面的结论：反馈分布必须控制的一阶统计量如何随优化器 warmth 改变，并给出显式的残差。

## 依赖与出处

- 依赖：[N05](N05_continuation_response_ZH.md)（response [N05](N05_continuation_response_ZH.md)/[N05](N05_continuation_response_ZH.md)，telescoping 恒等式 [N05(a)](N05_continuation_response_ZH.md)），[N07](N07_affine_moment_order_ZH.md)（分布 [N07(c)](N07_affine_moment_order_ZH.md)），[N10](N10_update_composition_ZH.md)（SGD 展开 [N10(c)](N10_update_composition_ZH.md)），[N11](N11_adam_cold_start_ZH.md)（冷启动映射 [N11(a)](N11_adam_cold_start_ZH.md)）。
- 被使用于：[N17](N17_ced_method_ZH.md)。

本页把以下四点写明了：
  - W1 中的假设 (W)；
  - W0 中 $`v_i=0`$ 的情形；
  - W2(b) 中的矩匹配分布族；
  - W3 中精确的平均场界，以及哪些是精确、哪些是近似。

**出处：** [warm-Adam §2](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md#2-warm-one-step-approximation); [warm-Adam §3](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md#3-cold-start-separates-equal-mean-feedback-at-first-order); [warm-Adam §4](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md#4-how-warmth-evolves-and-a-free-way-to-warm-the-state) — Lemma W0; Props. W1–W2; Cor. W3. [Warm_Adam_Response_EN.md](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md)（Lemma W0，Props. W1–W2，Cor. W3）。
