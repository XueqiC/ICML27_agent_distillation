# Corollary 7.2 · 平稳情形的幅度定律

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-02_stationary_amplitude_law_ZH.md) · [证明](07-02_stationary_amplitude_law_ZH.md) · [实验](../experiments/07-02_stationary_amplitude_law_ZH.md) · [English](07-02_stationary_amplitude_law.md)

**本页证明什么.** 在 Adam 的 warm、平稳、线性 response 区间里，一条幅度为 $`\lambda`$（以 running RMS 为单位）的消息造成的最终结果变化是一个显式的有限和。增量小时，它是 $`\lambda`$ 的凹二次函数，带显式误差界，有最优幅度 $`\lambda^\star`$，并在 $`2\lambda^\star`$ 附近变号；幅度很大时它有显式极限，极限的符号决定巨大的消息是否有害。证明把尾段梯度的矩代入 Adam 递推，对得到的各步求和，再对 $`(1+x)^{-1/2}`$ 的 Taylor 余项给出界。

**本页用到的设定与记号.**

- *单个坐标 $`i`$ 上不做 bias correction 的 Adam*：$`m'=\beta_1m+(1-\beta_1)g`$，$`v'=\beta_2v+(1-\beta_2)g^2`$，参数步为 $`-\eta m'/\sqrt{v'}`$，$`\epsilon=0`$，无 weight decay。
- *消息与尾段.* 第 $`t`$ 轮有一条 consumed gradient 为 $`g_i`$ 的消息，之后是 $`T=R-1-t`$ 次尾段更新；尾段梯度 $`g_{k,i}`$ 与 $`\theta`$ 无关，平稳，均值 $`\bar g_i\ne0`$、平方的均值 $`\bar v_i>0`$。最终得分 $`F`$ 关于 $`\theta_i`$ 仿射，导数为 $`(\nabla F)_i`$。
- *warm 状态.* 消息到达时 $`m_{t,i}=\bar g_i`$，$`v_{t,i}=\bar v_i`$。
- *归一化的量.* $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$；$`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$；$`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}\in(0,1]`$；$`b_j=(1-\beta_1)\beta_1^{\,j}`$，$`c_j=(1-\beta_2)\beta_2^{\,j}`$，$`j=0,\dots,T`$；$`\omega_T=1-\beta_2^{\,T+1}=\sum_{j\le T}c_j`$。
- *mean-field response* $`\Delta F_i(\lambda)`$：用幅度为 $`\lambda`$ 的消息替换一条零梯度消息时 $`F`$ 的变化，其中每一步的 $`m_k/\sqrt{v_k}`$ 换成 $`\mathbb Em_k/\sqrt{\mathbb Ev_k}`$。
- *一阶价值* $`\mathrm{Val}_i`$ 与线性 response 下的 costate $`\chi^m,\chi^v`$ 来自 [Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1)(d)。
- $`\kappa_{\rm iso}=(1-\beta_1)/\big(\sqrt{1-\beta_2}\,(1-\beta_1/\sqrt{\beta_2})\big)`$ 是 [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)(c) 的孤立访问常数。

**陈述（[Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)）.** 在上述设定下：

(a) 对任意 $`\lambda`$，

```math
\Delta F_i(\lambda)=\alpha_i\eta\sum_{j=0}^{T}\Big[\frac{\rho_i+b_j(\lambda-\rho_i)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho_i(1-b_j)}{\sqrt{1-c_j}}\Big];
```

(b) 令 $`Q_i(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho_i\omega_T\lambda^2`$，$`S_i(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho_i\omega_T\lambda^2`$；若 $`\beta_2\ge\tfrac12`$，则对任意 $`\lambda`$

```math
\big\lvert\Delta F_i(\lambda)-\alpha_i\eta\,Q_i(\lambda)\big\rvert\le11\,(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\;\lvert\alpha_i\rvert\eta\,S_i(\lambda),
```

且 $`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$ 满足同样的界；$`Q_i`$ 是凹的，在 $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$ 处最大；令 $`u=11(1-\beta_2)\max\lbrace1,9\lambda^{\star2}\rbrace`$，若 $`\alpha_i>0`$ 且 $`u\le1/8`$，则 $`\Delta F_i`$ 在 $`[\lambda^\star,2\lambda^\star(1-4u)]`$ 上为正、在 $`[2\lambda^\star(1+4u),3\lambda^\star]`$ 上为负；阈值 $`2\lambda^\star`$ 只在 $`(1-\beta_2)(2\lambda^\star)^2\ll1`$ 时可用，剩余步数短与长时的形式如陈述页所列；

(c) 当 $`\lambda\to\infty`$ 时，

```math
\Delta F_i(\lambda)\to\alpha_i\eta\big[\kappa_{\rm iso}(T)-\rho_i\,\Lambda(T)\big],\qquad \kappa_{\rm iso}(T)=\sum_{j=0}^{T}\frac{b_j}{\sqrt{c_j}},\qquad \Lambda(T)=\sum_{j=0}^{T}\frac{1-b_j}{\sqrt{1-c_j}},
```

所以对 $`\alpha_i>0`$：若 $`\rho_i\Lambda(T)>\kappa_{\rm iso}(T)`$，所有足够大的同向消息都劣于零梯度消息（且 $`\Delta F_i`$ 有正零点）；若 $`\rho_i\Lambda(T)<\kappa_{\rm iso}(T)`$，则都更好；$`\Lambda(T)\approx T+1`$，且 $`\beta_1<\sqrt{\beta_2}`$ 时 $`\kappa_{\rm iso}(T)\to\kappa_{\rm iso}`$；

(d) SGD 下 $`\Delta F_i(\lambda)=\alpha_i\eta\sqrt{\bar v_i}\,\lambda`$。

**证明.**

*(a)* 在 mean-field 替换下，尾段梯度被换成它们的矩，所以消息造成的偏差按几何速度衰减。对 $`j\ge0`$，

```math
\mathbb Em_{t+j,i}=\bar g_i+b_j(g_i-\bar g_i),\qquad \mathbb Ev_{t+j,i}=\bar v_i\big(1+c_j(g_i^2/\bar v_i-1)\big),
```

这里下标 $`t+j`$ 表示消息之后第 $`j`$ 次更新所用的矩（$`j=0`$ 即消息那次更新）。$`j=0`$ 是从 $`m_t=\bar g`$、$`v_t=\bar v`$ 出发的消息更新。尾段梯度的均值为 $`\bar g_i`$、平方的均值为 $`\bar v_i`$，所以此后每次更新把 $`\mathbb Em`$ 的偏差乘以 $`\beta_1`$，把 $`\mathbb Ev`$ 的偏差乘以 $`\beta_2`$。$`F`$ 关于 $`\theta_i`$ 仿射，所以结果的变化为 $`-(\nabla F)_i\,\eta\sum_{j=0}^{T}\big[\mathbb Em_{t+j}(g)/\sqrt{\mathbb Ev_{t+j}(g)}-\mathbb Em_{t+j}(0)/\sqrt{\mathbb Ev_{t+j}(0)}\big]`$。

分子分母同除以 $`\sqrt{\bar v_i}`$，并代入 $`\bar g_i/\sqrt{\bar v_i}=\mathrm{sign}(\bar g_i)\rho_i`$、$`g_i/\sqrt{\bar v_i}=\mathrm{sign}(\bar g_i)\lambda`$ 与 $`-(\nabla F)_i\,\mathrm{sign}(\bar g_i)=\alpha_i`$ 即得。由于 $`c_j\le1-\beta_2<1`$，所有分母都为正。

*(b)* 记 $`\gamma=1-\beta_2\le\tfrac12`$，$`f_j(x)=(1+c_j(x-1))^{-1/2}`$。(a) 中的方括号等于 $`\rho_i(1-b_j)\big(f_j(\lambda^2)-f_j(0)\big)+b_j\lambda f_j(\lambda^2)`$，所以

```math
\frac{\Delta F_i}{\alpha_i\eta}-Q_i=\underbrace{\lambda\sum_jb_j\big(f_j(\lambda^2)-1\big)}_{E_1}+\underbrace{\rho_i\sum_j(1-b_j)\big(f_j(\lambda^2)-f_j(0)+\tfrac12c_j\lambda^2\big)}_{E_2}+\underbrace{\tfrac12\rho_i\lambda^2\sum_jb_jc_j}_{E_3}.
```

对 $`w\ge-\gamma\ge-\tfrac12`$，有两个初等不等式：

- $`\lvert(1+w)^{-1/2}-1\rvert\le\tfrac12(1-\gamma)^{-3/2}\lvert w\rvert\le1.42\lvert w\rvert`$；
- $`\lvert1-(1+w)^{-3/2}\rvert\le\tfrac32(1-\gamma)^{-5/2}\lvert w\rvert\le8.49\lvert w\rvert`$。

由 $`1+w\ge1-\gamma`$ 和中值定理即得两式。这里的自变量是 $`w=c_j(x-1)`$，$`x\in[0,\lambda^2]`$，所以 $`\lvert w\rvert\le c_j\max\lbrace1,\lambda^2\rbrace`$。另外 $`\sum_jb_jc_j\le\gamma\sum_jb_j=\gamma(1-\beta_1^{\,T+1})`$，$`\sum_jc_j^2\le\gamma\omega_T`$。

- $`\lvert E_1\rvert\le1.42\,\gamma\max\lbrace1,\lambda^2\rbrace\,\lvert\lambda\rvert(1-\beta_1^{\,T+1})`$。
- $`f_j(x)-f_j(0)+\tfrac12c_jx=\int_0^x\tfrac12c_j\big(1-(1+c_j(y-1))^{-3/2}\big)\mathrm dy`$，所以 $`\lvert E_2\rvert\le\rho_i\sum_j\tfrac12c_j\lambda^2\cdot8.49\,c_j\max\lbrace1,\lambda^2\rbrace\le8.49\,\gamma\max\lbrace1,\lambda^2\rbrace\cdot\tfrac12\rho_i\omega_T\lambda^2`$。
- $`\lvert E_3\rvert\le\tfrac12\rho_i\lambda^2\gamma(1-\beta_1^{\,T+1})\le\tfrac12\gamma\max\lbrace1,\lambda^2\rbrace\lvert\lambda\rvert(1-\beta_1^{\,T+1})`$，这里用了 $`\rho_i\le1`$ 与 $`\lvert\lambda\rvert\le\max\lbrace1,\lambda^2\rbrace`$。

三项相加，$`\lvert\Delta F_i/(\alpha_i\eta)-Q_i\rvert\le(1.42+8.49+0.5)\,\gamma\max\lbrace1,\lambda^2\rbrace S_i(\lambda)`$。

一阶价值方面，[Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1)(d)（线性 response 下的闭式）在消息前状态上的 mean-field costate 为 $`\chi^m=-\eta(\nabla F)_i\sum_{j=1}^T\beta_1^{\,j}/\sqrt{\bar v_i}`$ 与 $`\chi^v=\tfrac12\eta(\nabla F)_i\bar g_i\sum_{j=1}^T\beta_2^{\,j}/\bar v_i^{3/2}`$（此时 $`D_1=\bar v_i^{-1/2}`$，$`D_2=\bar g_i/(2\bar v_i^{3/2})`$）。归一化后，$`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$ 等于 (a) 中 $`j=0`$ 的精确方括号加上 $`\sum_{j=1}^T\big(b_j\lambda-\tfrac12\rho_ic_j\lambda^2\big)`$。它与 $`Q_i`$ 之差是 $`E_1+E_2+E_3`$ 中 $`j=0`$ 的部分，满足同样的界。

各条推论：

- $`Q_i''=-\rho_i\omega_T<0`$，$`Q_i'(\lambda^\star)=0`$。
- 变号点：设 $`\alpha_i>0`$，$`\lambda=2\lambda^\star(1+\delta)`$，$`\lvert\delta\rvert\le\tfrac12`$。此时 $`\lambda^\star\le\lambda\le3\lambda^\star`$，误差不超过 $`u\,S_i(\lambda)`$；并且 $`Q_i(\lambda)=-(1-\beta_1^{\,T+1})\lambda\delta`$，$`S_i(\lambda)=(1-\beta_1^{\,T+1})\lambda(2+\delta)`$。
  - 当 $`\delta\in[4u,\tfrac12]`$：$`\Delta F_i/(\alpha_i\eta)\le(1-\beta_1^{\,T+1})\lambda\,[-\delta+u(2+\delta)]<0`$，因为 $`u<\tfrac12`$ 时 $`\delta(1-u)\ge4u(1-u)>2u`$。
  - 当 $`\delta\in[-\tfrac12,-4u]`$：$`\Delta F_i/(\alpha_i\eta)\ge(1-\beta_1^{\,T+1})\lambda\,[\lvert\delta\rvert-u(2-\lvert\delta\rvert)]>0`$，因为 $`\lvert\delta\rvert(1+u)>2u`$。
  - $`u\le1/8`$ 保证 $`4u\le\tfrac12`$，两个区间都非空。
- 关于剩余步数的两条由 $`\omega_T=1-(1-\gamma)^{T+1}`$ 得到。

*(c)* 当 $`\lambda\to\infty`$ 时，第 $`j`$ 项的第一个分式与 $`b_j\lambda/(\sqrt{c_j}\lambda)`$ 等价，趋于 $`b_j/\sqrt{c_j}`$（$`b_j=0`$ 时它趋于 $`0=b_j/\sqrt{c_j}`$）。第二个分式与 $`\lambda`$ 无关。由此得到极限。

- 若极限为负且 $`\alpha_i>0`$，则对所有足够大的 $`\lambda`$ 有 $`\Delta F_i(\lambda)<0`$。又 $`\Delta F_i(0)=0`$，$`\Delta F_i'(0)=\alpha_i\eta\sum_jb_j/\sqrt{1-c_j}>0`$（对 (a) 求导；$`b_0=1-\beta_1>0`$），所以小的 $`\lambda>0`$ 上 $`\Delta F_i>0`$，从而存在正零点。极限为正的情形对称。
- $`\Lambda(T)=\sum_j(1-b_j)(1+O(c_j))=(T+1)-(1-\beta_1^{\,T+1})+O(\omega_T)`$。
- $`\kappa_{\rm iso}(T)=\frac{1-\beta_1}{\sqrt{1-\beta_2}}\sum_{j\le T}(\beta_1/\sqrt{\beta_2})^j`$，在 $`\beta_1<\sqrt{\beta_2}`$ 时收敛到 $`\kappa_{\rm iso}`$。
- 含义一条直接由 (a) 读出：$`\lambda`$ 很大、且滞后 $`j`$ 处 $`b_j\lambda`$ 可以忽略时，消息一侧的项为 $`\rho_i(1-b_j)/\sqrt{1+c_j(\lambda^2-1)}`$，对照一侧为 $`\rho_i(1-b_j)/\sqrt{1-c_j}`$。

*(d)* 普通 SGD 把 $`\theta_i`$ 移动 $`-\eta g_i`$，不改变以后的步长，所以 $`\Delta F_i=-(\nabla F)_i\eta g_i=\alpha_i\eta\sqrt{\bar v_i}\lambda`$。$`\square`$

**关于证明的说明.**

- mean-field 替换忽略了 $`m_k`$、$`v_k`$ 围绕均值的波动。这是 (a) 里唯一的近似，它的精度在 [Check 7.2](../experiments/07-02_stationary_amplitude_law_ZH.md#c7-2) 中测量。
- (b) 中的常数 11 成立但偏保守；实测中 $`2\lambda^\star`$ 与真实变号点的吻合程度远好于这个界（[Check 7.3](../experiments/07-02_stationary_amplitude_law_ZH.md#c7-3)）。
- (c) 关于正零点的结论是单向蕴含。本文不声称极限为正时不存在零点；在检验过的设定中两者总是一致（[Check 7.2](../experiments/07-02_stationary_amplitude_law_ZH.md#c7-2)）。
