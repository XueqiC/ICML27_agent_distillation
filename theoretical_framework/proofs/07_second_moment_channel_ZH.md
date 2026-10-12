# §7 Adam 的二阶矩通道（Theorem B）— 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/07_second_moment_channel_ZH.md) · [English](07_second_moment_channel.md)

**概要.**

- [Theorem 7.1](#s7-1) 用反向（costate）递推计算 Adam/AdamW 下 continuation value 的梯度。由此，一条消息的一阶价值精确分解为参数通道、动量通道和**二阶矩通道**三项。二阶矩通道只通过 $`g\odot g`$ 依赖于消息。
- [Corollary 7.2](#s7-2) 在 warm、平稳、线性 response 的情形下给出闭式解。消息的幅度 $`\lambda`$ 以 running RMS 为单位度量梯度；对任意 $`\lambda`$，mean-field response 都是一个显式求和。
  - 增量小时，价值是 $`\lambda`$ 的凹二次函数，有最优幅度 $`\lambda^\star`$，并在 $`2\lambda^\star`$ 附近变号。
  - 幅度很大时，直接效应饱和。很大的同向消息最终是否劣于零梯度消息，由一个显式判据决定。
- [Proposition 7.3](#s7-3) 推出影响分数的两条结论。
  - task-blind 统计量不可能对所有任务都把 feedback law 排对。
  - 在 Corollary 7.2 的区间里，只看参数的一阶分数会把两条同向消息排反，余弦分数则区分不了它们。
- [Remark 7.5](#s7-5) 说明任何一阶排序器在哪里不可靠。
- "Theorem B"指 [Theorem 7.1](#s7-1) 与 [Corollary 7.2](#s7-2) 的合称。
- 用到：complete learner state 与 continuation value（[Definition 1.3–1.4](../README_ZH.md#s1-3)）；一阶表示（[Proposition 5.1](05_optimizer_state_ZH.md#s5-1)）；Adam 的增量映射（[Proposition 5.5(a)](05_warm_adam_ZH.md#s5-5)）；孤立访问常数 $`\kappa_{\rm iso}`$（[Proposition 6.11(c)](06_learner_relative_information_ZH.md#s6-11)）；[Proposition 3.1(b)](03_learning_consequence_ZH.md#s3-1)。

## 设定

**状态与更新.** complete learner state 记为 $`z=(\theta,o)`$，其中 optimizer state $`o=(m,v,s)`$，$`s`$ 是绝对步数计数器。超参数为 $`\beta_1\in[0,1)`$、$`\beta_2\in(0,1)`$、$`\eta>0`$、$`\epsilon\ge0`$，decoupled weight decay 系数 $`\lambda_{\rm wd}\ge0`$。AdamW 在第 $`s`$ 步把 consumed gradient $`g`$ 映为

```math
m'=\beta_1m+(1-\beta_1)g,\qquad v'=\beta_2v+(1-\beta_2)\,g\odot g,\qquad \theta'=(1-\eta\lambda_{\rm wd})\,\theta-\eta\,q_{1,s}\,\frac{m'}{\sqrt{q_{2,s}v'}+\epsilon},
```

其中 bias correction 因子为 $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$、$`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$；不做 bias correction 时 $`q_{1,s}=q_{2,s}=1`$。向量运算都按坐标进行。增量为 $`\Delta(z,g)=(\theta'-\theta,\;m'-m,\;v'-v)`$，即 [Proposition 5.5(a)](05_warm_adam_ZH.md#s5-5) 的映射。计数器加一，求导时不对它求导。

**消息与参考尾段.**

- 第 $`t`$ 轮学习器处在 $`z_t=(\theta_t,m_t,v_t,s_t)`$。一条 consumed gradient 为 $`g`$ 的消息把它移到 $`z_t+\Delta_t(z_t,g)`$。
- 之后参考尾段执行第 $`k=t+1,\dots,R-1`$ 次更新。第 $`k`$ 次更新用它自己数据上的 consumed gradient $`g_k(\theta)`$：即损失梯度 $`\nabla\ell_k(\theta)`$，或者它经全局范数裁剪后的 $`\mathrm{clip}_c(\nabla\ell_k(\theta))`$。
- 最终得分为 $`F(\theta_R)`$。continuation value $`V_{t+1}(z)`$ 是尾段从 $`z`$ 出发所得的得分（[Definition 1.4](../README_ZH.md#s1-4)）。尾段带随机性 $`\omega`$（数据顺序、采样）时，$`V_{t+1}=\mathbb E_\omega V_{t+1,\omega}`$。

**消息的一阶价值.** 按 [Proposition 5.1](05_optimizer_state_ZH.md#s5-1)，消息 $`g`$ 在 $`z_t`$ 处的一阶价值为

```math
\mathrm{Val}(g)=\big\langle\nabla V_{t+1}(z_t),\,\Delta_t(z_t,g)\big\rangle ,
```

即 $`V_{t+1}(z_t+\Delta_t(z_t,g))-V_{t+1}(z_t)`$ 的一阶项。对消息律 $`P`$，$`\mathrm{Val}(P)=\mathbb E_P\mathrm{Val}(G)=\langle\nabla V_{t+1}(z_t),\mu^P_t(z_t)\rangle`$，正是 [Proposition 5.1](05_optimizer_state_ZH.md#s5-1) 逐轮求和的那一项。比较时只用到同一状态下的差 $`\mathrm{Val}(g)-\mathrm{Val}(g')`$。

## 状态通道分解

<a id="s7-1"></a>

**Theorem 7.1（状态通道分解）.** 固定尾段随机性的一个实现。假设：

**(D1)** $`F`$ 属于 $`C^1`$。在从 $`z_t`$ 出发的尾段轨迹附近，每个 consumed gradient $`g_k`$ 关于 $`\theta`$ 属于 $`C^1`$，Jacobian 记为 $`J_k=\partial g_k/\partial\theta`$。不裁剪时 $`J_k=H_k=\nabla^2\ell_k(\theta_k)`$。全局范数裁剪以阈值 $`c`$ 生效、且轨迹上 $`\lVert\nabla\ell_k(\theta_k)\rVert\ne c`$ 时，

```math
J_k=\frac{c}{\lVert r_k\rVert}\Big(I-\frac{r_kr_k^\top}{\lVert r_k\rVert^2}\Big)H_k,\qquad r_k=\nabla\ell_k(\theta_k).
```

**(D2)** 每次尾段更新后都有 $`v'_k>0`$（逐坐标）。

沿从 $`z_t`$ 出发的尾段，记第 $`k`$ 次更新后的矩为 $`m'_k,v'_k`$，并令

```math
D_1^{(k)}=\mathrm{diag}\Big(\frac{q_{1,s_k}}{\sqrt{q_{2,s_k}v'_k}+\epsilon}\Big),\qquad D_2^{(k)}=\mathrm{diag}\Big(\frac{q_{1,s_k}\,q_{2,s_k}\,m'_k}{2\sqrt{q_{2,s_k}v'_k}\,\big(\sqrt{q_{2,s_k}v'_k}+\epsilon\big)^2}\Big).
```

定义 costate $`\chi_k=(\chi^\theta_k,\chi^m_k,\chi^v_k)`$：$`\chi_R=(\nabla F(\theta_R),0,0)`$，对 $`k=R-1,\dots,t+1`$，

```math
\begin{aligned}
\chi^m_k&=\beta_1\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big),\qquad \chi^v_k=\beta_2\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big),\\
\chi^\theta_k&=(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}+J_k^\top\Big[(1-\beta_1)\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big)+2(1-\beta_2)\,\mathrm{diag}(g_k)\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big)\Big].
\end{aligned}
```

则：

**(a) costate 恒等式.** $`V_{t+1}`$ 在 $`z_t`$ 处关于 $`(\theta,m,v)`$ 可微，且 $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$。

**(b) 三个通道.** 对每条消息 $`g`$，

```math
\mathrm{Val}(g)=\underbrace{\big\langle\chi^\theta_{t+1},\Delta\theta(g)\big\rangle}_{\text{参数通道}}+\underbrace{\big\langle\chi^m_{t+1},(1-\beta_1)(g-m_t)\big\rangle}_{\text{动量通道}}+\underbrace{\big\langle\chi^v_{t+1},(1-\beta_2)(g\odot g-v_t)\big\rangle}_{\text{二阶矩通道}} .
```

**(c) 各通道读取什么.**

- 二阶矩通道只通过 $`g\odot g`$ 依赖于 $`g`$。它描述的是：消息改变了留下的 $`v`$，从而改变此后所有的步长。
- 动量通道关于 $`g`$ 是仿射的。
- 参数通道包含消息自己这一步。由于 $`g`$ 也进入它自己的分母，这一项是 $`g`$ 的非线性函数。
- 对消息律 $`P`$，二阶矩通道只通过 $`\mathbb E_P\,G\odot G`$ 依赖于 $`P`$。

**(d) 线性 response 下的闭式.** 若所有尾段更新都有 $`J_k=0`$（尾段梯度与 $`\theta`$ 无关），且 $`\lambda_{\rm wd}=0`$，则 $`\chi^\theta_k\equiv\nabla F`$，并且

```math
\chi^m_{t+1}=-\eta\sum_{k=t+1}^{R-1}\beta_1^{\,k-t}D_1^{(k)}\nabla F,\qquad \chi^v_{t+1}=\eta\sum_{k=t+1}^{R-1}\beta_2^{\,k-t}D_2^{(k)}\nabla F .
```

**(e) 随机尾段.** 若尾段只有有限多个随机分支、每个分支都满足 (D1)–(D2)，或者更一般地，各分支的 costate 有可积的控制函数、使求导与期望可交换，则 $`\nabla V_{t+1}(z_t)=\mathbb E_\omega\chi_{t+1,\omega}`$，且 (b) 对 $`\mathbb E_\omega\chi_{t+1,\omega}`$ 成立。

*Proof.* (a) 一次尾段更新是映射 $`\Phi_k:(\theta,m,v)\mapsto(\theta',m',v')`$，其中 $`g=g_k(\theta)`$。在 (D1)–(D2) 下，它在轨迹附近属于 $`C^1`$。记 $`D_1=D_1^{(k)}`$、$`D_2=D_2^{(k)}`$、$`J=J_k`$，其 Jacobian 各块如下。

- $`\partial m'/\partial m=\beta_1I`$，$`\partial m'/\partial\theta=(1-\beta_1)J`$，$`\partial m'/\partial v=0`$。
- $`\partial v'/\partial v=\beta_2I`$，$`\partial v'/\partial\theta=2(1-\beta_2)\,\mathrm{diag}(g)\,J`$，$`\partial v'/\partial m=0`$。
- $`\partial\theta'/\partial m'=-\eta D_1`$。$`(\sqrt{q_2v'}+\epsilon)^{-1}`$ 对 $`v'`$ 的导数是 $`-q_2/\big(2\sqrt{q_2v'}(\sqrt{q_2v'}+\epsilon)^2\big)`$，所以 $`\partial\theta'/\partial v'=\eta D_2`$。
- 于是 $`\partial\theta'/\partial m=-\eta\beta_1D_1`$，$`\partial\theta'/\partial v=\eta\beta_2D_2`$，$`\partial\theta'/\partial\theta=(1-\eta\lambda_{\rm wd})I-\eta(1-\beta_1)D_1J+2\eta(1-\beta_2)D_2\,\mathrm{diag}(g)\,J`$。

$`V_{t+1}(z)=F(\pi_\theta\,\Phi_{R-1}\circ\cdots\circ\Phi_{t+1}(z))`$，其中 $`\pi_\theta`$ 取出参数部分。由链式法则，$`\nabla V_{t+1}(z_t)=D\Phi_{t+1}^\top\cdots D\Phi_{R-1}^\top(\nabla F,0,0)`$。从右往左累乘，得 $`\chi_k=D\Phi_k^\top\chi_{k+1}`$，$`\chi_R=(\nabla F,0,0)`$。把转置后的各块相乘（所有 $`D`$ 都是对角阵）就得到所列递推：$`m`$ 行收集 $`-\eta\beta_1D_1\chi^\theta+\beta_1\chi^m`$，$`v`$ 行收集 $`\eta\beta_2D_2\chi^\theta+\beta_2\chi^v`$，其余各项都归入 $`\theta`$ 行。裁剪生效时 $`g=c\,r/\lVert r\rVert`$，而 $`r\mapsto r/\lVert r\rVert`$ 的导数是 $`(I-rr^\top/\lVert r\rVert^2)/\lVert r\rVert`$，由此得到所列的 $`J_k`$。递推里用的是 $`J_k^\top`$，它不一定对称。

(b) 就是 $`\mathrm{Val}`$ 的定义，代入 $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$ 以及 [Proposition 5.5(a)](05_warm_adam_ZH.md#s5-5) 的增量 $`\Delta m=(1-\beta_1)(g-m_t)`$、$`\Delta v=(1-\beta_2)(g\odot g-v_t)`$。

(c) 直接从 (b) 读出；对消息律，在 (b) 两边取期望。

(d) $`J_k=0`$ 且 $`\lambda_{\rm wd}=0`$ 时，$`\theta`$ 行给出 $`\chi^\theta_k=\chi^\theta_{k+1}=\nabla F`$。于是 $`\chi^m_k=\beta_1\chi^m_{k+1}-\eta\beta_1D_1^{(k)}\nabla F`$，且 $`\chi^m_R=0`$。展开递推得 $`\chi^m_{t+1}=-\eta\sum_{k=t+1}^{R-1}\beta_1^{\,k-t}D_1^{(k)}\nabla F`$；$`v`$ 行完全相同，只是把系数换成 $`+\eta\beta_2D_2^{(k)}`$。

(e) 对每个分支用 (a)，再交换 $`\nabla`$ 与 $`\mathbb E_\omega`$。$`\square`$

- 求导时步数计数器保持固定；decoupled decay 用未经修正的学习率把 $`\theta`$ 乘以 $`1-\eta\lambda_{\rm wd}`$。
- 在裁剪阈值上映射不可微，那里不声称任何导数。

## 平稳情形的幅度定律

下面在二阶矩通道能写成闭式的最简单区间里计算 [Theorem 7.1](#s7-1)：单个坐标、warm 的平稳状态、尾段梯度与 $`\theta`$ 无关。

<a id="s7-2"></a>

**Corollary 7.2（平稳情形的幅度定律）.** 对坐标 $`i`$ 假设：

- 尾段梯度 $`g_{k,i}`$ 与 $`\theta`$ 无关（线性 response 区间，$`J_k=0`$），平稳，均值 $`\bar g_i\ne0`$、平方的均值 $`\bar v_i>0`$；$`F`$ 关于 $`\theta_i`$ 是仿射的；
- $`\epsilon=0`$，不做 bias correction，$`\lambda_{\rm wd}=0`$；
- 消息到达时状态 warm：$`m_{t,i}=\bar g_i`$，$`v_{t,i}=\bar v_i`$；消息那次更新之后还有 $`T=R-1-t`$ 次尾段更新。

记

- $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$：沿 $`-\bar g_i`$ 方向移动对结果有利时 $`\alpha_i>0`$；
- $`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$：消息相对于 running RMS 的同向幅度；
- $`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}\in(0,1]`$：该坐标的梯度信噪比；
- $`b_j=(1-\beta_1)\beta_1^{\,j}`$，$`c_j=(1-\beta_2)\beta_2^{\,j}`$，$`j=0,\dots,T`$（$`j=0`$ 是消息那次更新）；$`\omega_T=1-\beta_2^{\,T+1}=\sum_{j\le T}c_j`$。

**mean-field response** $`\Delta F_i(\lambda)`$ 定义为：用幅度为 $`\lambda`$ 的消息替换一条零梯度消息时 $`F`$ 的变化，其中每一步的 $`m_k/\sqrt{v_k}`$ 换成 $`\mathbb Em_k/\sqrt{\mathbb Ev_k}`$，期望对尾段梯度取。

**(a) 精确的 mean-field response（任意 $`\lambda`$）.**

```math
\Delta F_i(\lambda)=\alpha_i\eta\sum_{j=0}^{T}\Big[\frac{\rho_i+b_j(\lambda-\rho_i)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho_i(1-b_j)}{\sqrt{1-c_j}}\Big].
```

**(b) 小增量：二次律.** 令 $`Q_i(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho_i\omega_T\lambda^2`$，$`S_i(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho_i\omega_T\lambda^2`$。若 $`\beta_2\ge\tfrac12`$，则对所有 $`\lambda`$，

```math
\big\lvert\Delta F_i(\lambda)-\alpha_i\eta\,Q_i(\lambda)\big\rvert\le11\,(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\;\lvert\alpha_i\rvert\eta\,S_i(\lambda).
```

[Theorem 7.1](#s7-1) 的 mean-field 一阶价值之差 $`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$ 满足同样的界。由此：

- $`Q_i`$ 是凹的，在 $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$ 处取最大值。
- 令 $`u=11(1-\beta_2)\max\lbrace1,9\lambda^{\star2}\rbrace`$，并设 $`\alpha_i>0`$。若 $`u\le1/8`$，则 $`\Delta F_i`$ 在 $`[\lambda^\star,\,2\lambda^\star(1-4u)]`$ 上为正，在 $`[2\lambda^\star(1+4u),\,3\lambda^\star]`$ 上为负。所以变号点离 $`2\lambda^\star`$ 的相对距离不超过 $`4u`$，略高于它的同向消息劣于零梯度消息。
- 因此阈值 $`2\lambda^\star`$ 只在 $`(1-\beta_2)(2\lambda^\star)^2\ll1`$ 时可用。否则变号点（如果存在）要用 (a) 和 (c) 计算。
- 剩余步数短（$`T(1-\beta_2)\ll1`$）. 此时 $`\omega_T\approx(T+1)(1-\beta_2)`$，所以 $`\lambda^\star`$ 很大，且 $`(1-\beta_2)(2\lambda^\star)^2\approx4(1-\beta_1^{\,T+1})^2/\big(\rho_i^2(T+1)^2(1-\beta_2)\big)`$。除非这个量很小，$`2\lambda^\star`$ 落在 (b) 的适用范围之外。
- 剩余步数长（$`T(1-\beta_2)\gg1`$）. 此时 $`\omega_T\to1`$，$`\lambda^\star\to(1-\beta_1^{\,T+1})/\rho_i\approx1/\rho_i`$，$`(1-\beta_2)(2\lambda^\star)^2\approx4(1-\beta_2)/\rho_i^2`$。除非该坐标噪声很大，这个量都很小。

**(c) 大幅度.** 当 $`\lambda\to\infty`$ 时，

```math
\Delta F_i(\lambda)\to\alpha_i\eta\big[\kappa_{\rm iso}(T)-\rho_i\,\Lambda(T)\big],\qquad \kappa_{\rm iso}(T)=\sum_{j=0}^{T}\frac{b_j}{\sqrt{c_j}},\qquad \Lambda(T)=\sum_{j=0}^{T}\frac{1-b_j}{\sqrt{1-c_j}} .
```

- 设 $`\alpha_i>0`$。若 $`\rho_i\Lambda(T)>\kappa_{\rm iso}(T)`$，则所有足够大的同向消息都劣于零梯度消息，且 $`\Delta F_i`$ 有正零点。若 $`\rho_i\Lambda(T)<\kappa_{\rm iso}(T)`$，则所有足够大的同向消息都更好。精确的临界信噪比是 $`\kappa_{\rm iso}(T)/\Lambda(T)`$。
- 由于 $`\Lambda(T)=(T+1)\big(1+O(\tfrac1{T+1}+(1-\beta_2))\big)`$，判据近似为 $`\rho_i(T+1)>\kappa_{\rm iso}(T)`$。
- $`\kappa_{\rm iso}(T)`$ 是 [Proposition 6.11(c)](06_learner_relative_information_ZH.md#s6-11) 中孤立访问常数 $`\kappa_{\rm iso}=(1-\beta_1)/\big(\sqrt{1-\beta_2}\,(1-\beta_1/\sqrt{\beta_2})\big)`$ 的有限 horizon 版本；当 $`\beta_1<\sqrt{\beta_2}`$ 时，$`T\to\infty`$ 下收敛到它。
- 含义：幅度很大的消息自己决定了分母，所以它的直接效应饱和在 $`\eta\kappa_{\rm iso}(T)`$ 附近。与此同时，它压低了此后每一步：滞后 $`j`$ 时，对照的步长约为 $`\rho_i`$，消息一侧的步长约为 $`\rho_i/\sqrt{1+c_j\lambda^2}`$。只要 $`c_j\lambda^2\gtrsim1`$，这种压低就持续，即约 $`\log((1-\beta_2)\lambda^2)/(1-\beta_2)`$ 步，最多 $`T+1`$ 步。

**(d) 与 SGD 对比.** 同一区间下，SGD 的 $`\Delta F_i(\lambda)=\alpha_i\eta\sqrt{\bar v_i}\,\lambda`$ 是线性的：同向消息越大越好。

*Proof.* (a) 在 mean-field 替换下，尾段梯度被换成它们的矩，所以消息造成的偏差按几何速度衰减。对 $`j\ge0`$，

```math
\mathbb Em_{t+j,i}=\bar g_i+b_j(g_i-\bar g_i),\qquad \mathbb Ev_{t+j,i}=\bar v_i\big(1+c_j(g_i^2/\bar v_i-1)\big),
```

这里下标 $`t+j`$ 表示消息之后第 $`j`$ 次更新所用的矩（$`j=0`$ 即消息那次更新）。$`j=0`$ 是从 $`m_t=\bar g`$、$`v_t=\bar v`$ 出发的消息更新。尾段梯度的均值为 $`\bar g_i`$、平方的均值为 $`\bar v_i`$，所以此后每次更新把 $`\mathbb Em`$ 的偏差乘以 $`\beta_1`$，把 $`\mathbb Ev`$ 的偏差乘以 $`\beta_2`$。$`F`$ 关于 $`\theta_i`$ 仿射，所以结果的变化为 $`-(\nabla F)_i\,\eta\sum_{j=0}^{T}\big[\mathbb Em_{t+j}(g)/\sqrt{\mathbb Ev_{t+j}(g)}-\mathbb Em_{t+j}(0)/\sqrt{\mathbb Ev_{t+j}(0)}\big]`$。

分子分母同除以 $`\sqrt{\bar v_i}`$，并代入 $`\bar g_i/\sqrt{\bar v_i}=\mathrm{sign}(\bar g_i)\rho_i`$、$`g_i/\sqrt{\bar v_i}=\mathrm{sign}(\bar g_i)\lambda`$ 与 $`-(\nabla F)_i\,\mathrm{sign}(\bar g_i)=\alpha_i`$ 即得。由于 $`c_j\le1-\beta_2<1`$，所有分母都为正。

(b) 记 $`\gamma=1-\beta_2\le\tfrac12`$，$`f_j(x)=(1+c_j(x-1))^{-1/2}`$。(a) 中的方括号等于 $`\rho_i(1-b_j)\big(f_j(\lambda^2)-f_j(0)\big)+b_j\lambda f_j(\lambda^2)`$，所以

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

一阶价值方面，[Theorem 7.1(d)](#s7-1) 在消息前状态上的 mean-field costate 为 $`\chi^m=-\eta(\nabla F)_i\sum_{j=1}^T\beta_1^{\,j}/\sqrt{\bar v_i}`$ 与 $`\chi^v=\tfrac12\eta(\nabla F)_i\bar g_i\sum_{j=1}^T\beta_2^{\,j}/\bar v_i^{3/2}`$（此时 $`D_1=\bar v_i^{-1/2}`$，$`D_2=\bar g_i/(2\bar v_i^{3/2})`$）。归一化后，$`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$ 等于 (a) 中 $`j=0`$ 的精确方括号加上 $`\sum_{j=1}^T\big(b_j\lambda-\tfrac12\rho_ic_j\lambda^2\big)`$。它与 $`Q_i`$ 之差是 $`E_1+E_2+E_3`$ 中 $`j=0`$ 的部分，满足同样的界。

各条推论：

- $`Q_i''=-\rho_i\omega_T<0`$，$`Q_i'(\lambda^\star)=0`$。
- 变号点：设 $`\alpha_i>0`$，$`\lambda=2\lambda^\star(1+\delta)`$，$`\lvert\delta\rvert\le\tfrac12`$。此时 $`\lambda^\star\le\lambda\le3\lambda^\star`$，误差不超过 $`u\,S_i(\lambda)`$；并且 $`Q_i(\lambda)=-(1-\beta_1^{\,T+1})\lambda\delta`$，$`S_i(\lambda)=(1-\beta_1^{\,T+1})\lambda(2+\delta)`$。
  - 当 $`\delta\in[4u,\tfrac12]`$：$`\Delta F_i/(\alpha_i\eta)\le(1-\beta_1^{\,T+1})\lambda\,[-\delta+u(2+\delta)]<0`$，因为 $`u<\tfrac12`$ 时 $`\delta(1-u)\ge4u(1-u)>2u`$。
  - 当 $`\delta\in[-\tfrac12,-4u]`$：$`\Delta F_i/(\alpha_i\eta)\ge(1-\beta_1^{\,T+1})\lambda\,[\lvert\delta\rvert-u(2-\lvert\delta\rvert)]>0`$，因为 $`\lvert\delta\rvert(1+u)>2u`$。
  - $`u\le1/8`$ 保证 $`4u\le\tfrac12`$，两个区间都非空。
- 关于剩余步数的两条由 $`\omega_T=1-(1-\gamma)^{T+1}`$ 得到。

(c) 当 $`\lambda\to\infty`$ 时，第 $`j`$ 项的第一个分式与 $`b_j\lambda/(\sqrt{c_j}\lambda)`$ 等价，趋于 $`b_j/\sqrt{c_j}`$（$`b_j=0`$ 时它趋于 $`0=b_j/\sqrt{c_j}`$）。第二个分式与 $`\lambda`$ 无关。由此得到极限。

- 若极限为负且 $`\alpha_i>0`$，则对所有足够大的 $`\lambda`$ 有 $`\Delta F_i(\lambda)<0`$。又 $`\Delta F_i(0)=0`$，$`\Delta F_i'(0)=\alpha_i\eta\sum_jb_j/\sqrt{1-c_j}>0`$（对 (a) 求导；$`b_0=1-\beta_1>0`$），所以小的 $`\lambda>0`$ 上 $`\Delta F_i>0`$，从而存在正零点。极限为正的情形对称。
- $`\Lambda(T)=\sum_j(1-b_j)(1+O(c_j))=(T+1)-(1-\beta_1^{\,T+1})+O(\omega_T)`$。
- $`\kappa_{\rm iso}(T)=\frac{1-\beta_1}{\sqrt{1-\beta_2}}\sum_{j\le T}(\beta_1/\sqrt{\beta_2})^j`$，在 $`\beta_1<\sqrt{\beta_2}`$ 时收敛到 $`\kappa_{\rm iso}`$。
- 含义一条直接由 (a) 读出：$`\lambda`$ 很大、且滞后 $`j`$ 处 $`b_j\lambda`$ 可以忽略时，消息一侧的项为 $`\rho_i(1-b_j)/\sqrt{1+c_j(\lambda^2-1)}`$，对照一侧为 $`\rho_i(1-b_j)/\sqrt{1-c_j}`$。

(d) 普通 SGD 把 $`\theta_i`$ 移动 $`-\eta g_i`$，不改变以后的步长，所以 $`\Delta F_i=-(\nabla F)_i\eta g_i=\alpha_i\eta\sqrt{\bar v_i}\lambda`$。$`\square`$

- mean-field 替换忽略了 $`m_k`$、$`v_k`$ 围绕均值的波动。这是 (a) 里唯一的近似，它的精度在 [Check 7.2](../experiments/07_second_moment_channel_ZH.md#c7-2) 中测量。
- (b) 中的常数 11 成立但偏保守；实测中 $`2\lambda^\star`$ 与真实变号点的吻合程度远好于这个界（[Check 7.3](../experiments/07_second_moment_channel_ZH.md#c7-3)）。
- (c) 关于正零点的结论是单向蕴含。本文不声称极限为正时不存在零点；在检验过的设定中两者总是一致（[Check 7.2](../experiments/07_second_moment_channel_ZH.md#c7-2)）。

## 对影响分数的推论

<a id="s7-3"></a>

**Proposition 7.3（两类 proxy 的不一致性）.**

- **(a) task-blind 统计量.** 设 $`\mathcal T`$ 是只依赖于当前状态与 feedback law、不依赖于部署任务的统计量，并按它的值给反馈律排序。则存在两个任务，以及同一学习器、同一状态下的两个消息律 $`P,Q`$：$`P,Q`$ 的真实顺序在两个任务上相反，而 $`\mathcal T(P),\mathcal T(Q)`$ 在两个任务上都相同。因此 $`\mathcal T`$ 至少在一个任务上排错，或者分不出一个非零的差。
- **(b) 只看参数的分数与余弦分数.** 在 [Corollary 7.2](#s7-2) 的设定下取 $`\alpha_i>0`$，考虑两条同向消息，幅度 $`0<\lambda_1<\lambda_2`$。
  - 只看参数的分数 $`S_{\rm PO}(g)=\langle\chi^\theta,\Delta\theta(g)\rangle`$ 满足 $`S_{\rm PO}=\alpha_i\eta\,h(\lambda)`$，其中 $`h(\lambda)=(\beta_1\rho_i+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$。$`h`$ 在 $`\lambda<\lambda_{\rm PO}:=(1-\beta_1)\beta_2/(\beta_1\rho_i(1-\beta_2))`$ 上严格递增。这一类包括用 Adam 步长的 TracIn 式分数和 fixed-state 的 Adam data attribution。若 $`\lambda_2<\lambda_{\rm PO}`$，它把较大的消息排在前面。
  - 余弦分数只依赖于消息梯度的方向；在单个坐标上，也可以只依赖于参数步的方向。这一类包括 LESS 式分数。任何这样的分数都给两条消息相同的值。
  - 只要 $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$，真实价值就偏好较小的消息。由 [Corollary 7.2(b)](#s7-2)，当 $`\lambda_1+\lambda_2>2\lambda^\star`$ 且 $`Q_i(\lambda_1)-Q_i(\lambda_2)>11(1-\beta_2)\max\lbrace1,\lambda_2^2\rbrace\big(S_i(\lambda_1)+S_i(\lambda_2)\big)`$ 时这一点成立；一般情形可由 [Corollary 7.2(a)](#s7-2) 直接读出。
  - 在这一区域，只看参数的分数把这对消息排反，余弦分数区分不了它们。

*Proof.* (a) 任取得分为 $`F`$ 的任务及满足 $`L_F(P)\ne L_F(Q)`$ 的两个消息律，再取得分为 $`F'=1-F\in[0,1]`$ 的任务。则 $`L_{F'}(P)-L_{F'}(Q)=-(L_F(P)-L_F(Q))`$，而 $`\mathcal T`$ 不变。[Proposition 3.1(b)](03_learning_consequence_ZH.md#s3-1) 给出一对更自然、性质相同的任务：两个边缘分布相同、平均梯度相同的标签包，它们学习价值之差的符号在 $`J(p)=p^2`$ 与 $`J(p)=1-p^2`$ 下相反。

(b) 此时 $`\chi^\theta=\nabla F`$，消息自己这一步为

```math
\Delta\theta_i(g)=-\eta\,\frac{\beta_1\bar g_i+(1-\beta_1)g_i}{\sqrt{\beta_2\bar v_i+(1-\beta_2)g_i^2}}=-\eta\,\mathrm{sign}(\bar g_i)\,h(\lambda),
```

所以 $`S_{\rm PO}=(\nabla F)_i\Delta\theta_i=\alpha_i\eta\,h(\lambda)`$。求导得

```math
h'(\lambda)=\frac{(1-\beta_1)\beta_2-\beta_1\rho_i(1-\beta_2)\lambda}{\big(\beta_2+(1-\beta_2)\lambda^2\big)^{3/2}},
```

它在 $`\lambda<\lambda_{\rm PO}`$ 时为正。

- 同一梯度的正倍数方向相同；在单个坐标上，两条消息的参数步都指向 $`-\mathrm{sign}(\bar g_i)`$。所以任何余弦分数都给出平局。
- 真实价值方面，$`Q_i(\lambda_2)-Q_i(\lambda_1)=(\lambda_2-\lambda_1)\big[(1-\beta_1^{\,T+1})-\tfrac12\rho_i\omega_T(\lambda_1+\lambda_2)\big]`$，它为负当且仅当 $`\lambda_1+\lambda_2>2\lambda^\star`$。所列 margin 吸收了 [Corollary 7.2(b)](#s7-2) 在两个幅度上的误差界。$`\square`$

- (b) 通过 [Corollary 7.2(b)](#s7-2) 的二次律，是关于小增量的陈述；增量大时要从 [Corollary 7.2(a)](#s7-2) 读 response。
- 在常见设定下，满足 $`2\lambda^\star<\lambda_1+\lambda_2`$ 且 $`\lambda_2<\lambda_{\rm PO}`$ 的区间非空。剩余步数长时 $`2\lambda^\star\approx2/\rho_i`$，而 $`\lambda_{\rm PO}\approx(1-\beta_1)/(\beta_1\rho_i(1-\beta_2))`$；$`\beta_1=0.9`$、$`\beta_2=0.99`$ 时约为 $`11/\rho_i`$。

<a id="s7-4"></a>

**Remark 7.4（与已有工作的关系，以及裁剪的作用）.**

- Liu 与 Li（2026）把 AdamW 写成 input–state–output 系统，分析一个 minibatch 扰动在短 horizon（8–32 步）内如何通过动量与二阶矩状态传播。他们没有给出幅度定律、最优幅度、阈值，也没有讨论对数据选择的含义。[Corollary 7.2](#s7-2) 在长 horizon 的平稳情形下补上了这些。
- Adam 版 In-Run Data Shapley（Ding et al., 2026）采用 fixed-state 近似，只给当前这一步的预条件更新打分，因此属于 [Proposition 7.3(b)](#s7-3) 中只看参数的一类。用 Adam 步长计算的 TracIn 式分数（Pruthi et al., 2020）也属于这一类。LESS（Xia et al., 2024）按 Adam 预条件后的训练梯度与验证梯度的余弦排序，属于 [Proposition 7.3(b)](#s7-3) 中的余弦一类。
- 全局范数裁剪限制了单条消息的幅度 $`\lambda`$，因此也限制了二阶矩通道能造成的损失（[Corollary 7.2(a),(c)](#s7-2)）。这为"裁剪为什么有帮助"提供了一个可检验的机制，但这里不把它作为结论。

<a id="s7-5"></a>

**Remark 7.5（一阶排序器在哪里失效）.**

- 定义一对候选的二阶矩占比：两者增量之差在 $`v`$ 通道上的缩并的绝对值，除以 $`\theta`$、$`m`$、$`v`$ 三个通道缩并的绝对值之和。
- 占比高通常意味着增量相对于 $`v`$ 很大。这正是 [Corollary 7.2(a),(c)](#s7-2) 描述的非线性、会饱和的区域：一阶价值仍按二次律走，真实 response 却不再如此。线性化在那里高估了 $`v`$ 通道造成的伤害。
- [Theorem 7.1](#s7-1) 的分解是精确的，但它的一阶用法恰好在 $`v`$ 通道重要的地方最不可靠。
- 二阶矩占比高时，排序应当用有限替换的测量（[Theorem 8.8](08_certificate_ZH.md#s8-8)），或者用 [Corollary 7.2(a)](#s7-2) 那样的非线性公式，而不是任何一阶分数。只看参数的分数不够，用精确 continuation gradient 的完整状态分数也不够。
- [Proposition 7.3(b)](#s7-3) 是 [Corollary 7.2(b)](#s7-2) 小增量区间内的陈述。
- 这一点在受控面板上的证据见 [Check 7.5](../experiments/07_second_moment_channel_ZH.md#c7-5)。

## Related work

- Ding, M., Zhang, Z., Wang, D., Hu, L. (2026). In-Run Data Shapley for Adam optimizer. arXiv:2602.00329.
- Kingma, D. P., Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.
- Liu, K., Li, S. (2026). Finite-horizon input–output dynamics of minibatch perturbations in AdamW. arXiv:2608.19762.
- Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization. ICLR.
- Pruthi, G., Liu, F., Kale, S., Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. NeurIPS.
- Wang, J. T., Mittal, P., Song, D., Jia, R. (2025). Data Shapley in one training run. ICLR.
- Xia, M., Malladi, S., Gururangan, S., Arora, S., Chen, D. (2024). LESS: Selecting influential data for targeted instruction tuning. ICML.
