# Theorem 7.1 · 状态通道分解

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-01_state_channel_decomposition_ZH.md) · [证明](07-01_state_channel_decomposition_ZH.md) · [实验](../experiments/07-01_state_channel_decomposition_ZH.md) · [English](07-01_state_channel_decomposition.md)

**本页证明什么.** continuation value 关于完整 Adam/AdamW 状态 $`(\theta,m,v)`$ 的梯度，等于沿参考尾段反向递推得到的 costate。把它与一条消息的 Adam 增量配对，消息的一阶价值就精确地分成参数通道、动量通道与二阶矩通道。证明就是对各尾段更新的复合用链式法则，并把每个 Jacobian 块都写出来：这是贯穿训练的反向模式求导，与对带动量 SGD 的精确反向传播（[Maclaurin et al., 2015](#ref-maclaurin2015gradient)）同理。

**本页用到的设定与记号.**

- *complete learner state.* $`z=(\theta,o)`$，optimizer state $`o=(m,v,s)`$：一阶矩 $`m`$、二阶矩 $`v`$、绝对步数计数器 $`s`$（[Definition 1.3](../README_ZH.md#s1-3)）。
- *AdamW 更新*（[Kingma & Ba, 2015](#ref-kingma2015adam) 的 Adam，加上 [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled) 的 decoupled weight decay），超参数 $`\beta_1\in[0,1)`$、$`\beta_2\in(0,1)`$、$`\eta>0`$、$`\epsilon\ge0`$，decoupled weight decay $`\lambda_{\rm wd}\ge0`$，作用于第 $`s`$ 步的 consumed gradient $`g`$：

```math
m'=\beta_1m+(1-\beta_1)g,\qquad v'=\beta_2v+(1-\beta_2)\,g\odot g,\qquad \theta'=(1-\eta\lambda_{\rm wd})\,\theta-\eta\,q_{1,s}\,\frac{m'}{\sqrt{q_{2,s}v'}+\epsilon},
```

- bias correction 因子为 $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$、$`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$（或都取 $`1`$），所有运算按坐标进行。增量为 $`\Delta(z,g)=(\theta'-\theta,\;m'-m,\;v'-v)`$（[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)）；计数器加一，不对它求导。
- *尾段.* 第 $`t`$ 轮的消息之后，第 $`k=t+1,\dots,R-1`$ 次更新用 consumed gradient $`g_k(\theta)`$（损失梯度 $`\nabla\ell_k(\theta)`$，可能以阈值 $`c`$ 做全局范数裁剪，见 [Pascanu et al., 2013](#ref-pascanu2013difficulty)）。得分为 $`F(\theta_R)`$，$`V_{t+1}(z)`$ 是尾段从 $`z`$ 出发所得的得分（[Definition 1.4](../README_ZH.md#s1-4)）。
- *一阶价值.* $`\mathrm{Val}(g)=\langle\nabla V_{t+1}(z_t),\Delta_t(z_t,g)\rangle`$（[Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1)）。

**陈述（[Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1)）.** 固定尾段随机性的一个实现，并假设：

**(D1)** $`F`$ 属于 $`C^1`$；在从 $`z_t`$ 出发的尾段轨迹附近，每个 $`g_k`$ 关于 $`\theta`$ 属于 $`C^1`$，Jacobian 为 $`J_k=\partial g_k/\partial\theta`$；不裁剪时 $`J_k=H_k=\nabla^2\ell_k(\theta_k)`$，裁剪以阈值 $`c`$ 生效且轨迹上 $`\lVert\nabla\ell_k(\theta_k)\rVert\ne c`$ 时，

```math
J_k=\frac{c}{\lVert r_k\rVert}\Big(I-\frac{r_kr_k^\top}{\lVert r_k\rVert^2}\Big)H_k,\qquad r_k=\nabla\ell_k(\theta_k).
```

**(D2)** 每次尾段更新后都有 $`v'_k>0`$（逐坐标）。

记第 $`k`$ 次更新后的矩为 $`m'_k,v'_k`$，并令

```math
D_1^{(k)}=\mathrm{diag}\Big(\frac{q_{1,s_k}}{\sqrt{q_{2,s_k}v'_k}+\epsilon}\Big),\qquad D_2^{(k)}=\mathrm{diag}\Big(\frac{q_{1,s_k}\,q_{2,s_k}\,m'_k}{2\sqrt{q_{2,s_k}v'_k}\,\big(\sqrt{q_{2,s_k}v'_k}+\epsilon\big)^2}\Big),
```

定义 $`\chi_R=(\nabla F(\theta_R),0,0)`$，对 $`k=R-1,\dots,t+1`$，

```math
\begin{aligned}
\chi^m_k&=\beta_1\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big),\qquad \chi^v_k=\beta_2\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big),\\
\chi^\theta_k&=(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}+J_k^\top\Big[(1-\beta_1)\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big)+2(1-\beta_2)\,\mathrm{diag}(g_k)\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big)\Big].
\end{aligned}
```

则 (a) $`V_{t+1}`$ 在 $`z_t`$ 处可微，且 $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$；(b) 对每条消息 $`g`$，

```math
\mathrm{Val}(g)=\big\langle\chi^\theta_{t+1},\Delta\theta(g)\big\rangle+\big\langle\chi^m_{t+1},(1-\beta_1)(g-m_t)\big\rangle+\big\langle\chi^v_{t+1},(1-\beta_2)(g\odot g-v_t)\big\rangle ,
```

三项依次为参数、动量、二阶矩通道；(c) 二阶矩通道只通过 $`g\odot g`$ 依赖于 $`g`$（对消息律只通过 $`\mathbb E_P\,G\odot G`$），动量通道关于 $`g`$ 仿射，参数通道关于 $`g`$ 非线性；(d) 若所有尾段更新都有 $`J_k=0`$ 且 $`\lambda_{\rm wd}=0`$，则 $`\chi^\theta_k\equiv\nabla F`$，并且

```math
\chi^m_{t+1}=-\eta\sum_{k=t+1}^{R-1}\beta_1^{\,k-t}D_1^{(k)}\nabla F,\qquad \chi^v_{t+1}=\eta\sum_{k=t+1}^{R-1}\beta_2^{\,k-t}D_2^{(k)}\nabla F ;
```

(e) 随机尾段的各分支 costate 有控制函数时，$`\nabla V_{t+1}(z_t)=\mathbb E_\omega\chi_{t+1,\omega}`$，且 (b) 对 $`\mathbb E_\omega\chi_{t+1,\omega}`$ 成立。

**证明.**

*(a)* 一次尾段更新是映射 $`\Phi_k:(\theta,m,v)\mapsto(\theta',m',v')`$，其中 $`g=g_k(\theta)`$。在 (D1)–(D2) 下它在轨迹附近属于 $`C^1`$：(D1) 使 $`g_k`$ 可微，(D2) 使平方根远离 $`0`$。记 $`D_1=D_1^{(k)}`$、$`D_2=D_2^{(k)}`$、$`J=J_k`$，其 Jacobian 各块如下。

- $`\partial m'/\partial m=\beta_1I`$，$`\partial m'/\partial\theta=(1-\beta_1)J`$，$`\partial m'/\partial v=0`$。
- $`\partial v'/\partial v=\beta_2I`$，$`\partial v'/\partial\theta=2(1-\beta_2)\,\mathrm{diag}(g)\,J`$，$`\partial v'/\partial m=0`$。
- $`\partial\theta'/\partial m'=-\eta D_1`$。$`(\sqrt{q_2v'}+\epsilon)^{-1}`$ 对 $`v'`$ 的导数是 $`-q_2/\big(2\sqrt{q_2v'}(\sqrt{q_2v'}+\epsilon)^2\big)`$，所以 $`\partial\theta'/\partial v'=\eta D_2`$。
- 于是 $`\partial\theta'/\partial m=-\eta\beta_1D_1`$，$`\partial\theta'/\partial v=\eta\beta_2D_2`$，$`\partial\theta'/\partial\theta=(1-\eta\lambda_{\rm wd})I-\eta(1-\beta_1)D_1J+2\eta(1-\beta_2)D_2\,\mathrm{diag}(g)\,J`$。

$`V_{t+1}(z)=F(\pi_\theta\,\Phi_{R-1}\circ\cdots\circ\Phi_{t+1}(z))`$，其中 $`\pi_\theta`$ 取出参数部分。由链式法则，

```math
\nabla V_{t+1}(z_t)=D\Phi_{t+1}^\top\cdots D\Phi_{R-1}^\top\,(\nabla F,0,0).
```

从右往左累乘，得 $`\chi_k=D\Phi_k^\top\chi_{k+1}`$，$`\chi_R=(\nabla F,0,0)`$。把转置后的各块相乘（所有 $`D`$ 都是对角阵）：

- $`m`$ 行收集 $`-\eta\beta_1D_1\chi^\theta_{k+1}+\beta_1\chi^m_{k+1}`$；
- $`v`$ 行收集 $`\eta\beta_2D_2\chi^\theta_{k+1}+\beta_2\chi^v_{k+1}`$；
- $`\theta`$ 行收集 $`(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}`$，再加上 $`J^\top`$ 作用于 $`(1-\beta_1)\chi^m_{k+1}-\eta(1-\beta_1)D_1\chi^\theta_{k+1}+2(1-\beta_2)\mathrm{diag}(g)\big(\chi^v_{k+1}+\eta D_2\chi^\theta_{k+1}\big)`$。

这就是所列递推。裁剪生效时 $`g=c\,r/\lVert r\rVert`$，$`r=\nabla\ell_k(\theta)`$，而 $`r\mapsto r/\lVert r\rVert`$ 的导数是 $`(I-rr^\top/\lVert r\rVert^2)/\lVert r\rVert`$；再与 $`\partial r/\partial\theta=H_k`$ 复合即得所列的 $`J_k`$。递推里用的是 $`J_k^\top`$，它不一定对称。

*(b)* 按定义 $`\mathrm{Val}(g)=\langle\nabla V_{t+1}(z_t),\Delta_t(z_t,g)\rangle`$。代入 (a) 的 $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$，以及 Adam 增量映射中的 $`\Delta m=(1-\beta_1)(g-m_t)`$、$`\Delta v=(1-\beta_2)(g\odot g-v_t)`$（[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)：精确的平均 Adam 增量就是一阶统计量），再按块拆开内积。

*(c)* 直接从 (b) 读出：增量的 $`v`$ 块只通过 $`g\odot g`$ 依赖于 $`g`$，$`m`$ 块关于 $`g`$ 仿射，$`\Delta\theta(g)`$ 的分子分母都含 $`g`$。对消息律，在 (b) 两边取期望。

*(d)* $`J_k=0`$ 且 $`\lambda_{\rm wd}=0`$ 时，$`\theta`$ 行对每个 $`k`$ 给出 $`\chi^\theta_k=\chi^\theta_{k+1}=\nabla F`$。于是 $`\chi^m_k=\beta_1\chi^m_{k+1}-\eta\beta_1D_1^{(k)}\nabla F`$，且 $`\chi^m_R=0`$；展开递推得 $`\chi^m_{t+1}=-\eta\sum_{k=t+1}^{R-1}\beta_1^{\,k-t}D_1^{(k)}\nabla F`$。$`v`$ 行完全相同，只是系数换成 $`+\eta\beta_2D_2^{(k)}`$。

*(e)* 对每个分支用 (a)，再交换 $`\nabla`$ 与 $`\mathbb E_\omega`$；分支有限时可以交换，更一般地在各分支 costate 有控制函数时也可以。$`\square`$

**关于证明的说明.**

- 求导时步数计数器保持固定；decoupled decay 用未经修正的学习率把 $`\theta`$ 乘以 $`1-\eta\lambda_{\rm wd}`$。
- 在裁剪阈值上映射不可微，那里不声称任何导数。

## 参考文献

- [Kingma & Ba, 2015](../references_ZH.md#kingma2015adam)：Adam 优化器。
- [Loshchilov & Hutter, 2019](../references_ZH.md#loshchilov2019decoupled)：decoupled weight decay（AdamW）。
