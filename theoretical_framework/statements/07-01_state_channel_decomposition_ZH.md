# Theorem 7.1 · 状态通道分解

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](07-01_state_channel_decomposition_ZH.md) · [证明](../proofs/07-01_state_channel_decomposition_ZH.md) · [实验](../experiments/07-01_state_channel_decomposition_ZH.md) · [English](07-01_state_channel_decomposition.md)

**一段话概括.** 在 Adam 下，一条训练消息改变的不只是参数：它还改变 running moment $`m`$ 与 $`v`$，并通过它们影响此后训练的每一步。本定理对带 bias correction、decoupled weight decay、$`\epsilon`$ 与全局范数裁剪的 Adam/AdamW，沿剩余训练做一次反向（costate）递推，算出最终结果关于整个 optimizer state 的精确梯度。有了这个梯度，一条消息的一阶价值就精确地分成三部分：经由参数、经由动量、经由二阶矩。它与 [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 合称框架中的 **Theorem B**。

**本页用到的设定与记号.**

- *complete learner state.* $`z=(\theta,o)`$，参数为 $`\theta`$，optimizer state 为 $`o=(m,v,s)`$：Adam 的一阶矩 $`m`$、二阶矩 $`v`$（都按坐标）与绝对步数计数器 $`s`$（[Definition 1.3](../README_ZH.md#s1-3)）。
- *AdamW 更新*（[Kingma & Ba, 2015](#ref-kingma2015adam) 的 Adam，加上 [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled) 的 decoupled weight decay）。超参数 $`\beta_1\in[0,1)`$、$`\beta_2\in(0,1)`$、学习率 $`\eta>0`$、$`\epsilon\ge0`$，decoupled weight decay 系数 $`\lambda_{\rm wd}\ge0`$。第 $`s`$ 步的 consumed gradient $`g`$ 给出

```math
m'=\beta_1m+(1-\beta_1)g,\qquad v'=\beta_2v+(1-\beta_2)\,g\odot g,\qquad \theta'=(1-\eta\lambda_{\rm wd})\,\theta-\eta\,q_{1,s}\,\frac{m'}{\sqrt{q_{2,s}v'}+\epsilon},
```

- $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$ 与 $`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$ 是 bias correction 因子（不做 bias correction 时 $`q_{1,s}=q_{2,s}=1`$）。向量运算都按坐标进行。
- *增量.* $`\Delta(z,g)=(\theta'-\theta,\;m'-m,\;v'-v)`$，即 [Proposition 5.5](05-05_exact_first_order_statistic_adam_ZH.md#s5-5) 的 Adam 增量映射。计数器加一，求导时不对它求导。
- *消息与参考尾段.* 第 $`t`$ 轮学习器处在 $`z_t=(\theta_t,m_t,v_t,s_t)`$。consumed gradient 为 $`g`$ 的消息把它移到 $`z_t+\Delta_t(z_t,g)`$。之后参考尾段执行第 $`k=t+1,\dots,R-1`$ 次更新；第 $`k`$ 次更新用它自己数据上的 consumed gradient $`g_k(\theta)`$，即损失梯度 $`\nabla\ell_k(\theta)`$ 或其全局范数裁剪 $`\mathrm{clip}_c(\nabla\ell_k(\theta))`$（[Pascanu et al., 2013](#ref-pascanu2013difficulty)）。最终得分为 $`F(\theta_R)`$。
- *continuation value.* $`V_{t+1}(z)`$ 是尾段从 $`z`$ 出发所得的最终得分（[Definition 1.4](../README_ZH.md#s1-4)）。尾段带随机性 $`\omega`$（数据顺序、采样）时，$`V_{t+1}=\mathbb E_\omega V_{t+1,\omega}`$。
- *消息的一阶价值.* $`\mathrm{Val}(g)=\langle\nabla V_{t+1}(z_t),\Delta_t(z_t,g)\rangle`$，即 [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1) 展开中 $`V_{t+1}(z_t+\Delta_t(z_t,g))-V_{t+1}(z_t)`$ 的一阶项。对消息律 $`P`$，$`\mathrm{Val}(P)=\mathbb E_P\mathrm{Val}(G)=\langle\nabla V_{t+1}(z_t),\mu^P_t(z_t)\rangle`$，$`\mu^P_t`$ 为平均增量。

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

**说了什么.**

- 消息发出时，最终结果关于完整 Adam 状态的梯度，可以用上面的显式递推沿剩余训练反向扫一遍算出。
- 一条消息的一阶价值等于参数项、动量项、二阶矩项之和，别无其他。
- 二阶矩项只看平方梯度 $`g\odot g`$：平方梯度相同的两条消息经由 $`v`$ 的作用完全相同。

**直觉.**

- 消息改变当前这一步，同时也留下不同的 $`m`$ 与 $`v`$。更大的 $`v`$ 会压低此后的每一步，直到它衰减掉。
- costate $`\chi^m`$ 与 $`\chi^v`$ 是"现在多一点动量"或"现在多一点二阶矩"与最终结果之间的汇率。

**在框架中的作用.**

- 它让 [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1) 一阶表示中 optimizer state 的那一部分在 Adam 下可以计算。
- [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 把它算成闭式；[Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 用它说明只看参数的影响分数与余弦分数不一致；[Remark 7.5](07-05_where_first_order_rankers_fail_ZH.md#s7-5) 指出它的一阶用法在哪里失效。

**推导要点.**

- 一次 Adam 更新是 $`(\theta,m,v)`$ 的 $`C^1`$ 映射。它的 Jacobian 有 $`\partial m'/\partial m=\beta_1I`$、$`\partial v'/\partial v=\beta_2I`$、$`\partial\theta'/\partial m'=-\eta D_1`$、$`\partial\theta'/\partial v'=\eta D_2`$ 这些块，再加上经由 $`g_k(\theta)`$ 的 $`J_k`$ 项。
- continuation value 是 $`F`$ 与各尾段映射的复合。从末端开始累积链式法则（adjoint 过程），得 $`\chi_k=D\Phi_k^\top\chi_{k+1}`$；把转置后的各块相乘就得到递推。
- 裁剪的 Jacobian 是 $`c\,r/\lVert r\rVert`$ 的导数。

**用到.**

- [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1)：把一阶价值定义为与平均增量的配对。
- [Proposition 5.5](05-05_exact_first_order_statistic_adam_ZH.md#s5-5)：精确的 Adam 增量映射。

**被用于.**

- [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)：闭式的幅度定律。
- [Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)：两类 proxy 的不一致性。
- [Remark 7.5](07-05_where_first_order_rankers_fail_ZH.md#s7-5)：一阶排序器在哪里失效。
- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 与 [Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3)：方法中的幅度控制。

## Related work

- costate 递推就是贯穿训练的反向模式求导，这里对 Adam 的状态写成了闭式。同一原理也用于对带动量的 SGD 求精确梯度（[Maclaurin et al., 2015](#ref-maclaurin2015gradient)），以及在大规模训练中求 metagradient（[Engstrom et al., 2025](#ref-engstrom2025optimizing)），求导对象是超参数或数据权重。

## 参考文献

- [Kingma & Ba, 2015](../references_ZH.md#kingma2015adam)：Adam 优化器。
- [Loshchilov & Hutter, 2019](../references_ZH.md#loshchilov2019decoupled)：decoupled weight decay（AdamW）。
