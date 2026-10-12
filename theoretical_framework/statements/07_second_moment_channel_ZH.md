# §7 Theorem B：Adam 的二阶矩通道 — 精确陈述

[← Framework](../README_ZH.md) · [Proofs](../proofs/07_second_moment_channel_ZH.md) · [Experiments](../experiments/07_second_moment_channel_ZH.md) · [English](07_second_moment_channel.md)

### §7 · Theorem B：Adam 的二阶矩通道：精确陈述

[证明](../proofs/07_second_moment_channel_ZH.md) · [实验](../experiments/07_second_moment_channel_ZH.md)

**这一节做什么.**

- 用反向 costate 递推计算 Adam/AdamW 的 continuation value 梯度，并把一条消息的一阶价值拆成参数、动量、二阶矩三个通道（Theorem 7.1）。
- 在 warm、平稳、线性 response 的情形下，对任意消息幅度精确求解：最优幅度、变号点，以及大幅度判据（Corollary 7.2）。
- 由此推出，只看参数的影响分数和余弦分数在 Adam 下都不是一致的排序器（Proposition 7.3），并说明任何一阶排序器在哪里不可靠（Remark 7.5）。
- "Theorem B"指 Theorem 7.1 与 Corollary 7.2 的合称。

**设定.** 状态为 $`z_t=(\theta_t,m_t,v_t,s_t)`$。AdamW 是 [Proposition 5.5(a)](../proofs/05_warm_adam_ZH.md#s5-5) 的增量映射，bias correction 因子为 $`q_{1,s},q_{2,s}`$（不做 correction 时都为 1），decoupled decay 系数为 $`\lambda_{\rm wd}`$，consumed gradient 为 $`g_k(\theta)`$（可以经过全局范数裁剪，Jacobian 记为 $`J_k`$）。第 $`t`$ 轮的消息 $`g`$ 之后是参考尾段 $`k=t+1,\dots,R-1`$ 与得分 $`F(\theta_R)`$。消息的一阶价值为 $`\mathrm{Val}(g)=\langle\nabla V_{t+1}(z_t),\Delta_t(z_t,g)\rangle`$，即 [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1) 的一阶项。

#### [Theorem 7.1](../proofs/07_second_moment_channel_ZH.md#s7-1) · 状态通道分解

设 $`F\in C^1`$，每个 $`g_k`$ 在尾段轨迹附近关于 $`\theta`$ 属于 $`C^1`$（裁剪不落在阈值上），并且尾段上 $`v'_k>0`$。令 $`D_1^{(k)}=\mathrm{diag}\big(q_{1,s_k}/(\sqrt{q_{2,s_k}v'_k}+\epsilon)\big)`$，$`D_2^{(k)}=\mathrm{diag}\big(q_{1,s_k}q_{2,s_k}m'_k/(2\sqrt{q_{2,s_k}v'_k}(\sqrt{q_{2,s_k}v'_k}+\epsilon)^2)\big)`$。从 $`\chi_R=(\nabla F,0,0)`$ 出发反向计算 costate：

```math
\begin{aligned}
\chi^m_k&=\beta_1\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big),\qquad \chi^v_k=\beta_2\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big),\\
\chi^\theta_k&=(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}+J_k^\top\Big[(1-\beta_1)\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big)+2(1-\beta_2)\,\mathrm{diag}(g_k)\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big)\Big].
\end{aligned}
```

- **(a)** $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$。
- **(b)** $`\mathrm{Val}(g)=\langle\chi^\theta_{t+1},\Delta\theta(g)\rangle+\langle\chi^m_{t+1},(1-\beta_1)(g-m_t)\rangle+\langle\chi^v_{t+1},(1-\beta_2)(g\odot g-v_t)\rangle`$，三项依次为参数、动量、二阶矩通道。
- **(c)** 二阶矩通道只通过 $`g\odot g`$ 依赖于 $`g`$；对消息律，只通过 $`\mathbb E\,G\odot G`$。它就是消息对此后所有步长的影响。
- **(d)** 若尾段梯度与 $`\theta`$ 无关且 $`\lambda_{\rm wd}=0`$：$`\chi^\theta\equiv\nabla F`$，$`\chi^m_{t+1}=-\eta\sum_{k>t}\beta_1^{\,k-t}D_1^{(k)}\nabla F`$，$`\chi^v_{t+1}=\eta\sum_{k>t}\beta_2^{\,k-t}D_2^{(k)}\nabla F`$。
- **(e)** 随机尾段：在求导与期望可交换时，取 costate 的期望。

*推导要点.*

- 一次 Adam 更新是 $`(\theta,m,v)`$ 的 $`C^1`$ 映射。它的 Jacobian 有 $`\partial m'/\partial m=\beta_1I`$、$`\partial v'/\partial v=\beta_2I`$、$`\partial\theta'/\partial m'=-\eta D_1`$、$`\partial\theta'/\partial v'=\eta D_2`$ 这些块，再加上经由 $`g_k(\theta)`$ 的 $`J_k`$ 项。
- continuation value 是 $`F`$ 与各尾段映射的复合。从末端开始累积链式法则（adjoint 过程），得 $`\chi_k=D\Phi_k^\top\chi_{k+1}`$；把转置后的各块相乘就得到递推。
- 裁剪的 Jacobian 是 $`c\,r/\lVert r\rVert`$ 的导数。

*用到：* [Proposition 5.1](../proofs/05_optimizer_state_ZH.md#s5-1)、[Proposition 5.5](../proofs/05_warm_adam_ZH.md#s5-5)。*被用于：* Corollary 7.2、Proposition 7.3、Remark 7.5、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/07_second_moment_channel_ZH.md#s7-1) · [实验](../experiments/07_second_moment_channel_ZH.md#c7-1) · 已证明

#### [Corollary 7.2](../proofs/07_second_moment_channel_ZH.md#s7-2) · 平稳情形的幅度定律

考虑单个坐标，假设：尾段梯度与 $`\theta`$ 无关，平稳，均值 $`\bar g\ne0`$、平方的均值 $`\bar v>0`$；$`F`$ 关于 $`\theta_i`$ 仿射；$`\epsilon=0`$，不做 bias correction，没有 decay；状态 warm，$`m_t=\bar g`$、$`v_t=\bar v`$；消息之后还有 $`T`$ 次尾段更新。记

- $`\alpha=-(\nabla F)_i\,\mathrm{sign}(\bar g)`$，$`\lambda=g\,\mathrm{sign}(\bar g)/\sqrt{\bar v}`$（以 running RMS 为单位的消息幅度）；
- $`\rho=\lvert\bar g\rvert/\sqrt{\bar v}`$（该坐标的信噪比）；
- $`b_j=(1-\beta_1)\beta_1^{\,j}`$，$`c_j=(1-\beta_2)\beta_2^{\,j}`$，$`\omega_T=1-\beta_2^{\,T+1}`$。

**(a)** 用 $`g`$ 替换一条零梯度消息时的 mean-field response（把步长 $`m/\sqrt v`$ 换成 $`\mathbb Em/\sqrt{\mathbb Ev}`$）对任意 $`\lambda`$ 为

```math
\Delta F(\lambda)=\alpha\eta\sum_{j=0}^{T}\Big[\frac{\rho+b_j(\lambda-\rho)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho(1-b_j)}{\sqrt{1-c_j}}\Big].
```

- **(b)** 记 $`Q(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho\,\omega_T\lambda^2`$，$`S(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho\,\omega_T\lambda^2`$。若 $`\beta_2\ge\tfrac12`$，则 $`\lvert\Delta F-\alpha\eta Q\rvert\le11(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\,\lvert\alpha\rvert\eta S`$，一阶价值也满足同样的界。所以 $`Q`$ 是凹的，在 $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ 处最大。当 $`u=11(1-\beta_2)\max\lbrace1,9\lambda^{\star2}\rbrace\le1/8`$ 时，变号点离 $`2\lambda^\star`$ 的相对距离不超过 $`4u`$；阈值 $`2\lambda^\star`$ 只在 $`(1-\beta_2)(2\lambda^\star)^2\ll1`$ 时可用。
- **(c)** 当 $`\lambda\to\infty`$ 时 $`\Delta F\to\alpha\eta[\kappa_{\rm iso}(T)-\rho\Lambda(T)]`$，其中 $`\kappa_{\rm iso}(T)=\sum_jb_j/\sqrt{c_j}`$，$`\Lambda(T)=\sum_j(1-b_j)/\sqrt{1-c_j}\approx T+1`$。设 $`\alpha>0`$：若 $`\rho>\kappa_{\rm iso}(T)/\Lambda(T)`$，所有足够大的同向消息都劣于零梯度消息；若 $`\rho<\kappa_{\rm iso}(T)/\Lambda(T)`$，则都更好。
- **(d)** SGD 下 response 关于 $`\lambda`$ 是线性的。

*推导要点.*

- 在 mean-field 替换下，消息造成的偏差按几何速度衰减：$`\mathbb Em_{t+j}=\bar g+b_j(g-\bar g)`$，$`\mathbb Ev_{t+j}=\bar v(1+c_j(g^2/\bar v-1))`$。对有消息、无消息两种情形，把 $`T+1`$ 步的 $`\mathbb Em/\sqrt{\mathbb Ev}`$ 相加再相减，就得到 (a)。
- (b) 展开 $`(1+c_j(\lambda^2-1))^{-1/2}`$，对一阶、二阶余项用中值定理给出显式界；交叉项 $`\sum_jb_jc_j`$ 并入常数。
- (c) 逐项取极限，第一个分式趋于 $`b_j/\sqrt{c_j}`$。极限为负、又有 $`\Delta F'(0)>0`$，就必然存在正零点。

*例子（由公式解析算出）.* 取 $`\beta_1=0.9`$、$`\beta_2=0.99`$、$`\rho=0.5`$。

- 剩余 $`T=1000`$ 次更新时，$`\lambda^\star=2`$。精确公式在 $`\lambda\approx1.96`$ 处最大，在 $`\lambda\approx3.98`$ 处变号，紧挨着 $`2\lambda^\star=4`$。幅度为 running RMS 两倍的消息最好；幅度为 RMS 四倍以上的消息还不如不发。
- 剩余 $`T=20`$ 次更新时，$`2\lambda^\star\approx18.7`$，但精确的变号点在 $`\lambda\approx112`$，那里二次律早已不适用。大幅度极限为负，但只是略负（$`\rho\Lambda\approx10.1`$，$`\kappa_{\rm iso}\approx9.2`$）。

*用到：* Theorem 7.1、[Proposition 6.11(c)](../proofs/06_learner_relative_information_ZH.md#s6-11)（常数 $`\kappa_{\rm iso}`$）。*被用于：* Proposition 7.3、Remark 7.4–7.5、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/07_second_moment_channel_ZH.md#s7-2) · [实验](../experiments/07_second_moment_channel_ZH.md#c7-2) · 已证明（mean-field 替换是 $`\Delta F`$ 定义的一部分；它有多准，是一个要测量的问题）

#### [Proposition 7.3](../proofs/07_second_moment_channel_ZH.md#s7-3) · 两类 proxy 的不一致性

- **(a)** 只依赖于状态与反馈律、不依赖于任务的统计量，总会在某个任务上把某对反馈律排错，或者分不开它们：得分 $`F`$ 与得分 $`1-F`$ 要求相反的顺序。
- **(b)** 在 Corollary 7.2 的设定下，取 $`\alpha>0`$、$`0<\lambda_1<\lambda_2`$：
  - 只看参数的分数 $`\langle\chi^\theta,\Delta\theta(g)\rangle=\alpha\eta\,(\beta_1\rho+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$ 在 $`\lambda<(1-\beta_1)\beta_2/(\beta_1\rho(1-\beta_2))`$ 时递增，所以偏好 $`\lambda_2`$；
  - 任何余弦分数都给两条消息打平；
  - 只要 $`\Delta F(\lambda_2)<\Delta F(\lambda_1)`$，真实价值就偏好 $`\lambda_1`$；在 Corollary 7.2(b) 的误差范围内，$`\lambda_1+\lambda_2>2\lambda^\star`$ 时这一点成立。

*推导要点.*

- (a)：把任务反过来，所有真实顺序都反过来，而 task-blind 统计量一个也不变。[Proposition 3.1(b)](../proofs/03_learning_consequence_ZH.md#s3-1) 的标签包给出一对更自然的例子。
- (b)：只看参数的分数就是消息自己这一步，它对 $`\lambda`$ 的导数与 $`(1-\beta_1)\beta_2-\beta_1\rho(1-\beta_2)\lambda`$ 同号；真实价值沿着凹函数 $`Q`$ 走，过了 $`\lambda^\star`$ 就下降。

*用到：* Corollary 7.2、[Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1)。*被用于：* Remark 7.4–7.5、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/07_second_moment_channel_ZH.md#s7-3) · [实验](../experiments/07_second_moment_channel_ZH.md#c7-4) · 已证明

#### [Remark 7.4](../proofs/07_second_moment_channel_ZH.md#s7-4) · 与已有工作的关系，以及裁剪的作用

- Liu 与 Li（2026）分析了扰动在 AdamW 各状态中的短 horizon 传播，没有给出幅度定律或阈值。
- Adam 版 In-Run Data Shapley（Ding et al., 2026）和用 Adam 步长的 TracIn 式分数属于只看参数的一类；LESS（Xia et al., 2024）属于余弦分数。两类都在 Proposition 7.3(b) 的覆盖范围内。
- 全局范数裁剪限制了 $`\lambda`$，从而限制了二阶矩通道的损失。这是一个可检验的机制，不作为结论。

[证明](../proofs/07_second_moment_channel_ZH.md#s7-4) · remark

#### [Remark 7.5](../proofs/07_second_moment_channel_ZH.md#s7-5) · 一阶排序器在哪里失效

- 一对候选增量之差的二阶矩占比高时，消息相对于 $`v`$ 很大。这正是 Corollary 7.2(a)、(c) 所描述的非线性、会饱和的区域，线性化在那里高估了 $`v`$ 通道的伤害。
- 在那里任何一阶分数都不可靠，包括用精确 continuation gradient 的完整状态分数。
- 在那里排序应当用有限替换的测量（[Theorem 8.8](../proofs/08_certificate_ZH.md#s8-8)）或非线性公式。

[证明](../proofs/07_second_moment_channel_ZH.md#s7-5) · [实验](../experiments/07_second_moment_channel_ZH.md#c7-5) · remark
