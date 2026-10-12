# Proposition 7.3 · 两类 proxy 的不一致性

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/07-03_inconsistency_two_proxy_classes_ZH.md) · [证明](07-03_inconsistency_two_proxy_classes_ZH.md) · [实验](../experiments/07-03_inconsistency_two_proxy_classes_ZH.md) · [English](07-03_inconsistency_two_proxy_classes.md)

**本页证明什么.** (a) 不看部署任务的排序统计量，会在某个任务上排错或分不开：把任务反过来，所有真实顺序都反过来，而统计量不变。(b) 在幅度定律的 warm、平稳 Adam 区间里，消息的只看参数的分数在一个给定范围内是幅度的显式递增函数，余弦分数分不开同一梯度的两个正倍数，而真实价值过了最优幅度就下降；所以在一个显式的消息对集合上，前者排反，后者打平。

**本页用到的设定与记号.**

- *学习价值.* 对实际学习器在固定状态上消耗的反馈律 $`P`$，$`L_F(P)`$ 是得分为 $`F\in[0,1]`$ 的任务上的期望最终得分（[Definitions 1.3–1.4](../README_ZH.md#s1-3)）。
- *task-blind 统计量.* $`\mathcal T(P)`$ 只依赖于状态与反馈律、不依赖于任务，并按它的值排序。
- *Adam 区间.* warm、不做 bias correction 的 Adam 的单个坐标 $`i`$（$`\epsilon=0`$，无 decay）；尾段梯度与 $`\theta`$ 无关，平稳，均值 $`\bar g_i\ne0`$、平方的均值 $`\bar v_i`$；得分关于 $`\theta_i`$ 仿射；消息之后有 $`T`$ 次尾段更新；warm 状态 $`m_t=\bar g_i`$，$`v_t=\bar v_i`$。
- $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$，$`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$，$`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}`$，$`\omega_T=1-\beta_2^{\,T+1}`$，$`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$。
- $`\Delta F_i(\lambda)`$、$`Q_i(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho_i\omega_T\lambda^2`$、$`S_i(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho_i\omega_T\lambda^2`$，以及界 $`\lvert\Delta F_i-\alpha_i\eta Q_i\rvert\le11(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\lvert\alpha_i\rvert\eta S_i`$，都取自 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)。
- $`S_{\rm PO}(g)=\langle\chi^\theta,\Delta\theta(g)\rangle`$ 是 [Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1) 的参数通道；余弦分数是消息梯度方向（在单个坐标上，参数步方向）的函数。

**陈述（[Proposition 7.3](../statements/07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)）.**

- **(a)** 对任何 task-blind 排序统计量 $`\mathcal T`$，存在两个任务，以及同一学习器、同一状态下的两个消息律 $`P,Q`$：两者的真实顺序在两个任务上相反，而 $`\mathcal T(P),\mathcal T(Q)`$ 在两个任务上都相同；所以 $`\mathcal T`$ 至少在一个任务上排错，或分不出一个非零的差。
- **(b)** 在 Adam 区间中取 $`\alpha_i>0`$ 与同向幅度 $`0<\lambda_1<\lambda_2`$：$`S_{\rm PO}=\alpha_i\eta\,h(\lambda)`$，其中 $`h(\lambda)=(\beta_1\rho_i+(1-\beta_1)\lambda)/\sqrt{\beta_2+(1-\beta_2)\lambda^2}`$ 在 $`\lambda<\lambda_{\rm PO}=(1-\beta_1)\beta_2/(\beta_1\rho_i(1-\beta_2))`$ 上严格递增，所以 $`\lambda_2<\lambda_{\rm PO}`$ 时它把较大的消息排在前面；任何余弦分数都给两条消息打平；只要 $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$，真实价值就偏好较小的消息，而当 $`\lambda_1+\lambda_2>2\lambda^\star`$ 且 $`Q_i(\lambda_1)-Q_i(\lambda_2)>11(1-\beta_2)\max\lbrace1,\lambda_2^2\rbrace\big(S_i(\lambda_1)+S_i(\lambda_2)\big)`$ 时这一点成立。

**证明.**

*(a)* 任取得分为 $`F`$ 的任务及满足 $`L_F(P)\ne L_F(Q)`$ 的两个消息律，再取得分为 $`F'=1-F\in[0,1]`$ 的任务。由于 $`L_{F'}=1-L_F`$，有 $`L_{F'}(P)-L_{F'}(Q)=-(L_F(P)-L_F(Q))`$，而不看任务的 $`\mathcal T`$ 不变。若 $`\mathcal T(P)\ne\mathcal T(Q)`$，它在两个任务上给 $`P,Q`$ 相同的顺序，必在其中一个上排错；若 $`\mathcal T(P)=\mathcal T(Q)`$，它分不出一个非零的差。

[Proposition 3.1](../statements/03-01_static_value_not_learning_value_ZH.md#s3-1)(b)（边缘分布与平均梯度相同不能确定学习价值）给出一对更自然、性质相同的任务：两个边缘分布相同、平均梯度相同的标签包，它们学习价值之差的符号在 $`J(p)=p^2`$ 与 $`J(p)=1-p^2`$ 下相反。

*(b)* 在线性 response 区间里 $`\chi^\theta=\nabla F`$（[Theorem 7.1](../statements/07-01_state_channel_decomposition_ZH.md#s7-1)(d)：尾段梯度与 $`\theta`$ 无关时，参数 costate 为常数）。从 warm 状态出发，消息自己这一步为

```math
\Delta\theta_i(g)=-\eta\,\frac{\beta_1\bar g_i+(1-\beta_1)g_i}{\sqrt{\beta_2\bar v_i+(1-\beta_2)g_i^2}}=-\eta\,\mathrm{sign}(\bar g_i)\,h(\lambda),
```

所以 $`S_{\rm PO}=(\nabla F)_i\Delta\theta_i=\alpha_i\eta\,h(\lambda)`$。求导得

```math
h'(\lambda)=\frac{(1-\beta_1)\beta_2-\beta_1\rho_i(1-\beta_2)\lambda}{\big(\beta_2+(1-\beta_2)\lambda^2\big)^{3/2}},
```

它在 $`\lambda<\lambda_{\rm PO}`$ 时为正。取 $`\alpha_i>0`$ 与 $`\lambda_1<\lambda_2<\lambda_{\rm PO}`$，则 $`S_{\rm PO}(\lambda_2)>S_{\rm PO}(\lambda_1)`$。

- 同一梯度的正倍数方向相同；在单个坐标上，两条消息的参数步都指向 $`-\mathrm{sign}(\bar g_i)`$。所以任何余弦分数都给出平局。
- 真实价值方面，$`Q_i(\lambda_2)-Q_i(\lambda_1)=(\lambda_2-\lambda_1)\big[(1-\beta_1^{\,T+1})-\tfrac12\rho_i\omega_T(\lambda_1+\lambda_2)\big]`$，它为负当且仅当 $`\lambda_1+\lambda_2>2\lambda^\star`$。由 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)(b)（带显式误差界的二次律），在每个幅度上 $`\Delta F_i(\lambda)`$ 与 $`\alpha_i\eta Q_i(\lambda)`$ 之差不超过 $`11(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\lvert\alpha_i\rvert\eta S_i(\lambda)`$，且 $`\max\lbrace1,\lambda_1^2\rbrace\le\max\lbrace1,\lambda_2^2\rbrace`$。所以所列 margin 蕴含 $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$。$`\square`$

**关于证明的说明.**

- (b) 通过 Corollary 7.2(b) 的二次律，是关于小增量的陈述；增量大时要从 Corollary 7.2(a) 读 response，只要该公式给出 $`\Delta F_i(\lambda_2)<\Delta F_i(\lambda_1)`$，结论就成立。
- 在常见设定下，满足 $`2\lambda^\star<\lambda_1+\lambda_2`$ 且 $`\lambda_2<\lambda_{\rm PO}`$ 的区间非空。剩余步数长时 $`2\lambda^\star\approx2/\rho_i`$，而 $`\lambda_{\rm PO}\approx(1-\beta_1)/(\beta_1\rho_i(1-\beta_2))`$；$`\beta_1=0.9`$、$`\beta_2=0.99`$ 时约为 $`11/\rho_i`$。

## 参考文献

- [Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating)：TracIn。
- [Ding et al., 2026](../references_ZH.md#ding2026inrun)：Adam 版 In-Run Data Shapley（fixed-state）。
- [Xia et al., 2024](../references_ZH.md#xia2024less)：LESS。
