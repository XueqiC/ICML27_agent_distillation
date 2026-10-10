# N19 · 每种学习器要求反馈携带什么 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N19_learner_statistics_ZH.md)

## 设定与假设

沿用节点 [N18](N18_optimizer_transformed_response_ZH.md) 的设定。

- 完整状态为 $x=(\theta,s)$，消息 $y$ 的 consumed gradient 为 $G=G(y)$，优化器增量为 $\Delta(x,G)$。
- 对某状态处的两个分布 $P,Q$，一阶统计量是平均增量之差 $\mu^P-\mu^Q$。由节点 N18 的定理 C3，它在延续值梯度张成的 $U$ 上的投影，正是两个反馈分布在一阶上必须一致的量。
- 记 $\delta_1=\mathbb E_PG-\mathbb E_QG$，$\delta_2=\mathbb E_PG^{\odot2}-\mathbb E_QG^{\odot2}$。
- $\Delta$ 中与消息无关的项都会在 $\mu^P-\mu^Q$ 中抵消，下面略去：权重衰减、动量衰减，以及作用在旧状态上的 bias-correction 常数。

**优化器.** 所有运算都按坐标进行。

- *SGD：* $\theta^+=(1-\eta\lambda)\theta-\eta G$。
- *Heavy-ball：* $m^+=\beta m+G$，$\theta^+=\theta-\eta m^+$。*Nesterov：* $m^+=\beta m+G$，$\theta^+=\theta-\eta(G+\beta m^+)$。
- *Adam 与 AdamW*（带 bias correction，$\epsilon$ 在根号外，步数计数器为 $t$）：$m^+=\beta_1m+(1-\beta_1)G$，$v^+=\beta_2v+(1-\beta_2)G^{\odot2}$，$\theta^+=(1-\eta\lambda)\theta-\eta\,n(G)/(d(G)+\epsilon)$，其中 $n=m^+/(1-\beta_1^t)$，$d=\sqrt{v^+/(1-\beta_2^t)}$。*warm* 状态指对所有 $i$ 有 $v_i>0$；*冷*状态指 $t=1$ 时 $m=v=0$。
- *全局范数裁剪：* $\operatorname{clip}_c(G)=G\min\lbrace 1,c/\lVert G\rVert\rbrace$，在基础优化器之前作用。
- *Lion 形式的 sign 优化器：* $u=\operatorname{sign}(\beta_1m+(1-\beta_1)G)$，$\theta^+=\theta-\eta u$，$m^+=\beta_2m+(1-\beta_2)G$。
- *LoRA：* 被适配的权重 $W=W_0+\alpha BA$，$A\in\mathbb R^{r\times k}$，$B\in\mathbb R^{d\times r}$，全权重梯度为 $\Gamma=\nabla_W\ell$。可训练参数是 $(A,B)$。标准初始化为 $B_0=0$，$A_0$ 随机。

**warm Adam 记号**（节点 [N12](N12_optimizer_warmth_ZH.md)）：$a_i=\sqrt{\beta_2v_i/(1-\beta_2^t)}$，$P=\operatorname{diag}(1/(a_i+\epsilon))$，$c_1=(1-\beta_1)/(1-\beta_1^t)$，$\kappa_i=(1-\beta_2)G_i^2/(\beta_2v_i)$，$u_i=\lvert n_i\rvert/(a_i+\epsilon)$。

## 陈述

**定理 C4（每种学习器的一阶统计量）.**

**(a) SGD.** $\mu^P-\mu^Q=(-\eta\,\delta_1)$。一阶统计量是平均 consumed gradient；二阶统计量是 $\mathbb E\,GG^\top$。

**(b) Heavy-ball 与 Nesterov 动量.** 增量关于 $G$ 是仿射的：

- heavy-ball：$\Delta^\theta=-\eta(\beta m+G)$，$\Delta^m=(\beta-1)m+G$；
- Nesterov：$\Delta^\theta=-\eta\big((1+\beta)G+\beta^2m\big)$，$\Delta^m=(\beta-1)m+G$。

因此一阶项为
$$\langle\nabla V,\mu^P-\mu^Q\rangle=\big\langle-\eta\gamma\nabla_\theta V+\nabla_mV,\ \delta_1\big\rangle,\qquad\gamma=1\ \text{(heavy-ball)},\quad\gamma=1+\beta\ \text{(Nesterov)}.$$
统计量是平均 consumed gradient。动量改变的是它的权重，而不是统计量本身。

**(c) warm 状态下的 Adam 与 AdamW.** 存在依赖消息的 $e(G)$，满足 $\lvert e_i\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace$，使得在 $(\theta,m,v)$ 三块上
$$\mu^P-\mu^Q=\Big(-\eta c_1P\,\delta_1+\big(\mathbb E_Pe-\mathbb E_Qe\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big)$$
一阶统计量是 $(\mathbb E\,G,\ \mathbb E\,G^{\odot2})$ 这一对，误差为 warmth 误差，每个坐标至多 $\eta(\mathbb E_P+\mathbb E_Q)[u_i\min\lbrace \kappa_i/2,1\rbrace]$。按坐标的二阶矩通过 $v$ 在一阶上进入，权重为 $(1-\beta_2)\nabla_vV$。因此，除非 $\nabla_vV\perp\delta_2$，均值相同的分布对 warm Adam 在一阶上并不等价。

**(d) 冷状态下的 Adam.** 记 $h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)$，
$$\mu^P-\mu^Q=\Big(-\eta\big(\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big),\ \ (1-\beta_1)\,\delta_1,\ \ (1-\beta_2)\,\delta_2\Big).$$
参数块只看到 $\mathbb E\,h_\epsilon(G)$，当 $\epsilon\to0$ 时就是符号分布（节点 [N11](N11_adam_cold_start_ZH.md)）。矩块保留 $\mathbb E\,G$ 和 $\mathbb E\,G^{\odot2}$。若训练在这次更新后结束（$R=1$），延续值就是 $F_E(\theta)$，一阶项为
$$-\eta\,\big\langle\nabla_\theta V,\ \mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\big\rangle.$$

**(e) 梯度裁剪.** 对裁剪 SGD，$\mu^P-\mu^Q=-\eta\big(\mathbb E_P\operatorname{clip}_c(G)-\mathbb E_Q\operatorname{clip}_c(G)\big)$，所以统计量是裁剪后的均值。

- *见证.* 取 $c=1$，$P(G=4e_1)=\tfrac14$，$P(G=-\tfrac43e_1)=\tfrac34$，$Q=\delta_0$。则 $\delta_1=0$，但 $\mathbb E_P\operatorname{clip}_1G=-\tfrac12e_1$，一阶项为 $\tfrac\eta2\,\partial_{\theta_1}V$。
- *裁剪恒为 active.* 若在两个分布下几乎必然 $\lVert G\rVert>c$，统计量就是 $c\,\mathbb E[G/\lVert G\rVert]$，即平均单位方向。若两个分布还诱导出相同的 $G/\lVert G\rVert$ 分布，则增量分布完全相同，两者在每一阶上都等价。
- *带裁剪的基础优化器.* 对 (a)–(d) 与 (f) 中的任一基础优化器，统计量都取自裁剪后的梯度 $\operatorname{clip}_c(G)$。

**(f) 带动量的 sign 优化器（Lion 形式）.** 对 $m_i\ne0$，
$$\mathbb E\,\Delta^\theta_i=-\eta\operatorname{sign}(m_i)\,(1-2\pi_i),\qquad\pi_i=\Pr\lbrace \operatorname{sign}(\beta_1m_i+(1-\beta_1)G_i)\ne\operatorname{sign}(m_i)\rbrace ,$$
其中取零算作半次翻转；并且 $\Delta^m=(1-\beta_2)(G-m)$。参数块的统计量是翻转概率向量 $\pi$。

- 若动量占优，即在两个分布下几乎必然 $(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert$，则 $\pi_i=0$，坐标 $i$ 的参数增量与消息无关。
- $m$ 块仍然携带 $(1-\beta_2)\delta_1$。它的一阶效应是 $\langle\nabla_mV,(1-\beta_2)\delta_1\rangle$；在延续值关于 $m$ 局部为常数的地方，也就是 $m$ 的微小变化不会改变之后任何一步更新符号的地方，这一项为零。

**(g) 初始化时的 LoRA.** adapter 的梯度为 $G_A=\alpha B^\top\Gamma$，$G_B=\alpha\Gamma A^\top$。在 $B=B_0=0$ 处：

1. 对每条消息都有 $G_A=0$。在 SGD、从 $m=0$ 出发的动量以及冷启动 Adam 下，第一次更新对每个分布都不改变 $A$，在精确算术中逐位不变。
2. 对 SGD 与动量，$B$ 块的统计量是 $\alpha\,\mathbb E[\Gamma]A_0^\top$；对冷启动 Adam 是 $\mathbb E\,h_\epsilon(\alpha\Gamma A_0^\top)$。
3. 若两个分布的平均全权重梯度只相差一个满足 $\Delta\Gamma A_0^\top=0$ 的 $\Delta\Gamma$，则在 SGD 和动量下它们在第 1 步的一阶上一致。在冷启动 Adam 下，若 sketch $\Gamma A_0^\top$ 的符号分布相同，则当 $\epsilon\to0$ 时第一次更新完全相同。
4. 第 1 步之后 $B\ne0$，$G_A=\alpha B^\top\Gamma$ 带来第二个 sketch，所以统计量的维数随训练增长。

**汇总.** 每种学习器对反馈的一阶统计量：

| 学习器与状态 | 反馈分布的一阶统计量 | 进入渠道 |
|---|---|---|
| SGD | $\mathbb E\,G$ | $\theta$ |
| heavy-ball、Nesterov | $\mathbb E\,G$ | $\theta$ 与 $m$，权重为 $-\eta\gamma\nabla_\theta V+\nabla_mV$ |
| Adam/AdamW，warm | $\mathbb E\,G$ 与 $\mathbb E\,G^{\odot2}$，误差为 warmth 误差 $O(\eta\,u\,\kappa)$ | $\theta$（经 $P$ 预条件）、$m$、$v$ |
| Adam/AdamW，冷启动 | $\mathbb E\,h_\epsilon(G)$（$\epsilon\to0$ 时为符号分布）；$\mathbb E\,G$ 与 $\mathbb E\,G^{\odot2}$ | $\theta$；$m$ 与 $v$ |
| 裁剪 SGD | $\mathbb E\operatorname{clip}_c(G)$；裁剪恒为 active 时为平均单位方向 | $\theta$ |
| 带裁剪的基础优化器 | 基础优化器的统计量，取自 $\operatorname{clip}_c(G)$ | 同基础优化器 |
| Lion / 带动量的 sign | 翻转概率 $\pi$；动量占优时 $\theta$ 中没有 | $\theta$；$m$ |
| $B=0$ 时的 LoRA，第 1 步 | 秩 $r$ 的 sketch $\mathbb E[\Gamma]A_0^\top$（SGD），或其符号分布（冷启动 Adam）；$A$ 固定不动 | 只有 $B$ |

## 证明

全程有 $\mu^P-\mu^Q=\mathbb E_PD-\mathbb E_QD$，其中 $D(y)=\mathbb E_\xi\Delta(x,G(y,\xi))$。$\Delta$ 中任何仅是状态 $x$ 的固定函数的项，在两个分布下期望相同，因而抵消。每种情形都由更新规则算出 $\Delta$，再取期望之差。

**(a)** $\Delta^\theta=-\eta\lambda\theta-\eta G$。第一项抵消，由期望的线性得 $-\eta\delta_1$。二阶统计量来自取 $k=2$ 的定理 C1：除去只依赖状态的项，$\mathbb E\,\Delta^{\otimes2}=\eta^2\mathbb E\,GG^\top$。

**(b)** 对 heavy-ball，$\theta^+-\theta=-\eta m^+=-\eta(\beta m+G)$，$m^+-m=(\beta-1)m+G$。对 Nesterov，$\theta^+-\theta=-\eta(G+\beta m^+)=-\eta(G+\beta^2m+\beta G)$。两种情形中只依赖状态的项都抵消，依赖消息的部分在 $\theta$ 中是 $-\eta\gamma G$，在 $m$ 中是 $G$。因此 $\mu^P-\mu^Q=(-\eta\gamma\delta_1,\ \delta_1)$，与 $(\nabla_\theta V,\nabla_mV)$ 配对即得所示式子。

**(c)** 对矩块，$m^+-m=(1-\beta_1)(G-m)$，$v^+-v=(1-\beta_2)(G^{\odot2}-v)$，分别给出 $(1-\beta_1)\delta_1$ 与 $(1-\beta_2)\delta_2$。

对参数块，写出节点 [N12](N12_optimizer_warmth_ZH.md) 的冻结分母替代更新：
$$\tilde\theta^+=(1-\eta\lambda)\theta-\eta P\,n(G)=(1-\eta\lambda)\theta-\eta P\frac{\beta_1m}{1-\beta_1^t}-\eta c_1P\,G .$$
Lemma W0 给出 $\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta u_i\min\lbrace \kappa_i/2,1\rbrace$。定义 $e(G)=\theta^+-\tilde\theta^+$，则 $\Delta^\theta=(\text{state-only terms})-\eta c_1PG+e(G)$，取期望之差即得 $\theta$ 块。对 $\mathbb E_Pe_i-\mathbb E_Qe_i$ 的界来自 $\lvert\mathbb E_Pe_i-\mathbb E_Qe_i\rvert\le\mathbb E_P\lvert e_i\rvert+\mathbb E_Q\lvert e_i\rvert$。

最后一个论断：若 $\delta_1=0$，一阶项化为 $\langle\nabla_\theta V,\mathbb E_Pe-\mathbb E_Qe\rangle+(1-\beta_2)\langle\nabla_vV,\delta_2\rangle$，只要 $\nabla_vV\not\perp\delta_2$，第二项就非零。

**(d)** 在 $m=v=0$、$t=1$ 时，$m^+=(1-\beta_1)G$，且 $1-\beta_1^1=1-\beta_1$，所以 $n=G$；同理 $v^+=(1-\beta_2)G^{\odot2}$，$d=\lvert G\rvert$。于是 $\theta^+=(1-\eta\lambda)\theta-\eta G/(\lvert G\rvert+\epsilon)=(1-\eta\lambda)\theta-\eta h_\epsilon(G)$。三个增量分别是 $-\eta\lambda\theta-\eta h_\epsilon(G)$、$(1-\beta_1)G$ 和 $(1-\beta_2)G^{\odot2}$，即得所示式子。当 $\epsilon\to0$ 时，$h_\epsilon(g)_i\to\operatorname{sign}(g_i)$，且 $h_0(0)=0$（节点 N11 的命题 A2）。

若 $R=1$，更新后的延续值是 $F_E(\theta)$，于是 $\nabla_mV=\nabla_vV=0$，一阶项为 $\langle\nabla_\theta V,\mu^P_\theta-\mu^Q_\theta\rangle=-\eta\langle\nabla_\theta V,\mathbb E_Ph_\epsilon(G)-\mathbb E_Qh_\epsilon(G)\rangle$。

**(e)** 对裁剪 SGD，除权重衰减外 $\Delta^\theta=-\eta\operatorname{clip}_c(G)$，期望之差如所述。

- *见证.* $\mathbb E_PG=\tfrac14\cdot4e_1+\tfrac34\cdot(-\tfrac43e_1)=0=\mathbb E_QG$。又 $\operatorname{clip}_1(4e_1)=e_1$，$\operatorname{clip}_1(-\tfrac43e_1)=-e_1$，所以 $\mathbb E_P\operatorname{clip}_1G=(\tfrac14-\tfrac34)e_1=-\tfrac12e_1$，而 $\mathbb E_Q\operatorname{clip}_1G=0$。因此 $\mu^P-\mu^Q=-\eta(-\tfrac12e_1)=\tfrac\eta2e_1$，一阶项为 $\tfrac\eta2\,\partial_{\theta_1}V$。
- *裁剪恒为 active.* 若 $\lVert G\rVert>c$，则 $\operatorname{clip}_c(G)=cG/\lVert G\rVert$，所以 $\mathbb E\operatorname{clip}_cG=c\,\mathbb E[G/\lVert G\rVert]$。若两个分布诱导出相同的 $G/\lVert G\rVert$ 分布，则增量 $-\eta cG/\lVert G\rVert$ 同分布，定理 C1 的每一项以及精确 response 都相同。
- *带裁剪的基础优化器.* 它们把基础更新作用在 $\operatorname{clip}_c(G)$ 上；在基础优化器的计算中把 $G$ 换成 $\operatorname{clip}_c(G)$ 即可。

**(f)** 记 $c_i=\beta_1m_i+(1-\beta_1)G_i$。对 $m_i\ne0$：

- 不发生翻转时，$\operatorname{sign}(c_i)=\operatorname{sign}(m_i)$；
- 严格翻转时，$\operatorname{sign}(c_i)=-\operatorname{sign}(m_i)$；
- 当 $c_i=0$ 时，$\operatorname{sign}(c_i)=0=\operatorname{sign}(m_i)(1-2\cdot\tfrac12)$。

因此在零算作半次翻转的约定下，$\mathbb E\operatorname{sign}(c_i)=\operatorname{sign}(m_i)(1-2\pi_i)$，且 $\mathbb E\Delta^\theta_i=-\eta\operatorname{sign}(m_i)(1-2\pi_i)$。

若几乎必然 $(1-\beta_1)\lvert G_i\rvert<\beta_1\lvert m_i\rvert$，则 $c_i$ 几乎必然与 $m_i$ 同号，于是 $\pi_i=0$，且对每条消息 $\Delta^\theta_i=-\eta\operatorname{sign}(m_i)$。

$m$ 块的增量是 $(1-\beta_2)(G-m)$，给出 $(1-\beta_2)\delta_1$。之后的参数更新只通过之后各个 $c$ 的符号依赖于 $m$；在对 $m$ 作微小扰动时这些符号都不变的地方，延续值关于 $m$ 局部为常数，因此那里 $\nabla_mV=0$。

**(g)** 对 $W=W_0+\alpha BA$，链式法则给出 $\partial\ell/\partial B_{jl}=\alpha\sum_q\Gamma_{jq}A_{lq}$，即 $G_B=\alpha\Gamma A^\top$；同理 $\partial\ell/\partial A_{lq}=\alpha\sum_jB_{jl}\Gamma_{jq}$，即 $G_A=\alpha B^\top\Gamma$。

1. 在 $B_0=0$ 处 $G_A$ 恒为零。SGD 使 $A$ 移动 $-\eta\cdot0=0$；从 $m=0$ 出发的动量给出 $m^+=0$，因此不动；冷启动 Adam 使 $A$ 移动 $-\eta h_\epsilon(0)=0$，因为 $\epsilon>0$ 时 $h_\epsilon(0)=0/(0+\epsilon)=0$。三者对每条消息都成立，从而对每个分布都成立。
2. 在 SGD 下 $B$ 的增量是 $-\eta\alpha\Gamma A_0^\top$；从 $m=0$ 出发的动量第一步与 SGD 相同；冷启动 Adam 下是 $-\eta h_\epsilon(\alpha\Gamma A_0^\top)$。取期望即得统计量。
3. 若 $\mathbb E_P\Gamma-\mathbb E_Q\Gamma=\Delta\Gamma$ 且 $\Delta\Gamma A_0^\top=0$，则在 SGD 和动量下 $B$ 块的均值相同，$A$ 块的均值都为零，所以第 1 步 $\mu^P=\mu^Q$。在冷启动 Adam 下，若两个 sketch 的符号分布相同，极限 $h_0(\alpha\Gamma A_0^\top)$ 同分布，所以当 $\epsilon\to0$ 时第一次更新相同。
4. 一步之后一般 $B_1\ne0$，于是 $G_A=\alpha B_1^\top\Gamma$ 不再为零。$\square$

## 注记

1. **重要的是统计量，而不只是数据.** 同一对反馈分布，对一种学习器可以等价，对另一种却相去甚远：
   - C23 的 moment 分布均值相同，所以对 SGD 和动量在一阶上等价；
   - 但对冷启动 Adam、sign 优化器和裁剪 SGD，它们在一阶上不同，因为这些优化器的增量看到的是符号分布或裁剪均值。

   因此，评判 teacher 数据必须放在学生实际使用的优化器下进行。
2. **few-shot LoRA 配方.** 我们的 Seed-SFT 配方用 AdamW 从冷状态训练 LoRA，并按全局范数 1 裁剪。主表 K20 训练在最后一次更新时记录的裁剪前范数为 5.28，所以裁剪是 active 的，(c)–(e) 行的统计量都取自裁剪后的梯度。在初始化时适用 (g) 行：第一次更新只通过秩 $r$ sketch $\Gamma A_0^\top$ 的符号分布看到反馈。
3. **与前面节点的关系.**
   - (d) 行统一了节点 [N11](N11_adam_cold_start_ZH.md) 的命题 A2 与 A4：参数块携带符号分布，矩块携带幅度，而 A4 表明这些幅度能使第二步反向。
   - (c) 行是节点 [N12](N12_optimizer_warmth_ZH.md) 命题 W1 的线性化形式，它把 W1 中的 $L_v\kappa$ 残差细化成关于 $\mathbb E\,G^{\odot2}$ 的显式线性项。
4. **先例.** Adam 的冷启动第一步（Kingma 与 Ba，2015）、把 Adam 看作带方差自适应的 sign 下降（Balles 与 Hennig，2018）、signSGD（Bernstein 等，2018）以及 Lion（Chen 等，2023）都是已有结果。LESS（Xia 等，2024）在 LoRA warmup 之后用 Adam 预条件的梯度特征做数据选择，对应 (c) 行的参数块。这里的贡献是包含矩块的完整状态清单，它就是定理 C3 意义下的最小一阶统计量。

## 依赖与出处

- **依赖：** [N18](N18_optimizer_transformed_response_ZH.md)（定理 C1 与 C3）、[N11](N11_adam_cold_start_ZH.md)（冷启动 Adam 映射 (A7)）、[N12](N12_optimizer_warmth_ZH.md)（Lemma W0 与 warm Adam 记号）。
- **被用于：** [N20](N20_first_order_audit_ZH.md)、[N17](N17_ced_method_ZH.md)。
- **出处：** `docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md`，§4（定理 C4(a)–(g)）。
