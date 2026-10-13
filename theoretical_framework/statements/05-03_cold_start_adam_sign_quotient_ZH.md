# Proposition 5.3 · Cold-start Adam：符号商与 optimizer 记忆

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-03_cold_start_adam_sign_quotient_ZH.md) · [证明](../proofs/05-03_cold_start_adam_sign_quotient_ZH.md) · [实验](../experiments/05-03_cold_start_adam_sign_quotient_ZH.md) · [English](05-03_cold_start_adam_sign_quotient.md)

**一段话.** Adam 从零 optimizer state 开始时，第一步参数更新是 consumed gradient 的逐坐标符号映射。这个命题精确刻画这样的 cold 步需要梯度分布的哪些信息：$`\epsilon=0`$ 时是联合符号分布，$`\epsilon>0`$ 时是整个分布。它给出符号近似的误差界，并说明 optimizer state 保留了梯度的幅度，所以之后的步仍可能依赖它。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

- **状态.** 完整 learner state 为 $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$。$`\theta`$ 是可训练参数（例如 LoRA adapter），$`o`$ 是 optimizer state（例如 Adam 的矩 $`m,v`$ 与步数计数器）。需要用到计数器时显式写出。
- **反馈与梯度.** 第 $`t`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 给出消息 $`Y`$。学习器在 loss 归一化与累加之后计算 consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$，$`\xi`$ 是声明的更新随机性。若有裁剪，裁剪算作 optimizer 的一部分。
- **Optimizer.** optimizer 是一个可测的增量映射 $`z^+=z+\Delta_t(z,g)`$。被比较的 feedback law 共用同一个 optimizer 与同一个梯度映射，只在消息分布上不同。换一个 optimizer 就是换一个 $`\Delta`$，用同样的方式比较。
- **本节局部复用的字母.** 本节中 $`s`$ 表示 Adam 二阶矩已吸收的梯度计算次数（Proposition 5.8），与第 6 节的 agent 状态 $`s`$ 无关。裁剪阈值记为 $`c`$，weight decay 系数记为 $`\lambda_{\rm wd}`$。

分布 $`P`$ 的学习 kernel 为

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

对 $`(t,z)`$ 处的分布 $`P`$，消息层面的平均增量为 $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$，并记

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation value.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ 是协议 $`P`$ 的 continuation value：之后的更新都按 $`P`$ 进行，然后部署学生并用 $`F_E`$ 打分（[Definition 1.4](../README_ZH.md#s1-4)）。$`L(P)=\mathbb E\,F_E(Z_R^P)`$，$`d_t^Q`$ 是协议 $`Q`$ 下 $`Z_t`$ 的分布。
- **范数.** 在 $`\mathbb R^{p+q}`$ 上固定一个范数 $`\lVert\cdot\rVert`$，其对偶范数记为 $`\lVert\cdot\rVert_*`$。

**Adam 与 AdamW.** Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW；[Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- 超参数满足 $`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，$`\eta>0`$，$`\lambda_{\rm wd}\ge0`$，$`\epsilon\ge0`$。运算都是精确实数运算。
- 下一个完整状态为 $`z^+=(\theta^+,m^+,v^+,t)`$。
- *cold* 状态指 $`m=v=0`$ 且 $`t=1`$；*warm* 状态指对每个坐标 $`i`$ 有 $`v_i>0`$。

**符号映射.** 对 $`\epsilon>0`$，$`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$。对 $`\epsilon=0`$，*理想符号更新*为 $`h_0(g)_i=\mathrm{sign}(g_i)`$，约定 $`\mathrm{sign}(0)=0`$。这个约定是定义的一部分；字面上的浮点 Adam 在 $`\epsilon=0`$ 时会算出 $`0/0`$。

**效用.** *即时参数效用*是 $`\theta_1`$ 的有界可测函数；*完整状态效用*是下一个完整状态 $`(\theta_1,m_1,v_1,1)`$ 的有界可测函数。consumed gradient 的分布 $`P,Q`$ 是 $`\mathbb R^p`$ 上的概率分布；在有限情形的陈述中，它们支撑在有限字母表 $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$ 上。记 $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$。

<a id="s5-3"></a>

**Proposition 5.3（Cold-start Adam：符号商与 optimizer 记忆）.** 从 cold 状态 $`(\theta_0,0,0,0)`$ 启动 Adam。

- **(a) 第一步.** $`\theta_1=\bar\theta-\eta\,h_\epsilon(G)`$。
- **(b) $`\epsilon=0`$：联合符号分布.** 两个分布对所有即时参数效用给出相同期望，当且仅当它们的联合符号向量分布相同。在有 $`s`$ 个可达符号模式的有限字母表上，对 $`\mathcal A`$ 上不受限的分布，足以决定所有这类效用的线性期望统计量最少为 $`s-1`$ 个；这里 $`s\le3^p`$，没有零坐标时 $`s\le2^p`$。逐坐标的符号边缘分布不够：取 $`p=2`$，$`P`$ 在 $`\lbrace (1,1),(-1,-1)\rbrace`$ 上均匀，$`Q`$ 在 $`\lbrace (1,-1),(-1,1)\rbrace`$ 上均匀。每个坐标的符号边缘分布相同，两者的平均步长都是 $`0`$，但效用 $`F(\theta)=1\lbrace (\theta_1-\bar\theta_1)(\theta_2-\bar\theta_2)>0\rbrace`$ 在 $`P`$ 下期望为 $`1`$，在 $`Q`$ 下为 $`0`$。
- **(c) $`\epsilon>0`$：完整分布.** $`h_\epsilon`$ 是逐坐标的双射 $`\mathbb R\to(-1,1)`$，其逆与导数如下。所以对所有即时参数效用相等，等价于 consumed gradient 的完整分布相等。在 $`\mathcal A`$ 上最少需要 $`m-1`$ 个统计量。

```math
h_\epsilon^{-1}(u)_i=\frac{\epsilon u_i}{1-\lvert u_i\rvert},\qquad\frac{d}{du}h_\epsilon^{-1}(u)=\frac{\epsilon}{(1-\lvert u\rvert)^2}.
```

**(d) 对 margin 敏感的近似.** 设 $`F:\mathbb R^p\to[0,1]`$ 对所有 $`u=\bar\theta-\eta h_\epsilon(g)`$、$`w=\bar\theta-\eta h_0(g)`$（$`g`$ 属于支撑）满足 $`\lvert F(u)-F(w)\rvert\le\sum_iL_i\lvert u_i-w_i\rvert`$。对 $`a>0`$ 记 $`\beta_{P,i}(a)=P(0<\lvert G_i\rvert<a)`$，并令 $`\Delta_{\rm sign}(F)=\lvert\mathbb E_PF(\bar\theta-\eta h_0(G))-\mathbb E_QF(\bar\theta-\eta h_0(G))\rvert`$。则 $`\Delta_{\rm sign}(F)\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$，且

```math
\big\lvert\mathbb E_PF(\bar\theta-\eta h_\epsilon(G))-\mathbb E_QF(\bar\theta-\eta h_\epsilon(G))\big\rvert\le\Delta_{\rm sign}(F)+\eta\sum_iL_i\Big[\beta_{P,i}(a)+\beta_{Q,i}(a)+\frac{2\epsilon}{a+\epsilon}\Big].
```

**(e) 保留下来的 optimizer state.** 当 $`\beta_1<1`$ 时，第一个完整状态中含有 $`m_1=(1-\beta_1)G`$，它是 $`G`$ 的单射函数。

- *反例.* 取一维，$`\beta_1=\beta_2=\tfrac12`$，$`\lambda_{\rm wd}=0`$，$`\epsilon=0`$，$`\eta>0`$，cold 启动。比较梯度历史 $`(G_1,G_2)=(1,-\tfrac34)`$ 与 $`(G_1',G_2')=(2,-\tfrac34)`$。两者都给出 $`\theta_1=\theta_1'=\theta_0-\eta`$，符号历史都是 $`(+,-)`$。第 2 步的 bias-corrected 矩为 $`\widehat m_2=-\tfrac16`$、$`\widehat m_2'=+\tfrac16`$，以及 $`\widehat v_2=\tfrac{17}{24}`$、$`\widehat v_2'=\tfrac{41}{24}`$，所以下面的第二步方向相反。有界检验 $`F(\theta)=1\lbrace \theta>\theta_1\rbrace`$ 把两段历史分开，差为 $`1`$。这两段历史可以由平方损失 $`\tfrac12(\theta-a)^2`$ 在 $`\theta_0=0`$ 处实现：第一步目标为 $`a=-1`$ 与 $`a=-2`$，第二步取共同目标 $`a=\theta_1+\tfrac34`$。

```math
\theta_2=\theta_1+\frac{\eta}{6\sqrt{17/24}},\qquad\theta_2'=\theta_1-\frac{\eta}{6\sqrt{41/24}}.
```

- *记忆由 momentum 携带.* 若 $`\beta_1=0`$，任意两段 $`G_2`$ 符号相同的历史，第二步方向都相同，这类反例不存在；对每个 $`\beta_1\in(0,1)`$，这类反例都存在。

**说了什么.** 从零 optimizer state 出发，Adam 的第一步只依赖 $`g_i/(\lvert g_i\rvert+\epsilon)`$。$`\epsilon=0`$ 时必须匹配联合符号分布（只匹配符号边缘分布不够）；$`\epsilon>0`$ 时必须匹配整个梯度分布。momentum 保留了第一个梯度的幅度，所以符号相同的两段历史，第二步可以方向相反。

**直觉.** Adam 的第一步几乎就是 sign descent：幅度从步长里被除掉了，却保留在 optimizer state 中。

**在框架中的作用.** 说明矩充分性不能搬到 Adam 上，要匹配什么取决于 optimizer state。

**推导要点.**

- cold 状态下 bias-corrected 矩为 $`G`$ 与 $`G^{\odot2}`$，所以这一步是 $`G/(\lvert G\rvert+\epsilon)`$。
- 第一步是符号向量（$`\epsilon=0`$）或 $`G`$（$`\epsilon>0`$）的单射函数；统计量个数是在纤维上取常值的函数空间的维数，由 Theorem 4.2 得到。
- margin 界用到 $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\epsilon/(\lvert g_i\rvert+\epsilon)`$。
- 记忆的反例是精确的两步计算；momentum 项 $`\beta_1(1-\beta_1)G_1`$ 翻转了 $`m_2`$ 的符号。

**例示.** 两个坐标的梯度以相同概率取 $`(1,1)`$ 或 $`(-1,-1)`$，对比以相同概率取 $`(1,-1)`$ 或 $`(-1,1)`$：每个坐标的符号边缘分布相同，平均步长都是 $`0`$；但前者两个参数总是一起动，后者从不一起动。

**用到.**

- [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) — 充分性对偶与迁移

**被用于.**

- [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean_ZH.md#s5-7) — cold 步在一阶就分开均值相同的反馈
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10) — 各学习器的一阶统计量

## 相关工作

- Adam 的更新与 bias correction 来自 [Kingma & Ba, 2015](#ref-kingma2015adam)。把 Adam 看成带方差自适应的 sign descent 出自 [Balles & Hennig, 2018](#ref-balles2018dissecting)；基于符号的 optimizer 包括 signSGD（[Bernstein et al., 2018](#ref-bernstein2018signsgd)）和 Lion（[Chen et al., 2023](#ref-chen2023symbolic)）。Adam 自适应学习率在最初几步方差很大，这一点由 [Liu et al., 2020](#ref-liu2020variance) 分析。
- Proposition 5.3 新给出的是 cold 步的精确信息需求：$`\epsilon=0`$ 时是联合符号分布，$`\epsilon>0`$ 时是整个分布；以及 optimizer 状态保留了这一步除掉的梯度大小。
