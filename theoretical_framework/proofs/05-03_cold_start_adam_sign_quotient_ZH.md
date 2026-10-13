# Proposition 5.3 · Cold-start Adam：符号商与 optimizer 记忆

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-03_cold_start_adam_sign_quotient_ZH.md) · [证明](../proofs/05-03_cold_start_adam_sign_quotient_ZH.md) · [实验](../experiments/05-03_cold_start_adam_sign_quotient_ZH.md) · [English](05-03_cold_start_adam_sign_quotient.md)

**本页证明什么.** 五个部分都给出证明：(a) 第一步的公式；(b) 联合符号分布的刻画、精确的统计量个数，以及"逐坐标符号边缘分布不够"的反例；(c) $`\epsilon>0`$ 时需要整个分布；(d) 对 margin 敏感的近似误差界；(e) optimizer 记忆的反例：两个符号相同的历史，第二步方向相反。

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

**陈述（[Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)）.** 从 cold 状态 $`(\theta_0,0,0,0)`$ 启动 Adam。

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

*(a) 的证明.* cold 状态下 $`m_1=(1-\beta_1)G`$，$`v_1=(1-\beta_2)G^{\odot2}`$。除以 $`t=1`$ 的 bias correction 得 $`n=G`$、$`d^2=G^{\odot2}`$，所以逐坐标 $`d=\lvert G\rvert`$。代入得 $`\theta_1=\bar\theta-\eta\,G/(\lvert G\rvert+\epsilon)`$。$`\epsilon=0`$ 时由约定得 $`h_0=\mathrm{sign}`$。$`\square`$

*(b) 的证明.* 因为 $`\eta>0`$，映射 $`\sigma\mapsto\bar\theta-\eta\sigma`$（从 $`\lbrace -1,0,1\rbrace ^p`$ 到 $`\mathbb R^p`$）是单射，所以 $`\theta_1`$ 是符号向量 $`\sigma(G)`$ 的单射函数。

- *充分性.* 联合符号分布相同，则 $`\theta_1`$ 的分布相同，所有效用的期望也相同。
- *必要性.* 对每个可达模式 $`\sigma^\star`$，示性函数 $`F=1\lbrace \theta_1=\bar\theta-\eta\sigma^\star\rbrace`$ 是有界效用，期望为 $`\Pr(\sigma(G)=\sigma^\star)`$。所有这样的 $`F`$ 期望相同，就迫使各模式的概率相同。
- *维数.* 把每个效用看成 $`\mathcal A`$ 上的函数 $`u_F(g)=F(\bar\theta-\eta h_0(g))`$。族 $`\mathcal U=\lbrace u_F\rbrace`$ 恰好是 $`\mathcal A`$ 上在 $`\sigma`$ 的纤维上取常值的函数：这些函数都在其中，因为 $`F`$ 在 $`s`$ 个不同输出点上可以任意取值；其余函数都不在其中，因为 $`u_F`$ 只通过 $`\sigma(g)`$ 依赖 $`g`$。所以 $`\mathrm{span}(\lbrace 1\rbrace \cup\mathcal U)`$ 由 $`s`$ 个纤维示性函数张成，维数为 $`s`$。由 [Theorem 4.2(a)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)，对 $`\mathcal A`$ 上的所有分布，均值相等能推出每个 $`u\in\mathcal U`$ 均值相等的线性统计量最少为 $`s-1`$ 个；取 $`s-1`$ 个模式的示性函数即可达到。每个坐标符号至多取三个值，所以 $`s\le3^p`$；排除 $`0`$ 时 $`s\le2^p`$。
- *边缘分布.* 在反例的两个分布下，每个坐标都以概率 $`1/2`$ 取 $`\pm1`$，且 $`\mathbb E h_0(G)=0`$。在 $`P`$ 下 $`\theta_1-\bar\theta=-\eta\sigma`$ 的两个坐标总是同号，所以 $`F=1`$；在 $`Q`$ 下总是异号，所以 $`F=0`$。$`\square`$

*(c) 的证明.* 固定 $`\epsilon>0`$。在 $`g\ge0`$ 上 $`h_\epsilon(g)=g/(g+\epsilon)`$ 连续、严格递增，从 $`0`$ 升到 $`1`$；由奇性，$`h_\epsilon`$ 是双射 $`\mathbb R\to(-1,1)`$。

- 解 $`u=g/(\lvert g\rvert+\epsilon)`$：$`\mathrm{sign}u=\mathrm{sign}g`$ 且 $`\lvert u\rvert(\lvert g\rvert+\epsilon)=\lvert g\rvert`$，所以 $`\lvert g\rvert=\epsilon\lvert u\rvert/(1-\lvert u\rvert)`$，$`g=\epsilon u/(1-\lvert u\rvert)`$。对 $`u>0`$，导数为 $`\epsilon/(1-u)+\epsilon u/(1-u)^2=\epsilon/(1-u)^2`$；由对称性，$`u<0`$ 时公式相同。
- 因为 $`g\mapsto\bar\theta-\eta h_\epsilon(g)`$ 是单射，(b) 的论证在纤维为单点时同样适用：对所有有界效用相等等价于 $`G`$ 的分布相等，在 $`\mathcal A`$ 上维数为 $`m-1`$。$`\square`$

*(d) 的证明.* 固定坐标 $`i`$ 与梯度值 $`g`$。

- 若 $`g_i=0`$，则 $`h_\epsilon(g)_i=0=h_0(g)_i`$。
- 若 $`g_i\ne0`$，则 $`\lvert h_\epsilon(g_i)-\mathrm{sign}(g_i)\rvert=\lvert g_i\rvert\cdot\big\lvert\tfrac1{\lvert g_i\rvert+\epsilon}-\tfrac1{\lvert g_i\rvert}\big\rvert=\tfrac{\epsilon}{\lvert g_i\rvert+\epsilon}`$。它总不超过 $`1`$，当 $`\lvert g_i\rvert\ge a`$ 时不超过 $`\epsilon/(a+\epsilon)`$。

所以 $`\mathbb E_P\lvert h_\epsilon(G_i)-h_0(G_i)\rvert\le\beta_{P,i}(a)+\epsilon/(a+\epsilon)`$。对每对 $`\big(\bar\theta-\eta h_\epsilon(G),\,\bar\theta-\eta h_0(G)\big)`$ 逐点用 Lipschitz 假设，

```math
\big\lvert\mathbb E_PF(\bar\theta-\eta h_\epsilon(G))-\mathbb E_PF(\bar\theta-\eta h_0(G))\big\rvert\le\eta\sum_iL_i\Big[\beta_{P,i}(a)+\frac{\epsilon}{a+\epsilon}\Big],
```

在 $`Q`$ 下同样成立。经过两个理想符号期望用三角不等式即得。至于 TV 的陈述：$`F(\bar\theta-\eta h_0(G))=f(\sigma(G))`$，其中 $`f:\lbrace -1,0,1\rbrace ^p\to[0,1]`$，任何这样的 $`f`$ 都满足 $`\lvert\mathbb E_Pf-\mathbb E_Qf\rvert\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})`$。$`\square`$

*(e) 的证明.* $`\beta_1<1`$ 时 $`m_1=(1-\beta_1)G`$ 显然是单射。对反例，$`1-\beta_1=1-\beta_2=\tfrac12`$，$`t=1`$ 时 bias correction 为 $`\tfrac12`$，$`t=2`$ 时为 $`\tfrac34`$。

- *第 1 步.* $`m_1=G_1/2`$，$`v_1=G_1^2/2`$，所以 $`\widehat m_1=G_1`$，$`\widehat v_1=G_1^2`$。两段历史都有 $`G_1>0`$，所以 $`\theta_1=\theta_0-\eta`$。
- *第 2 步.* $`m_2=\tfrac12m_1+\tfrac12G_2=\tfrac{G_1}4-\tfrac38`$：$`G_1=1`$ 时为 $`-\tfrac18`$，$`G_1=2`$ 时为 $`+\tfrac18`$；于是 $`\widehat m_2=\tfrac43m_2=\mp\tfrac16`$。同样 $`v_2=\tfrac12v_1+\tfrac12G_2^2=\tfrac{G_1^2}4+\tfrac9{32}`$，即 $`\tfrac{17}{32}`$ 或 $`\tfrac{41}{32}`$，所以 $`\widehat v_2=\tfrac{17}{24}`$ 或 $`\tfrac{41}{24}`$。
- $`\epsilon=0`$ 时 $`\theta_2=\theta_1-\eta\widehat m_2/\sqrt{\widehat v_2}`$，即上面的式子。分母为正、分子异号，所以两步方向相反，$`F(\theta_2)=1`$，$`F(\theta_2')=0`$。
- *可实现性.* $`\tfrac12(\theta-a)^2`$ 的梯度为 $`\theta-a`$。在 $`\theta_0=0`$ 处，目标 $`a=-1,-2`$ 给出 $`G_1=1,2`$。两个第一步都落在 $`\theta_1=-\eta`$，所以共同目标 $`a=\theta_1+\tfrac34`$ 在两段历史中都给出 $`G_2=-\tfrac34`$。不需要依赖分支的未来梯度。
- *$`\epsilon>0`$.* 对同样的梯度序列，第二步的方向对每个 $`\epsilon\ge0`$ 都是 $`-\mathrm{sign}(\widehat m_2)`$，因为分母 $`\sqrt{\widehat v_2}+\epsilon`$ 为正；两个方向仍然相反。
- *momentum.* 若 $`\beta_1=0`$，则 $`m_2=G_2`$，无论第一个梯度是什么，第二步的符号都是 $`-G_2`$ 的符号。若 $`\beta_1>0`$，则 $`m_2=\beta_1(1-\beta_1)G_1+(1-\beta_1)G_2`$。对 $`0<G_1<G_1'`$，取 $`G_2<0`$ 且 $`\lvert G_2\rvert`$ 严格介于 $`\beta_1G_1`$ 与 $`\beta_1G_1'`$ 之间，$`m_2`$ 的符号相反，而 $`\epsilon=0`$ 时两个第一步 $`-\eta\,\mathrm{sign}(G_1)`$ 相同。$`\square`$

**关于 Proposition 5.3 的讨论.**

- *三个层次的充分性.* cold 步的即时参数充分性，在 $`\epsilon=0`$ 时需要联合符号分布，在 $`\epsilon>0`$ 时需要完整分布。完整状态的充分性需要 consumed gradient 的完整分布，因为 $`m_1`$ 是单射。任务层面的充分性只需要实际 continuation 的 response 类（[Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)），它可能小得多。上面精确的统计量个数针对不受限的分布；受限的任务族可能需要更少。
- *$`\epsilon=0`$ 处的不连续是精确意义上的，不是操作意义上的.* 对任意可测效用，任何 $`\epsilon>0`$ 都使幅度可辨识。对 Lipschitz 效用、且梯度在零附近质量很小时，(d) 说明符号近似是准确的。(c) 中的逆在 $`\lvert u\rvert\to1`$ 时病态。
- *warm 状态会移动阈值.* warm 状态下坐标 $`i`$ 的分子在 $`G_i=-\beta_1m_i/(1-\beta_1)`$ 处变号，分母依赖 $`v_i`$ 与 $`G_i^2`$。对标量交叉熵梯度 $`G=p-Z`$，cold 的符号概率就是 $`Z`$ 在当前预测 $`p`$ 之下、之上与等于 $`p`$ 的质量。允许所有阈值就能辨识 $`Z`$ 的分布函数；没有哪个固定阈值或边缘符号特征能在状态变化时始终充分。
- *对仿射理论的含义.* 两个标签分布可以匹配所有低于 $`N`$ 阶的矩（[Theorem 4.4](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)），却在关于当前预测的符号概率上不同。cold Adam 的一步就会在 $`\eta`$ 的一阶上把它们分开（Proposition 5.7）。

## 相关工作

- Adam 带 bias correction 的第一步及其 sign descent 行为是已知的：[Kingma & Ba, 2015](#ref-kingma2015adam)；[Balles & Hennig, 2018](#ref-balles2018dissecting) 把 Adam 看成带方差自适应的 sign descent；[Liu et al., 2020](#ref-liu2020variance)（RAdam）讨论早期步的方差；signSGD（[Bernstein et al., 2018](#ref-bernstein2018signsgd)）与 Lion（[Chen et al., 2023](#ref-chen2023symbolic)）。decoupled weight decay 见 [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-balles2018dissecting"></a>

**Lukas Balles, Philipp Hennig (2018).** Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:404–413. [proceedings.mlr.press/v80/balles18a](https://proceedings.mlr.press/v80/balles18a.html).

<a id="ref-bernstein2018signsgd"></a>

**Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, Animashree Anandkumar (2018).** signSGD: Compressed optimisation for non-convex problems. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:560–569. [proceedings.mlr.press/v80/bernstein18a](https://proceedings.mlr.press/v80/bernstein18a.html).

<a id="ref-chen2023symbolic"></a>

**Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le (2023).** Symbolic discovery of optimization algorithms. *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 49205–49233. arXiv [2302.06675](https://arxiv.org/abs/2302.06675).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-liu2020variance"></a>

**Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Jiawei Han (2020).** On the variance of the adaptive learning rate and beyond. *8th International Conference on Learning Representations (ICLR)*. arXiv [1908.03265](https://arxiv.org/abs/1908.03265).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).
