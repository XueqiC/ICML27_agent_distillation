# Corollary 8.16 · 由复用、估计与编译组合而成的保证

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](08-16_composed_guarantee_ZH.md) · [证明](../proofs/08-16_composed_guarantee_ZH.md) · [实验](../experiments/08-16_composed_guarantee_ZH.md) · [English](08-16_composed_guarantee.md)

**一段话概括.** §3、§6 与 §8 的各个部分可以拼成整次持续训练的一个保证。在每个被认证的状态上，目标反馈与编译出的合法反馈之间的局部 response 差，由四个可测项界住：信息–编译半径、锚点接口丢弃的谱、锚点的估计误差，以及从锚点到当前状态的漂移。沿训练过程用带停止的联合分解求和，它们与动态项、终点项、退出项一起界住最终差距；这几项都没有被设为零。这个保证成立。在实测量级下，对任何现实的密度界 $`B`$ 它都是空的：它的各项是有界取值集中不等式给出的绝对精度，不随单步效应缩小。10 次复现时，单步下限为 $`B\times1.296`$，在两候选的最坏情形 $`B=2`$ 下约为 2.6，而整个差距的平凡界是 1。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- *两个协议.* 参考（目标）协议发出 $`P^0_t(\cdot\mid z)`$，替代协议发出编译出的合法分布 $`P^S_t(\cdot\mid z)`$；两者使用同一个实际更新 $`U_t`$，从同一初始分布出发。$`L_E(0)`$、$`L_E(S)`$ 是它们的最终真实任务分数；$`\mathbb E_0`$、$`\mathbb E_S`$ 是沿两条链的期望。
- *近似价值*（同 [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4)）。固定函数 $`\widehat V_t\in[0,1]`$，$`t\le R`$；参考 Bellman 残差 $`r^0_t=K^0_t\widehat V_{t+1}-\widehat V_t`$；终点误差 $`e_R=F_E-\widehat V_R`$；近似 response $`\bar g_{t,z}(y)=\mathbb E_{\xi_t}\widehat V_{t+1}(U_t(z,y,\xi_t))`$；局部 response 差 $`d_t(z)=\int\bar g_{t,z}\,d(P^0_t-P^S_t)(\cdot\mid z)`$。
- *Response operator.* 合法消息空间上的共同参考测度 $`\rho`$，$`\mathcal H=L^2_0(\rho)`$，有限测试集 $`\mathcal A`$，权重 $`\omega_a>0`$，每个状态上有 [Definition 4.8](04-08_response_operator_ZH.md#s4-8) 的 response operator $`\mathcal R_{t,z}`$。对每个 $`(t,z)`$，测试集都包含一个测试 $`a_t`$，其 response 为 $`\bar g_{t,z}`$（执行实际更新，再读出 $`\widehat V_{t+1}`$）。
- *锚点.* 锚点 $`j=1,\dots,J`$，估计算子为 $`\hat{\mathcal R}_j`$（[Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)），真实算子为 $`\mathcal R_j`$。$`\hat{\mathcal R}_j`$ 的前 $`r_j`$ 个奇异三元组 $`(\hat\sigma_{j,k},\hat u_{j,k},\hat v_{j,k})`$ 给出消息特征 $`\phi_j=(\hat v_{j,1},\dots,\hat v_{j,r_j})`$、系数映射 $`C_j\mu=\sum_{k\le r_j}\hat\sigma_{j,k}\mu_k\hat u_{j,k}`$ 与模式均值 $`\mu^j_P=\mathbb E_P\phi_j`$。$`\hat\Pi_j`$ 投影到 $`\hat v_{j,1},\dots,\hat v_{j,r_j}`$ 上。
- *信息–编译半径*（同 [Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16)）。对目标模式均值的紧置信集 $`\mathcal M\subset\mathbb R^{r_j}`$ 与合法可达输出的紧集合 $`\mathcal K_j=\lbrace C_j\mu^j_Q:Q\text{ legal}\rbrace`$，

```math
\mathfrak r(\mathcal M,\mathcal K_j)=\min_{q\in\mathcal K_j}\ \sup_{\mu\in\mathcal M}\ \lVert C_j\mu-q\rVert_\omega .
```

- *被认证的状态.* 在审计数据的某个事件 $`\mathcal E`$ 上，若为第 $`t`$ 轮的状态 $`z`$ 选定锚点 $`j=j(t,z)`$ 后满足以下条件，就称 $`z`$ 对替代链被认证，记 $`z\in C^S_t`$：
  1. $`\lVert\hat{\mathcal R}_j-\mathcal R_j\rVert_{\rm op}\le\epsilon_j`$，且 $`\lVert\mathcal R_{t,z}-\mathcal R_j\rVert_{\rm op}\le\Delta_{t,z}`$；
  2. $`\mu^j_{P^0_t(\cdot\mid z)}\in\mathcal M_{t,z}`$，且编译出的分布满足 $`\sup_{\mu\in\mathcal M_{t,z}}\lVert C_j\mu-C_j\mu^j_{P^S_t(\cdot\mid z)}\rVert_\omega\le\mathfrak r_{t,z}+\eta_{\rm opt}`$，其中 $`\mathfrak r_{t,z}=\mathfrak r(\mathcal M_{t,z},\mathcal K_j)`$；
  3. 两个分布对 $`\rho`$ 都有密度，且 $`h_{t,z}=dP^0_t/d\rho-dP^S_t/d\rho`$ 满足 $`\lVert h_{t,z}\rVert_\rho\le B_{t,z}`$。

  $`\tau^S`$ 是替代链第一次落在 $`C^S_t`$ 之外的轮次；参考链可以用任意认证集合 $`C^0_t`$（例如整个空间），退出时刻为 $`\tau^0`$。

<a id="s8-16"></a>

**Corollary 8.16（由复用、估计与编译组合而成的保证）.** 在上述设定下，在事件 $`\mathcal E`$ 上：

**(a) 局部界.** 对每个被认证的 $`z\in C^S_t`$，令 $`j=j(t,z)`$，

```math
\lvert d_t(z)\rvert\le\beta_t(z):=\frac{\mathfrak r_{t,z}+\eta_{\rm opt}+B_{t,z}\big(\hat\sigma_{j,r_j+1}+\epsilon_j+\Delta_{t,z}\big)}{\sqrt{\omega_{a_t}}} .
```

**(b) 整次训练的界.**

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau^S}\beta_t(Z_t)+\mathcal E_{\rm dyn}+\mathcal E_{\rm terminal}+\mathcal E_{\rm exit},
```

其中 $`\mathcal E_{\rm dyn}=\mathbb E_S\sum_{t<\tau^S}\lvert r^0_t(Z_t)\rvert+\mathbb E_0\sum_{t<\tau^0}\lvert r^0_t(Z_t)\rvert`$，$`\mathcal E_{\rm terminal}=\mathbb E_0[\lvert e_R\rvert\mathbf 1\lbrace\tau^0=R\rbrace]+\mathbb E_S[\lvert e_R\rvert\mathbf 1\lbrace\tau^S=R\rbrace]`$，$`\mathcal E_{\rm exit}=\Pr_0(\tau^0<R)+\Pr_S(\tau^S<R)`$。没有任何一项被设为零。

**(c) 实测量级下的大小.** 取一个测试（$`\omega_{a_t}=1`$）、$`B_{t,z}=2`$，锚点误差 $`\epsilon_j=\varepsilon_n`$ 取 [Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) 在 20 个候选、每格 10 次复现时的值（$`\varepsilon_n=0.578`$），漂移半径取 [Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) 在实测评估噪声、10 对时的值（$`\hat\Delta=0.718`$）。那么在加上半径项、对各步求和之前，单步下限 $`B(\epsilon_j+\Delta_{t,z})`$ 就是 2.59（[Check 8.23](../experiments/08-16_composed_guarantee_ZH.md#c8-23)）；这些数值取 $`p=1`$ 个测试、$`m=20`$ 个候选、$`\delta=0.05`$，所以 $`\epsilon_j+\Delta_{t,z}\approx1.296`$。下限对 $`B_{t,z}`$ 是线性的。$`B=2`$ 是两个候选混合时的最坏情形；在 20 个候选的目录混合测度下最坏情形是 $`\sqrt{40}\approx6.3`$；离目标很近的编译律可以有小得多的 $`B`$，$`B=0.5`$ 时每步 0.65。即便如此，两个及以上被审计步的下限之和也超过 1，而且下限不随单步效应缩小。由于总有 $`\lvert L_E(0)-L_E(S)\rvert\le1`$，除非 $`B_{t,z}`$ 很小且只审计一步，这个界成立，但在该量级下为空。

**说了什么.**

- 接口复用、response 估计、信息获取与合法编译进入持续训练的同一个保证，各自通过自己的可测项，动态项、终点项与退出项都保留。
- 保证为完整的 response 漂移付费，所以锚点与当前状态之间保留模式系数的变化也被计入，而不只是被丢弃方向上的漂移。
- 用有界取值集中不等式时，每一步的各项都是绝对精度。即使每一步的效应极小，对各步求和后的界也随步数增长；在实测量级下，它远高于平凡界。

**直觉.** 这套记账是精确的；让它没有信息量的是：每个估计项被认证到的精度，并不知道它要界的量有多小。一个由很多小效应之和组成的量，用逐项的绝对精度去认证，得到的是很多大半径之和。

**在框架中的作用.**

- 它是框架维护链条的组合陈述：[Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16)（该获取什么、能编译什么）、[Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)（估计、复用、成本），以及 [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4)（沿训练过程累积）。
- 在当前量级下，它的实际作用是一份误差预算：它说明应该减小哪一项，而不是说明差距很小。

**关于陈述的说明.**

- **三个问题彼此不同.**
  1. 估计一对固定协议之间的收益：直接比较两个协议，或者用配对运行估计 Proposition 3.4(a) 的有符号恒等式。
  2. 同时认证每一个局部 response，这就是 (a)–(b) 做的事。
  3. 对一整类未来协议保证充分性。

  它们的成本不同。有 $`R`$ 项、每次查询噪声为 $`\sigma`$ 时，在均匀分配下把有符号的和估计到均方误差 $`\varepsilon^2`$ 约需 $`\sigma^2R^2/\varepsilon^2`$ 次查询；而把每一项认证到 $`\varepsilon/R`$ 再把绝对值相加，约需 $`\sigma^2R^3/\varepsilon^2`$ 次（[Check 8.23](../experiments/08-16_composed_guarantee_ZH.md#c8-23)：$`R=20`$、$`\sigma=\varepsilon=1`$ 时为 400 对 8,000）。当 response 接近可加时，像 datamodels（[Ilyas et al., 2022](#ref-ilyas2022datamodels)）那样对 $`R`$ 个选择做随机析因设计，也能一次性以 $`R^2`$ 的量级得到所有项。$`R^3`$ 是这种认证方案的成本，不是这个问题的下界。
- **什么能让它有信息量.** 现实的路线是让近似价值承担长程部分：这样每一项只需要单步更新与 $`\widehat V`$ 的评估，成本转移到 Bellman 残差与拟合 $`\widehat V`$ 上。这正是双重稳健离策略评估的结构，即价值模型加校正项，在某些情形下它有匹配的下界（[Jiang & Li, 2016](#ref-jiang2016doubly)）；面向持续训练的版本必须与它比较，也要与预测状态时序差分学习（[Boots & Gordon, 2010](#ref-boots2010predictive)）比较。
- **事件必须在哪里成立.** $`\mathcal E`$ 必须在每个被认证的状态上、对锚点与编译分布的每一种自适应选择都成立。如果条件只在有限个被审计的状态上建立，那么只有这些状态被认证：链必须真的经过它们（在它到达的状态上审计，并且有效性以访问为条件），否则退出项吸收其余部分。在别处需要一致的控制。
- **逐测试的代价.** 直接审计 $`a_t`$ 那一行，就可以避免因子 $`1/\sqrt{\omega_{a_t}}`$。

**推导要点.**

- 分解 $`\mathcal R_{t,z}h=\hat{\mathcal R}_j\hat\Pi_jh+\hat{\mathcal R}_j(I-\hat\Pi_j)h+(\mathcal R_{t,z}-\hat{\mathcal R}_j)h`$。
- 第一项等于 $`C_j(\mu^j_{P^0}-\mu^j_{P^S})`$；因为目标的模式均值落在 $`\mathcal M_{t,z}`$ 中，它由半径条件界住。
- 另两项分别由 $`B_{t,z}\hat\sigma_{j,r_j+1}`$ 与 $`B_{t,z}(\epsilon_j+\Delta_{t,z})`$ 界住。
- 在测试 $`a_t`$ 处取值，其 response 差就是 $`d_t(z)`$，代价为 $`1/\sqrt{\omega_{a_t}}`$。
- 代入 Proposition 3.4(d)。

**用到.**

- [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4) — 带停止的联合分解
- [Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16) — 信息–编译半径
- [Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) — 锚点估计
- [Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) — 锚点到状态的漂移
- [Proposition 8.15](08-15_two_fidelity_response_estimation_ZH.md#s8-15) — 更便宜的锚点估计

**被用于.**

- [Remark 8.17](08-17_open_problems_ZH.md#s8-17) — 要让组合保证有信息量还缺什么

## 相关工作

- 整次训练这一步是带近似价值的 performance-difference 记账，即 [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4)（[Kakade & Langford, 2002](#ref-kakade2002approximately)）。价值模型加校正项是离策略评估中的双重稳健构造（[Jiang & Li, 2016](#ref-jiang2016doubly)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-boots2010predictive"></a>

**Byron Boots, Geoffrey J. Gordon (2010).** Predictive state temporal difference learning. *Advances in Neural Information Processing Systems 23 (NIPS)*, Curran Associates. [proceedings.neurips.cc/paper/2010/hash/3473decccb0509fb264818a7512a8b9b](https://proceedings.neurips.cc/paper_files/paper/2010/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).
