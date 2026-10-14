# Proposition 4.12 · response 经过下一步 predictive state 分解

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-12_response_predictive_state_factorization_ZH.md) · [证明](../proofs/04-12_response_predictive_state_factorization_ZH.md) · [实验](../experiments/04-12_response_predictive_state_factorization_ZH.md) · [English](04-12_response_predictive_state_factorization.md)

**一段话.** [Definition 4.8](04-08_response_operator_ZH.md#s4-8) 的 response operator 与 [Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10) 的 predictive state，是用两种语言描述同一件事。若声明测试的 continuation value 落在由 predictive 特征张成的有限维空间里，response operator 就是一个固定的观测矩阵，作用在下一步 predictive state 期望的变化上。当测试张成这个空间时，两个 feedback law 在每个声明测试上的 response 相同，当且仅当它们产生相同的下一步 predictive state 期望；若这个空间还对声明的 continuation 封闭，它们此后在每个声明的期望上都一致。这个引理与 predictive state 的定义很接近；它的用处是让 §4 的压缩、闭合要求和 §8 的测量谈论同一个对象。

**本页用到的设定.**

- 检查点 $`(t,z)`$，参考消息测度 $`\rho`$，$`H=L_0^2(\rho)`$，有限个声明测试 $`a=1,\dots,p`$，权重 $`\omega_a>0`$，以及 response operator $`\mathcal R_z`$（[Definition 4.8](04-08_response_operator_ZH.md#s4-8)）。
- $`V_a`$ 是测试 $`a`$ 从本轮更新之后的状态出发的 continuation value，所以 $`g_a(y)=\mathbb E[V_a(U_t(z,y,\xi_t))\mid z,y]`$。
- *有限 predictive 基.* 有界特征 $`\psi:\mathcal Z\to\mathbb R^d`$，使 $`V_a=k_a+c_a^\top\psi`$，$`k_a`$ 为常数，$`c_a`$ 为向量。这是只针对本命题的有限维假设，不是框架默认的低秩假设。$`C\in\mathbb R^{p\times d}`$ 的行为 $`c_a^\top`$。
- 消息 $`y`$ 之后的*下一步 predictive state 期望*为 $`\Gamma_z(y)=\mathbb E[\psi(U_t(z,y,\xi_t))\mid z,y]\in\mathbb R^d`$，$`\mathcal T_z:H\to\mathbb R^d`$ 定义为 $`\mathcal T_zh=\mathbb E_\rho[\Gamma_z(Y)h(Y)]`$。对密度为 $`p,q`$ 的分布，$`\mathcal T_z(p-q)=\mathbb E_P\Gamma_z-\mathbb E_Q\Gamma_z`$。

<a id="s4-12"></a>

**Proposition 4.12（response 经过下一步 predictive state 分解）.** 在上述设定下：

**(a) 分解.**

```math
\mathcal R_z=C\,\mathcal T_z,\qquad\text{即对每个测试 }a\text{ 与每个 }h\in H,\ (\mathcal R_zh)(a)=c_a^\top\mathcal T_zh. \qquad(4.23)
```

**(b) response 相等即下一步 predictive state 相等.** 若 $`C`$ 列满秩 $`d`$，即声明的测试张成 predictive 基，则 $`\ker\mathcal R_z=\ker\mathcal T_z`$。对两个密度在 $`L^2(\rho)`$ 中的分布，在每个声明测试上期望相等，等价于 $`\mathbb E_P\Gamma_z=\mathbb E_Q\Gamma_z`$。

**(c) 继续下去.** 若此外 $`\mathrm{span}\lbrace\mathbf 1,\psi\rbrace`$ 在声明的合法 continuation 的 kernel 下不变（如 [Theorem 4.10(a)](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)），则下一步 predictive state 期望相等，推出在每个声明的开环控制词之后，每个声明测试的期望都相等。对观察自适应的 continuation，只要这个空间在每个观察算子下不变（如 [Theorem 4.10(c)](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)），结论同样成立。

**(d) 带残差的版本.** 若 $`V_a=k_a+c_a^\top\psi+e_a`$，残差 $`e_a`$ 有界，令 $`\bar e_a(y)=\mathbb E[e_a(U_t(z,y,\xi_t))\mid z,y]`$。则 $`\mathcal R_z=C\mathcal T_z+\mathcal E_z`$，其中 $`(\mathcal E_zh)(a)=\langle\bar e_a-\mathbb E_\rho\bar e_a,h\rangle_\rho`$，并且

```math
\lVert\mathcal E_zh\rVert_\omega\le\epsilon_{\rm pred}\,\lVert h\rVert_\rho,\qquad \epsilon_{\rm pred}^2=\sum_a\omega_a\,\mathrm{Var}_\rho\,\bar e_a\le\sum_a\omega_a\,\mathbb E_\rho\bar e_a^2 . \qquad(4.24)
```

**说了什么.**

- 在检查点上，feedback 差异对声明测试的影响，要经过一个中间对象：它在下一步 predictive state 期望上造成的变化。
- $`\mathcal T_z`$ 说明 feedback 能改变哪些 predictive 记忆；$`C`$ 说明这些记忆中哪些能被声明的未来测试看到；response operator 是两者的复合。
- 测试能看到整个 predictive 基时，response 匹配与下一步 predictive state 匹配是同一个条件；闭合则把这一步的匹配延续到声明的 continuation。

**直觉.** 一个 feedback law 对未来的影响，只能通过它留在 learner 状态里的东西；测试透过一个固定的镜头去量这些残留。response operator 的谱把两者混在一起；这个分解把它们分开。

**在框架中的作用.**

- 它是让 §4、§3 与 §8 谈论同一个对象的引理。[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 的谱压缩挑选复合 $`C\mathcal T_z`$ 的方向；[Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11) 的闭合决定这些方向在下一轮是否还能用；[Proposition 6.16](06-16_information_compilation_radius_ZH.md#s6-16) 的获取，为 $`C`$ 能看到、而数据还没分辨的方向收集证据。
- 它是 Hankel 算子分解为可观性与可控性的单步情形（[Proposition 4.13](04-13_finite_horizon_gramians_ZH.md#s4-13)）。

**推导要点.** 以消息为条件得 $`g_a(y)=k_a+c_a^\top\Gamma_z(y)`$。由于 $`h`$ 的 $`\rho`$ 均值为零，常数项消失，$`\langle\tilde g_a,h\rangle_\rho=c_a^\top\mathcal T_zh`$。列满秩使 $`C`$ 是单射，给出两个核相等。在 continuation kernel 下的不变性把 $`\mathbb E\psi`$ 的相等沿每个声明控制词传下去。带残差的版本对每个测试用 Cauchy–Schwarz。

**它不给出什么.**

- *接近定义.* 声明测试张成 predictive 基时，"在每个测试上 response 相同"与"predictive state 相同"几乎是同一句话；这里的内容在于把算子拆成 $`\mathcal T_z`$ 与 $`C`$，以及拆分成立的条件，而不是一个新的等价关系。
- *没有定量的逆.* 列满秩给出核相等。要从小的 response 误差推出小的 predictive state 误差，需要 $`C`$ 的最小奇异值，而它可能很差；predictive 坐标只确定到可逆线性变换，所以保证应当用任务范数表述。
- *一个检查点不等于一次训练.* 在一个检查点上匹配，代替不了在训练实际访问的所有状态上匹配；对整次训练，[Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2) 需要在每个访问状态上都匹配，并付出其 (c) 部分的 coverage 代价。
- *成本.* 一次 continuation 协议的运行会输出好几个指标。若每条分支给出 $`p_{\rm branch}`$ 个指标，张成 $`d`$ 维的基至少需要 $`\lceil d/p_{\rm branch}\rceil`$ 个不同的 continuation 协议，而不是 $`d`$ 个；识别闭合还需要额外的延伸测试。一条完整的 continuation 分支能提供多少独立信息，还没有解决（[Remark 8.17](08-17_open_problems_ZH.md#s8-17)）。

**用到.**

- [Definition 4.8](04-08_response_operator_ZH.md#s4-8)：response operator.
- [Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)：predictive state、闭合与条件化.

**被用于.**

- [Proposition 4.13](04-13_finite_horizon_gramians_ZH.md#s4-13)：分解的有限时域版本.

## 相关工作

- 用对未来测试的预测表示状态，是 predictive state representation 的前提 ([Littman, Sutton & Singh, 2001](#ref-littman2001predictive))；(b) 是这个前提在学习过程 response operator 上的重述。
- 只要求模型在重要的价值函数上正确，是 value-equivalence 原则 ([Grimm et al., 2020](#ref-grimm2020value))；这里的价值函数是声明的 continuation value，模型是 learner 的下一步 predictive state。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-grimm2020value"></a>

**Christopher Grimm, André Barreto, Satinder Singh, David Silver (2020).** The value equivalence principle for model-based reinforcement learning. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 5541–5552. arXiv [2011.03506](https://arxiv.org/abs/2011.03506).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).
