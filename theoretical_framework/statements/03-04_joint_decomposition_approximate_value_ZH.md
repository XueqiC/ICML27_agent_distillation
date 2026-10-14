# Proposition 3.4 · 用近似价值做的联合分解

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](03-04_joint_decomposition_approximate_value_ZH.md) · [证明](../proofs/03-04_joint_decomposition_approximate_value_ZH.md) · [实验](../experiments/03-04_joint_decomposition_approximate_value_ZH.md) · [English](03-04_joint_decomposition_approximate_value.md)

**一段话.** [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2) 用精确的参照 continuation value $`V^0`$ 写出两个训练过程之间的差，可是 $`V^0`$ 谁也拿不到。实际手里只有一个近似价值 $`\widehat V`$，可能来自 predictive-state 模型，也可能来自别的办法。这个命题把 $`\widehat V`$ 放在 $`V^0`$ 的位置上，记账依然精确：差距拆成局部 response 项、$`\widehat V`$ 在两条过程上的参照 Bellman 残差、以及两条过程上的终点误差。带停止的版本让每条过程的离开概率各付一次。它属于 performance-difference 一族的恒等式，本身不产生小误差：每一项仍然要估计，而取绝对值的界会丢掉估计量本该保留的抵消。

**本页用到的设定.**

- complete learner state $`z\in\mathcal Z`$、feedback law、实际更新 $`U_t(z,y,\xi_t)`$ 与学习 kernel 见 [Definition 1.3](../README_ZH.md#s1-3)。$`K_t^0`$ 是参照协议，$`K_t^S`$ 是备选协议 $`S`$，两者从同一初始分布出发。$`\mathbb E_0`$、$`\mathbb E_S`$ 分别是沿参照链 $`Z_t^0`$ 与备选链 $`Z_t^S`$ 的期望。
- $`F_E(z)\in[0,1]`$ 是最终真实任务得分，$`L_E(S)=\mathbb E_SF_E(Z_R)`$（[Definition 1.4](../README_ZH.md#s1-4)）。
- *近似价值* 是任意一族有界可测函数 $`\widehat V_0,\dots,\widehat V_R`$，在比较之前固定。它可以来自 predictive-state 模型（[Theorem 4.10](04-10_hankel_rank_predictive_realization_ZH.md#s4-10)），也可以来自其他任何办法。
- $`\widehat V`$ 的*参照 Bellman 残差*、*终点误差*与*局部 response 差*为

```math
r_t^0=K_t^0\widehat V_{t+1}-\widehat V_t,\qquad e_R=F_E-\widehat V_R,\qquad d_t=(K_t^0-K_t^S)\widehat V_{t+1}\quad(t<R). \qquad(3.6)
```

- 当两个协议共用同一个更新映射、只在 feedback law $`P_t^0,P_t^S`$ 上不同时，$`d_t(z)=\int\bar g_{t,z}\,d(P_t^0-P_t^S)(\cdot\mid z)`$，其中*近似 response* 为 $`\bar g_{t,z}(y)=\mathbb E_{\xi_t}\widehat V_{t+1}(U_t(z,y,\xi_t))`$。$`\bar g`$ 里的更新是实际更新，被近似的只有 continuation。
- $`d_t^S`$ 是 $`Z_t^S`$ 的分布，即备选过程在第 $`t`$ 轮的占据分布，与 [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2) 相同。它是状态上的测度，不要与局部 response 差 $`d_t`$ 混淆。

<a id="s3-4"></a>

**Proposition 3.4（用近似价值做的联合分解）.** 在上述设定下：

**(a) 精确恒等式.**

```math
L_E(0)-L_E(S)=\mathbb E_S\sum_{t<R}d_t(Z_t)+\mathbb E_0\sum_{t<R}r_t^0(Z_t)-\mathbb E_S\sum_{t<R}r_t^0(Z_t)+\mathbb E_0e_R(Z_R)-\mathbb E_Se_R(Z_R). \qquad(3.7)
```

**(b) 绝对值界.** 若对 $`d_t^S`$-几乎所有 $`z`$ 有 $`\lvert d_t(z)\rvert\le\beta_t(z)`$，则

```math
\lvert L_E(0)-L_E(S)\rvert\le\sum_{t<R}\mathbb E_S\beta_t(Z_t)+\sum_{t<R}\big(\mathbb E_0\lvert r_t^0(Z_t)\rvert+\mathbb E_S\lvert r_t^0(Z_t)\rvert\big)+\mathbb E_0\lvert e_R(Z_R)\rvert+\mathbb E_S\lvert e_R(Z_R)\rvert. \qquad(3.8)
```

**(c) 闭合残差形式.** 若 $`\widehat V_t=c_t^\top\psi_t`$，其中 $`\psi_t:\mathcal Z\to\mathbb R^{d_t}`$ 为有界特征，且参照 kernel 逐分量满足 $`K_t^0\psi_{t+1}=A_t\psi_t+\epsilon_t`$（$`A_t`$ 为矩阵，$`\epsilon_t`$ 为残差函数），则

```math
r_t^0=\big(A_t^\top c_{t+1}-c_t\big)^\top\psi_t+c_{t+1}^\top\epsilon_t .
```

**(d) 带停止的版本.** 设对每个 $`t`$ 有 $`\widehat V_t\in[0,1]`$。对每条过程 $`j\in\lbrace0,S\rbrace`$ 选认证集合 $`C_t^j\subseteq\mathcal Z`$，令 $`\tau^j`$ 为该过程自己的链上第一个满足 $`Z_t^j\notin C_t^j`$ 的 $`t<R`$（若不存在则 $`\tau^j=R`$）。若对 $`z\in C_t^S`$ 有 $`\lvert d_t(z)\rvert\le\beta_t(z)`$，则

```math
\begin{aligned}
\lvert L_E(0)-L_E(S)\rvert\le{}&\mathbb E_S\sum_{t<\tau^S}\big(\beta_t+\lvert r_t^0\rvert\big)(Z_t)+\mathbb E_0\sum_{t<\tau^0}\lvert r_t^0(Z_t)\rvert\\
&+\mathbb E_0\big[\lvert e_R(Z_R)\rvert\mathbf 1\lbrace\tau^0=R\rbrace\big]+\mathbb E_S\big[\lvert e_R(Z_R)\rvert\mathbf 1\lbrace\tau^S=R\rbrace\big]+\Pr_0(\tau^0<R)+\Pr_S(\tau^S<R).
\end{aligned} \qquad(3.9)
```

取 $`C_t^0=\mathcal Z`$ 即可去掉参照过程的离开项。

**说了什么.**

- 无论用哪个近似 continuation value，学习差距都精确等于：备选过程访问状态上的局部 response 项，加上两条过程上累计的参照 Bellman 残差之差，再加上终点误差之差。
- response 项用的是实际更新；只有 continuation 换成了 $`\widehat V`$。
- 当 $`\widehat V`$ 对 predictive 特征是线性的，它的残差拆成价值递推误差与特征闭合误差两部分。
- 保证只在认证集合上成立时，每条过程的离开概率各付一次。

**直觉.** Theorem 3.2 用精确的 continuation 度量每一轮的替换。改用近似的 continuation 是可以的，只要同时为它偏离参照递推的程度付账，而且两条过程都要付，因为两条过程访问的状态不同。

**在框架中的作用.**

- 它是所有组合保证背后的记账工具：[Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) 用的是 (d)，把 §8 的接口误差与估计误差放进 response 项 $`d_t`$。当 $`\widehat V`$ 由 predictive 特征构成时，[Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11) 的闭合残差通过 (c) 进入 $`r_t^0`$。
- 它不会让任何一项变小。把所有项都估计出来，代价可能比直接跑两个协议再比较还高；而且 (b) 偏松：两个残差和在 (3.7) 中符号相反，分别取绝对值就把这部分抵消丢了。估计量应当用配对运行去估计 (3.7) 中带符号的各项。

**推导要点.**

- 对每条过程 $`j`$，把 $`\widehat V_t(Z_t^j)`$ 从 $`t=0`$ telescope 到 $`R`$。由 Markov 性质，每个增量的条件均值是 $`(K_t^j\widehat V_{t+1}-\widehat V_t)(Z_t^j)`$：对 $`j=0`$ 它是 $`r_t^0`$，对 $`j=S`$ 它是 $`r_t^0-d_t`$。两式相减，公共的初始项抵消，得到 (a)；三角不等式给出 (b)。
- (c) 是 $`K_t^0`$ 作用在 $`c_{t+1}^\top\psi_{t+1}`$ 上的线性性。
- (d) 在每条过程各自的离开时刻停止 telescoping。离开之前的增量按 (a) 控制；离开之后剩下的差 $`F_E(Z_R)-\widehat V_{\tau}(Z_{\tau})`$ 落在 $`[-1,1]`$，因为两个函数都取值于 $`[0,1]`$。

**说明.**

- 取 $`\widehat V=V^0`$ 时所有残差与终点误差都为零，(a) 变成 (3.2)，(b) 变成 (3.3)。对精确价值的停止界，若参照过程的认证集取整个状态空间，(3.9) 与 (3.5) 一致：两者都只付备选过程的离开概率。
- 若为了 (d) 把 $`\widehat V`$ 截断到 $`[0,1]`$，残差要对截断后的函数重新计算；截断不保持 (c) 所用的线性形式。
- 只在某个 audit 分布上测得的残差，要到达实际访问的状态，还需另付 transport 或 coverage 项（[Theorem 8.2](08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)）。

**用到.**

- [Definitions 1.3–1.4](../README_ZH.md#s1-3)：learner state、kernel、continuation value.
- [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2)：精确情形与停止论证.

**被用于.**

- [Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11)：把闭合残差当作 Bellman 残差.
- [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16)：由复用、估计与编译组合出的保证.

## 相关工作

- 这个恒等式属于强化学习中的 performance-difference 与 simulation lemma 一族 ([Kakade & Langford, 2002](#ref-kakade2002approximately))，这里允许任意近似价值，并且比较的是训练过程而不是策略。它是记账恒等式，不是新的引理。
- 用价值模型加上沿被评估过程的残差校正，正是 doubly robust off-policy evaluation 的结构 ([Jiang & Li, 2016](#ref-jiang2016doubly))，那条线在一些设定下还给出了匹配的下界。本框架若沿价值模型路线建立成本理论，必须与它比较。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).
