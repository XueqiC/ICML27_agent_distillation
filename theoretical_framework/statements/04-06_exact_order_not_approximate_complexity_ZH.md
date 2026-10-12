# Proposition 4.6 · 精确阶不等于近似复杂度

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](04-06_exact_order_not_approximate_complexity_ZH.md) · [证明](../proofs/04-06_exact_order_not_approximate_complexity_ZH.md) · [实验](../experiments/04-06_exact_order_not_approximate_complexity_ZH.md) · [English](04-06_exact_order_not_approximate_complexity.md)

**一段话.** [Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) 说明，对至多 $`N`$ 次决策的部署，精确可互换需要直到 $`N`$ 阶的每一个矩。但精确相等是很强的要求。若可以接受一个小的结果差 $`\varepsilon`$，需要多少个矩？这个命题说明答案取决于任务，而不是它的次数：对光滑的价值 $`p^N`$，大约 $`\sqrt{N\log(1/\varepsilon)}`$ 个矩就够了；而对 parity 价值，任何低于 $`N`$ 的阶都毫无帮助。所以起作用的复杂度是 response 类的近似维数。

**本页用到的设定.**

- 标量任务：一个 context 被访问 $`N`$ 次，全部动作为 1 时成功，所以动作概率为 $`p\in[0,1]`$ 的学生的部署价值为 $`J(p)=p^N`$。
- $`N`$ 个 context 上的 parity 任务：每个 context 访问一次，动作 $`a_j\sim\mathrm{Bernoulli}(p_j)`$ 相互独立，$`\sum_ja_j`$ 为偶数时回报 1：$`[0,1]^N`$ 上的 $`J_{\rm even}(p)=\tfrac12\big(1+\prod_{j=1}^N(1-2p_j)\big)`$（[Theorem 4.4(c)](04-04_affine_learners_moment_order_ZH.md#s4-4)）。
- 带 IID 块的凸仿射 learner 如 [Definition 4.3](04-03_binary_context_student_affine_learner_ZH.md#s4-3)：$`p_R=w_0p_0+\sum_iw_iZ_i`$，凸权重固定。
- $`T_j`$ 是第一类 Chebyshev 多项式（[Rivlin, 1990](../references_ZH.md#rivlin1990chebyshev)）：$`T_j(\cos\theta)=\cos(j\theta)`$，所以在 $`[-1,1]`$ 上 $`\lvert T_j\rvert\le1`$，且 $`T_j(1)=1`$。$`\mathcal P_q`$ 是总次数不超过 $`q`$ 的多项式集合，$`\lVert\cdot\rVert_{\infty,S}`$ 是集合 $`S`$ 上的上确界范数。
- $`B\sim\mathrm{Bin}(2N,\tfrac12)`$ 是一个二项随机变量。

<a id="s4-6"></a>

**Proposition 4.6（精确阶不等于近似复杂度）.** 令 $`T_j`$ 为第一类 Chebyshev 多项式（$`T_j(\cos\theta)=\cos(j\theta)`$，在 $`[-1,1]`$ 上 $`\lvert T_j\rvert\le1`$，$`T_j(1)=1`$），$`\mathcal P_q`$ 为总次数不超过 $`q`$ 的多项式。考虑 $`[0,1]`$ 上的标量任务 $`J(p)=p^N`$（同一 context 访问 $`N`$ 次，全部动作为 1 时成功），以及 Theorem 4.4(c) 中 $`[0,1]^N`$ 上的 parity 任务 $`J_{\rm even}`$。对 $`p\in[0,1]`$，

```math
p^N=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1). \qquad(4.7)
```

- **(a) 光滑的价值只需要低的近似阶.** 设 $`q<N`$，$`h_q\in\mathcal P_q`$ 为 (4.7) 截断到 $`q`$ 阶的结果，$`B\sim\mathrm{Bin}(2N,\tfrac12)`$。则：
  - $`\lVert p^N-h_q\rVert_{\infty,[0,1]}=\Pr\lbrace\lvert B-N\rvert>q\rbrace\le2e^{-q^2/N}`$；
  - 若 $`[0,1]`$ 上两个分布 $`P,Q`$ 的 $`\ell\le q`$ 阶矩都相等，则 $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert\le4e^{-q^2/N}`$；
  - 对任何带独立同分布标量块的凸仿射 learner，学习价值 $`\mathbb E\,p_R^N`$ 满足同样的界。
- 因此，匹配到 $`q\ge\min\lbrace N,\lceil\sqrt{N\log(4/\varepsilon)}\rceil\rbrace`$ 阶的矩就能保证差距不超过 $`\varepsilon`$，尽管要对所有分布精确相等需要 $`N`$ 阶（Theorem 4.4(c)）。
- **(b) Parity 在次数低于 $`N`$ 时不可压缩.** 对每个 $`q<N`$，

```math
\min_{h\in\mathcal P_q}\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}=\tfrac12 ,
```

- 且常数 $`h\equiv\tfrac12`$ 取到最小值。

**说了什么.**

- (a) 对光滑的价值 $`p^N`$，把它的 Chebyshev 展开截断到 $`q`$ 阶，误差恰为一个二项分布的尾概率，按 $`e^{-q^2/N}`$ 下降。所以大约 $`\sqrt{N\log(4/\varepsilon)}`$ 个匹配的矩就已保证精度 $`\varepsilon`$，对分布和平均化 learner 都一样。
- (b) 对 parity，任何次数低于 $`N`$ 的多项式都逼近不过平凡常数 $`\tfrac12`$，所以低于 $`N`$ 的阶都无法把它分辨到优于 $`\tfrac12`$。

**直觉.** 光滑的 response 有好的低阶逼近，组合型的 response 没有。$`p^N`$ 与 $`J_{\rm even}`$ 的精确次数都是 $`N`$，近似需求却完全不同。

**在框架中的作用.** 它说明起作用的复杂度是 response 类的近似维数 $`d_\varepsilon`$（[Theorem 4.2(b)](04-02_sufficiency_duality_transfer_ZH.md#s4-2)），而不是它的次数。近似匹配引出系数稳定性的问题，由 [Theorem 8.1](08-01_stability_approximation_tradeoff_ZH.md#s8-1) 精确回答。

**说明.**

- (a) 是对这个标量任务与凸仿射 learner 所需阶数的上界；不声称近似阶有匹配的下界。
- 精确的低阶匹配不是有限样本证书：若矩只是估计值，$`h_q`$ 在幂基下的系数会放大估计误差，而这些系数可以很大。有界的 Chebyshev 特征对单个标量消息避免了这一点，但经过多次仿射更新拉回之后，它们也需要自己的系数界；一般情形下相关的是 Theorem 4.2(c) 中的 $`(B,\tau)`$，由 Theorem 8.1 精确化。
- 两族见证都是可实现的有限反馈协议（reset 实验可以按指定分布发出一个块）。这说明这样的环境存在，不说明某个 benchmark 允许它们。

**推导要点.**

- (a) 令 $`2p-1=\cos\theta`$，$`p^N=\cos^{2N}(\theta/2)`$ 展开成以二项式系数为系数的 Chebyshev 多项式。非负系数的尾部就是上确界误差（在 $`p=1`$ 处取到），它是一个二项分布的尾概率，用 Hoeffding 不等式界住（[Hoeffding, 1963](../references_ZH.md#hoeffding1963probability)）。匹配的矩消去截断部分，剩下至多两倍的上确界误差。
- (b) 偶、奇 parity 分布在每个次数低于 $`N`$ 的多项式上都一致，在 $`J_{\rm even}`$ 上却相差 1；由 Theorem 4.2(b)，逼近误差至少为 $`\tfrac12`$。

**例示.** $`N=100`$、$`\varepsilon=0.01`$ 时，大约 $`\sqrt{100\log400}\approx25`$ 个匹配的矩就已经能分辨 $`p^{100}`$；而 100 个 context 上的 parity 需要全部 100 阶。

**用到.**

- [Theorem 4.2](04-02_sufficiency_duality_transfer_ZH.md#s4-2) — 差距等于两倍逼近误差。
- [Theorem 4.4](04-04_affine_learners_moment_order_ZH.md#s4-4) — 精确阶与 parity 见证。
- [Definition 4.3](04-03_binary_context_student_affine_learner_ZH.md#s4-3) — 凸仿射 learner。

**被用于.**

- [Theorem 8.1](08-01_stability_approximation_tradeoff_ZH.md#s8-1) — 近似接口的系数稳定性。

## 参考文献

- [Hoeffding, 1963](../references_ZH.md#hoeffding1963probability) — 有界变量之和的 Hoeffding 不等式。
- [Rivlin, 1990](../references_ZH.md#rivlin1990chebyshev) — Chebyshev 多项式。
