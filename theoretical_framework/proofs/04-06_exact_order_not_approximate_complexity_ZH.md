# Proposition 4.6 · 精确阶不等于近似复杂度

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/04-06_exact_order_not_approximate_complexity_ZH.md) · [证明](04-06_exact_order_not_approximate_complexity_ZH.md) · [实验](../experiments/04-06_exact_order_not_approximate_complexity_ZH.md) · [English](04-06_exact_order_not_approximate_complexity.md)

**本页证明什么.** 光滑的部署价值 $`p^N`$ 与 parity 价值 $`J_{\rm even}`$ 的精确次数都是 $`N`$，所以两者的精确可互换都需要直到 $`N`$ 阶的全部矩。但在近似意义下它们完全不同：把 $`p^N`$ 的 Chebyshev 展开截断到 $`q`$ 阶，误差等于一个二项分布的尾概率，所以大约 $`\sqrt{N\log(1/\varepsilon)}`$ 个匹配的矩就足以达到精度 $`\varepsilon`$；而 parity 用任何次数低于 $`N`$ 的多项式都逼近不到 $`\tfrac12`$ 以内。

**本页用到的设定.**

- 标量任务：一个 context 被访问 $`N`$ 次，全部动作为 1 时成功，所以动作概率为 $`p\in[0,1]`$ 的学生的部署价值为 $`J(p)=p^N`$。
- $`N`$ 个 context 上的 parity 任务：每个 context 访问一次，动作 $`a_j\sim\mathrm{Bernoulli}(p_j)`$ 相互独立，$`\sum_ja_j`$ 为偶数时回报 1：$`J_{\rm even}(p)=\tfrac12\big(1+\prod_{j=1}^N(1-2p_j)\big)`$（[Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)）。
- $`T_j`$ 是第一类 Chebyshev 多项式（[Rivlin, 1990](../references_ZH.md#rivlin1990chebyshev)）：$`T_j(\cos\theta)=\cos(j\theta)`$，所以在 $`[-1,1]`$ 上 $`\lvert T_j\rvert\le1`$，且 $`T_j(1)=1`$。$`\mathcal P_q`$ 是总次数不超过 $`q`$ 的多项式集合。
- $`[0,1]`$ 上（或块 $`Z\in[0,1]^k`$ 上）的分布 $`P,Q`$，以及带 IID 块的凸仿射 learner $`p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i`$，如 [Definition 4.3](../statements/04-03_binary_context_student_affine_learner_ZH.md#s4-3)。对 $`\lvert\alpha\rvert=a`$ 的多重指标 $`\alpha`$（标量情形下 $`\alpha=\ell`$），[Theorem 4.4](04-04_affine_learners_moment_order_ZH.md) 证明中的块矩展开为

```math
\mathbb E\,p_R^\alpha=\sum_s\Big(\prod_lw_{s(l)}\Big)\prod_{l:s(l)=0}p_{0,j(l)}\prod_{i=1}^Rm(\beta_i(s)), \qquad(4.4)
```

- 其中求和取遍来源映射 $`s:\lbrace1,\dots,a\rbrace\to\lbrace0,\dots,R\rbrace`$，$`\beta_i(s)`$ 收集分配给块 $`i`$ 的因子（所以 $`\lvert\beta_i(s)\rvert\le a`$），$`m(\beta)=\mathbb EZ^\beta`$ 是块矩。
- parity 分布：$`P`$ 为 $`\lbrace0,1\rbrace^N`$ 中偶 parity 顶点上的均匀分布，$`Q`$ 为奇 parity 顶点上的均匀分布。在两种分布下，每个真坐标子集都是均匀的（Theorem 4.4(c) 的证明）。

**陈述（[Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6)）.** 令 $`T_j`$ 为第一类 Chebyshev 多项式（$`T_j(\cos\theta)=\cos(j\theta)`$，在 $`[-1,1]`$ 上 $`\lvert T_j\rvert\le1`$，$`T_j(1)=1`$），$`\mathcal P_q`$ 为总次数不超过 $`q`$ 的多项式。考虑 $`[0,1]`$ 上的标量任务 $`J(p)=p^N`$（同一 context 访问 $`N`$ 次，全部动作为 1 时成功），以及 Theorem 4.4(c) 中 $`[0,1]^N`$ 上的 parity 任务 $`J_{\rm even}`$。对 $`p\in[0,1]`$，

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

**证明.** *Claim 1（恒等式 (4.7)）.* 令 $`2p-1=\cos\theta`$，$`\theta\in[0,\pi]`$，则 $`p=\cos^2(\theta/2)`$，

```math
p^N=\Big(\frac{e^{i\theta/2}+e^{-i\theta/2}}{2}\Big)^{2N}=4^{-N}\sum_{b=0}^{2N}\binom{2N}{b}e^{i(b-N)\theta}.
```

对 $`j\ge1`$ 把 $`b=N\pm j`$ 合并，并用 $`\binom{2N}{N+j}=\binom{2N}{N-j}`$，得 $`p^N=4^{-N}\binom{2N}N+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}\cos(j\theta)`$，而 $`\cos(j\theta)=T_j(2p-1)`$。两边都是多项式且在 $`[0,1]`$ 上相等，所以 (4.7) 是多项式恒等式。

*(a)，第一条.* 记 $`a_0=4^{-N}\binom{2N}N`$，$`j\ge1`$ 时 $`a_j=2\cdot4^{-N}\binom{2N}{N-j}`$。则 $`p^N-h_q(p)=\sum_{j=q+1}^Na_jT_j(2p-1)`$。

- 所有 $`a_j\ge0`$ 且 $`\lvert T_j\rvert\le1`$，所以 $`\lvert p^N-h_q(p)\rvert\le\sum_{j>q}a_j`$；在 $`p=1`$ 处每个 $`T_j(1)=1`$，等号成立。
- 把 $`b=N-j`$ 与 $`b=N+j`$ 配对，$`\sum_{j=q+1}^Na_j=\sum_{b:\lvert b-N\rvert>q}\binom{2N}b4^{-N}=\Pr\lbrace\lvert B-N\rvert>q\rbrace`$。
- $`B`$ 是 $`2N`$ 个独立 $`\lbrace0,1\rbrace`$ 变量之和，均值为 $`N`$，由 Hoeffding 不等式（[Hoeffding, 1963](../references_ZH.md#hoeffding1963probability)），$`\Pr\lbrace\lvert B-N\rvert\ge q\rbrace\le2\exp(-2q^2/(2N))=2e^{-q^2/N}`$。

*(a)，第二条.* 由于 $`h_q\in\mathcal P_q`$ 且两个分布的矩匹配到 $`q`$ 阶，$`\mathbb E_Ph_q(Z)=\mathbb E_Qh_q(Z)`$，所以 $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert=\lvert(P-Q)(p^N-h_q)\rvert\le\lVert P-Q\rVert_1\lVert p^N-h_q\rVert_\infty\le4e^{-q^2/N}`$。

*(a)，第三条.* 对 $`\ell\le q`$，(4.4) 把 $`\mathbb Ep_R^\ell`$ 写成不超过 $`\ell`$ 阶块矩之积的固定组合。所以 $`p_R\in[0,1]`$ 在 $`P^R`$ 与 $`Q^R`$ 下的分布矩匹配到 $`q`$ 阶，第二条对它们适用。

*所需阶数.* $`4e^{-q^2/N}\le\varepsilon`$ 当且仅当 $`q\ge\sqrt{N\log(4/\varepsilon)}`$；$`q\ge N`$ 时由 [Theorem 4.4(b)](../statements/04-04_affine_learners_moment_order_ZH.md#s4-4)（矩匹配到部署长度时学习价值相等）差距恰为 0。

*(b).* *上界.* $`J_{\rm even}`$ 是一个概率，所以 $`\lVert J_{\rm even}-\tfrac12\rVert_\infty\le\tfrac12`$。

*下界.* 设 $`h\in\mathcal P_q`$，$`q<N`$；令 $`P,Q`$ 分别为 $`\lbrace0,1\rbrace^N`$ 中偶、奇 parity 顶点上的均匀分布。

- 在二元顶点上，$`a\ge1`$ 时 $`p_j^a=p_j`$，所以 $`h`$ 在 $`\lbrace0,1\rbrace^N`$ 上等于一个多重线性多项式，其单项式 $`\prod_{j\in S}p_j`$ 满足 $`\lvert S\rvert\le q<N`$。
- 两种分布下每个真坐标子集都是均匀的，所以 $`\mathbb E_Ph=\mathbb E_Qh`$。
- 在顶点上 $`J_{\rm even}(Z)=\mathbf 1\lbrace Z\ \text{even}\rbrace`$，所以 $`\mathbb E_PJ_{\rm even}=1`$、$`\mathbb E_QJ_{\rm even}=0`$。

因此

```math
1=(P-Q)J_{\rm even}=(P-Q)(J_{\rm even}-h)\le\lVert P-Q\rVert_1\,\lVert J_{\rm even}-h\rVert_{\infty,\lbrace0,1\rbrace^N}\le2\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N},
```

即 $`\lVert J_{\rm even}-h\rVert_\infty\ge\tfrac12`$。等价地，由 [Theorem 4.2(b)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2)（最坏匹配对的差距等于两倍一致逼近误差），匹配对 $`(P,Q)`$ 证明了 $`\mathrm{dist}_\infty(J_{\rm even},\mathcal P_q)\ge\tfrac12`$。$`\square`$

## 附注

- **对比.** $`p^N`$ 与 $`J_{\rm even}`$ 的精确次数都是 $`N`$，两者的精确充分性都需要直到 $`N`$ 阶的全部矩。但它们的近似需求完全不同：
  - $`p^N`$ 用 $`O(\sqrt{N\log(1/\varepsilon)})`$ 阶就能分辨到精度 $`\varepsilon`$；
  - $`J_{\rm even}`$ 用任何低于 $`N`$ 的阶都分辨不到 $`\tfrac12`$ 以内。
- 所以起作用的复杂度是 response 类的近似维数 $`d_\varepsilon`$（Theorem 4.2(b)），而不是它的次数。
- **适用范围.** (a) 是对这个标量任务与凸仿射 learner 所需阶数的上界；不声称近似阶有匹配的下界。
- **稳定性.**
  - 精确的低阶匹配不是有限样本证书。若矩只是估计值，$`h_q`$ 在幂基下的系数会放大估计误差，而这些系数可以很大。
  - 有界的 Chebyshev 特征对单个标量消息避免了幂基放大，但经过多次仿射更新拉回之后，它们也需要自己的系数界。
  - 一般情形下相关的是 [Theorem 4.2(c)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 中的 $`(B,\tau)`$，[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1)（带测量容差时稳定性与逼近之间的精确权衡）把这一权衡精确化。
- **可实现性.** 两族见证都是可实现的有限反馈协议：reset 实验可以按指定分布发出一个块。这说明这样的环境存在，不说明某个 benchmark 允许它们。

## 相关工作

- Chebyshev 展开（[Rivlin, 1990](../references_ZH.md#rivlin1990chebyshev)）、Hoeffding 不等式（[Hoeffding, 1963](../references_ZH.md#hoeffding1963probability)）以及 parity 与真坐标子集的独立性都是经典工具，在此注明，不作为新结果。
- 本页新增的是用它们把 response 类的精确信息阶数与它的近似维数分开。

## 参考文献

- [Hoeffding, 1963](../references_ZH.md#hoeffding1963probability) — Hoeffding 不等式。
- [Rivlin, 1990](../references_ZH.md#rivlin1990chebyshev) — Chebyshev 多项式。
