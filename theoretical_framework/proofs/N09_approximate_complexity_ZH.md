# N09 · 精确阶数不等于近似复杂度 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/N09_approximate_complexity_ZH.md) · [English](N09_approximate_complexity.md)

## 设定与假设

- **标量任务。** 同一个 context 访问 $`N`$ 次、全部动作为 1 才算成功，其价值在 $`[0,1]`$ 上为 $`J(p)=p^N`$（N07，[N07(c)](N07_affine_moment_order_ZH.md)）。
- **Parity 任务。** $`k=N`$ 个 context 各访问一次，动作向量为偶 parity 时成功，其价值为 [N07(c)](N07_affine_moment_order_ZH.md) 中定义在 $`[0,1]^N`$ 上的 $`J_{\rm even}(p)=\tfrac12(1+\prod_{j=1}^N(1-2p_j))`$。
- **Chebyshev 多项式。** $`T_j`$ 是第一类 Chebyshev 多项式：$`T_j(\cos\theta)=\cos(j\theta)`$，在 $`[-1,1]`$ 上 $`\lvert T_j\rvert\le1`$，且 $`T_j(1)=1`$。
- **仿射 learner。** 即 N07 中的凸仿射平均 [N07](N07_affine_moment_order_ZH.md)，块 $`Z_i\in[0,1]`$ 为独立同分布的标量。
- **逼近空间。** $`\mathcal P_q`$ 表示总次数至多 $`q`$ 的多项式。
- **矩匹配。** "$`P,Q`$ 矩匹配到 $`q`$ 阶"在标量情形指对 $`\ell\le q`$ 有 $`\mathbb E_PZ^\ell=\mathbb E_QZ^\ell`$；在向量情形指所有总次数 $`\le q`$ 的联合矩都相等。

## 陈述

**引理（$`p^N`$ 的 Chebyshev 展开，式 (32)）。** 对 $`p\in[0,1]`$，

```math
p^N=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}T_j(2p-1). \qquad(32)
```

**(a)** **命题。** 设 $`q<N`$，$`h_q\in\mathcal P_q`$ 是 (32) 截断到 $`q`$ 阶的结果，$`B\sim\mathrm{Bin}(2N,\tfrac12)`$。则

1. $`\displaystyle\lVert p^N-h_q\rVert_{\infty,[0,1]}=\Pr\lbrace \lvert B-N\rvert>q\rbrace \le2e^{-q^2/N}`$。
2. 若 $`[0,1]`$ 上的两个标量分布 $`P,Q`$ 矩匹配到 $`q`$ 阶，则 $`\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert\le4e^{-q^2/N}`$。
3. 对任意固定的凸仿射 learner [N07](N07_affine_moment_order_ZH.md)，学习价值 $`\mathbb E\,p_R^N`$ 满足同样的界。

因此，只要矩匹配到

```math
q\ \ge\ \min\Big\lbrace N,\ \Big\lceil\sqrt{N\log(4/\varepsilon)}\Big\rceil\Big\rbrace
```

阶，就能保证差距不超过 $`\varepsilon`$；而要对所有分布都精确相等，则必须匹配到 $`N`$ 阶（N07，[N07(c)](N07_affine_moment_order_ZH.md)）。

**(b)** **命题（parity 在 $`N`$ 阶以下不可压缩）。** 对每个 $`q<N`$，

```math
\min_{h\in\mathcal P_q}\ \lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}=\tfrac12,
```

常数 $`h\equiv\tfrac12`$ 取到最小值。

## 证明

### 引理的证明

令 $`2p-1=\cos\theta`$，$`\theta\in[0,\pi]`$。则 $`p=\tfrac{1+\cos\theta}{2}=\cos^2(\theta/2)`$，于是

```math
p^N=\cos^{2N}(\theta/2)=\Big(\frac{e^{i\theta/2}+e^{-i\theta/2}}2\Big)^{2N}=4^{-N}\sum_{b=0}^{2N}\binom{2N}{b}e^{i(b-N)\theta}.
```

利用 $`\binom{2N}{N+j}=\binom{2N}{N-j}`$，把 $`j\ge1`$ 时 $`b=N\pm j`$ 的两项合并：

```math
p^N=4^{-N}\binom{2N}{N}+4^{-N}\sum_{j=1}^N\binom{2N}{N-j}\big(e^{ij\theta}+e^{-ij\theta}\big)=4^{-N}\binom{2N}{N}+2\cdot4^{-N}\sum_{j=1}^N\binom{2N}{N-j}\cos(j\theta).
```

由于 $`\cos(j\theta)=T_j(\cos\theta)=T_j(2p-1)`$，(32) 对每个 $`p\in[0,1]`$ 都成立。两边都是多项式，所以它是一个多项式恒等式。$`\square`$

### (a) 的证明

*(1)，等式部分。* 记 (32) 中 $`T_j(2p-1)`$ 的系数为 $`a_j`$：$`a_0=4^{-N}\binom{2N}N`$，$`j\ge1`$ 时 $`a_j=2\cdot4^{-N}\binom{2N}{N-j}`$。则

```math
p^N-h_q(p)=\sum_{j=q+1}^Na_jT_j(2p-1).
```

所有 $`a_j\ge0`$ 且 $`\lvert T_j\rvert\le1`$，所以 $`\lvert p^N-h_q(p)\rvert\le\sum_{j>q}a_j`$。在 $`p=1`$ 处每个 $`T_j(1)=1`$，上界被取到，于是

```math
\lVert p^N-h_q\rVert_\infty=\sum_{j=q+1}^Na_j=\sum_{j=q+1}^N2\cdot4^{-N}\binom{2N}{N-j}=\sum_{b:\lvert b-N\rvert>q}\binom{2N}{b}4^{-N}=\Pr\lbrace \lvert B-N\rvert>q\rbrace .
```

第三个等号把 $`b=N-j`$ 与 $`b=N+j`$ 配对。

*(1)，不等式部分。* $`B`$ 是 $`2N`$ 个独立 $`\lbrace 0,1\rbrace`$ 变量之和，均值为 $`N`$。由 Hoeffding 不等式，

```math
\Pr\lbrace \lvert B-N\rvert\ge t\rbrace \le2\exp(-2t^2/(2N)).
```

取 $`t=q`$ 得 $`\Pr\lbrace \lvert B-N\rvert>q\rbrace \le2e^{-q^2/N}`$。

*(2)。* 因为 $`h_q\in\mathcal P_q`$，且两个分布矩匹配到 $`q`$ 阶，有 $`\mathbb E_Ph_q(Z)=\mathbb E_Qh_q(Z)`$。于是

```math
\lvert\mathbb E_PZ^N-\mathbb E_QZ^N\rvert=\lvert(P-Q)(p^N-h_q)\rvert\le\lVert P-Q\rVert_1\,\lVert p^N-h_q\rVert_\infty\le2\cdot2e^{-q^2/N}.
```

*(3)。* 对 $`\ell\le q`$，N07 的展开 [N07(b)](N07_affine_moment_order_ZH.md) 把 $`\mathbb E\,p_R^\ell`$ 写成若干块矩乘积的固定组合，这些块矩的阶都 $`\le\ell\le q`$。所以 $`p_R`$ 在 $`P^R`$ 下与在 $`Q^R`$ 下的分布矩匹配到 $`q`$ 阶，且 $`p_R\in[0,1]`$。对 $`p_R`$ 的分布应用 (2) 即可。

*所需阶数。* $`4e^{-q^2/N}\le\varepsilon`$ 当且仅当 $`q\ge\sqrt{N\log(4/\varepsilon)}`$。当 $`q\ge N`$ 时，由 N07 的 [N07(b)](N07_affine_moment_order_ZH.md)，差距恰为 0。$`\square`$

### parity 命题的证明

*上界。* $`J_{\rm even}`$ 是一个事件的概率，所以 $`J_{\rm even}\in[0,1]`$，$`\lVert J_{\rm even}-\tfrac12\rVert_\infty\le\tfrac12`$。

*下界。* 设 $`h\in\mathcal P_q`$，$`q<N`$。令 $`P,Q`$ 分别为 $`\lbrace 0,1\rbrace ^N`$ 中偶、奇 parity 顶点上的均匀分布。

1. 在二元顶点上，$`a\ge1`$ 时 $`p_j^a=p_j`$。所以 $`h`$ 在 $`\lbrace 0,1\rbrace ^N`$ 上等于一个多重线性多项式，它的单项式 $`\prod_{j\in S}p_j`$ 都满足 $`\lvert S\rvert\le q<N`$。
2. 由 N07 的 [N07(c)](N07_affine_moment_order_ZH.md)(ii)，每个真坐标子集在两个分布下都是均匀的。因此 $`\mathbb E_Ph=\mathbb E_Qh`$。
3. 在顶点上，$`J_{\rm even}(Z)=\mathbf 1\lbrace Z\text{ even}\rbrace`$，所以 $`\mathbb E_PJ_{\rm even}=1`$、$`\mathbb E_QJ_{\rm even}=0`$。

于是

```math
1=(P-Q)J_{\rm even}=(P-Q)(J_{\rm even}-h)\le\lVert P-Q\rVert_1\,\lVert J_{\rm even}-h\rVert_{\infty,\lbrace 0,1\rbrace ^N}\le2\lVert J_{\rm even}-h\rVert_{\infty,[0,1]^N}.
```

所以 $`\lVert J_{\rm even}-h\rVert_\infty\ge\tfrac12`$。换个角度看，由 [N06(b)](N06_sufficiency_duality_ZH.md)，匹配的分布对 $`(P,Q)`$ 证明了 $`\mathrm{dist}_\infty(J_{\rm even},\mathcal P_q)\ge\tfrac12`$。$`\square`$

## 附注

1. **对比。** $`p^N`$ 和 $`J_{\rm even}`$ 的精确次数都是 $`N`$，要精确充分，两者都需要所有 $`N`$ 阶以内的矩（N07）。但它们的近似需求完全不同：
   - $`p^N`$ 只需 $`O(\sqrt{N\log(1/\varepsilon)})`$ 阶就能达到 $`\varepsilon`$ 精度；
   - $`J_{\rm even}`$ 用任何低于 $`N`$ 的阶都无法把误差降到 1/2 以下。

   所以起作用的复杂度是 response 类的逼近宽度 $`d_\varepsilon`$（[N06(b)](N06_sufficiency_duality_ZH.md)），而不是它的次数。
2. **适用范围。** (a) 针对的是这个标量任务和仿射 learner。它给出的是所需阶数的*上界*，并不声称有匹配的近似阶下界。
3. **稳定性。**
   - 精确的低阶矩匹配本身不是有限样本证书。若矩只是估计出来的，$`h_q`$ 在幂基下的系数会放大估计误差，而这些系数可能很大。
   - 对单条标量消息，有界的 Chebyshev 特征可以避免幂基放大；但经多次仿射更新拉回之后，它们需要自己的系数界。
   - 一般情况下，相关的量是 [N06(c)](N06_sufficiency_duality_ZH.md) 中的 $`(B,\tau)`$，这一权衡由 N13（[N13](N13_stability_tradeoff_ZH.md)）精确刻画。
4. **可实现性。** 两类见证都是可实现的有限反馈协议：一次 reset 实验输出一个按指定分布抽取的块（原文 [可实现性与访问限制](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#141-realizability-and-access-restrictions)）。这只说明所构造的环境存在，并不说明某个具体 benchmark 允许这样的实验。

## 依赖与来源

- **用到：** N06（逼近宽度 [N06(b)](N06_sufficiency_duality_ZH.md)）、N07（[N07(b)](N07_affine_moment_order_ZH.md)、[N07(c)](N07_affine_moment_order_ZH.md) 与块展开）。
- **被用于：** [N13](N13_stability_tradeoff_ZH.md)（稳定性–近似权衡）。

**出处：** [v0.14.1 §8](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#8-exact-order-is-not-approximate-information-complexity) — eqs. (32)–(33); Prop. 16; parity paragraph. v0.14.1 主框架 §8，式 (32)、Proposition 16 与 parity 段落，[Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md)。独立审阅：[Independent_Proof_Audit_EN.md](../../docs/direction/v014_handoff/Independent_Proof_Audit_EN.md) B2、B3。Chebyshev 展开、Hoeffding 不等式和 parity 独立性是经典工具。
