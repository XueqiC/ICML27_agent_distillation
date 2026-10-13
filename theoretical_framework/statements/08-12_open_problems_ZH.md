# Remark 8.12 · 未解决的问题

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-12_open_problems_ZH.md) · [实验](../experiments/08-12_open_problems_ZH.md) · [English](08-12_open_problems.md)

**一段话概括.** Theorem C 按所述形式是成立的，但实际中能否下结论取决于两个问题。这个 Remark 把它们记下来：一个远小于先验区间的、可证明的配对分数范围，以及一个可检验的条件，使 surrogate 的校准能迁移到新的候选对与后续状态。

**本页用到的设定与记号.**

全节沿用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 轮次 $`t=0,\dots,R-1`$，$`z`$ 是 complete learner state。对一阶 optimizer，实际更新为 $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$。
- 在第 $`t`$ 轮状态 $`z`$ 处，参照发出 $`Y\sim P^0_{t,z}`$，候选 $`j\in\lbrace1,\dots,M\rbrace`$ 发出 $`Y\sim P^j_{t,z}`$。候选 $`j`$ 的 kernel 记为 $`K^j_t`$。
- 参照 continuation 为 $`V^0_{t+1}`$，$`V^0_R=F_E\in[0,1]`$。response 为 $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$。对消息积分得到 kernel 恒等式

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- 对由候选 $`j`$ 驱动的过程，$`L_E(j)=\mathbb E\,F_E(Z^j_R)`$。所有过程从相同的 $`Z_0`$ 初始律出发。
- 候选 $`j`$ 在 $`(t,z)`$ 处的**局部优势**为

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- 它表示：把一轮参照替换成一轮候选 $`j`$，之后走参照尾段，对最终得分的影响。由 [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2)，整次训练的差距就是这些项在替代过程到达的状态上的和。

*配对替换分数*（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）.

- 在产生它的历史条件下，固定第 $`t`$ 轮的一个被 audit 状态 $`z`$（complete learner state，含 optimizer state）。候选为 $`j=0,1,\dots,M`$，$`M\ge1`$，其中 $`j=0`$ 是参照。候选 $`j`$ 的消息律为 $`P^j=P^j_t(\cdot\mid z)`$，消息 $`Y`$ 产生增量 $`\Delta^j=\Delta_t(z,G_t(z,Y,\xi))`$（含更新随机性）。
- **surrogate 尾段** $`T_\omega`$ 用随机性 $`\omega`$ 运行一段声明的（可以截短的）参照尾段，把一个状态映到最终状态。**surrogate 目标** $`\tilde F`$ 给最终状态打分（例如留出成功轨迹的 teacher-forced log-likelihood，映到 $`[0,1]`$）。记

```math
\varphi_\omega(u)=\tilde F\big(T_\omega(z+u)\big),\qquad\tilde L(P^j)=\mathbb E\,\varphi_\omega(\Delta^j),
```

其中 $`\omega`$ 与 $`\Delta^j`$ 独立。$`\tilde L(P^j)`$ 是"这一轮用候选 $`j`$、之后走 surrogate 尾段"的 surrogate 结果。

- **副本** $`\ell=1,\dots,r`$，$`r\ge2`$：抽一个新的 $`\omega_\ell`$，并为每个候选抽一条消息，得到增量 $`\Delta^j_\ell`$。同一副本内所有候选共用 $`\omega_\ell`$（**common random numbers**）；它们的消息也可以用共同的随机数耦合，只要每个 $`\Delta^j_\ell`$ 的边缘分布正确。不同副本相互独立。
- 若候选的消息律支撑有限，可以用支撑上的精确加权平均代替抽样，此时随机性只来自 $`\omega`$。
- 有序对 $`(j,k)`$ 的分数为

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_{\ell=1}^rZ^{jk}_\ell .
```

- 任务一侧，$`L_F(P^j)=\mathbb E\,F_E\big(T^{\rm full}(z+\Delta^j)\big)`$ 是"这一轮用候选 $`j`$、之后走完整参照尾段、按任务目标打分"的结果。用设定中的记号，$`L_F(P^j)=(K^j_tV^0_{t+1})(z)`$，所以 $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$。

*它的假设*（[Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7)）.

- **(I) 副本独立.** 在产生 $`z`$ 的历史条件下，向量 $`(Z^{jk}_\ell)_{j,k}`$（$`\ell=1,\dots,r`$）独立同分布；$`r`$ 在观察分数之前固定。
- **(B) 分数有界.** 对每一对，$`Z^{jk}_\ell`$ 几乎必然落在一个已知的、长度为 $`B`$ 的区间里。例如 $`\tilde F\in[0,1]`$ 时 $`B\le2`$。
- **(A) surrogate 对齐.** 存在 $`\alpha>0`$ 与 $`\varepsilon_{\rm sur}\ge0`$，使对所有候选对，

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ 同时吸收了 surrogate 与任务目标之间的差别，以及尾段截短带来的差别。

<a id="s8-12"></a>

**Remark 8.12（未解决的问题）.**

- **可证明的取值范围.** [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 在先验范围下成立（$`\tilde F\in[0,1]`$ 时 $`B\le2`$）。这时范围项的量级为 $`7B\log(4M^2/\delta)/(3r)`$，对任何负担得起的 $`r`$，都可能远大于相邻候选之间的差距。分数的观测范围不能代替它。缺的是一个可证明的、远小于先验范围的界，例如 [Proposition 8.10(c)](../statements/08-10_cost_and_crn_ZH.md#s8-10) 中的 pathwise Lipschitz 界，配上一个对真实训练尾段被证明的 $`\bar\lambda`$。
- **校准的迁移.** 需要一个可检验的条件，使在校准对上拟合的 $`(\alpha,\varepsilon_{\rm sur})`$ 在新的候选对与之后的状态上仍然有效。
- 在这两点解决之前，Theorem 8.8 在实践中是一个无偏、低方差的排序分数；只有当 $`B`$ 与 $`(\alpha,\varepsilon_{\rm sur})`$ 被独立论证时，它才是严格的证书。

**说了什么.** 两个未解决问题：一个远小于先验区间的可证明分数范围（例如来自经过认证的 pathwise Lipschitz 常数）；一个可检验的校准迁移条件。两者都解决之前，Theorem 8.8 在实践中是一个无偏、低方差的排序分数；只有在范围与校准都独立得到论证时，它才是严格的证书。

**直觉.** 瓶颈已经不是统计量，而是范围与校准。

**在框架中的作用.** 它精确说明还缺什么，才能让可测量的证书既严格又有用。

**用到.**

- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) — 证书
- [Proposition 8.10](../statements/08-10_cost_and_crn_ZH.md#s8-10) — 通往可证明范围的 pathwise Lipschitz 途径
- [Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11) — 校准及其迁移
- [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) — 范围与对齐假设

**被用于.**

- [Remark 9.4](../statements/09-04_scope_of_evidence_ZH.md#s9-4) — 证据的范围

## Related work

- 自适应于方差的置信界，仍要通过一个量级为 $`B/r`$ 的项为先验取值范围付出代价（[Audibert et al., 2009](#ref-audibert2009exploration)；[Maurer & Pontil, 2009](#ref-maurer2009empirical)）。所以更小的取值范围只能来自训练问题本身，例如经过认证的 Lipschitz 常数，而不是更精细的估计量。
