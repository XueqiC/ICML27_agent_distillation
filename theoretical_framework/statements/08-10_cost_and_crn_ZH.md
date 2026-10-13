# Proposition 8.10 · 代价，以及 common random numbers 带来的好处

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-10_cost_and_crn_ZH.md) · [证明](../proofs/08-10_cost_and_crn_ZH.md) · [实验](../experiments/08-10_cost_and_crn_ZH.md) · [English](08-10_cost_and_crn.md)

**一段话概括.** Theorem C 只有在负担得起时才有用。这个命题计算它的代价，说明要多少副本才能让半径中的范围项变小，并说明 common random numbers 带来了什么：在 pathwise Lipschitz 条件下，配对分数的大小与两个候选增量之差同阶，而独立尾段会保留全部尾段噪声。

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

<a id="s8-10"></a>

**Proposition 8.10（代价，以及 common random numbers 带来的好处）.**

- **(a) 代价.** 一个被 audit 的状态需要 $`(M+1)r`$ 条 surrogate 尾段，外加校准用的任务评测尾段（[Remark 8.11](../statements/08-11_surrogate_calibration_and_transfer_ZH.md#s8-11)）。以 [Theorem 8.3(d)](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) 的半径做完整尾段的配对 audit，每个候选需要 $`2n_t`$ 条完整尾段。
- **(b) 范围项所需的副本数.** $`e_{\rm stat}`$ 的范围项不超过 $`\zeta B`$，当且仅当

```math
r\ \ge\ 1+\frac{7\log(4M^2/\delta)}{3\zeta}.
```

例如 $`M=8`$、$`\delta=0.05`$、$`\zeta=0.2`$ 时，$`\log(4M^2/\delta)=\log5120\approx8.54`$，需要 $`r\ge101`$。副本更少时，范围项会主导半径。

- **(c) common random numbers 下的方差.** 设 $`\varphi_\omega`$ 在到达的增量上 pathwise Lipschitz，常数为 $`\lambda_\omega`$：$`\lvert\varphi_\omega(u)-\varphi_\omega(u')\rvert\le\lambda_\omega\lVert u-u'\rVert`$。则

```math
\lvert Z^{jk}_\ell\rvert\le\lambda_{\omega_\ell}\,\lVert\Delta^j_\ell-\Delta^k_\ell\rVert,\qquad\mathrm{Var}\,Z^{jk}_\ell\le\mathbb E\big[\lambda_\omega^2\,\lVert\Delta^j-\Delta^k\rVert^2\big].
```

若两臂用独立的尾段 $`\omega,\omega'`$（消息也独立），则 $`\varphi_\omega(\Delta^j)-\varphi_{\omega'}(\Delta^k)`$ 的方差为 $`\mathrm{Var}\,\varphi_\omega(\Delta^j)+\mathrm{Var}\,\varphi_{\omega'}(\Delta^k)\ge\mathbb E\,\mathrm{Var}(\varphi_\omega(\Delta^j)\mid\Delta^j)+\mathbb E\,\mathrm{Var}(\varphi_{\omega'}(\Delta^k)\mid\Delta^k)`$，其中含有尾段本身的噪声，不会随两个候选靠近而消失。若进一步 $`\sup_\omega\lambda_\omega\le\bar\lambda`$，且在耦合下几乎必然 $`\lVert\Delta^j-\Delta^k\rVert\le b_{jk}`$，则 $`B=2\bar\lambda b_{jk}`$ 是 [Assumption 8.7(B)](../statements/08-07_certificate_assumptions_ZH.md#s8-7) 中合法的取值范围。

**说了什么.** 证书的代价是每个候选、每个副本一个 surrogate 尾段。要让范围项变小，副本数需要在一百左右；common random numbers 把每个配对分数的方差降到两个候选增量之差的量级。

**直觉.** 共享随机性的配对尾段只因候选不同而不同；独立尾段还会因训练本身的噪声而不同。

**在框架中的作用.** 它确定方法的副本预算，并指出 pathwise Lipschitz 界是得到可证明范围的途径。

**推导要点.**

计数；整理范围项；在配对增量处用 Lipschitz 条件，对独立的两臂用全方差公式。

**用到.**

- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) — 完整尾段 audit 的代价
- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) — 要确定范围项大小的半径
- [Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7) — 范围假设

**被用于.**

- [Remark 8.12](../statements/08-12_open_problems_ZH.md#s8-12) — 可证明范围这一未解决问题
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2) — 方法的副本预算

## Related work

- (b) 所确定的 range 项，就是 [Maurer & Pontil, 2009](#ref-maurer2009empirical) 的 empirical Bernstein 界的第二项：$`7\log(\cdot)/(3(r-1))`$ 乘以取值范围。
- (c) 是 common random numbers 的标准论证：当两个臂对共享随机性的响应方向一致时（例如 Lipschitz 或单调的响应），配对才有帮助（[Glasserman, 2004](#ref-glasserman2004monte)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-glasserman2004monte"></a>

**Paul Glasserman (2004).** Monte Carlo Methods in Financial Engineering. Springer, New York (Stochastic Modelling and Applied Probability 53). DOI [10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).
