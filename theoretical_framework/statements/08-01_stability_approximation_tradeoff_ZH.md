# Theorem 8.1 · 稳定性–近似的权衡

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-01_stability_approximation_tradeoff_ZH.md) · [证明](../proofs/08-01_stability_approximation_tradeoff_ZH.md) · [实验](../experiments/08-01_stability_approximation_tradeoff_ZH.md) · [English](08-01_stability_approximation_tradeoff.md)

**一段话概括.** 有限次 audit 永远测不准特征均值，只能测到某个容差以内。这个定理计算：当两个消息律的特征均值在这些容差内一致时，它们的 response 差距最大还能有多大。答案是一个精确的权衡：一边是特征对 response 近似得多好，一边是这个近似的系数有多大。它是 §8（验证）的第一个工具：说明哪些特征、哪些 response 模型可以充当证书。

**本页用到的设定与记号.**

设 $`\mathcal Y`$ 是有限消息字母表，$`\lvert\mathcal Y\rvert=m\ge2`$。

- $`\mathcal Y`$ 上的函数是 $`\mathbb R^m`$ 中的向量，概率律是单纯形 $`\Delta(\mathcal Y)`$ 中的向量。对 $`h\in\mathbb R^m`$ 记 $`(P-Q)h=\sum_y(P(y)-Q(y))h(y)`$。
- $`g\in\mathbb R^m`$ 是一个 response。$`\phi_1,\dots,\phi_d\in\mathbb R^m`$ 是固定的特征，$`\Phi\in\mathbb R^{m\times d}`$ 以 $`\phi_j`$ 为列，$`\Phi b=\sum_jb_j\phi_j`$。
- $`\mathbf{\epsilon}=(\epsilon_1,\dots,\epsilon_d)`$ 是容差，$`\epsilon_j\ge0`$。允许零容差，也允许特征线性相关。
- $`h`$ 的振幅（oscillation）为 $`\mathrm{osc}(h)=\max_yh(y)-\min_yh(y)`$。

$`g`$ 的**容差差距**（tolerant discrepancy）定义为

```math
W_{\mathbf{\epsilon}}(g)=\sup\Big\lbrace\lvert(P-Q)g\rvert:\ P,Q\in\Delta(\mathcal Y),\ \lvert(P-Q)\phi_j\rvert\le\epsilon_j\ \ (j=1,\dots,d)\Big\rbrace .
```

它是特征均值之差不超过所声明容差的两个消息律之间，response 差距的最大值。一次有限 audit 把每个特征均值测到精度 $`\epsilon_j`$ 之后，面对的正是这种情形。

<a id="s8-1"></a>

**Theorem 8.1（稳定性–近似的权衡）.**

**(a) 精确权衡.** 下面两个最小值都能取到，并且

```math
W_{\mathbf{\epsilon}}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big].
```

特别地 $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$；当 $`g\in[0,1]^m`$ 时它不超过 $`1`$。

- **(b) 精确情形.** $`\mathbf{\epsilon}=0`$ 时，$`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$，其中 $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$。这是 [Theorem 4.2(b)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 的单个 response 版本。
- **(c) 稳定性反例.** 设 $`g\in\lbrace0,1\rbrace^m`$ 不是常数，$`d=1`$，$`\phi_1=\alpha g`$，$`\alpha>0`$。则 $`W_\epsilon(g)=\min\lbrace1,\epsilon/\alpha\rbrace`$。
- **(d) 有限测量的桥梁.** 固定一个同时成立的事件，在其上每个被 audit 的特征均值满足 $`\lvert\widehat\mu_{P,j}-\mu_{P,j}\rvert\le e_{P,j}`$ 与 $`\lvert\widehat\mu_{Q,j}-\mu_{Q,j}\rvert\le e_{Q,j}`$。令 $`\epsilon_j=\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}`$。则在该事件上 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(g)`$。若还有 $`\lVert\widehat g-g\rVert_\infty\le\eta`$，则 $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$。

**说了什么.** 每个特征均值只知道到某个容差时，无法排除的最大 response 差距，恰好等于两件事的最优权衡：特征对 response 的近似误差，以及近似系数的大小（每个系数按其特征的容差加权）。

**直觉.** 测不准一个特征的代价，就是它在 response 模型里的系数。残差小但系数巨大的拟合什么也证明不了。

**在框架中的作用.** 它决定 audit 应该测哪些特征、哪些 response 模型可以充当证书。Theorem 8.2 的冻结 audit 用到它的线性项，方法中的拟合规则最小化它的右端。

**关于陈述的说明.**

- **含义.** $`\mathrm{osc}(g-\Phi b)`$ 是近似误差，$`\sum_j\epsilon_j\lvert b_j\rvert`$ 是测量误差的代价。一组特征即使能精确表示 $`g`$，若需要很大的系数，也不是好的证书：(c) 中 $`W=\epsilon/\alpha`$，随特征尺度 $`\alpha`$ 变小而变大。
- **紧性.** (a) 是对不受限制的消息律对的等式。若固定参照、只允许一族可实现的律，或者只允许固定边缘分布的 coupling（[Proposition 9.1](../statements/09-01_coupling_width_ZH.md#s9-1)），右端仍是上界，但不一定取到。
- **用途.** [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) 用到线性项，系数界 $`B_t\ge\lVert b\rVert_1`$。最小化 (a) 的右端，就是方法中拟合 response 模型的目标（[Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3)）。

**推导要点.**

- 概率律之差恰好是和为零、$`\ell_1`$ 范数不超过 $`2`$ 的向量，而这个集合的支撑函数就是振幅（oscillation）。
- 特征约束把 $`W_{\mathbf{\epsilon}}`$ 变成线性规划；弱对偶对每个系数向量 $`b`$ 给出上界，对 Lagrangian 用 Sion minimax 定理得到等号。
- 对偶目标是凸多面体函数且有下界，所以最小值能取到。

**示例.** 一个等于 response 乘以 $`\alpha=0.01`$ 的特征能精确表示 response，但在容差 $`\epsilon=0.005`$ 下只能认证 $`W=0.5`$：系数 $`1/\alpha`$ 放大了测量误差。

**用到.**

- [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) — 精确情形：一致近似误差的两倍
- [Proposition 4.6](../statements/04-06_exact_order_not_approximate_complexity_ZH.md#s4-6) — response 类的近似宽度

**被用于.**

- [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2) — 冻结 audit 中的系数界
- [Remark 9.3](../statements/09-03_derivation_design_rules_ZH.md#s9-3) — response 模型的拟合规则

## Related work

- [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff_ZH.md#s8-1) 是把线性规划对偶（[Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)）与范数对偶用在可操作的差距上，等号来自 Sion minimax 定理（[Sion, 1958](#ref-sion1958general)），不主张新的对偶原理。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-bertsimas1997introduction"></a>

**Dimitris Bertsimas, John N. Tsitsiklis (1997).** Introduction to Linear Optimization. Athena Scientific, Belmont, MA (Athena Scientific Series in Optimization and Neural Computation 6). ISBN 978-1-886529-19-9.

<a id="ref-sion1958general"></a>

**Maurice Sion (1958).** On general minimax theorems. *Pacific Journal of Mathematics* 8, no. 1, 171–176. DOI [10.2140/pjm.1958.8.171](https://doi.org/10.2140/pjm.1958.8.171).
