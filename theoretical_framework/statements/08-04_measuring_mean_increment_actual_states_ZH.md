# Proposition 8.4 · 在实际状态上测量平均增量

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-04_measuring_mean_increment_actual_states_ZH.md) · [证明](../proofs/08-04_measuring_mean_increment_actual_states_ZH.md) · [实验](../experiments/08-04_measuring_mean_increment_actual_states_ZH.md) · [English](08-04_measuring_mean_increment_actual_states.md)

**一段话概括.** 一阶 audit 通过平均 optimizer 增量比较两个 feedback law。这个命题说明：在实际状态上，平均增量只用梯度计算就能估计，误差在 Hilbert 范数下按样本数平方根的倒数下降；再把估计转成一轮效应与整次训练差距的界。它也明确指出样本测不到的东西：continuation value 的梯度范数与曲率。

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

沿用 [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) 的对象：consumed gradient $`G_t(z,Y,\xi)`$，增量 $`\Delta_t(z,g)`$，平均增量 $`\mu^P_t(z)`$，增量二阶矩 $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$，以及 continuation $`V^P_{t+1}`$。范数取状态空间的 Hilbert 范数（有限维时为 Euclidean 范数）。

- **增量有界 (B).** 存在 $`b<\infty`$，使在被 audit 的状态上、对被比较的律，几乎必然 $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$。例如：参数块上的裁剪 SGD，$`b=\eta c`$；$`p`$ 个参数上的 sign 类更新，$`b=\eta\sqrt p`$；梯度以 $`\bar G`$ 为界的 SGD，$`b=\eta\bar G`$。
- **曲率 (S).** 在被 audit 的状态上，$`V:=V^P_{t+1}`$ 沿增量到达的线段 $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ 二阶可微，且 $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$；记 $`\Lambda=\lVert\nabla V(z)\rVert`$。
- **audit 样本.** 在被 audit 的状态 $`z`$ 上，独立抽取 $`n_P`$ 条消息 $`Y\sim P_t(\cdot\mid z)`$ 与 $`n_Q`$ 条消息 $`Y\sim Q_t(\cdot\mid z)`$，每条配独立的更新随机性，计算增量 $`X=\Delta_t(z,G_t(z,Y,\xi))`$。经验均值为 $`\bar X_P,\bar X_Q`$，经验二阶矩为 $`\hat m^P_2,\hat m^Q_2`$。这需要在 $`z`$ 处做 $`n_P+n_Q`$ 次 consumed gradient 计算，不需要训练尾段。

<a id="s8-4"></a>

**Proposition 8.4（在实际状态上测量平均增量）.**

**(a) Hilbert 空间中的集中性.** 设 $`X_1,\dots,X_n`$ 是实 Hilbert 空间中的独立同分布随机元，几乎必然 $`\lVert X\rVert\le b`$，均值为 $`\mu`$。则以至少 $`1-\delta`$ 的概率，

```math
\lVert\bar X-\mu\rVert\le\frac b{\sqrt n}\Big(1+\sqrt{2\log(1/\delta)}\Big).
```

对大小为 $`n_P,n_Q`$ 的两个独立样本，以至少 $`1-\delta`$ 的概率，

```math
\big\lVert(\bar X_P-\bar X_Q)-(\mu^P-\mu^Q)\big\rVert\le b\Big(1+\sqrt{2\log(2/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big).
```

**(b) 在一个实际状态上的一阶 audit.** 在产生被 audit 状态 $`z`$ 的历史条件下固定 $`z`$，并设 (B) 与 (S) 在该处成立。则在 audit 样本上以至少 $`1-\delta`$ 的概率，

```math
\big\lvert(K^P_t-K^Q_t)V^P_{t+1}(z)\big\rvert\le\Lambda\big(\lVert\bar X_P-\bar X_Q\rVert+e_1\big)+\tfrac{M_2}{2}\big(\hat m^P_2+\hat m^Q_2+e_2\big),
```

其中

```math
e_1=b\Big(1+\sqrt{2\log(4/\delta)}\Big)\Big(\tfrac1{\sqrt{n_P}}+\tfrac1{\sqrt{n_Q}}\Big),\qquad e_2=b^2\Big(\sqrt{\tfrac{\log(4/\delta)}{2n_P}}+\sqrt{\tfrac{\log(4/\delta)}{2n_Q}}\Big).
```

**(c) 沿实际训练的整次 audit.** 独立运行协议 $`Q`$ 共 $`J`$ 次。在每次运行的每一轮 $`t`$，在访问到的状态上以置信度 $`\delta_{\rm a}/(JR)`$ 应用 (b)，把 (b) 的右端截断到 $`1`$，记为 $`\hat\varepsilon_{j,t}`$。设 (B) 与 (S) 在每个访问到的状态上以所述常数成立，且抽取 audit 样本不改变训练过程。则以至少 $`1-\delta_{\rm a}-\delta_{\rm mc}`$ 的概率，

```math
\lvert L(P)-L(Q)\rvert\le\frac1J\sum_{j=1}^J\sum_{t=0}^{R-1}\hat\varepsilon_{j,t}+R\sqrt{\frac{\log(1/\delta_{\rm mc})}{2J}}.
```

**测了什么，没测什么.**

- 依赖数据的部分是 $`\bar X_P-\bar X_Q`$ 与 $`\hat m_2`$，它们只需要在访问到的状态上计算 consumed gradient。
- $`\Lambda`$ 与 $`M_2`$ 是 continuation value 的性质，与消息无关，样本不估计它们。它们必须解析地给界，或者用完整尾段的 audit 校准（[Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3)），并把校准误差加进界里。

**说了什么.** 一个 feedback law 的平均 optimizer 增量只用梯度计算就能估计，误差在 Hilbert 范数下按样本数平方根的倒数下降。结合 continuation 的梯度范数与曲率，就得到一轮效应的界，沿抽样的训练过程还得到整次训练差距的界。

**直觉.** 一阶证书的数据一侧很便宜；continuation 一侧（梯度与曲率）这些样本根本测不到。

**在框架中的作用.** 它是一阶 audit 中便宜的那一半，并说明一阶 audit 为什么仍然需要校准常数。

**推导要点.**

- 在 Hilbert 空间中 $`\lVert\bar X-\mu\rVert^2`$ 的交叉项为零，所以平均偏差不超过 $`b/\sqrt n`$；McDiarmid 不等式再补上偏离项。
- 二阶 Taylor 界把一轮的效应写成一阶项加余项，Cauchy–Schwarz 用 $`\Lambda`$ 乘以平均增量之差控制一阶项。
- 整次训练的版本结合了 telescoping 恒等式、对各次 audit 的 union bound，以及对运行次数的 Hoeffding。

**用到.**

- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) — 一轮效应的一阶展开
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10) — 各学习器产生的增量
- [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) — 整次训练的 telescoping 恒等式
- [Theorem 8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) — 用完整尾段 audit 校准常数

**被用于.**

- [Remark 8.5](../statements/08-05_norms_constants_ZH.md#s8-5) — 界中的范数与常数
- [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) — 一阶证书及其上限

## Related work

- [Theorems 8.2](../statements/08-02_frozen_finite_interface_selection_transport_ZH.md#s8-2)–[8.3](../statements/08-03_actual_state_adaptive_audit_fidelity_ZH.md#s8-3) 与 [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4) 用到 Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)）、union bound 与 McDiarmid 不等式（[McDiarmid, 1989](#ref-mcdiarmid1989method)）。按置信下界选择以保证不退化，是 high-confidence policy improvement（[Thomas et al., 2015](#ref-thomas2015high)）的思路。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-mcdiarmid1989method"></a>

**Colin McDiarmid (1989).** On the method of bounded differences. In *Surveys in Combinatorics, 1989*, London Mathematical Society Lecture Note Series 141, 148–188, Cambridge University Press. DOI [10.1017/CBO9781107359949.008](https://doi.org/10.1017/CBO9781107359949.008).

<a id="ref-thomas2015high"></a>

**Philip S. Thomas, Georgios Theocharous, Mohammad Ghavamzadeh (2015).** High confidence policy improvement. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, PMLR 37:2380–2388. [proceedings.mlr.press/v37/thomas15](https://proceedings.mlr.press/v37/thomas15.html).
