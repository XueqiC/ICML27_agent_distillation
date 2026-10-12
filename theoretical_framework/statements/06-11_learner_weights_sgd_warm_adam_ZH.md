# Proposition 6.11 · SGD 与 warm Adam 的学习器权重

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md) · [证明](../proofs/06-11_learner_weights_sgd_warm_adam_ZH.md) · [实验](../experiments/06-11_learner_weights_sgd_warm_adam_ZH.md) · [English](06-11_learner_weights_sgd_warm_adam.md)

**一段话概括.** 这个命题在每个 agent 状态拥有一个参数坐标的模型中，计算 SGD 与 warm Adam 的学习器相对权重 $`W_s`$。SGD 按访问频率的平方给状态加权；频繁访问区间的 warm Adam 按访问频率本身加权，并带有一个二阶矩项；对极少访问的状态，Adam 回到平方标度，但只看反馈的符号。

**本页用到的设定与记号.**

- **agent 状态** $`s\in\mathcal S`$，有限集；本页中 $`s`$ 表示 agent 的状态（[Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)）。
- **逐状态的 oracle 分布.** 在状态 $`s`$ 上 oracle 消息 $`Y\sim P^*_s`$（例如 teacher 在该状态的动作或修正），取值于有限字母表；各状态的分布互不相关，不假设状态之间有泛化。
- **influence function** $`f_s(y)`$：$`\Psi`$ 在 $`s`$ 处、把 $`P^*_s`$ 向 $`y`$ 处点质量移动时的导数；在 $`P^*_s`$ 下均值为零。
- **学习器相对权重** $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$：在 $`s`$ 上多一个 oracle 样本，对估计 $`\Psi`$ 值多少。
- **样本数.** 状态 $`s`$ 共得到 $`N_s=k_s+n_s`$ 个 oracle 样本：$`k_s`$ 来自示范（期望为 $`KH\,d_T(s)`$，$`H`$ 为轨迹长度，$`d_T`$ 为 teacher 的状态访问分布），$`n_s`$ 来自查询，$`\sum_sn_s=n`$。
- **state-owned 坐标.** 每个 agent 状态 $`s`$ 有自己的标量参数 $`\theta_s`$；每步按学生的访问概率 $`d(s)`$ 独立抽一个状态 $`s_k\sim d`$，并在那里消费一条 oracle 消息。
- **梯度.** consumed gradient $`g=G(s,Y)`$ 与参数无关（线性 response 区间）；在状态 $`s`$ 上 $`m_1=\mathbb Eg`$，$`m_2=\mathbb Eg^2`$。
- **结果** $`F=\langle\zeta,\theta_R\rangle`$，共 $`R`$ 步，学习率 $`\eta`$；$`\Psi`$ 是结果的期望变化。
- **Adam**：动量系数 $`\beta_1`$、二阶矩衰减 $`\beta_2`$，$`\beta_1<\sqrt{\beta_2}`$，$`\epsilon`$ 可忽略（Adam 的精确增量见 [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)）。

<a id="s6-11"></a>

**Proposition 6.11（SGD 与 warm Adam 的学习器权重）.** 在 [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9) 之外，再设：

- 每个 agent 状态拥有自己的参数坐标（state-owned coordinates）；
- 每个训练步独立抽取一个状态 $`s_k\sim d`$，并在那里给一条 oracle 消息；
- consumed gradient 与参数无关（线性 response 区间）：$`g=G(s,Y)`$；在状态 $`s`$ 上记 $`m_1=\mathbb Eg`$，$`m_2=\mathbb Eg^2`$；
- 结果为 $`F=\langle\zeta,\theta_R\rangle`$，$`\Psi`$ 取为期望结果变化 $`\mathbb E F(\theta_R)-F(\theta_0)`$，它是 $`(P^*_s)_s`$ 的泛函；
- Adam 的 $`\epsilon`$ 相对 $`\sqrt v`$ 可以忽略，且 $`\beta_1<\sqrt{\beta_2}`$。

则在各自的区间内、当 $`R\to\infty`$ 时，在首项意义下：

- **(a) SGD.** $`f_s^{\rm SGD}(y)=-\eta R\,d(s)\,\zeta_s\,(g-m_1)`$，所以 $`W_s^{\rm SGD}=\eta^2R^2\zeta_s^2\,d(s)^2\,\mathrm{Var}(g)`$。
- **(b) warm Adam，频繁访问区间（frequent regime）.** 设二阶矩的有效平均窗口内访问次数很多：有 bias correction 时 $`d(s)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$，无 bias correction 时还要求 $`t\gg1/(1-\beta_2)`$。则

```math
f_s^{\rm Adam}(y)=-\eta R\sqrt{d(s)}\,\zeta_s\Big[\frac{g-m_1}{\sqrt{m_2}}-\frac{m_1(g^2-m_2)}{2m_2^{3/2}}\Big],\qquad W_s^{\rm Adam}=\eta^2R^2\zeta_s^2\,d(s)\,\mathrm{Var}\Big(\frac{g}{\sqrt{m_2}}-\frac{m_1g^2}{2m_2^{3/2}}\Big).
```

  括号中的第二项是二阶矩通道：一条抬高 $`m_2`$ 的消息会压低此后的每一个步长。

- **(c) warm Adam，孤立访问区间（isolated regime）.** 设 $`d(s)\ll\min\lbrace1-\beta_1,1-\beta_2\rbrace`$，且无 bias correction 或 $`t\gg1/(1-\beta_2)`$。此时每次访问遇到的都是 $`m\approx0`$、$`v\approx0`$，坐标移动 $`-\eta\,\kappa_{\rm iso}\,\mathrm{sign}(g)`$，其中

```math
\kappa_{\rm iso}=\frac{1-\beta_1}{\sqrt{1-\beta_2}\,\big(1-\beta_1/\sqrt{\beta_2}\big)} .
```

  因此 $`f_s(y)=-\eta R\,d(s)\,\zeta_s\,\kappa_{\rm iso}\,(\mathrm{sign}\,g-\mathbb E\,\mathrm{sign}\,g)`$，$`W_s\propto d(s)^2\,\mathrm{Var}(\mathrm{sign}\,g)`$：关于 $`d`$ 的标度与 SGD 相同，但消息只通过符号进入。有 bias correction 时，若在第 $`t\gg1/(1-\beta_1)`$ 步访问一个 $`m`$、$`v`$ 仍为零的坐标（例如第一次访问），位移为 $`-\eta\,\kappa_{\rm iso}\sqrt{1-\beta_2^{\,t}}\,\mathrm{sign}(g)`$：常数改变，关于 $`d`$ 的标度不变。

- **(d) 推论.** 设 $`s`$ 是高频状态、$`s'`$ 是较稀有的状态，$`d(s)>d(s')`$，两者都处在频繁访问区间，方差因子相同。权重比 $`W_{s'}/W_s`$ 在 SGD 下是 $`(d(s')/d(s))^2`$，在 warm Adam 下是 $`d(s')/d(s)`$。所以相对于高频状态，warm Adam 对稀有状态的权重是 SGD 的 $`d(s)/d(s')`$ 倍。覆盖了高频状态、漏掉学生长尾状态的示范，可以对 SGD 充分，对 warm Adam 不充分。

**说了什么.**

- SGD 下一个状态的权重随访问频率的平方增长；warm Adam 在频繁访问区间只随访问频率线性增长，并带有一个二阶矩项；对极少访问的状态，Adam 又回到平方标度，但只看反馈的符号。

**直觉.**

- Adam 按 running second moment 归一化，少见状态的步长和常见状态一样大，所以这些状态上的反馈比在 SGD 下更重要。

**在框架中的作用.**

- 它说明 optimizer 改变了信息该在哪里收集，并给出常数 $`\kappa_{\rm iso}`$；它的有限 horizon 版本出现在 [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2) 中。

**推导要点.**

- SGD 的期望位移关于该状态的消息分布线性，所以 influence function 正比于 $`d(s)(g-m_1)`$。
- 频繁访问区间中二阶矩平均为 $`d(s)m_2`$，期望位移为 $`-\eta R\sqrt{d(s)}\,m_1/\sqrt{m_2}`$；对 $`m_1`$、$`m_2`$ 求导（delta method）得到 influence function，其中对 $`m_2`$ 的导数就是二阶矩通道。
- 孤立访问区间中，两个矩在每次访问之前都已衰减殆尽；访问本身带来的动量与二阶矩随后按几何速度衰减，它们的比值求和得到 $`\kappa_{\rm iso}\,\mathrm{sign}(g)`$。
- 在 $`d(s)\approx1-\beta_2`$ 附近，两个闭式都不适用。

**示例.** [Example 6.13](../statements/06-13_long_tail_example_ZH.md#s6-13)：一个常见状态以概率 $`1-q`$ 被访问，$`N`$ 个长尾状态各以概率 $`q/N`$ 被访问，都处在频繁访问区间。相对于常见状态，长尾在 $`\sum_sW_s/N_s`$ 中的份额在 warm Adam 下比在 SGD 下大 $`N(1-q)/q`$ 倍；例如 $`N=20`$、$`q=0.2`$ 时为 $`80`$ 倍。

**用到.**

- [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)：状态覆盖模型。
- [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)：权重进入信息公式。
- [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)：Adam 的精确增量。
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10)：各学习器的一阶统计量。

**被用于.**

- [Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)：长尾分离。
- [Example 6.13](../statements/06-13_long_tail_example_ZH.md#s6-13)：算例。
- [Corollary 7.2](../statements/07-02_stationary_amplitude_law_ZH.md#s7-2)：有限 horizon 的孤立区间常数。
- [Algorithm 9.2](../statements/09-02_ced_algorithm_ZH.md#s9-2)：CED 中的学习器相对分配。

**状态.** 在所述各区间内首项意义下已证明。
