# Proposition 6.11 · SGD 与 warm Adam 的学习器权重

*证明*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md) · [证明](../proofs/06-11_learner_weights_sgd_warm_adam_ZH.md) · [实验](../experiments/06-11_learner_weights_sgd_warm_adam_ZH.md) · [English](06-11_learner_weights_sgd_warm_adam.md)

**本页证明什么.** SGD 以及频繁、孤立两个区间中 warm Adam 的 influence function 与权重 $`W_s`$，以及由此得到的学习器之间的比值。

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

**陈述（[Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)）.** 在 [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9) 之外，再设：

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

**证明用到的结果.**

- [Assumption 6.9](../statements/06-09_state_coverage_model_ZH.md#s6-9)：逐状态独立的分布与 influence function。
- [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)：(c)、(d)，用于结论。

**用到的经典工具.** 大数定律与 delta method（[van der Vaart, 1998](#ref-vandervaart1998asymptotic)）。Adam 的递推（含 bias correction）按 [Kingma & Ba, 2015](#ref-kingma2015adam)。

**证明.** 坐标之间没有相互作用，所以 $`\Psi=\sum_s\zeta_s\,\mathbb E[\theta_{s,R}-\theta_{s,0}]`$，只需对每个 $`s`$ 计算期望位移关于 $`P^*_s`$ 的 Gateaux 导数。记 $`\delta m_1(y)=g(y)-m_1`$，$`\delta m_2(y)=g(y)^2-m_2`$，它们是 $`m_1,m_2`$ 沿 $`\delta_y-P^*_s`$ 方向的导数。

(a) SGD 的位移为 $`-\eta\sum_k1\lbrace s_k=s\rbrace g_k`$，均值 $`-\eta Rd(s)m_1`$，关于 $`P^*_s`$ 线性，导数为 $`-\eta Rd(s)\,\delta m_1`$。

(b) 访问构成 Bernoulli($`d(s)`$) 序列。二阶矩是逐步平方梯度 $`g_k^21\lbrace s_k=s\rbrace`$ 在约 $`\min\lbrace t,1/(1-\beta_2)\rbrace`$ 步有效窗口上的加权平均（bias correction 之后）。在频繁访问区间，窗口内访问很多，由大数定律 $`v_k=d(s)m_2(1+o(1))`$；单次访问使 $`v`$ 相对变化 $`(1-\beta_2)g^2/(d(s)m_2)=o(1)`$，所以 $`m_k`$ 与 $`v_k`$ 之间的相关是低阶的。动量关于梯度线性，$`\sum_km_k=\sum_jg_j1\lbrace s_j=s\rbrace(1-\beta_1^{R-j})`$，均值为 $`Rd(s)m_1(1+o(1))`$。于是期望位移为

```math
\Phi(m_1,m_2)=-\eta R\sqrt{d(s)}\,\frac{m_1}{\sqrt{m_2}}\,\big(1+o(1)\big).
```

$`\Phi`$ 是 $`(m_1,m_2)`$ 的光滑函数，由链式法则（delta method），它的 influence function 是 $`\partial_{m_1}\Phi\,\delta m_1+\partial_{m_2}\Phi\,\delta m_2`$，其中 $`\partial_{m_1}\Phi=-\eta R\sqrt{d}/\sqrt{m_2}`$，$`\partial_{m_2}\Phi=\tfrac12\eta R\sqrt d\,m_1/m_2^{3/2}`$。乘以 $`\zeta_s`$ 即得所列的 $`f_s`$；来自 $`\partial_{m_2}\Phi`$ 的那一项就是二阶矩通道。

(c) 两次访问之间，$`m`$ 与 $`v`$ 每步分别乘以 $`\beta_1`$ 与 $`\beta_2`$；间隔约为 $`1/d(s)`$ 步，所以 $`\beta_1^{1/d}`$ 与 $`\beta_2^{1/d}`$ 可以忽略，每次访问遇到 $`m\approx0`$、$`v\approx0`$。一次梯度为 $`g`$ 的访问之后第 $`j`$ 步，$`m=(1-\beta_1)\beta_1^{\,j}g`$，$`v=(1-\beta_2)\beta_2^{\,j}g^2`$，位移为

```math
-\eta\,\frac{(1-\beta_1)\beta_1^{\,j}g}{\sqrt{(1-\beta_2)\beta_2^{\,j}}\,\lvert g\rvert}=-\frac{\eta(1-\beta_1)}{\sqrt{1-\beta_2}}\Big(\frac{\beta_1}{\sqrt{\beta_2}}\Big)^{j}\mathrm{sign}(g).
```

由于 $`\beta_1<\sqrt{\beta_2}`$，几何级数收敛，各项在 $`O(1/(1-\beta_1/\sqrt{\beta_2}))`$ 步内衰减，远小于访问间隔；对 $`j\ge0`$ 求和得 $`-\eta\kappa_{\rm iso}\mathrm{sign}(g)`$。总位移为 $`-\eta\kappa_{\rm iso}\sum_k1\lbrace s_k=s\rbrace\mathrm{sign}(g_k)`$，均值 $`-\eta Rd(s)\kappa_{\rm iso}\mathbb E\,\mathrm{sign}(g)`$，关于 $`P^*_s`$ 线性，由此得到 $`f_s`$。有 bias correction 时，第 $`t`$ 步访问之后第 $`j`$ 步的位移要除以 $`(1-\beta_1^{\,t+j})`$、乘以 $`\sqrt{1-\beta_2^{\,t+j}}`$；当 $`t\gg1/(1-\beta_1)`$ 且 $`j`$ 在衰减范围内时，这两个因子分别是 $`1+o(1)`$ 与 $`\sqrt{1-\beta_2^{\,t}}\,(1+o(1))`$。

(d) 把 (a) 与 (b) 的权重相除，再用 [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)(c)、(d)。$`\square`$

**附注.**

- 两个区间在 $`d(s)\approx1-\beta_2`$ 附近相接：在那里一次访问对 $`v`$ 的影响既平均不掉，也衰减不完。两个闭式在那里都不适用。
- 在孤立区间里，位移只通过 $`\mathrm{sign}(g)`$ 依赖于消息：这是 [Balles & Hennig, 2018](#ref-balles2018dissecting) 分析的 Adam 的 sign descent 一面。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-balles2018dissecting"></a>

**Lukas Balles, Philipp Hennig (2018).** Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:404–413. [proceedings.mlr.press/v80/balles18a](https://proceedings.mlr.press/v80/balles18a.html).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-vandervaart1998asymptotic"></a>

**A. W. van der Vaart (1998).** Asymptotic Statistics. Cambridge University Press, Cambridge (Cambridge Series in Statistical and Probabilistic Mathematics 3). DOI [10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
