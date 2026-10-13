# Example 6.13 · 一个常见状态加 N 个长尾状态

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](../statements/06-13_long_tail_example_ZH.md) · [English](06-13_long_tail_example.md)

**一段话概括.** [Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)(b) 的算例：一个常见状态与 $`N`$ 个长尾状态，都处在 Adam 的频繁访问区间。相对于常见状态，长尾在信息需求中所占的份额，在 warm Adam 下比在 SGD 下大 $`N(1-q)/q`$ 倍。

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

<a id="s6-13"></a>

**Example 6.13（一个常见状态加 $`N`$ 个长尾状态）.** 学生以概率 $`1-q`$ 访问常见状态，以概率 $`q/N`$ 访问 $`N`$ 个长尾状态中的每一个。teacher 对常见状态的覆盖为 $`1-q_T`$，对每个长尾状态为 $`q_T/N`$，所以 $`N_s\propto KH\,d_T(s)`$。各状态的方差因子与部署权重相同，长尾状态处在 warm Adam 的频繁访问区间，即 $`(q/N)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$。

- **对 $`\sum_sW_s/N_s`$ 的贡献**（略去公共因子）：
  - 长尾状态：SGD 下为 $`N(q/N)^2/(q_T/N)=q^2/q_T`$，warm Adam 下为 $`N(q/N)/(q_T/N)=Nq/q_T`$；
  - 常见状态：SGD 下为 $`(1-q)^2/(1-q_T)`$，warm Adam 下为 $`(1-q)/(1-q_T)`$。
- **放大倍数.** 相对于常见状态，长尾项在 warm Adam 下比在 SGD 下大 $`N(1-q)/q`$ 倍。例如 $`N=20`$、$`q=0.2`$ 时放大 $`80`$ 倍；取 $`\beta_2=0.999`$，每个长尾状态在一个二阶矩窗口内的期望访问数为 $`(q/N)/(1-\beta_2)=10`$，处在频繁访问区间。
- **分配.** 因此 [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)(c) 的 water-filling 规则在 warm Adam 下会比在 SGD 下把更多查询预算分给长尾状态。

**说了什么.**

- 一个常见状态与 $`N`$ 个长尾状态：相对于常见状态，长尾在信息需求中的份额在 warm Adam 下比在 SGD 下大 $`N(1-q)/q`$ 倍（例如 $`N=20`$、$`q=0.2`$ 时为 $`80`$）。

**在框架中的作用.**

- 它说明在一个现实的访问分布下，学习器相对的放大可以有多大，以及为什么在 Adam 下 water-filling 会把更多查询送到长尾。

**用到.**

- [Proposition 6.10](../statements/06-10_water_filling_allocation_ZH.md#s6-10)：信息公式与 water-filling。
- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11)：学习器权重。

**被用于.**

- [Corollary 6.12](../statements/06-12_learner_separations_ZH.md#s6-12)：说明 (b)。

**状态.** 例子（由 Proposition 6.10 与 6.11 精确算出）。

## 相关工作

- 用于计算频繁区间的 $`\beta_2=0.999`$ 是 Adam 推荐的默认值（[Kingma & Ba, 2015](#ref-kingma2015adam)）。
- 把放大效应转成查询分配的 water-filling 规则，在没有 teacher 覆盖时退化为 Neyman 分配 $`n_s\propto\sqrt{W_s}`$（[Neyman, 1934](#ref-neyman1934two)）。
