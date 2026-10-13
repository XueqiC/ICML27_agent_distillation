# Remark 7.5 · 一阶排序器在哪里失效

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](07-05_where_first_order_rankers_fail_ZH.md) · [实验](../experiments/07-05_where_first_order_rankers_fail_ZH.md) · [English](07-05_where_first_order_rankers_fail.md)

**一段话概括.** [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 的状态通道分解是精确的，但用它给候选反馈律排序时只保留了一阶项。本 remark 指出这种用法在哪里最不可靠：二阶矩通道在两个候选之差中占比很大的地方。那里消息相对于 $`v`$ 很大，正是幅度定律的饱和区域；任何一阶分数，即使用精确的 continuation 梯度算，都会误判这个差。在这一区域，候选必须用有限测量来排序。

**本页用到的设定与记号.**

- *候选.* 两个反馈律 $`j,k`$ 作用在同一个完整 Adam 状态 $`z=(\theta,m,v,s)`$ 上（[Definition 1.3](../README_ZH.md#s1-3)），完整状态的平均增量为 $`\mu^j,\mu^k`$；它们的*增量对比*是 $`\mu^j-\mu^k`$。
- *通道缩并.* 把对比与 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 的 continuation 梯度 $`\nabla V=(\chi^\theta,\chi^m,\chi^v)`$ 配对，得到三个数：$`\theta`$、$`m`$、$`v`$ 通道的缩并 $`\langle\chi^\theta,\mu^j_\theta-\mu^k_\theta\rangle`$、$`\langle\chi^m,\mu^j_m-\mu^k_m\rangle`$、$`\langle\chi^v,\mu^j_v-\mu^k_v\rangle`$。三者之和就是一阶差。
- *一阶分数.* 任何按这样的一阶差或其一部分给候选排序的排序器：只看参数的分数、用精确 continuation 梯度的完整状态分数，或沿平均增量测得的方向差。
- *有限替换测量.* [Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 的 CRN 配对替换分数：把一轮训练分别替换成各候选，走共同的参考尾段，再比较结果。让各候选共用同一尾段，就是经典的方差缩减手段 common random numbers（[Glasserman, 2004](#ref-glasserman2004monte)）。
- *饱和区域.* [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)(a)、(c) 的大幅度区域，即消息相对于 running RMS $`\sqrt v`$ 很大的情形。

<a id="s7-5"></a>

**Remark 7.5（一阶排序器在哪里失效）.**

- 定义一对候选的二阶矩占比：两者增量之差在 $`v`$ 通道上的缩并的绝对值，除以 $`\theta`$、$`m`$、$`v`$ 三个通道缩并的绝对值之和。
- 占比高通常意味着增量相对于 $`v`$ 很大。这正是 [Corollary 7.2(a),(c)](07-02_stationary_amplitude_law_ZH.md#s7-2) 描述的非线性、会饱和的区域：一阶价值仍按二次律走，真实 response 却不再如此。线性化在那里高估了 $`v`$ 通道造成的伤害。
- [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 的分解是精确的，但它的一阶用法恰好在 $`v`$ 通道重要的地方最不可靠。
- 二阶矩占比高时，排序应当用有限替换的测量（[Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)），或者用 [Corollary 7.2(a)](07-02_stationary_amplitude_law_ZH.md#s7-2) 那样的非线性公式，而不是任何一阶分数。只看参数的分数不够，用精确 continuation gradient 的完整状态分数也不够。
- [Proposition 7.3(b)](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 是 [Corollary 7.2(b)](07-02_stationary_amplitude_law_ZH.md#s7-2) 小增量区间内的陈述。

**说了什么.**

- 一对候选的二阶矩占比是一个可计算的诊断量，用来判断这对候选的一阶排序能否信任。
- 占比高时，要测量每个候选的有限效应，或用非线性的幅度公式来排序。

**直觉.**

- 大消息既走了一大步，又抬高了 $`v`$。一阶视角把这两种效应都线性外推；真实的 $`v`$ 效应会饱和，所以线性视角给消息记上的伤害远多于实际。
- 在单坐标的幅度定律里，一阶价值已经比真实 response 早好几倍就变负。

**在框架中的作用.**

- 它把方法中一阶分数的使用限制在一阶项占主导、占比低的候选对上（[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 第 4–5 步），其余交给 [Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 的有限测量。
- 它解释了为什么在 SGD 下准确的一阶 data attribution 方法到了 Adam 下会失效。

**用到.**

- [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)：三个通道缩并。
- [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)：饱和区域，以及一阶价值的提前变号。
- [Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)：小增量区间内的不一致性。
- [Theorem 8.8](08-08_crn_paired_replacement_certificate_ZH.md#s8-8)：有限配对替换测量。

**被用于.**

- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)（第 4–5 步）与 [Remark 9.3](09-03_derivation_design_rules_ZH.md#s9-3)（Rule 3 与 Rule 5）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-glasserman2004monte"></a>

**Paul Glasserman (2004).** Monte Carlo Methods in Financial Engineering. Springer, New York (Stochastic Modelling and Applied Probability 53). DOI [10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1).
