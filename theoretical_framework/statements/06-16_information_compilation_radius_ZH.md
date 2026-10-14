# Proposition 6.16 · 信息—编译半径

*陈述*

[← 框架总览](../README_ZH.md) · [陈述](06-16_information_compilation_radius_ZH.md) · [证明](../proofs/06-16_information_compilation_radius_ZH.md) · [实验](../experiments/06-16_information_compilation_radius_ZH.md) · [English](06-16_information_compilation_radius.md)

**一段话概括.** 有两种不同的原因会让蒸馏器复现不了 oracle 反馈的后果。一种是证据还说不清 oracle 会带来哪些后果；另一种是证据已经够了，但手头消息的任何合法混合都产生不了这些后果。Theorem A 衡量前者，[Proposition 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14) 衡量后者。这个命题把两者放进同一个数：在 response 接口上计算的鲁棒逼近半径，即在证据仍然允许的所有目标下，最好的合法反馈离它们最远有多远。半径为零，当且仅当两种缺口都已补上；它的两个下界指出哪一种缺口还开着，从而决定下一份预算该用于获取证据，还是扩充合法消息池。

**本页用到的设定与记号.**

**校准过的 response 接口.** 固定一个 checkpoint，以及有限个声明的测试 $`a`$，权重 $`\omega_a>0`$（权重为零的测试先去掉）。冻结的特征映射 $`\phi(y)\in\mathbb R^r`$ 把每个测试的 response 写成

```math
g_a(y)=k_a+c_a^\top\phi(y)+e_a(y),
```

其中系数 $`c_a`$ 已知，组成 $`C\in\mathbb R^{p\times r}`$（第 $`a`$ 行为 $`c_a^\top`$），$`e_a`$ 是残差。对分布 $`P`$，$`\mu_P=\mathbb E_P\phi(Y)`$。在 $`\mathbb R^p`$ 上，$`\lVert x\rVert_\omega^2=\sum_a\omega_ax_a^2`$。

**Response operator.** 同 [Definition 4.8](04-08_response_operator_ZH.md#s4-8)，参照测度为 $`\rho`$：对 $`h=dP/d\rho-dQ/d\rho`$，$`(\mathcal Rh)(a)=\langle\tilde g_a,h\rangle_\rho`$。残差算子 $`(\mathcal Eh)(a)=\langle\tilde e_a,h\rangle_\rho`$ 用的是中心化残差 $`\tilde e_a=e_a-\mathbb E_\rho e_a`$；$`\epsilon_{\rm tail}`$ 是它从 $`L^2_0(\rho)`$ 到 $`(\mathbb R^p,\lVert\cdot\rVert_\omega)`$ 的算子范数的上界，并设 $`\phi\in L^2(\rho)`$。

**未知的目标.** 目标反馈 $`P`$（例如真实环境中的 oracle 律）未知。由示范与查询构成的 transcript 给出一个非空紧集 $`\mathcal M\subset\mathbb R^r`$，满足 $`\Pr(\mu_P\in\mathcal M)\ge1-\delta`$。

**合法集合.** $`\mathcal Q_{\rm legal}`$ 是有限已核验消息池的一个非空合法混合集合，可以带密度约束（[Proposition 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14)），其特征均值集合为紧集。它在测试上的像为 $`\mathcal K=\lbrace C\mu_Q:Q\in\mathcal Q_{\rm legal}\rbrace`$。

**半径.** 对 $`q\in\mathcal K`$，$`\Gamma(q)=\sup_{\mu\in\mathcal M}\lVert C\mu-q\rVert_\omega`$，并且

```math
\mathfrak r(\mathcal M,\mathcal K)=\min_{q\in\mathcal K}\Gamma(q).
```

<a id="s6-16"></a>

**Proposition 6.16（信息—编译半径）.** 在上述设定下：

**(a) 存在性与凸性.** 最小值存在。$`\Gamma`$ 是凸的 1-Lipschitz 函数，所以只要 $`\mathcal K`$ 是凸集（例如合法混合构成的多胞形），外层问题就是凸规划。

**(b) 两个下界.**

```math
\mathfrak r(\mathcal M,\mathcal K)\ \ge\ \max\Big\lbrace\tfrac12\,\mathrm{diam}_\omega(C\mathcal M),\ \sup_{\mu\in\mathcal M}\mathrm{dist}_\omega(C\mu,\mathcal K)\Big\rbrace .
```

**(c) 半径何时为零.** $`\mathfrak r(\mathcal M,\mathcal K)=0`$，当且仅当 $`C\mathcal M`$ 是单点且这个点属于 $`\mathcal K`$。

**(d) 对半径最优编译的保证.** 设 $`q^*`$ 取到最小值，$`Q^*\in\mathcal Q_{\rm legal}`$ 满足 $`C\mu_{Q^*}=q^*`$。若 $`\lVert dP/d\rho-dQ^*/d\rho\rVert_\rho\le B`$，则以至少 $`1-\delta`$ 的概率

```math
\Big\lVert\mathcal R\Big(\frac{dP}{d\rho}-\frac{dQ^*}{d\rho}\Big)\Big\rVert_\omega\ \le\ \mathfrak r(\mathcal M,\mathcal K)+B\,\epsilon_{\rm tail}.
```

若外层问题只解到精度 $`\eta_{\rm opt}`$（$`\Gamma(\hat q)\le\mathfrak r+\eta_{\rm opt}`$），再加上 $`\eta_{\rm opt}`$。

**(e) 三种不同的补救.**

- 若 $`\mathcal M`$ 只沿 $`\ker C`$ 变化，$`C\mathcal M`$ 就是一个点：关于这些方向的更多证据降低不了半径，把环境重建到 $`C`$ 看得见的范围之外也没有好处。
- 若 $`C\mathcal M`$ 很大，无论消息池多丰富，至少还剩它直径的一半；只有能区分各后果方向的证据才能降低它。[Proposition 6.15](06-15_linear_gaussian_target_ZH.md#s6-15) 给出可以计算的线性高斯情形。
- 若 $`C\mathcal M`$ 是 $`\mathcal K`$ 之外的一个点，同类证据再多也没用；误差至少是到 $`\mathcal K`$ 的距离，只有更大的消息池或不同的接口能降低它。

**(f) 内层上确界的计算.** $`\Gamma(q)`$ 是一个关于 $`\mu`$ 的凸函数在 $`\mathcal M`$ 上的最大值。

- 若是有限个情景，它是有限个范数的最大值；当 $`\mathcal K`$ 是多面体（或更一般地可用二阶锥表示）时，外层问题是二阶锥规划。
- 若是单个椭球，二次函数 $`\lVert C\mu-q\rVert_\omega^2`$ 的内层最大值可由 S-lemma 精确算出（[Pólik & Terlaky, 2007](#ref-polik2007survey)），当 $`\mathcal K`$ 是多面体（或更一般地可用线性矩阵不等式表示）时，外层问题成为半定规划。
- 若是盒子或一般的多胞形，最大值在某个顶点取到，但在盒子上最大化凸二次函数一般是困难的（它包含 maximum cut）。$`r`$ 较小时可以枚举盒子的 $`2^r`$ 个顶点。

**说了什么.**

- 一个数，即这个半径，衡量在现有证据下，最好的合法反馈离 oracle 反馈可能带来的后果最远有多远。只有当相关后果已被确定且能合法达到时，它才为零。
- 它的两个下界把信息缺口（仍被允许的后果集合直径的一半）与可实现性缺口（被允许的后果到合法反馈所能产生之物的距离）分开，两者要求不同的补救：获取证据，或者扩充消息池。
- 接口看不见的不确定性不产生任何代价。

**直觉.**

- 蒸馏器必须在知道哪个目标为真之前就选定一个合法反馈。它的最坏误差，是以合法集合中某点为球心、覆盖证据所允许的全部目标的最小球的半径。若被允许的目标很分散，就没有一个球心能同时靠近它们；若它们集中在合法集合之外的一点，球心就够不着。

**在框架中的作用.**

- 它把获取（§6）、压缩为接口（§4）与合法编译（[Proposition 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14)）放进同一组测试上的同一个目标，而不是把"更多数据"与"更好的反馈"当作互不相干的模块。
- 它的保证 (d) 是逐 checkpoint、加权范数下的界。放进 [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) 的组合保证时，必须先化为逐测试的界，并且仍要支付 [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4) 中的动态项与覆盖项。
- 它以固定且已校准的 $`C`$ 为条件。在未知环境中，$`C`$、终点评价与访问分布也可能随环境变化；这时必须为联合对象构造不确定集合，或者把 $`C`$ 的校准误差加进界里。置信集合是频率意义下的覆盖陈述，不是未经校准的信心；消息池之外的消息也不在它的覆盖范围内。
- 把下一次查询选为"使期望或最坏半径下降最多、扣除成本"是一个方法提案；对一般非线性学习系统，这里没有证明它能高效求解或全局最优。构造 $`C`$、$`\mathcal M`$ 与 $`\mathcal K`$ 需要逐条目的 response 日志与声明的目标，必须为此专门记录。

**推导要点.**

- (a)：$`\Gamma`$ 是关于 $`q`$ 的凸 1-Lipschitz 函数的上确界，由于 $`\mathcal M`$ 紧而有限；连续函数在紧集 $`\mathcal K`$ 上取到最小值。
- (b)：对任意球心 $`q`$，两个目标之间相距至多 $`2\Gamma(q)`$，每个目标离 $`q`$ 至少有它到 $`\mathcal K`$ 的距离。
- (c)：半径为零意味着某个合法点等于每一个被允许的目标。
- (d)：response 差分解为 $`C(\mu_P-\mu_{Q^*})`$ 与残差 $`\mathcal Eh`$；前者在置信事件上不超过 $`\Gamma(q^*)`$，后者不超过 $`B\epsilon_{\rm tail}`$。
- (f)：多胞形上的凸函数在顶点取到最大值；maximum cut 是半正定二次型 $`x^\top Lx/4`$ 在 $`\lbrace-1,1\rbrace^n`$ 上的最大值，$`L`$ 为图的 Laplacian。

**用到.**

- [Definition 4.8](04-08_response_operator_ZH.md#s4-8)：response operator 及其参照测度。
- [Proposition 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14)：合法集合及其特征均值。
- [Proposition 6.15](06-15_linear_gaussian_target_ZH.md#s6-15)：可计算的获取情形。

**被用于.**

- [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16)：组合保证中的编译项。
- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：在获取与扩充消息池之间分配预算。

**状态.** 已证明，以设定中固定且已校准的接口为条件。

## 相关工作

- 这个半径是一个鲁棒（minimax）逼近问题：限制在可行集上的不确定集合的 Chebyshev 中心。两个下界是经典的半径—直径与距离论证；信息缺口背后的两点思想来自 [Le Cam, 1973](#ref-lecam1973convergence)。
- 让单椭球情形精确可解的 S-lemma，综述见 [Pólik & Terlaky, 2007](#ref-polik2007survey)。
- 这里特有的是两者的耦合：不确定性是关于声明测试上的学习后果的，可行集是已核验合法消息池的像，于是同一个数决定了是去获取证据，还是改变消息池。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-polik2007survey"></a>

**Imre Pólik, Tamás Terlaky (2007).** A survey of the S-lemma. *SIAM Review* 49, no. 3, 371–418. DOI [10.1137/S003614450444614X](https://doi.org/10.1137/S003614450444614X).
