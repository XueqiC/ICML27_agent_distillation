# §2 可行性与信息获取 — 精确陈述

[← Framework](../README_ZH.md) · [Proofs](../proofs/02_feasibility_and_acquisition_ZH.md) · [Experiments](../experiments/02_feasibility_and_acquisition_ZH.md) · [English](02_feasibility_and_acquisition.md)

### §2 · 可行性与信息获取：精确陈述

[证明](../proofs/02_feasibility_and_acquisition_ZH.md) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md)

**本节做什么.**

- 它问的是：现有数据能否确定学生该怎么做，并把这个问题与学生学得好不好分开（[Theorem 2.2](../proofs/02_feasibility_and_acquisition_ZH.md#s2-2)）。
- 它说明任何对现有数据的加工都补不上缺失的事实（[Proposition 2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4)），而即使是最优 teacher 的示范也可能留下这样的缺口（[Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)）。
- 它给缺失信息定价：主动查询需要 $`\Delta^{-2}\log(1/\delta)`$ 量级，编译另算（[Proposition 2.7](../proofs/02_feasibility_and_acquisition_ZH.md#s2-7)）。
- 它把示范数 $`K`$ 显式写进结论，并说明示范与查询共用一份信息预算（[Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10)、[Proposition 2.11](../proofs/02_feasibility_and_acquisition_ZH.md#s2-11)）。

**设定（[Definition 2.1](../proofs/02_feasibility_and_acquisition_ZH.md#s2-1)）.** 有限的环境集合 $`\mathcal E`$ 带有以公共初始信息 $`I_0`$ 为条件的先验。学生类 $`\Pi`$ 包含在参照类 $`\Pi_{\rm ref}`$ 中，取值 $`J_E(\pi)\in[0,1]`$。transcript $`D`$ 包含学习器可以做的所有实验，学习器输出 $`\widehat\pi\sim\lambda(\cdot\mid D)`$。记 $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$，$`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$，并令 $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ 为 Bayes 值。

#### Theorem 2.2 · regret 分解与决策歧义

**(a)** 对任何学习器，

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\underbrace{\mathbb E[J_E^{\rm ref}-J_E^\Pi]}_{\mathcal R_{\rm rep}}+\underbrace{\mathbb E J_E^\Pi-\mathbb E_D B(D)}_{\mathcal R_{\rm info}}+\underbrace{\mathbb E_D\big[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\big]}_{\mathcal R_{\rm alg}},
```

且三项都非负。

**(b)** $`\mathcal R_{\rm info}=0`$，当且仅当对每条概率为正的 transcript $`D`$，后验概率为正的环境有共同的最优策略：$`\bigcap_{E:\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing`$。

*推导要点.*

- 加减 $`\mathbb EJ_E^\Pi`$ 与 $`\mathbb E_DB(D)`$ 即得恒等式。
- 表示项非负，因为 $`\Pi\subseteq\Pi_{\rm ref}`$。信息项非负，因为对每个策略都有 $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$。算法项非负，因为给定 $`D`$ 时学习器的输出是策略的凸组合，不会超过其中最好的那个。
- 对 (b)，$`\mathcal R_{\rm info}`$ 是对 transcript 求和的一组非负括号 $`\sum_E\mu(E\mid D)(J_E^\Pi-J_E(\pi_D))`$；一个括号为零，恰好当后验最优策略在每个后验概率为正的环境中都最优。

*用到* [Definition 2.1](../proofs/02_feasibility_and_acquisition_ZH.md#s2-1)。*被用于* [Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)、[Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10)、[Proposition 3.1](../proofs/03_learning_consequence_ZH.md#s3-1)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/02_feasibility_and_acquisition_ZH.md#s2-2) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md#c2-1) · 已证明

#### Proposition 2.3 · 稳健歧义的对偶

对有限的环境类 $`C`$，

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big].
```

若没有允许的实验能区分 $`C`$ 中的成员，则每个学习器在 $`C`$ 上的最坏期望 regret 至少为 $`r(C)`$，且 minimax 混合策略能达到它；$`r(C)=0`$ 当且仅当有某个策略在每个 $`E\in C`$ 中都最优。

*推导要点.*

- regret 矩阵 $`M_{E,\pi}=J_E^\Pi-J_E(\pi)`$ 定义了一个有限零和博弈：学习器选策略的混合，对手选 $`C`$ 上的先验。
- 由 von Neumann minimax 定理交换 min 与 max；内层最小值在纯策略处取到，即得对偶形式。
- 当实验分不清 $`C`$ 中的成员时，学习器的输出分布在它们中都相同，所以其最坏 regret 至少等于博弈的值。

*例示.* 两个环境偏好相反的策略，regret 矩阵为 $`\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}`$。minimax 混合是 $`(\tfrac12,\tfrac12)`$，最不利先验也是 $`(\tfrac12,\tfrac12)`$，两边都等于 $`0.1`$。这是混合平均的 regret：单个抽样策略的 regret 是 0 或 0.2。

*用到* [Definition 2.1](../proofs/02_feasibility_and_acquisition_ZH.md#s2-1)。*被用于* [Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)（一个 $`r(C)=\Delta`$ 的实例）。

[证明](../proofs/02_feasibility_and_acquisition_ZH.md#s2-3) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md#c2-2) · 已证明

#### Proposition 2.4 · 后处理不能降低信息 regret

若 $`D'\sim\kappa(\cdot\mid D)`$，且 kernel $`\kappa`$ 不依赖于 $`E`$，则 $`\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D]`$，即 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。往 $`D`$ 中加入满足 $`Z\perp E\mid(D,I_0)`$ 的变量 $`Z`$，不改变 $`B`$ 与 $`\mathcal R_{\rm info}`$。

*推导要点.*

- $`E\to D\to D'`$ 是 Markov 链，所以 $`\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D]\,\big|\,D'\big]`$。
- 条件期望的最大值不超过最大值的条件期望，所以 $`D'`$ 下的 Bayes 值不超过 $`D`$ 下的。

*例示.* 用与环境无关的流程对 $`K`$ 条示范做 replay、转述或改写，就是这样的 $`\kappa`$。它可以让学生更好地拟合已有信息（降低 $`\mathcal R_{\rm alg}`$），但不能降低 $`\mathcal R_{\rm info}`$。

*用到* [Definition 2.1](../proofs/02_feasibility_and_acquisition_ZH.md#s2-1)。*被用于* [Theorem 2.6(a)](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)、[Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10)（replay 的价值为零）、[Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md#c2-1) · 已证明

#### Theorem 2.6 · 被动不可能，主动可达

**设定（[Definition 2.5](../proofs/02_feasibility_and_acquisition_ZH.md#s2-5)）.** 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 选定两个世界之一，差距 $`0<\Delta\le1/8`$ 已知。均匀的隐藏状态 $`s\in\lbrace0,1\rbrace`$ 对 teacher 可见、对混叠的学生不可见，学生以概率 $`p`$ 选动作 $`A`$。teacher 在每个状态都选成功概率为 1 的动作，所以在两个世界中都最优；学生的值为 $`J_v(p)=\tfrac34+v\Delta(2p-1)`$，最优动作取决于 $`v`$。在查询接口下，学习器可以自适应地 reset 到 $`s=1`$、执行 $`A`$ 并观察结果，查询次数是停时 $`N`$。

- **(a)** 只看示范（及其后处理）的学习器都满足 $`\max_v\mathbb E[J_v^\Pi-J_v(\widehat p)]\ge\Delta`$。
- **(b)** 若在每个世界中，学习器以至少 $`1-\delta`$ 的概率输出 regret 不超过 $`\Delta/2`$ 的学生，则

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}.
```

- **(c)** $`\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ 次查询就够了。

*推导要点.*

- (a) 在两个世界中，一条示范都以各一半的概率是"在 $`s=0`$ 选 $`A`$ 并成功"或"在 $`s=1`$ 选 $`B`$ 并成功"，所以示范的分布、进而学习器输出的分布都与 $`v`$ 无关。值关于 $`p`$ 是仿射的；两个 regret 分别为 $`2\Delta(1-m)`$ 与 $`2\Delta m`$（$`m`$ 是平均输出），最大值不小于平均值 $`\Delta`$。
- (b) 把"在两个世界中都 $`\Delta/2`$ 最优"化为对事件 $`\lbrace\widehat p>1/2\rbrace`$ 的二元检验。对停时用 Wald 恒等式，整条 transcript 的 KL 散度等于期望查询次数乘以单次查询的 KL；data processing 使它至少为 $`\mathrm{kl}(1-\delta,\delta)`$。常数 64 来自 $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le16u^2`$。
- (c) 对独立的查询结果取平均，用单侧 Hoeffding 不等式。

*用到* [Theorem 2.2](../proofs/02_feasibility_and_acquisition_ZH.md#s2-2)、[Propositions 2.3–2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-3)。*被用于* [Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10)、[Proposition 2.11](../proofs/02_feasibility_and_acquisition_ZH.md#s2-11)（change of measure 技术）、[Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7)、[Remark 9.3](../proofs/09_mechanism_and_method_ZH.md#s9-3)。

[证明](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md#c2-3) · 已证明

#### Proposition 2.7 · 编译代价

估计出符号之后，从目标 $`q\in\lbrace\tfrac18,\tfrac78\rbrace`$ 生成 $`m\ge\lceil32\log(2/\delta_L)\rceil`$ 个标签并拟合其频率，则以至少 $`1-\delta_L`$ 的概率有 $`\lvert\widehat p-q\rvert\le\tfrac18`$。编译出的学生以至少 $`1-\delta-\delta_L`$ 的概率 regret 不超过 $`\Delta/2`$。

*推导要点.* 编译标签的经验频率由双侧 Hoeffding 不等式控制，再对识别失败与编译失败取 union bound。标签数与 $`\Delta`$ 无关。

*例示.* 识别消耗的是真实的环境查询，量级 $`\Delta^{-2}\log(1/\delta)`$；编译消耗的是合成标签，量级 $`\log(1/\delta_L)`$。$`\Delta`$ 小时，预算几乎全部花在获取上，两者必须分开记账。

*用到* [Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)。*被用于* [Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/02_feasibility_and_acquisition_ZH.md#s2-7) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md#c2-3) · 已证明

#### Lemma 2.9 与 Theorem 2.10 · 隐藏比特、信息下限与最优查询

**设定（[Definition 2.8](../proofs/02_feasibility_and_acquisition_ZH.md#s2-8)）.** 比特 $`\Theta_1,\dots,\Theta_d`$ 独立且均匀；上下文 $`j`$ 的部署权重为 $`w_j`$，正确决策为 $`\Theta_j`$。$`K`$ 条示范中的每一条揭示一个随机的比特子集，子集的抽取与比特独立，比特 $`j`$ 被揭示的概率为 $`q_j`$。硬预算允许 $`n`$ 次查询，每次揭示一个选定的比特。$`U_K`$ 是没有被任何示范揭示的比特集合，$`M_K=\lvert U_K\rvert`$。

**[Lemma 2.9](../proofs/02_feasibility_and_acquisition_ZH.md#s2-9).** 给定完整 transcript，未揭示的比特仍然独立、均匀，并且与 transcript 独立。

**[Theorem 2.10](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10).**

- **(a)** 不查询时，$`K`$ 条示范之后可达到的最小 regret 为

```math
R_{\rm info}(K)=\tfrac12\textstyle\sum_jw_j(1-q_j)^K .
```

- **(b)** 有 $`n`$ 次查询时，$`R_{\rm info}(K,n)=\tfrac12\,\mathbb E\big[\sum_{r>n}w_{(r)}\big]`$，对未见比特的排序权重求和；先查权重最大的未见比特是最优的，按值自适应地查询没有收益。一般而言 $`\mathbb E[(M_K-n)_+]\ne(\mathbb EM_K-n)_+`$。

*推导要点.*

- *Lemma 2.9.* 联合分布中唯一依赖于未揭示比特的因子是常数 $`2^{-d}`$。
- *(a)* 任何动作猜对一个未揭示比特的概率都是一半，已揭示的比特可以照抄，所以条件 regret 是未见比特权重的一半。比特 $`j`$ 在 $`K`$ 条独立示范中都没被揭示的概率是 $`(1-q_j)^K`$。
- *(b)* 对剩余预算归纳。一次查询之后，由 Lemma 2.9，剩下的问题与查到的值无关，所以按值分支没有收益，每一步贪心地查权重最大的未见比特就是最优的。$`x\mapsto(x-n)_+`$ 是凸函数，由 Jensen 不等式，两个表达式有大小关系，在小例子中是严格的。

*例示.* $`d=8`$ 个等权上下文、每条示范均匀地揭示其中一个时，下限为 $`\tfrac12(7/8)^K`$：一个事实以 $`(7/8)^K`$ 的概率一直躲过示范；一次定向查询必定消除一个未见事实，而多一条示范只以 $`1/8`$ 的概率消除每个未见事实。

*用到* [Theorem 2.2](../proofs/02_feasibility_and_acquisition_ZH.md#s2-2)、[Proposition 2.4](../proofs/02_feasibility_and_acquisition_ZH.md#s2-4)。*被用于* [Proposition 6.10](../proofs/06_learner_relative_information_ZH.md#s6-10)（学习器相对的 coverage 模型）、[Algorithm 9.2](../proofs/09_mechanism_and_method_ZH.md#s9-2)。

[证明](../proofs/02_feasibility_and_acquisition_ZH.md#s2-10) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md#c2-4) · 已证明

#### Proposition 2.11 · 示范与查询共用一份信息预算

在两世界的自适应获取中，设 $`K`$ 条独立同分布的示范每条 KL 信息至多 $`\kappa_D`$，给定任何历史时每次查询的 KL 信息至多 $`\kappa_X`$，查询–停止规则与世界无关且 $`N\le N_{\max}`$，最终决策在每个世界中出错的概率至多 $`\delta<\tfrac12`$。则

```math
K\kappa_D+\kappa_X\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta).
```

若 $`\kappa_X=0`$ 且 $`K\kappa_D<\mathrm{kl}(1-\delta,\delta)`$，则任何查询预算都不够。

*推导要点.* 查询选择、停止与随机化 kernel 在两个世界中相同，在似然比中相消，只剩示范项与查询结果项。前者的期望至多为 $`K\kappa_D`$，后者至多为 $`\kappa_X\,\mathbb EN`$；data processing 不等式要求两者之和至少为 $`\mathrm{kl}(1-\delta,\delta)`$。

*用到* [Theorem 2.6](../proofs/02_feasibility_and_acquisition_ZH.md#s2-6)（技术）。*被用于* [Corollary 6.8](../proofs/06_learner_relative_information_ZH.md#s6-8)。

[证明](../proofs/02_feasibility_and_acquisition_ZH.md#s2-11) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md#c2-5) · 已证明

#### Remark 2.12 · 学习器相对的版本

[Corollary 6.7](../proofs/06_learner_relative_information_ZH.md#s6-7) 推广了 Theorem 2.6(a) 与 Proposition 2.4：若示范分不清两个在学习器相关泛函 $`\Psi`$ 上不同的环境，则任何蒸馏器（包括 replay、改写与合成）都不能做得比差值的一半更好，对任何 $`K`$ 都如此。[Corollary 6.8](../proofs/06_learner_relative_information_ZH.md#s6-8) 把 Proposition 2.11 推广成逐方向的预算，其中方向由学习器加权。对象从策略的决策 regret 变成了反馈律相对于学习器更新的 fidelity。

[证明](../proofs/02_feasibility_and_acquisition_ZH.md#s2-12)
