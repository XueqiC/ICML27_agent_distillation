# §2 可行性与信息获取 — 证明

[← 框架总览](../README_ZH.md) · [实验](../experiments/02_feasibility_and_acquisition_ZH.md) · [English](02_feasibility_and_acquisition.md)

**概要.**

- [Theorem 2.2](#s2-2) 把学生的期望 regret 精确拆成表示项、信息项和算法项，三项都非负。信息项为零，当且仅当所有与数据相容的环境有共同的最优策略。
- [Proposition 2.3](#s2-3) 给出信息项稳健版本的 minimax 对偶。[Proposition 2.4](#s2-4) 说明，任何与环境无关的数据后处理都不能降低信息项：replay、改写和合成只能降低算法项。
- [Theorem 2.6](#s2-6) 给出一个环境族：teacher 在两个世界里都是最优的，它的示范在两个世界里分布相同，但学生的最优策略不同。于是每个被动学习器至少损失 $`\Delta`$。允许 reset 查询时，$`\Delta^{-2}\log(1/\delta)`$ 量级的查询既必要又充分。[Proposition 2.7](#s2-7) 说明，把获取到的事实编译进学生是另一项独立的代价。
- [Theorem 2.10](#s2-10) 在隐藏比特模型中把示范数 $`K`$ 显式写进结论：$`K`$ 条示范之后的信息下限，以及硬查询预算的最优用法。[Proposition 2.11](#s2-11) 说明，示范与自适应查询共用一份信息预算。
- **用到.** [Definitions 1.1–1.2](../README_ZH.md#s1-1) 的设定（环境、学生类与参照类、公共初始信息 $`I_0`$、transcript $`D`$）。
- **被用于.** [Proposition 3.1](03_learning_consequence_ZH.md#s3-1)（那里的差距来自学习器提取了什么）；学习器相对的版本 [Corollary 6.7](06_learner_relative_information_ZH.md#s6-7) 与 [Corollary 6.8](06_learner_relative_information_ZH.md#s6-8)（[Remark 2.12](#s2-12)）；方法中的获取步骤（[Algorithm 9.2](09_mechanism_and_method_ZH.md#s9-2)、[Remark 9.3](09_mechanism_and_method_ZH.md#s9-3)）。

## Regret 分解

<a id="s2-1"></a>

**Definition 2.1（Bayes 值与三项 regret）.** 考虑一个有限决策问题。

- $`\mathcal E`$ 是有限的环境集合，先验为 $`\mu`$。先验以公共初始信息 $`I_0`$ 为条件：预训练模型、已知规则和 $`K`$ 条可信示范。
- $`\Pi\subseteq\Pi_{\rm ref}`$ 是有限策略类，取值 $`J_E(\pi)\in[0,1]`$。$`\Pi`$ 是学生的部署类，$`\Pi_{\rm ref}`$ 是参照类。
- transcript $`D`$ 取值于有限字母表 $`\mathcal D`$，分布为 $`P(D\mid E)`$。学习器可以做的所有实验，无论是否自适应，都包含在 $`D`$ 中。
- 学习器是一个 Markov kernel $`\lambda(\cdot\mid D)\in\Delta(\Pi)`$，输出 $`\widehat\pi\sim\lambda(\cdot\mid D)`$。给定 $`D`$ 时，它的私有随机性与 $`E`$ 条件独立。
- $`\mu(E\mid D)`$ 是后验。

定义

```math
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi),\qquad B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D],
```

以及表示、信息、算法三项 regret

```math
\mathcal R_{\rm rep}=\mathbb E[J_E^{\rm ref}-J_E^\Pi],\qquad \mathcal R_{\rm info}=\mathbb E J_E^\Pi-\mathbb E_D B(D),\qquad \mathcal R_{\rm alg}=\mathbb E_D\big[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\big].
```

策略类有限，所以所有最大值都能取到。$`B(D)`$ 是 Bayes 值：给定 transcript 时，$`\Pi`$ 中的学生能达到的最好期望回报。

<a id="s2-2"></a>

**Theorem 2.2（regret 分解与决策歧义）.** 在 Definition 2.1 下：

**(a) 精确分解.** 相对参照最优的期望 regret 为

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\mathcal R_{\rm rep}+\mathcal R_{\rm info}+\mathcal R_{\rm alg},\qquad \mathcal R_{\rm rep},\ \mathcal R_{\rm info},\ \mathcal R_{\rm alg}\ge0. \qquad(2.1)
```

**(b) 决策歧义.** $`\mathcal R_{\rm info}=0`$，当且仅当对每个满足 $`\Pr(D)>0`$ 的 $`D`$，

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2.2)
```

*Proof.* **(a)** 加减 $`\mathbb EJ_E^\Pi`$ 与 $`\mathbb E_DB(D)`$ 即得恒等式。下面逐项证明非负。

- $`\mathcal R_{\rm rep}\ge0`$：由 $`\Pi\subseteq\Pi_{\rm ref}`$，对每个 $`E`$ 有 $`J_E^\Pi\le J_E^{\rm ref}`$。
- $`\mathcal R_{\rm info}\ge0`$：固定满足 $`\Pr(D)>0`$ 的 $`D`$。逐点有 $`J_E(\pi)\le J_E^\Pi`$，所以对每个 $`\pi\in\Pi`$ 有 $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$。对 $`\pi`$ 取最大得 $`B(D)\le\mathbb E[J_E^\Pi\mid D]`$。对 $`D`$ 取期望并用 tower property，得 $`\mathbb E_DB(D)\le\mathbb EJ_E^\Pi`$。
- $`\mathcal R_{\rm alg}\ge0`$：给定 $`D`$ 时 $`\widehat\pi`$ 与 $`E`$ 条件独立，所以

```math
\mathbb E[J_E(\widehat\pi)\mid D]=\sum_{\pi\in\Pi}\lambda(\pi\mid D)\,\mathbb E[J_E(\pi)\mid D]\le\max_\pi\mathbb E[J_E(\pi)\mid D]=B(D),
```

因为凸组合不会超过其中最大的一项。

**(b)** 由 tower property，

```math
\mathcal R_{\rm info}=\sum_{D:\Pr(D)>0}\Pr(D)\,\big(\mathbb E[J_E^\Pi\mid D]-B(D)\big).
```

由 (a)，每个括号都非负，所以 $`\mathcal R_{\rm info}=0`$ 当且仅当每个括号都为零。固定这样的 $`D`$，取 $`\pi_D\in\arg\max_\pi\mathbb E[J_E(\pi)\mid D]`$，则

```math
\mathbb E[J_E^\Pi\mid D]-B(D)=\sum_{E}\mu(E\mid D)\big(J_E^\Pi-J_E(\pi_D)\big),
```

这是一组非负项之和。

- ($`\Leftarrow`$) 若 $`\pi^\circ`$ 属于交集 (2.2)，则 $`\mathbb E[J_E(\pi^\circ)\mid D]=\mathbb E[J_E^\Pi\mid D]\ge B(D)\ge\mathbb E[J_E(\pi^\circ)\mid D]`$，括号为零。
- ($`\Rightarrow`$) 若括号为零，则每个 $`\mu(E\mid D)>0`$ 的求和项都为零。于是 $`\pi_D`$ 在每个后验概率为正的环境中都最优，属于交集。$`\square`$

<a id="s2-3"></a>

**Proposition 2.3（稳健歧义的对偶）.** 对有限的歧义类 $`C\subseteq\mathcal E`$，令

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]. \qquad(2.3)
```

则

```math
r(C)=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big]. \qquad(2.4)
```

此外：

- **(a)** 若每个允许的实验在所有 $`E\in C`$ 下分布相同，则任何学习器在 $`C`$ 上的最坏期望 regret 至少为 $`r(C)`$。若 $`C`$ 已知，minimax 混合策略可以达到 $`r(C)`$。
- **(b)** $`r(C)=0`$ 当且仅当存在某个 $`\pi\in\Pi`$ 在所有 $`E\in C`$ 中都最优。

*Proof.* 定义 regret 矩阵 $`M_{E,\pi}=J_E^\Pi-J_E(\pi)\in[0,1]`$。对混合策略 $`\rho`$，映射 $`\mu\mapsto\mu^\top M\rho`$ 是线性的，它在 $`\Delta(C)`$ 上的最大值在顶点处取到，所以

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{\mu\in\Delta(C)}\mu^\top M\rho .
```

这是一个有限二人零和博弈的值。由 von Neumann minimax 定理，可以交换 min 与 max：

```math
r(C)=\max_{\mu\in\Delta(C)}\min_{\rho\in\Delta(\Pi)}\mu^\top M\rho=\max_\mu\min_{\pi\in\Pi}(\mu^\top M)_\pi .
```

单纯形上的内层最小值在纯策略处取到，而 $`(\mu^\top M)_\pi=\sum_E\mu_EJ_E^\Pi-\sum_E\mu_EJ_E(\pi)`$，即得 (2.4)。

- **(a)** 若没有实验能区分 $`C`$ 中的成员，则 $`D`$ 的分布在每个 $`E\in C`$ 下都相同，输出分布 $`\bar\rho=\sum_D\Pr(D)\lambda(\cdot\mid D)`$ 也相同。学习器在 $`E`$ 中的期望 regret 是 $`(M\bar\rho)_E`$，所以它在 $`C`$ 上的最坏情况至少为 $`\min_\rho\max_E(M\rho)_E=r(C)`$。若 $`C`$ 已知，输出一个最小化者 $`\rho^\star`$ 即可；$`\Delta(\Pi)`$ 紧且目标连续，所以最小化者存在。
- **(b)** 若 $`\pi^\circ`$ 在每个 $`E\in C`$ 中都最优，则 $`\pi^\circ`$ 处的点质量最坏 regret 为 0。反之，若 $`r(C)=0`$ 由 $`\rho^\star`$ 取到，则对每个 $`E\in C`$ 有 $`\sum_\pi\rho^\star(\pi)M_{E,\pi}=0`$。由于 $`M\ge0`$，$`\rho^\star`$ 支撑中的每个 $`\pi`$ 对所有 $`E\in C`$ 都有 $`M_{E,\pi}=0`$，因此是共同最优。$`\square`$

<a id="s2-4"></a>

**Proposition 2.4（后处理不能降低信息 regret）.** 设 $`D'\sim\kappa(\cdot\mid D)`$ 是随机后处理，kernel $`\kappa`$ 不依赖于 $`E`$，即 $`E\to D\to D'`$ 是 Markov 链。则

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(2.5)
```

即 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。若 $`Z\perp E\mid(D,I_0)`$，则 $`B(D,Z)=B(D)`$：把 $`Z`$ 加进 transcript 不改变 $`\mathcal R_{\rm info}`$。

*Proof.* 令 $`h_\pi(D)=\mathbb E[J_E(\pi)\mid D]`$。由 Markov 性质（$`E\perp D'\mid D`$）与 tower property，

```math
\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D,D']\,\big|\,D'\big]=\mathbb E[h_\pi(D)\mid D'] .
```

条件期望的最大值不超过最大值的条件期望：

```math
\max_\pi\mathbb E[h_\pi(D)\mid D']\le\mathbb E\big[\max_\pi h_\pi(D)\,\big|\,D'\big]=\mathbb E[B(D)\mid D'] .
```

对 $`D'`$ 取期望即得 (2.5)。$`\mathbb EJ_E^\Pi`$ 与 transcript 无关，所以 (2.5) 等价于 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$。对于加入 $`Z`$：$`Z\perp E\mid D`$ 给出 $`\mu(E\mid D,Z)=\mu(E\mid D)`$，因此对每个 $`\pi`$ 有 $`\mathbb E[J_E(\pi)\mid D,Z]=h_\pi(D)`$，从而 $`B(D,Z)=B(D)`$。$`\square`$

**关于 Theorem 2.2 与 Propositions 2.3–2.4 的说明.**

- **每一项分离出什么.**
  - $`\mathcal R_{\rm rep}`$ 是任何信号都消除不了的部分：学生执行不了参照最优策略。
  - $`\mathcal R_{\rm info}`$ 是 transcript 无法确定一个共同最优决策时剩下的部分。
  - $`\mathcal R_{\rm alg}`$ 是学习器没有用好已有信息。因此某一个学习器的失败不构成信息下界；对固定的学习器，额外信息也未必有帮助，因为它只能降低 $`\mathcal R_{\rm info}`$。
- **生成的数据.** 由 Proposition 2.4，对 $`D`$ 的合成改写、转述和 replay 都不能降低 $`\mathcal R_{\rm info}`$。它们可以让已有信息更容易学到，从而降低 $`\mathcal R_{\rm alg}`$。预训练 teacher 内部已有的知识属于 $`I_0`$；新的外部查询才扩大 $`D`$。
- **(2.3) 的概率含义.** 设置信集 $`C_D`$ 以概率 $`1-\delta`$ 包含真实环境，某个混合策略在 $`C_D`$ 上的最坏差距不超过 $`\varepsilon`$。这界定的是该混合策略在这一数据事件上的*期望* regret，而不是单个抽样策略的 regret。
- **等式只在 Bayes 意义下可加.** 对三项分别取最大（例如逐项取最坏情况）不保持 (2.1)。
- **范围.** 有限性保证最大值可取到、条件期望有定义；一般可测空间需要可测选择与可积性条件。这个分解是经典的决策论恒等式，它在这里的作用是把三种误差来源分开，框架后面分别处理它们。

## 被动不可能与主动获取

<a id="s2-5"></a>

**Definition 2.5（两世界混叠族）.**

- 未知符号 $`v\in\lbrace -1,+1\rbrace`$ 选定世界。差距 $`0<\Delta\le1/8`$ 已知。
- 隐藏状态 $`s\in\lbrace 0,1\rbrace`$ 均匀分布。
- 学生只有一个混叠的上下文，看不到 $`s`$，做一次决策：以概率 $`p`$ 选动作 $`A`$，否则选 $`B`$。
- teacher 能看到 $`s`$：在 $`s=0`$ 选 $`A`$，在 $`s=1`$ 选 $`B`$。
- 成功概率：

| 状态 | $`A`$ 的成功概率 | $`B`$ 的成功概率 |
|---|---:|---:|
| $`s=0`$ | $`1`$ | $`\tfrac12-2v\Delta`$ |
| $`s=1`$ | $`\tfrac12+2v\Delta`$ | $`1`$ |

考虑两种接口。

- **被动接口.** 学习器只能看到 teacher 示范（可能附带揭示 $`s`$）以及对它们的随机后处理。
- **查询接口.** 此外，学习器可以 reset 到 $`s=1`$，执行 $`A`$，观察 Bernoulli 结果。查询可以自适应地选择。查询次数 $`N`$ 是满足 $`\mathbb E_vN<\infty`$ 的停时。学习器的算法与私有随机性在两个世界中相同。

记二元 KL 散度为 $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$。

*这一族中的值.* 对 $`s`$ 取平均，

```math
J_v(p)=\tfrac12\big[p+(1-p)(\tfrac12-2v\Delta)\big]+\tfrac12\big[p(\tfrac12+2v\Delta)+(1-p)\big]=\tfrac34+v\Delta(2p-1).
```

- $`J_v`$ 关于 $`p`$ 线性，所以 $`J_v^\Pi=\tfrac34+\Delta`$：$`v=+1`$ 时在 $`p=1`$ 取到，$`v=-1`$ 时在 $`p=0`$ 取到。
- teacher 总是选成功概率为 1 的动作。其余状态–动作的成功概率都等于 $`\tfrac12\pm2\Delta\in[\tfrac14,\tfrac34]`$，因为 $`2\Delta\le\tfrac14`$。所以 teacher 在两个世界中都严格最优。

<a id="s2-6"></a>

**Theorem 2.6（被动不可能，主动可达）.** 在 Definition 2.5 的族中：

- **(a) 被动不可能.** 在被动接口下，每个学习器都满足 $`\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta`$。
- **(b) 主动下界.** 在查询接口下，设 $`0<\delta<1/2`$，并设在每个世界中，学习器以至少 $`1-\delta`$ 的概率输出 regret 不超过 $`\Delta/2`$ 的学生。则对每个 $`v`$，

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(2.6)
```

- **(c) 主动上界.** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ 次独立查询足以在每个世界中以至少 $`1-\delta`$ 的概率输出最优端点 $`p\in\lbrace 0,1\rbrace`$。

*Proof.* **(a)** 一条示范记录 $`(s,\text{动作},\text{结果})`$，在两个世界中都以各 $`\tfrac12`$ 的概率等于 $`(0,A,1)`$ 或 $`(1,B,1)`$。示范的分布与 $`v`$ 无关，对它的任何随机后处理（包括学习器的输出 $`\widehat p`$）的分布也与 $`v`$ 无关。令 $`m=\mathbb E\widehat p`$，两个世界中相同。$`J_v`$ 关于 $`p`$ 是仿射的，所以 $`\mathbb EJ_v(\widehat p)=J_v(m)`$，两个期望 regret 为

```math
J_{+1}^\Pi-J_{+1}(m)=2\Delta(1-m),\qquad J_{-1}^\Pi-J_{-1}(m)=2\Delta m .
```

两者的最大值不小于平均值 $`\Delta`$。$`m=\tfrac12`$ 时取等，例如 $`\widehat p\equiv\tfrac12`$。

**(b)** *第 1 步（化为二元检验）.* 在世界 $`+1`$ 中，regret $`2\Delta(1-\widehat p)\le\Delta/2`$ 恰好等价于 $`\widehat p\ge3/4`$；在世界 $`-1`$ 中，$`2\Delta\widehat p\le\Delta/2`$ 恰好等价于 $`\widehat p\le1/4`$。令 $`A=\lbrace \widehat p>1/2\rbrace`$，则 $`P_{+1}(A)\ge1-\delta`$，$`P_{-1}(A)\le\delta`$。

*第 2 步（停止后 transcript 的 KL）.* 令 $`T`$ 为完整 transcript：示范、查询结果 $`Y_1,\dots,Y_N`$ 和学习器的私有随机性，其分布记为 $`P_v^T`$。

- 示范与私有随机性在两个世界中分布相同。给定过去，每次查询的结果是 Bernoulli $`(\tfrac12+2v\Delta)`$。记 $`d_\Delta=\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)`$。
- 令 $`\ell(y)=\log\frac{P_{+1}(Y=y)}{P_{-1}(Y=y)}`$ 为单次查询的对数似然比。它有界：$`\lvert\ell\rvert\le\log\frac{1/2+2\Delta}{1/2-2\Delta}=:c`$。
- 示范因子与随机化因子相消，transcript 的对数似然比为 $`\Lambda(T)=\sum_{t=1}^{N}\ell(Y_t)`$。
- 令 $`\mathcal F_{t-1}`$ 为第 $`t`$ 次查询之前的历史。事件 $`\lbrace t\le N\rbrace`$ 是 $`\mathcal F_{t-1}`$ 可测的，并且在该事件上 $`\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]=d_\Delta`$。又 $`\sum_t\mathbb E_{+1}\lvert1\lbrace t\le N\rbrace \ell(Y_t)\rvert\le c\,\mathbb E_{+1}N<\infty`$，所以可用 Fubini 定理（Wald 恒等式）：

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})=\mathbb E_{+1}\Lambda(T)=\sum_{t\ge1}\mathbb E_{+1}\big[1\lbrace t\le N\rbrace \,\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]\big]=\mathbb E_{+1}[N]\,d_\Delta .
```

*第 3 步（data processing）.* 事件 $`A`$ 是 $`T`$ 的函数，所以

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})\ge\mathrm{kl}(P_{+1}(A),P_{-1}(A))\ge\mathrm{kl}(1-\delta,\delta).
```

第二个不等号成立，是因为当 $`y\le x`$ 时 $`\mathrm{kl}(x,y)`$ 关于 $`x`$ 不减、关于 $`y`$ 不增；在矩形 $`x\ge1-\delta>\tfrac12>\delta\ge y`$ 上最小值在 $`(1-\delta,\delta)`$ 处取到。第 2、3 步给出 $`v=+1`$ 时 (2.6) 的第一个不等式。$`v=-1`$ 时论证相同，因为 $`\mathrm{kl}(\tfrac12-2\Delta,\tfrac12+2\Delta)=d_\Delta`$。

*第 4 步（常数）.* 对 $`0<u\le\tfrac14`$，

```math
\mathrm{kl}(\tfrac12+u,\tfrac12-u)=2u\log\frac{1+2u}{1-2u}=4u\,\mathrm{artanh}(2u)=8u^2\sum_{j\ge0}\frac{(4u^2)^j}{2j+1}.
```

级数系数为正，关于 $`u`$ 递增；在 $`u=\tfrac14`$ 处它等于 $`2\,\mathrm{artanh}(\tfrac12)=\log3`$。所以 $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le8\log3\cdot u^2\le16u^2`$。取 $`u=2\Delta\le\tfrac14`$ 得 $`d_\Delta\le64\Delta^2`$，即 (2.6) 的第二个不等式。

**(c)** 抽 $`n`$ 个独立的查询结果：若 $`\bar Y>\tfrac12`$ 令 $`\widehat v=+1`$，否则 $`\widehat v=-1`$（分析中平局算作错误）。输出 $`\widehat p=1\lbrace \widehat v=+1\rbrace`$。在世界 $`+1`$ 中 $`\mathbb E\bar Y=\tfrac12+2\Delta`$，单侧 Hoeffding 不等式给出

```math
P_{+1}(\bar Y\le\tfrac12)\le\exp(-2n(2\Delta)^2)=\exp(-8n\Delta^2)\le\delta
```

只要 $`n\ge\log(1/\delta)/(8\Delta^2)`$。世界 $`-1`$ 对称。在成功事件上输出就是最优端点，regret 为 0。$`\square`$

<a id="s2-7"></a>

**Proposition 2.7（编译代价）.** 在 Definition 2.5 的族中，符号估计为 $`\widehat v`$ 之后：

- 若 $`\widehat v=+1`$，取目标 $`q=7/8`$，否则取 $`q=1/8`$；
- 生成 $`m`$ 个 Bernoulli $`(q)`$ 目标标签，令学生为它们的经验频率 $`\widehat p`$。

若 $`m\ge\lceil32\log(2/\delta_L)\rceil`$，则以至少 $`1-\delta_L`$ 的概率有 $`\lvert\widehat p-q\rvert\le1/8`$。若 $`\widehat v`$ 以至少 $`1-\delta`$ 的概率正确（Theorem 2.6(c)），则编译出的学生以至少 $`1-\delta-\delta_L`$ 的概率 regret 不超过 $`\Delta/2`$。

*Proof.* 双侧 Hoeffding 不等式给出

```math
\Pr(\lvert\widehat p-q\rvert>\tfrac18)\le2\exp(-2m/64)=2e^{-m/32}\le\delta_L
```

只要 $`m\ge32\log(2/\delta_L)`$。设 $`\widehat v=v=+1`$ 且 $`\lvert\widehat p-q\rvert\le\tfrac18`$，则 $`\widehat p\ge\tfrac34`$，regret 为 $`2\Delta(1-\widehat p)\le\Delta/2`$。$`v=-1`$ 时对称，此时 $`\widehat p\le\tfrac14`$。对识别失败（$`\le\delta`$）与编译失败（$`\le\delta_L`$）取 union bound 即得所述概率。$`\square`$

**关于 Theorem 2.6 与 Proposition 2.7 的说明.**

- **完美示范也可能信息不足.** teacher 在两个世界中都以概率 1 成功，但对每个被动学习器都有 $`\mathcal R_{\rm info}\ge\Delta`$（Theorem 2.2）。缺失的事实是 $`v`$ 的符号，即使示范揭示了 $`s`$，它也不在示范里：示范展示的是 teacher 做了什么，而不是能力不同的学生应该怎么做。
- **获取与编译是两项独立的代价.**
  - 识别需要 $`\Theta(\Delta^{-2}\log(1/\delta))`$ 次真实环境查询；(2.6) 与 Theorem 2.6(c) 在常数意义下吻合。
  - 编译还需要 $`O(\log(1/\delta_L))`$ 个合成目标标签，与 $`\Delta`$ 无关。这些标签是编译出来的，不是新的 teacher 执行，要单独计数。
- **紧性.** [Check 2.3](../experiments/02_feasibility_and_acquisition_ZH.md#c2-3) 的精确计算表明，调好参数的 SPRT 在部分设定上恰好达到 (2.6)，所以一般而言它的常数不能改进。
- **用到的假设.** $`\Delta`$ 已知用于 (c) 的停止规则。(b) 需要声明的查询接口、与世界无关的算法以及 $`\mathbb E_vN<\infty`$。论证是标准的 change of measure，作用在学生–teacher 的信息错配上；它不是新的检验不等式。

## 示范的数量

<a id="s2-8"></a>

**Definition 2.8（隐藏比特模型）.** 所有概率都以一个固定的、声明过的预训练信息状态 $`I_{\rm pre}`$ 为条件。额外的 teacher 反馈从不计入 $`K`$ 条示范。

- **比特与部署.** 未知比特 $`\Theta=(\Theta_1,\dots,\Theta_d)`$ 相互独立且均匀。上下文 $`j`$ 在部署时以权重 $`w_j\ge0`$ 出现，$`\sum_jw_j=1`$；上下文 $`j`$ 处的正确决策是 $`\Theta_j`$。学生类包含所有"上下文到动作"的表，所以 $`\mathcal R_{\rm rep}=0`$。效用为 $`J_\Theta(\pi)=\sum_jw_j\Pr_\pi(A_j=\Theta_j)`$，知情最优为 1。
- **不泄露的示范.** 第 $`k`$ 条轨迹揭示一个随机子集 $`S_k\subseteq[d]`$ 及其标签 $`\Theta_{S_k}`$。$`S_1,\dots,S_K`$ 独立同分布于 $`L`$，且与 $`\Theta`$ 独立；同一条轨迹内揭示的坐标之间可以相关。transcript 的其余内容的条件分布只依赖于已揭示的标签和独立的随机性。记 $`q_j=\Pr(j\in S)`$。
- **硬查询预算.** 示范之后，学习器至多做 $`n`$ 次无噪声、单位成本的查询，每次揭示一个选定的比特。选择可以依赖于此前观察到的一切。$`n`$ 是每条 transcript 的硬预算，不是期望预算。
- $`U_K`$ 是 $`K`$ 条轨迹都没有揭示的比特集合，$`M_K=\lvert U_K\rvert`$。

<a id="s2-9"></a>

**Lemma 2.9（示范不泄露未揭示比特）.** 在 Definition 2.8 的模型中，给定完整 transcript $`T`$，未揭示的比特 $`(\Theta_j)_{j\in U_K}`$ 独立同分布、均匀，并且与 $`T`$ 以及学习器的私有随机性独立。

*Proof.* 联合分布分解为

```math
\Pr(S_{1:K},\Theta,X)=\prod_kL(S_k)\cdot2^{-d}\cdot\Pr(X\mid S_{1:K},\Theta_{\cup_kS_k}),
```

其中 $`X`$ 是 transcript 的其余内容。固定观察到的 $`(S_{1:K},\Theta_{\cup S_k},X)`$，唯一依赖于 $`\Theta_{U_K}`$ 的因子是常数 $`2^{-d}`$。所以给定 $`T`$ 时 $`\Theta_{U_K}`$ 在 $`\lbrace 0,1\rbrace ^{\lvert U_K\rvert}`$ 上均匀。学习器的随机性与其余一切独立。$`\square`$

<a id="s2-10"></a>

**Theorem 2.10（少样本信息下限与硬预算下的最优查询）.** 在 Definition 2.8 的模型中：

**(a) 信息下限.** 不查询时，$`K`$ 条轨迹之后可达到的最小期望 regret 为

```math
R_{\rm info}(K)=\frac12\sum_{j=1}^dw_j(1-q_j)^K, \qquad(2.7)
```

约定 $`(1-q)^0=1`$。再多一条轨迹的边际价值为

```math
R_{\rm info}(K)-R_{\rm info}(K+1)=\frac12\sum_jw_jq_j(1-q_j)^K. \qquad(2.8)
```

**(b) 最优查询.** 把未揭示比特的权重排序为 $`w_{(1)}\ge\dots\ge w_{(M_K)}`$，则

```math
R_{\rm info}(K,n)=\frac12\,\mathbb E\Big[\sum_{r=n+1}^{M_K}w_{(r)}\Big], \qquad(2.9)
```

并且查询权重最大的 $`n`$ 个未揭示比特是最优的；根据揭示出的值调整查询没有帮助。在均匀情形（$`w_j=1/d`$，每条轨迹揭示一个均匀抽取的上下文）下，

```math
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K. \qquad(2.10)
```

一般有 $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$，且不等号可以严格。

*Proof.* **(a)**

- *下界.* 固定任意学习器。对 $`j\in U_K`$，由 Lemma 2.9，对任何（可能随机的）动作 $`A_j`$ 有 $`\Pr(A_j=\Theta_j\mid T)=\tfrac12`$；对 $`j\notin U_K`$，成功概率至多为 1。所以条件 regret 至少为 $`\tfrac12\sum_{j\in U_K}w_j`$。
- *可达性.* 照抄每个已揭示的标签、其余随机猜的表达到这个下界，且它属于学生类。
- *取期望.* $`S_k`$ 独立同分布，所以 $`\Pr(j\in U_K)=\Pr(j\notin S_1)^K=(1-q_j)^K`$，从而 $`\mathbb E\sum_{j\in U_K}w_j=\sum_jw_j(1-q_j)^K`$，即 (2.7)。
- *边际价值.* 由 $`(1-q_j)^K-(1-q_j)^{K+1}=q_j(1-q_j)^K`$ 得 (2.8)。

**(b)** 以示范 transcript 为条件，令 $`V(U,b)`$ 为未揭示集合为 $`U`$、剩余预算为 $`b`$ 时的最优条件 regret。对 $`b`$ 归纳证明

```math
V(U,b)=\tfrac12\Big(\sum_{j\in U}w_j-\mathrm{top}_b(U)\Big),
```

其中 $`\mathrm{top}_b(U)`$ 是 $`U`$ 中最大的 $`\min(b,\lvert U\rvert)`$ 个权重之和。

- *基础情形.* $`b=0`$ 时就是条件形式的 (a)。
- *归纳步.* $`b\ge1`$ 时，学习器可以停止、查询一个已知比特，或查询某个 $`j\in U`$。
  - 停止得到 $`V(U,0)`$，查询已知比特得到 $`V(U,b-1)`$。由归纳假设，两者都不小于右端，因为 $`\mathrm{top}_b`$ 关于 $`b`$ 单调。
  - 查询 $`j\in U`$ 揭示 $`\Theta_j`$。由 Lemma 2.9，剩余的未揭示比特仍然独立同分布、均匀，且与揭示出的值独立，所以无论观察到什么值，后续问题都是 $`V(U\setminus\lbrace j\rbrace ,b-1)`$；按值分支没有收益。

因此

```math
V(U,b)=\min_{j\in U}\tfrac12\Big(\sum_{U\setminus\lbrace j\rbrace }w-\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\Big)=\tfrac12\Big(\sum_Uw-\max_{j\in U}\big[w_j+\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\big]\Big).
```

最大值等于 $`\mathrm{top}_b(U)`$，取权重最大的 $`j`$ 即可达到。归纳完成，并且贪心规则最优。对示范取 $`V(U_K,n)`$ 的平均即得 (2.9)。

- *均匀情形.* 权重为 $`1/d`$ 时 $`\sum_{r>n}w_{(r)}=(M_K-n)_+/d`$，即 (2.10)；$`n=0`$ 时由 (a) 取 $`q_j=1/d`$ 得 $`R_{\rm info}(K,0)=\tfrac12(1-1/d)^K`$。
- *Jensen.* $`x\mapsto(x-n)_+`$ 是凸函数，所以 $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$。
- *严格性.* 取 $`d=3`$、$`K=3`$、$`n=1`$。则 $`\Pr(M_3=0)=6/27`$，$`\Pr(M_3=1)=18/27`$，$`\Pr(M_3=2)=3/27`$，$`\mathbb EM_3=8/9<1`$。右端为 $`0`$，而 $`R_{\rm info}(3,1)=\Pr(M_3=2)/(2\cdot3)=1/54`$。$`\square`$

<a id="s2-11"></a>

**Proposition 2.11（示范与查询共用一份信息预算）.** 考虑两世界的自适应获取。

- 世界为 $`v\in\lbrace 0,1\rbrace`$，两个世界要求不同的最终决策。
- $`K`$ 条独立同分布的示范中，每条满足 $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})\le\kappa_D^{(v)}<\infty`$。
- 一个共同的、依赖历史的规则选择查询，并在 $`N\le N_{\max}<\infty`$ 时停止。该规则的选择、停止与私有随机化 kernel 都不依赖于 $`v`$。
- 对每个允许的查询 $`x`$ 和相关支撑上的每段之前的历史 $`h`$，$`\mathrm{KL}(P_v(\cdot\mid h,x)\Vert P_{1-v}(\cdot\mid h,x))\le\kappa_X^{(v)}<\infty`$。
- 最终决策在每个世界中出错的概率至多为 $`\delta`$，$`0<\delta<1/2`$。

则

```math
K\kappa_D^{(v)}+\kappa_X^{(v)}\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta). \qquad(2.11)
```

若 $`0<\kappa_X^{(v)}<\infty`$，

```math
\mathbb E_vN\ge\frac{[\mathrm{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}{\kappa_X^{(v)}}. \qquad(2.12)
```

若 $`\kappa_X^{(v)}=0`$ 且 $`K\kappa_D^{(v)}<\mathrm{kl}(1-\delta,\delta)`$，则任何查询预算都达不到所述误差。

*Proof.* *补齐.* 停止后用固定的空符号把 transcript 补齐到长度 $`N_{\max}`$，记为 $`T=(O_1,\dots,O_K,(X_1,Y_1),\dots,(X_N,Y_N))`$，连同停止指示与私有随机性。

*似然比.* 选择、停止与随机化 kernel 在两个世界中相同，所以它们相消：

```math
\log\frac{P_v(T)}{P_{1-v}(T)}=\sum_{k=1}^K\log\frac{P^{\rm demo}_v(O_k)}{P^{\rm demo}_{1-v}(O_k)}+\sum_{t=1}^{N_{\max}}1\lbrace t\le N\rbrace \log\frac{P_v(Y_t\mid H_{t-1},X_t)}{P_{1-v}(Y_t\mid H_{t-1},X_t)}.
```

有限的 KL 上界蕴含相关支撑上的绝对连续性，所以每一项几乎必然有限。

*取 $`\mathbb E_v`$.*

- 第一个和的期望为 $`K\,\mathrm{KL}(P^{\rm demo}_v\Vert P^{\rm demo}_{1-v})\le K\kappa_D^{(v)}`$。
- 第二个和中，$`\lbrace t\le N\rbrace`$ 与 $`X_t`$ 关于 $`(H_{t-1},\text{随机性})`$ 可测。逐项取条件期望，每一项的期望为 $`\mathbb E_v[1\lbrace t\le N\rbrace \,\mathrm{KL}(P_v(\cdot\mid H_{t-1},X_t)\Vert P_{1-v}(\cdot\mid H_{t-1},X_t))]\le\kappa_X^{(v)}\Pr_v(t\le N)`$。
- 由于 $`N_{\max}<\infty`$，和有限。对 $`t`$ 求和得 $`\kappa_X^{(v)}\mathbb E_vN`$。

所以 $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\le K\kappa_D^{(v)}+\kappa_X^{(v)}\mathbb E_vN`$。

*data processing.* 对"最终决策对 $`v`$ 正确"这一指示变量用 data processing。它在 $`v`$ 下的概率 $`x\ge1-\delta`$，在 $`1-v`$ 下的概率 $`y\le\delta`$，所以由二元 kl 在矩形 $`x\ge1-\delta>\tfrac12>\delta\ge y`$ 上的单调性，$`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\ge\mathrm{kl}(x,y)\ge\mathrm{kl}(1-\delta,\delta)`$。两个界合起来即得 (2.11)，移项得 (2.12)。

*$`\kappa_X^{(v)}=0`$ 的情形.* (2.11) 左端等于 $`K\kappa_D^{(v)}`$。若它小于 $`\mathrm{kl}(1-\delta,\delta)`$，则 (2.11) 对任何 $`N`$ 都不成立，这样的算法不存在。$`\square`$

**关于 Theorem 2.10 与 Proposition 2.11 的说明.**

- **replay 的价值为零.** 重新展示已揭示的标签不改变 $`U_K`$，因而不改变 $`R_{\rm info}`$。这是 Proposition 2.4 在这个模型中的特例。replay 仍可能降低学习器自身的误差 $`\mathcal R_{\rm alg}`$。
- **关于 $`K`$ 没有普适速率.** $`q_j=0`$ 的坐标留下不可约的被动误差 $`\tfrac12w_j`$，很小的 $`q_j>0`$ 会让收敛任意慢。任何关于 $`K`$ 的速率都需要对 $`(q_j)`$ 的 coverage 假设。
- **每个假设的作用.**
  - 比特独立使查询之间互不影响。比特相关时，一次查询会提供关于另一个比特的信息，贪心结论随之改变。
  - 硬预算必不可少：期望预算约束允许在不同 transcript 之间调配预算。
  - 不泄露的揭示机制必不可少：若揭示集合依赖于比特本身，它就会泄露信息。
- **(2.11) 的范围.** 它是二元识别的界。策略学习版本需要互不相交的近似最优输出集合。(2.12) 右端为零并不说明可达。
- **是下限，不是实际风险.** $`R_{\rm info}(K)`$ 是 Bayes 意义的信息下限。实际学习器的风险可以更高；超出的部分是 Theorem 2.2 中它的算法项 $`\mathcal R_{\rm alg}`$。

<a id="s2-12"></a>

**Remark 2.12（学习器相对的版本）.** §6 针对蒸馏真正关心的量重新表述本节的结果：一个反馈律对一个声明的学习器造成的一阶后果。

- [Corollary 6.7](06_learner_relative_information_ZH.md#s6-7) 推广了 Theorem 2.6(a) 与 Proposition 2.4。若两个环境下示范的分布相同，而它们在学习器相关泛函 $`\Psi`$ 上不同，则任何蒸馏器（包括 replay、改写与合成）的最坏误差至少为 $`\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$，对每个 $`K`$ 都成立。
- [Corollary 6.8](06_learner_relative_information_ZH.md#s6-8) 推广了 Proposition 2.11。渐近地，对每个方向 $`h`$ 有 $`K\,e_D(h)+n\,e_Q(h)\ge(c^\top h)^2/(16\varepsilon^2)`$，其中 $`c`$ 由学习器决定。所以示范替代查询的比率取决于学习器，而不只取决于数据。
- 对象上的区别：本节度量学生策略的决策 regret；§6 度量反馈律相对于学习器更新的 fidelity。两者都说明，缺失的信息只能由新的实验补上。

## 相关工作

- Theorem 2.2 的分解与 Proposition 2.3 的对偶属于经典决策论（Bayes 风险与 von Neumann minimax 定理）。Proposition 2.4 是 Bayes 风险的 data processing 性质。
- Theorem 2.6(b) 与 Proposition 2.11 用的是自适应采样的标准 change of measure 论证（Wald 恒等式与 KL 的 data processing 不等式），与序贯检验、best-arm identification 中的下界相同。SPRT 是 Wald 的序贯概率比检验。
- 本节的新内容在于设定：学生–teacher 信息错配下完美示范也不充分；获取与编译的分离；以及对示范数 $`K`$ 的显式依赖。
