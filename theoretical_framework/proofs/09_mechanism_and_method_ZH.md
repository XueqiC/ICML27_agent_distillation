# §9 机制与方法 — 证明

[← 框架总图](../README_ZH.md) · [实验](../experiments/09_mechanism_and_method_ZH.md) · [English](09_mechanism_and_method.md)

**概要.**

- [Proposition 9.1](#s9-1) 是一个带严格零假设（strict null）的机制。固定每个反馈单元（feedback unit）的边缘分布，只改变这些单元怎样组合成联合训练 packet。学习结果对任意重组都不变，当且仅当联合 packet 的 response 是可加的。在同一参数处做一步 SGD，对重组严格不变；连续多步则不然。
- [Algorithm 9.2](#s9-2) 把 §2–§8 推出的方法 Consequence-Calibrated Experience Distillation（CED）写成编号的步骤。每一步都注明支撑它的结论，算法最后列出每项输出带有什么保证。
- [Remark 9.3](#s9-3) 推导算法背后的六条设计规则，每条都是前面结论的推论。
- [Remark 9.4](#s9-4) 区分受控证据覆盖了什么，以及对 Theorem A、B、C 计划中的真实模型检验。
- 用到：[Theorem 3.2](03_learning_consequence_ZH.md#s3-2) 的 response；§2 与 §6 的信息结果；§5 与 §7 的学习器统计量；§8 的 audit 与证书。

## 设定

学习器、feedback law、增量映射（increment map）与 continuation value 同 [Definition 1.3–1.4](../README_ZH.md#s1-3)。Proposition 9.1 的机制设定如下：

- $`\mathcal Y_1=\lbrace 1,\dots,m\rbrace`$ 与 $`\mathcal Y_2=\lbrace 1,\dots,n\rbrace`$ 是有效反馈单元的有限集合，例如填入训练 packet 两个槽位的单元。
- $`\mu\in\Delta(\mathcal Y_1)`$ 与 $`\nu\in\Delta(\mathcal Y_2)`$ 是两个槽位的边缘分布，假设严格为正；若有零分量，下文所有结论在正支撑的乘积上成立。
- 除非另有说明，每一对 $`(i,j)`$ 都是有效的 packet。
- **耦合多面体（coupling polytope）** 为

```math
\mathcal C(\mu,\nu)=\lbrace \Gamma\in\mathbb R^{m\times n}_{\ge0}:\ \Gamma\mathbf 1=\mu,\ \Gamma^\top\mathbf 1=\nu\rbrace .
```

- 在完整学习器状态（complete learner state）$`z`$ 上，$`H_z(i,j)`$ 是：对联合 packet $`(i,j)`$ 执行实际更新、再走完参考尾段之后，最终任务回报的期望。它是 [Theorem 3.2](03_learning_consequence_ZH.md#s3-2) 意义下的一轮边界 response，消息就是这个 packet。
- **耦合宽度（coupling width）** 为

```math
w(H_z)=\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle .
```

它是在两个槽位的边缘分布（因而每个单元的内容）都固定时，仅靠重新安排哪些单元打包在一起，所能造成的期望学习价值的最大变化。

对方法而言，**候选 feedback law**（也称 provider）$`j\in\lbrace 0,1,\dots,M\rbrace`$ 是一条可执行的规则：在完整状态 $`z`$、第 $`t`$ 轮，它发出消息 $`Y\sim P^j_t(\cdot\mid z)`$，即训练单元、packet 或目标。候选 $`0`$ 是参考（例如原始的 teacher 示范）。它的平均 response 和相对参考的带号优势（signed advantage）为

```math
\bar g_t(z,j)=\mathbb E_{Y\sim P^j_t(\cdot\mid z)}\,g_{t,z}(Y),\qquad A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0),
```

其中 $`g_{t,z}`$ 是 [Definition 1.4](../README_ZH.md#s1-4) 的一轮边界 response。

## 一个带严格零假设的机制

<a id="s9-1"></a>

**Proposition 9.1（耦合宽度）.** 在上述设定下：

- **(a) 宽度为零.** $`w(H_z)=0`$ 当且仅当存在 $`a\in\mathbb R^m`$、$`b\in\mathbb R^n`$，使 $`H_z(i,j)=a_i+b_j`$。
- **(b) 用可加残差控制.** 对所有 $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$，

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty .
```

- (b) 中的不等式可以严格成立。
- **(c) 受限配对.** 若禁止某些配对，可行耦合构成 $`\mathcal C(\mu,\nu)`$ 的一个真面，此时可行宽度为零并不推出可加。
- **(d) 严格可加零假设.** 在同一参数处、对所有 packet 的并集、以相同的逐样本权重做一步 SGD，结果与样本怎样分组成 packet 无关。连续两步及以上一般不再不变；对二次损失，差别是 $`\eta^2`$ 阶，具体形式见证明。

*Proof.*

**(a)，"充分性".** 对 $`H=a\mathbf 1^\top+\mathbf 1b^\top`$，

```math
\langle\Gamma,H\rangle=\sum_ia_i\sum_j\Gamma_{ij}+\sum_jb_j\sum_i\Gamma_{ij}=\mu^\top a+\nu^\top b,
```

对每个 $`\Gamma\in\mathcal C(\mu,\nu)`$ 都相同。

**(a)，"必要性".**

- $`\Gamma_0=\mu\nu^\top`$ 属于 $`\mathcal C(\mu,\nu)`$，且严格为正。
- 取行 $`i\ne i'`$、列 $`j\ne j'`$，令 $`E=e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$。$`E`$ 的行和与列和都为零，所以当 $`0<s\le\min_{k,l}(\Gamma_0)_{kl}`$ 时 $`\Gamma_0\pm sE\in\mathcal C(\mu,\nu)`$。
- 宽度为零意味着 $`\langle\Gamma,H\rangle`$ 在 $`\mathcal C(\mu,\nu)`$ 上为常数。于是 $`\langle\Gamma_0+sE,H\rangle=\langle\Gamma_0-sE,H\rangle`$，即 $`\langle E,H\rangle=0`$：

```math
H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}=0\quad\text{对所有矩形成立}.
```

- 令 $`a_i=H_{i1}`$，$`b_j=H_{1j}-H_{11}`$。矩形 $`(i,1,j,1)`$ 给出 $`H_{ij}=H_{i1}+H_{1j}-H_{11}=a_i+b_j`$；$`i=1`$ 或 $`j=1`$ 时这是平凡的。

**(b).** $`\Delta=\Gamma-\Gamma'`$ 的行和与列和都为零，所以对所有 $`a,b`$ 有 $`\langle\Delta,a\mathbf 1^\top+\mathbf 1b^\top\rangle=0`$。又 $`\lVert\Delta\rVert_1\le\lVert\Gamma\rVert_1+\lVert\Gamma'\rVert_1=2`$。因此对每个 $`a,b`$，

```math
\lvert\langle\Delta,H\rangle\rvert=\lvert\langle\Delta,H-a\mathbf 1^\top-\mathbf 1b^\top\rangle\rvert\le\lVert\Delta\rVert_1\,\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty\le2\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty .
```

对 $`(a,b)`$ 取下确界即得。

*严格性.* 令 $`\mu=\nu=(0.9,0.1)`$，$`H=e_{11}`$。

- 每个耦合都形如 $`\Gamma_{11}=x`$，$`\Gamma_{12}=\Gamma_{21}=0.9-x`$，$`\Gamma_{22}=x-0.8`$，其中 $`x\in[0.8,0.9]`$。所以 $`\langle\Gamma,H\rangle=x`$，宽度为 $`0.1`$。
- 每个可加表的矩形和为 $`0`$，而 $`H`$ 的矩形和为 $`1`$。所以残差 $`R=H-a\mathbf 1^\top-\mathbf 1b^\top`$ 的矩形和为 $`1`$，必有某个分量满足 $`\lvert R_{kl}\rvert\ge1/4`$。
- 棋盘形残差 $`R=\tfrac14\begin{pmatrix}1&-1\cr -1&1\end{pmatrix}`$ 取到这个值，对应 $`a=(\tfrac34,\tfrac14)`$，$`b=(0,-\tfrac12)`$。
- 因此 (b) 的右端是 $`2\cdot\tfrac14=0.5>0.1`$。

耦合是一个受限的可行集，不同于 [Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2) 中不受限制的分布族，所以 (b) 只是上界，不是对偶等式。

**(c).** 令 $`\mu=\nu=(\tfrac12,\tfrac12)`$，禁止配对 $`(2,2)`$，并令 $`H=e_{22}`$，即表 $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$。每个可行耦合都有 $`\Gamma_{22}=0`$，所以 $`\langle\Gamma,H\rangle=0`$，可行宽度为 $`0`$。但 $`H`$ 的矩形和为 $`1`$，不可加。

**(d).** 设样本为 $`\lbrace (x_k,w_k)\rbrace _{k\in I}`$，逐样本损失为 $`\ell_k`$；一种分组就是把 $`I`$ 划分成若干 packet。

*在同一参数处走一步.* 更新为

```math
\theta^+=\theta-\eta\sum_{k\in I}w_k\nabla\ell_k(\theta).
```

它只依赖于样本与权重构成的多重集，与划分无关。所以重组不改变任何东西，这是严格的零假设：任何测到的差别都说明实现或随机性有差异。

*连续两步.* 先用 packet $`P_1`$，再用 $`P_2`$。取二次损失 $`\ell_k(\theta)=\tfrac12(x_k^\top\theta-y_k)^2`$，记 $`g_P(\theta)=\sum_{k\in P}w_k\nabla\ell_k(\theta)`$，$`H_P=\sum_{k\in P}w_kx_kx_k^\top`$。则

```math
\theta_1=\theta_0-\eta\,g_{P_1}(\theta_0),\qquad g_{P_2}(\theta_1)=g_{P_2}(\theta_0)-\eta H_{P_2}g_{P_1}(\theta_0),
```

因此

```math
\theta_2=\theta_0-\eta\big(g_{P_1}(\theta_0)+g_{P_2}(\theta_0)\big)+\eta^2H_{P_2}\,g_{P_1}(\theta_0).
```

- 一阶项只依赖于 $`P_1\cup P_2`$。
- 二阶项 $`H_{P_2}g_{P_1}(\theta_0)`$ 依赖于哪些样本落在哪个 packet 里，所以重组一般会改变它。
- 非线性网络和 Adam 会带来更多交互。$`\square`$

*如何理解 Proposition 9.1.*

- **宽度不为零意味着什么.** 在边缘分布固定的前提下重组真实的单元，只能通过联合 packet response $`H_z`$ 的不可加部分改变学习结果。这是学习后果层面的效应，与单个单元的质量、梯度方差或动作层面的代价是两回事。
- **$`H_z`$ 必须是什么.** 它是声明的学习实验下、对整个 packet 的全尾段 response。token 级损失的表格不是学习后果的表格。
- **范围.** 这一结果是耦合上的线性代数，不主张新的对偶。它的作用是把"packet 的组织方式对这个学习器有没有影响"变成一个可测、并带有精确零假设的量。

## 方法

<a id="s9-2"></a>

**Algorithm 9.2（Consequence-Calibrated Experience Distillation，CED）.**

*输入.*

- 学生：初始完整学习器状态 $`z_0=(\theta_0,o_0)`$；声明的训练配方（更新 $`U_t`$、optimizer、horizon $`R`$、参考尾段）；声明的部署任务及得分 $`F_E`$。
- 证据：每个任务族 $`K`$ 条可信示范（demonstration），以及 $`n`$ 次允许的额外实验预算（在学生访问的状态上查询 teacher、环境分支、reset）。
- 候选生成器：产生 feedback law $`P^1,\dots,P^M`$（例如示范的 replay、teacher 在学生状态上的修正、经环境奖励筛选的自生成轨迹、teacher 对示范的改写），以及参考律 $`P^0`$。
- audit 预算：检查点 $`t_1<\dots<t_A`$，每个候选、每个检查点 $`r`$ 个副本（replica），一个 surrogate 目标 $`\tilde F`$，少量用于校准的全尾段任务评测配对 audit，以及置信水平 $`\delta`$。

*输出.*

- 蒸馏后的学生 $`z_R`$。
- 证据台账：把新证据与 replay、改写、合成派生分开记录。
- 每个检查点上：配对替换分数 $`\hat S^{jk}`$ 及其半径、被选中的 feedback law，以及这次选择是否被认证。
- 冻结的选择器，以及分开报告的方法迁移与策略迁移结果。

*步骤.*

1. **记录完整学习器状态.** 每个检查点都保存可恢复的完整状态 $`z_t=(\theta_t,o_t)`$，包括 $`m`$、$`v`$、步数计数器、采样器与随机状态。按 [Remark 5.11](05_learner_statistics_ZH.md#s5-11) 记录 warmth：敏感度加权的 warmth 及其最大项。

   *依据：* feedback law 必须携带的统计量取决于 optimizer state，而不只取决于 $`\theta`$（[Proposition 5.3](05_optimizer_state_ZH.md#s5-3)、[Proposition 5.6–5.7](05_warm_adam_ZH.md#s5-6)、[Theorem 5.10](05_learner_statistics_ZH.md#s5-10)）；coverage 与 transport 都是关于完整状态的陈述（[Theorem 8.2](08_verification_ZH.md#s8-2)）。
2. **先获取，再编译.**
   - 先做一个 pilot，在学生访问的状态上估计学习器相对权重（learner-relative weight）$`W_s`$ 与 teacher 覆盖 $`k_s=KH\,d_T(s)`$。参数坐标按状态分开（state-owned）时用 [Proposition 6.11](06_learner_relative_information_ZH.md#s6-11) 的闭式；否则在状态簇上用分组 CRN 替换分数（[Definition 8.6](08_certificate_ZH.md#s8-6)）测量 $`W_s`$。
   - 剩余的查询按 water-filling 分配：$`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$（[Proposition 6.10](06_learner_relative_information_ZH.md#s6-10)）；一般模型中这就是以 $`c=\sum_tJ_t^*g_t`$ 为目标向量的 c-optimal design（[Theorem 6.5](06_learner_relative_information_ZH.md#s6-5)）。
   - 维护证据台账。只有新证据计入信息预算。

   *依据：* 后处理不能降低信息 regret（[Proposition 2.4](02_feasibility_and_acquisition_ZH.md#s2-4)、[Corollary 6.7](06_learner_relative_information_ZH.md#s6-7)）；被动数据可能留下只有查询才能补上的缺口（[Theorem 2.6](02_feasibility_and_acquisition_ZH.md#s2-6)）；示范与查询共用一份预算，兑换率是学习器相对的（[Proposition 2.11](02_feasibility_and_acquisition_ZH.md#s2-11)、[Corollary 6.8](06_learner_relative_information_ZH.md#s6-8)）；最小误差与 $`\mathfrak n_{K,n}`$ 同阶，由 c-optimal design 达到（[Theorem 6.3 与 6.5](06_learner_relative_information_ZH.md#s6-3)）。
3. **编译候选.** 把证据整理成有限的候选 feedback law 菜单 $`P^0,\dots,P^M`$。每个候选由它实际发出的 packet 确定：单元内容与分组方式。

   *依据：* 分组恰好通过联合 packet response 的不可加部分改变结果（[Proposition 9.1](#s9-1)），所以候选是它的 packet 律，而不是一袋单元。
4. **记录每个候选的学习器统计量，并控制幅度.**
   - 在 $`z_t`$ 上，为每个候选记录在当前 optimizer 下决定其一阶效应的量：平均 optimizer 增量 $`\mathbb E\,\Delta`$（[Theorem 5.10](05_learner_statistics_ZH.md#s5-10)；AdamW 下是 [Proposition 5.5](05_warm_adam_ZH.md#s5-5) 的平均 Adam 增量）、裁剪的激活率（[Proposition 5.9](05_learner_statistics_ZH.md#s5-9)），以及逐坐标的幅度比 $`\lambda`$（相对 running RMS）和从 optimizer state 读出的信噪比 $`\rho`$（[Corollary 7.2](07_second_moment_channel_ZH.md#s7-2)）。
   - 若一个候选在高 $`\rho`$ 坐标上的质量落在价值峰值 $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$ 之外，或落在 Corollary 7.2(c) 的大 $`\lambda`$ 区间，就标记它。候选经缩放或裁剪得到的变体是新的候选，进入第 5 步的菜单。

   *依据：* 平均梯度相同的候选在一阶上就可以不同（[Theorem 5.10](05_learner_statistics_ZH.md#s5-10)）；warm Adam 下，同方向、幅度更大的消息价值可能更低（[Corollary 7.2](07_second_moment_channel_ZH.md#s7-2)）；只看参数的分数和余弦分数在那里会排错或打平（[Proposition 7.3](07_second_moment_channel_ZH.md#s7-3)）；二阶矩通道占主导处一阶分数不可靠（[Remark 7.5](07_second_moment_channel_ZH.md#s7-5)）。所以被标记的候选只能用第 5 步的有限测量来排序。全局范数裁剪给 $`\lambda`$ 设了上限（[Remark 7.4](07_second_moment_channel_ZH.md#s7-4)）。
5. **在实际状态上测量后果.**
   - 在 $`z_t`$ 上用 common random numbers（CRN）抽 $`r`$ 个副本，对所有候选对（包括每个候选对参考）计算配对替换分数 $`\hat S^{jk}`$（[Definition 8.6](08_certificate_ZH.md#s8-6)）及半径 $`e_{\rm stat}^{jk}`$（[Theorem 8.8](08_certificate_ZH.md#s8-8)）。
   - 在少数候选对上，用全尾段任务评测的配对 audit 校准 $`(\alpha,\varepsilon_{\rm sur})`$（[Theorem 8.3](08_verification_ZH.md#s8-3)、[Remark 8.11](08_certificate_ZH.md#s8-11)）。
   - 一阶分数只能作为筛选，且只用于一阶 margin 超过二阶余项（[Corollary 8.9](08_certificate_ZH.md#s8-9)）、第 4 步未标记的候选对。
   - 若在特征上拟合 response 模型，用 [Theorem 8.1](08_verification_ZH.md#s8-1) 的有界系数目标。
6. **选择、训练、重新 audit、冻结、检验迁移.**
   - 选择 $`j_t\in\arg\max_j\big[\hat S^{j0}-e_{\rm stat}^{j0}-\varepsilon_{\rm sur}\big]`$。若没有候选相对参考的下界为正，就保留参考；按点估计做出的选择报告为未认证。
   - 用 $`P^{j_t}`$ 训练到下一个检查点。把选择用到未经 audit 的状态上，需要 [Theorem 8.2](08_verification_ZH.md#s8-2) 的 transport 项与退出项，或者在下一个检查点重新 audit、核对排序是否仍然成立（[Remark 8.11](08_certificate_ZH.md#s8-11)）。
   - 在评测数据之前冻结候选菜单、特征、cell、surrogate、校准与选择器。在留出的任务族、配方与状态上，分别评测方法迁移（在新任务族上、用它自己的 $`K`$ 条示范重跑冻结的流程）与策略迁移（把蒸馏好的学生直接部署到新任务族、不再更新）。

*保证.* 每项保证都是把所引定理用于相应步骤的输出，不需要额外论证。

- **(G1) 单个检查点.** 在 [Assumption 8.7](08_certificate_ZH.md#s8-7) 下，以至少 $`1-\delta`$ 的概率、对 $`z_t`$ 上所有候选对同时成立：若第 6 步因下界为正而选中 $`j_t\ne0`$，则在 $`z_t`$ 上把一轮参考换成 $`P^{j_t}`$、之后走参考尾段，不会降低任务价值（[Theorem 8.8](08_certificate_ZH.md#s8-8)）。
- **(G2) 整次训练.** 若第 5 步的 audit 是实际状态上的全尾段配对 audit，第 6 步按其下置信界选择，则 $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$（[Theorem 8.3](08_verification_ZH.md#s8-3)）。
- **(G3) 信息.** 只对 transcript 做后处理的步骤都不会降低信息 regret（[Proposition 2.4](02_feasibility_and_acquisition_ZH.md#s2-4)）。在 [Theorem 6.5](06_learner_relative_information_ZH.md#s6-5) 的假设下，第 2 步的获取在其中所述的常数因子内达到学习器相对的信息界。
- **(G4) 其余情形为经验结论.** 按点估计做出的选择、迁移未经核实的校准、或在未 audit 的状态上不带 transport 项使用的选择，都不带证书。此时结论是关于冻结方法在声明的任务与 seed 总体上的平均结果，由整次训练的随机对照支持，并标注为经验结论。
- **已知局限.** 用 [Assumption 8.7](08_certificate_ZH.md#s8-7) 的先验区间时，Theorem 8.8 的半径可能超过候选之间的典型差距，这是一个未解决的问题（[Remark 8.12](08_certificate_ZH.md#s8-12)）。在这种情况下第 6 步属于 (G4)。

## 设计规则的推导

<a id="s9-3"></a>

**Remark 9.3（设计规则的推导）.** Algorithm 9.2 实现了六条规则。每条规则先说明：一个声称具有相应保证的方法必须做什么，再给出迫使它成立的结论。这些规则是获得这些保证的必要条件；它们本身并不说明某个 CED 实例优于某个给定的 baseline。

**Rule 1（先获取，再编译）.** transcript 无法确定、但与决策相关的事实，只能由新的允许实验提供；获取多少、在哪里获取，由学习器决定。

- 设 $`D'=C(D)`$ 是 transcript 的任意随机化后处理：replay、改写、转述或合成派生。则 $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$（[Proposition 2.4](02_feasibility_and_acquisition_ZH.md#s2-4)）。由 [Theorem 2.2](02_feasibility_and_acquisition_ZH.md#s2-2) 的分解，编译只能降低算法项。
- 即使 teacher 最优，被动数据也可能留下缺口：在两世界族中最坏 regret 至少为 $`\Delta`$（[Theorem 2.6](02_feasibility_and_acquisition_ZH.md#s2-6)(a)）。更一般地，若两个环境的示范分布相同、学习器相关泛函不同，则对每个 $`K`$，任何蒸馏器（包括 replay）的最坏误差至少是两者差的一半（[Corollary 6.7](06_learner_relative_information_ZH.md#s6-7)）。
- 需要获取的量，在显式模型中由 [Theorem 2.6](02_feasibility_and_acquisition_ZH.md#s2-6)(b)、[Theorem 2.10](02_feasibility_and_acquisition_ZH.md#s2-10) 与 [Proposition 2.11](02_feasibility_and_acquisition_ZH.md#s2-11) 给出下界，一般情形下的下界是 $`\tfrac14\mathfrak n_{K,n}`$（[Theorem 6.3](06_learner_relative_information_ZH.md#s6-3)）。c-optimal design 达到它（[Theorem 6.5](06_learner_relative_information_ZH.md#s6-5)），在状态覆盖模型中就是 water-filling（[Proposition 6.10](06_learner_relative_information_ZH.md#s6-10)）。
- 目标向量 $`c=\sum_tJ_t^*g_t`$ 取决于 optimizer，所以同一批示范可以对一个学习器充分、对另一个不充分（[Corollary 6.12](06_learner_relative_information_ZH.md#s6-12)）。获取必须针对实际使用的学习器来规划。
- 把获取到的事实编译进学生另有代价（[Proposition 2.7](02_feasibility_and_acquisition_ZH.md#s2-7)），与获取分开核算。

**Rule 2（按实测后果选择）.** 在声明的族上，选择规则与学习价值的排序一致，仅当它是平均 response $`\lbrace\bar g_t(z,j)\rbrace_j`$ 的函数，或是某个在该族上已证明对它们充分的统计量的函数。

- *response 是不变量.* 对每一次一轮边界替换加声明尾段，两个消息律可以互换，当且仅当它们在声明类的每个 response 下均值相等（[Theorem 3.2](03_learning_consequence_ZH.md#s3-2)(b)）。在 $`z`$ 上把参考换成候选 $`j`$ 的价值恰为 $`A_t(z,j)`$，一个候选策略的总效应为 $`\sum_t\mathbb E_{d^S_t}A_t`$（[Theorem 8.3](08_verification_ZH.md#s8-3)(b)）。
- *静态价值与平均梯度确定不了排序.* [Proposition 3.1](03_learning_consequence_ZH.md#s3-1)(a) 给出两个数据源：它们的 teacher 价值相同，每个混叠学生的静态策略价值也相同，蒸馏出的学生价值却不同。只依赖这些统计量的规则，对这个实例和它交换标签后的实例做出同样的选择，所以在其中一个上选到较差的源。[Proposition 3.1](03_learning_consequence_ZH.md#s3-1)(b) 给出两个 packet：标签边缘分布与平均梯度都相同，学习价值之差的符号却在两个任务之间相反；只依赖这些统计量的规则至少在一个任务上出错。
- *两类 proxy 不一致.* task-blind 统计量不能对所有任务都排对；只看参数的一阶分数和余弦分数在 [Corollary 7.2](07_second_moment_channel_ZH.md#s7-2) 描述的区域里会排错或打平（[Proposition 7.3](07_second_moment_channel_ZH.md#s7-3)）。对其他静态 proxy 没有证明不一致性；在受控学习器上测试过的静态 proxy 未能可靠地选对（[实验，Check 9.4](../experiments/09_mechanism_and_method_ZH.md#c9-4)）。
- *packet 的组织是候选的一部分.* 在边缘分布固定时重组单元，恰好通过 $`H_z`$ 的不可加部分改变学习价值（[Proposition 9.1](#s9-1)）。所以测量的 response 必须是对实际发出的 packet 的 response，而不是逐单元分数之和。
- *不存在通用的小统计量.* 对不受限制的学习器，任何少于 $`m-1`$ 个统计量的固定族都不能确定全部 response 均值（[Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2)(d)）。小的充分统计量只在受限的族上存在，例如仿射学习器的 $`N`$ 阶以内联合矩（[Theorem 4.4](04_complexity_of_sufficiency_ZH.md#s4-4)）。

**Rule 3（匹配学习器的统计量，而不是原始梯度）.** 候选在一阶上必须匹配的，是完整学习器状态的平均增量在 continuation value 梯度方向上的部分。因此 response 及其统计量要按完整状态及其 warmth 来索引。

- 一阶表示通过完整状态的平均增量写出结果差（[Proposition 5.1](05_optimizer_state_ZH.md#s5-1)）。[Theorem 5.10](05_learner_statistics_ZH.md#s5-10) 对每种学习器算出这个统计量：SGD 是平均梯度；裁剪下是裁剪后的均值，只看平均方向的误差不超过 $`c(1-p_{\rm act})`$（[Proposition 5.9](05_learner_statistics_ZH.md#s5-9)）；cold Adam 一步是符号分布，幅度保留在 $`m`$ 中（[Proposition 5.3](05_optimizer_state_ZH.md#s5-3)）；warm 状态下是平均 Adam 增量，只有在冻结分母近似下才化为 $`(\mathbb EG,\mathbb E\,G\odot G)`$（[Proposition 5.5](05_warm_adam_ZH.md#s5-5)、[Proposition 5.6](05_warm_adam_ZH.md#s5-6)）。
- 在 cold 状态上，均值相同的分布在一阶上就分开（[Proposition 5.7](05_warm_adam_ZH.md#s5-7)）。$`\beta_2`$ 固定时，warmth 收敛到平台 $`(1-\beta_2)/\beta_2`$，而不是趋于零（[Proposition 5.8](05_warm_adam_ZH.md#s5-8)）。所以冻结分母描述每一步都有一个误差下限，warmth 必须测量，不能假定它很小。
- warm Adam 下，消息还通过二阶矩通道起作用（[Theorem 7.1](07_second_moment_channel_ZH.md#s7-1)）；在所述区间内，它的价值关于相对 running RMS 的幅度是凹的，超过一个显式阈值后变负（[Corollary 7.2](07_second_moment_channel_ZH.md#s7-2)）。所以相对于 $`\sqrt v`$ 的幅度也是统计量的一部分。
- 参数相同、optimizer 矩不同的两个状态会走到不同的下一状态，response 也可以不同，所以 transport 条件要在完整状态上陈述（[Theorem 8.2](08_verification_ZH.md#s8-2)）。

**Rule 4（拟合稳定的 response 模型）.** 特征以容差 $`\epsilon_j`$ 测得时，response 模型 $`c+b^\top\phi`$ 按最小化 $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$ 来拟合。

- [Theorem 8.1](08_verification_ZH.md#s8-1) 表明，这个目标就是被认证的未分辨差距的精确值；若 response 估计在 sup 范数下误差不超过 $`\eta`$，被认证的界就是它的值加 $`2\eta`$。对每个固定的 $`b`$，界是方括号在 $`b`$ 处的值，所以取最小的 $`b`$ 给出最好的证书。
- 残差更小、但 $`\sum_j\epsilon_j\lvert b_j\rvert`$ 更大的模型，只要两项之和更大，给出的证书就严格更差。特征是 response 的小倍数时，残差为零的模型也可能给出空洞的证书（[Theorem 8.1](08_verification_ZH.md#s8-1)）。

**Rule 5（计入覆盖；用配对替换做认证；一阶 audit 只用在 margin 占优处）.** 对被选中候选的学习价值给出的保证，取以下四种形式之一：

- (i) 冻结 audit 的证书，含实测的 transport 与退出概率（[Theorem 8.2](08_verification_ZH.md#s8-2)(b)）；
- (ii) 实际状态上、基于全尾段配对 audit 的下置信界保证（[Theorem 8.3](08_verification_ZH.md#s8-3)(c)）；
- (iii) 被 audit 状态上、带校准 surrogate 的 CRN 配对替换证书（[Theorem 8.8](08_certificate_ZH.md#s8-8)），沿训练推广只能靠 transport 项或重新 audit（[Remark 8.11](08_certificate_ZH.md#s8-11)）；
- (iv) 冻结方法的整次训练随机对照，标注为经验结论。

推导：

- 去掉 transport 项是无效的：audit 可以报告零差距，而学习差距为 $`1`$（[Theorem 8.2](08_verification_ZH.md#s8-2)(d)）。
- 实际状态上的全尾段 audit 保证损失不超过 $`\delta`$，但每个候选需要上千个配对副本（[Theorem 8.3](08_verification_ZH.md#s8-3)(c)–(d)）。
- 配对替换分数在期望下精确，不需要 continuation value 梯度，也不需要曲率，半径只含统计项与校准项（[Theorem 8.8](08_certificate_ZH.md#s8-8)）。在 CRN 下，它的方差由两个候选增量之差决定（[Proposition 8.10](08_certificate_ZH.md#s8-10)）。
- 只有当候选对的一阶项超过二阶余项界时，一阶分数才能认证它。所以平均增量沿 surrogate 梯度方向相同的候选对，无论真实差距多大，一阶上都认证不了（[Corollary 8.9](08_certificate_ZH.md#s8-9)）；Theorem 8.8 可以认证这类候选对。
- 按点估计选择、或不带 transport 的外推，都不带 (i)–(iii) 中的任何保证。

**Rule 6（先冻结，再检验迁移）.** 选择器、特征、cell、surrogate 与候选都在评测它们的数据之前冻结；迁移只能通过目标 response 的已核实分解来保证，否则就要测量。

- *冻结.* [Theorem 8.2](08_verification_ZH.md#s8-2)(a) 的 audit 事件是对 audit 之前固定的候选集同时成立的。复用 audit 样本来构造或训练候选会改变候选的分布，证书就需要一次新的独立 audit。在评测数据上调过的选择器同理。
- *迁移.* 若目标族的每个 response 都满足 $`g=c+b^\top\phi+r`$，$`\lVert b\rVert_1\le B`$，$`\lVert r\rVert_\infty\le\tau`$，则 $`\sup_g\lvert(P-Q)g\rvert\le B\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau`$（[Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2)(c)）。反之，若某个目标 response 不在特征张成的空间中，就存在特征匹配、却被它区分的分布（[Theorem 4.2](04_complexity_of_sufficiency_ZH.md#s4-2)(a)），只在训练族上做 audit 无法认证目标族。
- *两种迁移.* 方法迁移评测 $`L_{E'}(S)`$：把冻结的流程 $`S`$ 用到新任务族 $`E'`$、用它自己的 $`K`$ 条示范；它的检验类是 $`E'`$ 经由更新定义的 response 类。策略迁移评测 $`J_{E'}(\pi_z)`$：蒸馏好的学生 $`z`$ 不再更新；它的检验函数是 $`E'`$ 的部署得分。两者是检验类不同的两个泛函，一个的界不能推出另一个的界，所以分开报告。

## 证据的范围

<a id="s9-4"></a>

**Remark 9.4（受控证据覆盖了什么，以及计划中的真实模型检验）.**

- **已覆盖.** Proposition 9.1 由精确有限计算和一个受控小网络检验；选择规则在该网络的冻结留出面板上检验（[实验](../experiments/09_mechanism_and_method_ZH.md)）。Theorem A、B、C 的检验见 [§6](../experiments/06_learner_relative_information_ZH.md)、[§7](../experiments/07_second_moment_channel_ZH.md)、[§8](../experiments/08_verification_ZH.md) 的实验页。这些都是精确有限计算或受控小网络。
- **未覆盖.** 本页的任何陈述都没有真实模型或 benchmark 结果的支持。下面的真实模型检验已经设计好，尚未运行。
- **计划中的真实模型检验**（ALFWorld；用 AdamW 加梯度裁剪训练的 LoRA 学生，主表配方）。探索阶段每个配置单 seed；每个排序都在它所预测的评测之前预先注册。
  - *R0（已有 pilot 的再分析）.* 报告敏感度加权的 warmth 及其最大项（[Remark 5.11](05_learner_statistics_ZH.md#s5-11)），并用 CRN surrogate 尾段测量 **response** 的不可加性，与已测的 update map 不可加性并列。
  - *R1（Theorem C：同状态、同预算的排序）.* 从保存的完整 optimizer state 出发，比较四个候选 feedback law：$`K`$ 条示范的 replay、teacher 在学生状态上的修正（查询）、经环境奖励筛选的自生成轨迹、teacher 对示范的改写。用带校准 surrogate 的 CRN 配对替换分数排序（[Theorem 8.8](08_certificate_ZH.md#s8-8)），预先注册排序，再用每个候选继续训练并评测。对照排序器：LESS、GRACE、梯度范数、静态 proxy。判据是被认证的候选对与实际排序是否一致、秩相关，以及对校准迁移的单独检验。
  - *R2（Theorem A：查询分配）.* 查询预算固定，比较不查询、在学生状态上均匀分配、SGD 式分配，以及用分组 CRN 分数测得 $`W_s`$ 的学习器相对 water-filling（[Proposition 6.10](06_learner_relative_information_ZH.md#s6-10)）。各臂的预测顺序与幅度事先由 Proposition 6.10 的方差公式写下。
  - *R3（Theorem B：二阶矩通道）.* 构造若干消息批：它们的 LESS 余弦分数相同，但在高信噪比坐标上的幅度 $`\lambda`$ 不同。预测：$`\lambda_1+\lambda_2>2\lambda^\star`$ 的批对会被余弦分数和只看参数的分数排错（[Proposition 7.3](07_second_moment_channel_ZH.md#s7-3)）；由 R1 的 CRN 证书测量。
  - baseline 只用完整的官方代码运行。

## 相关工作

- **数据与 teacher 选择.** LESS（Xia et al., 2024）按与目标的梯度余弦相似度给训练数据排序；GRACE（Panigrahi et al., 2025）为蒸馏给 teacher 打分。Algorithm 9.2 则在实际的 optimizer state 上，按实测后果给候选 feedback law 排序；[Proposition 7.3](07_second_moment_channel_ZH.md#s7-3) 指出余弦类分数在哪里不一致。
- **主动获取.** DAgger（Ross et al., 2011）在学生访问的状态上查询 teacher。第 2 步按学习器相对权重分配这些查询，而不是均匀分配（[Proposition 6.10](06_learner_relative_information_ZH.md#s6-10)）。
- **影响与 metagradient.** TracIn（Pruthi et al., 2020）、influence functions（Koh & Liang, 2017）、metagradient（Engstrom et al., 2025）与 In-Run Data Shapley（Wang et al., 2025）都是一阶或 fixed-state 的归因。在 Algorithm 9.2 中，它们只能在 [Corollary 8.9](08_certificate_ZH.md#s8-9) 的条件下充当第 5 步的筛选。
- **packet 的组织.** Proposition 9.1 是标准的耦合线性代数；在这里它的用途是一个带精确零假设的机制检验。
