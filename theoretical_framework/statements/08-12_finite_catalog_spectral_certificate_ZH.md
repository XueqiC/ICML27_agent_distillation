# Proposition 8.12 · 有限目录的谱证书

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](08-12_finite_catalog_spectral_certificate_ZH.md) · [证明](../proofs/08-12_finite_catalog_spectral_certificate_ZH.md) · [实验](../experiments/08-12_finite_catalog_spectral_certificate_ZH.md) · [English](08-12_finite_catalog_spectral_certificate.md)

**一段话概括.** [Definition 4.8](04-08_response_operator_ZH.md#s4-8) 的 response operator 是总体层面的对象。实际做法是在一个 checkpoint 上冻结一份有限目录：若干测试、若干候选 feedback law，每格跑几次，然后用估计出来的矩阵。这个命题说明这样一份目录能认证什么：逐格的 Hoeffding 半径变成加权、中心化矩阵上同样大小的算子范数半径；每个奇异值的移动不超过这个半径；截断接口的残差有界，而且不需要任何谱间隙。它也确定了成本的计数单位：续训次数等于候选数 × 复现数 × *不同的续训协议数*，而不是乘以测试数。在负担得起的复现数下，半径远大于 response 的量级，所以这个证书成立，但在实测量级下是空的。

**本页用到的设定与记号.**

全文使用 [Definitions 1.3–1.4](../README_ZH.md#s1-3) 中的对象。

- 固定一个 checkpoint $`(t,z)`$：完整的 learner state，包括 optimizer state。
- *测试.* 声明的 $`p`$ 个测试 $`i=1,\dots,p`$，权重 $`w_i>0`$，$`\sum_iw_i=1`$。每个测试是一个声明的任务分数加一个声明的续训协议。共用同一续训协议的测试从同一次运行中读出；$`p_{\rm train}\le p`$ 表示这些测试中不同续训协议的个数。
- *候选.* 冻结的 $`m`$ 个候选 feedback law $`j=1,\dots,m`$，权重 $`q_j>0`$，$`\sum_jq_j=1`$。候选可以是一个可执行的反馈提供者；这时目录压缩的是提供者的混合权重，而不是它们内部的消息。
- *格均值.* $`G_{ij}\in[0,1]`$ 是：在 $`z`$ 处用候选 $`j`$ 走一轮，再接测试 $`i`$ 的续训之后，测试 $`i`$ 的期望最终分数。这就是 [Definition 4.8](04-08_response_operator_ZH.md#s4-8) 的 response，以候选为消息、$`\rho=q`$。
- *加权、中心化矩阵.* 记 $`D_a=\mathrm{diag}(a)`$，$`\sqrt q`$ 逐元素取，$`\Pi_q=I-\sqrt q\sqrt q^{\top}`$，

```math
M=D_{\sqrt w}\,G\,D_{\sqrt q}\,\Pi_q .
```

- 对候选上的一个分布 $`P`$（权重为 $`P_j`$ 的混合）与另一个分布 $`Q`$，密度差为 $`h_j=(P_j-Q_j)/q_j`$，$`x=D_{\sqrt q}h`$。于是 $`\sqrt q^{\top}x=0`$，$`\lVert x\rVert_2=\lVert h\rVert_q`$，并且 $`(Mx)_i=\sqrt{w_i}\,a_i`$，其中 $`a_i=\sum_j(P_j-Q_j)G_{ij}`$ 是测试 $`i`$ 上的分数差。所以 $`\lVert Mx\rVert_2=\lVert a\rVert_\omega`$，其中 $`\lVert a\rVert_\omega^2=\sum_iw_ia_i^2`$：矩阵 $`M`$ 就是 response operator 在正交坐标下的表示。它的奇异值为 $`\sigma_1\ge\sigma_2\ge\dots`$，且 $`\lVert M\rVert_{\rm HS}^2\le\tfrac14`$。
- *复现.* 每格跑 $`n`$ 次，$`\hat G_{ij}`$ 是格内平均。复现之间相互独立；同一次复现内，候选之间可以共享随机数，从同一次运行读出的测试之间也可以相关。$`\hat M`$ 由 $`\hat G`$ 按同样方式构造；$`\hat\sigma_j`$、$`\hat v_j`$ 是它的奇异值与右奇异向量，$`\hat\Pi_r`$ 投影到 $`\hat v_1,\dots,\hat v_r`$ 上。
- *估计的行系数.* 对测试 $`i`$，令 $`\hat g_i`$ 为 $`\hat G`$ 的第 $`i`$ 行，看作候选上的函数并在 $`q`$ 下中心化；$`\hat\phi_k=D_{\sqrt q}^{-1}\hat v_k`$ 是第 $`k`$ 个估计模式在候选上的函数；$`\hat b_k(i)=\langle\hat g_i,\hat\phi_k\rangle_q`$，$`\hat e_r(i)=\lVert\hat g_i-\sum_{k\le r}\hat b_k(i)\hat\phi_k\rVert_q`$。$`P`$ 与 $`Q`$ 的模式均值差为 $`\alpha_k=\sum_j(P_j-Q_j)\hat\phi_k(j)`$。

<a id="s8-12"></a>

**Proposition 8.12（有限目录的谱证书）.** 在上述设定下，令

```math
\varepsilon_n=\sqrt{\frac{\log(2pm/\delta)}{2n}} .
```

**(a) 逐格半径.** 以至少 $`1-\delta`$ 的概率，所有格同时满足 $`\lvert\hat G_{ij}-G_{ij}\rvert\le\varepsilon_n`$。

**(b) 算子半径与谱.** 在 (a) 的事件上，

```math
\lVert\hat M-M\rVert_{\rm op}\le\varepsilon_n,\qquad\lvert\hat\sigma_k-\sigma_k\rvert\le\varepsilon_n\quad(\forall k) .
```

**(c) 无需间隙的截断接口.** 在同一事件上，对每个 $`r`$，

```math
\lVert M(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}+\varepsilon_n\le\sigma_{r+1}+2\varepsilon_n .
```

**(d) 可逐测试检查的界.** 在同一事件上，对候选上任意一对满足 $`\lVert h\rVert_q\le B`$ 的分布 $`P,Q`$ 与每个测试 $`i`$，

```math
\Big\lvert\sum_j(P_j-Q_j)G_{ij}\Big\rvert\le\Big\lvert\sum_{k\le r}\hat b_k(i)\,\alpha_k\Big\rvert+B\,\hat e_r(i)+2\varepsilon_n .
```

如果只有估计值 $`\hat\alpha_k`$ 及同时成立的误差 $`\lvert\hat\alpha_k-\alpha_k\rvert\le\eta_k`$，就把 $`\alpha_k`$ 换成 $`\hat\alpha_k`$，并加上 $`\sum_{k\le r}\lvert\hat b_k(i)\rvert\eta_k`$。

**(e) 模式冻结后的截断.** 设 $`M`$ 精确的奇异三元组 $`(\sigma_k,u_k,v_k)`$ 已知。记 $`\phi_k=D_{\sqrt q}^{-1}v_k`$、$`\alpha_k=\mathbb E_P\phi_k-\mathbb E_Q\phi_k`$，令 $`\hat\alpha_k`$ 为分别从 $`P`$ 与 $`Q`$ 独立抽取 $`n'`$ 次（候选下标）后 $`\phi_k`$ 样本均值之差。令 $`f=Mx=\sum_k\sigma_k\alpha_ku_k`$ 为真实的加权 response 差，$`\hat f_r=\sum_{k\le r}\sigma_k\hat\alpha_ku_k`$。则

```math
\mathbb E\,\lVert\hat f_r-f\rVert_2^2=\sum_{k\le r}\frac{\sigma_k^2}{n'}\big(\mathrm{Var}_P\phi_k+\mathrm{Var}_Q\phi_k\big)+\sum_{k>r}\sigma_k^2\alpha_k^2 .
```

若改用由一个与抽样独立、已冻结的估计 $`\hat M`$ 得到的三元组 $`(\hat\sigma_k,\hat u_k,\hat v_k)`$，并令 $`\hat f_r=\sum_{k\le r}\hat\sigma_k\hat\alpha_k\hat u_k`$（$`\hat\alpha_k`$ 为 $`\hat\phi_k=D_{\sqrt q}^{-1}\hat v_k`$ 的样本均值之差），方差项是加帽后的同一表达式，偏差项则变成 $`\lVert(M-\hat M\hat\Pi_r)x\rVert_2^2`$，而不是 $`\sum_{k>r}\sigma_k^2\alpha_k^2`$。

**(f) 成本单位.** $`m`$ 个候选、$`n`$ 次复现的完整目录需要 $`m\,n\,p_{\rm train}`$ 次续训，外加 $`p`$ 个测试的评估。只有当每个测试都需要自己的续训时，才是 $`mnp`$。

**说了什么.**

- 冻结一份有限目录，就能以一致、无需间隙的保证估计 response operator：逐格误差变成同样大小的算子范数误差，截断接口保留残差界。
- 这个界可以逐个测试检查；用估计格代替真实格的代价是明确的 $`2\varepsilon_n`$。
- 续训次数由不同的续训协议数决定，不由每次运行读出多少个指标决定。

**直觉.** 权重 $`w`$ 与 $`q`$ 都归一，所以误差矩阵的加权 Frobenius 范数不超过最大的单格误差；算子范数又不超过 Frobenius 范数。只有当你想把某一根估计出来的奇异向量解释成一个稳定的"机制"时，才需要谱间隙；残差界用不到它。

**在框架中的作用.**

- 它是 [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 在单个 checkpoint 上的有限样本版本，也是 [Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) 跨 checkpoint 复用的锚点估计。
- 它的半径 $`\varepsilon_n`$ 是 [Proposition 8.15](08-15_two_fidelity_response_estimation_ZH.md#s8-15) 想用更低成本达到的量，也是 [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) 中的锚点误差 $`\epsilon_j`$。
- 在 $`p=10`$ 个测试、$`m=20`$ 个候选、$`\delta=0.05`$ 时，每格 4、16、64、256 次复现对应的半径是 1.06、0.53、0.26、0.13（[Check 8.18](../experiments/08-12_finite_catalog_spectral_certificate_ZH.md#c8-18)）。256 次复现时 $`2\varepsilon_n=0.265`$，已低于先验界 $`\tfrac12`$，所以对这个先验界而言并不空；它空的是相对于实测效应的量级：真实模型配对的 surrogate 差约为 $`10^{-3}`$（[Check 8.17](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-17)）。严格证书成立，但在这个量级下没有信息量。

**关于陈述的说明.**

- **不覆盖的部分.** 没观测过的候选、测试和 checkpoint 不受 (a) 保护。低秩矩阵也不一定能从少数格子里恢复：某个重要方向可能只出现在一个很少被抽到的格子里。任何补全、泛化或离策略外推都需要自己的覆盖条件。
- **自适应目录.** (a) 的 union bound 针对的是在抽取复现之前就冻结的目录。根据审计结果生成的新候选，需要新的确认样本，或者一个在自适应选择下仍成立的界。
- **(e) 中的选择.** 如果 $`r`$ 或模式是从估计 $`\hat\alpha`$ 的同一批数据中选出来的，上式不能原样套用；需要样本分割或交叉拟合。

**推导要点.**

- (a) 对每格用 Hoeffding 不等式（[Hoeffding, 1963](#ref-hoeffding1963probability)），再对 $`pm`$ 个格取 union bound。
- (b) 用加权 Frobenius 范数控制算子范数，由于权重归一，前者不超过 $`\varepsilon_n`$；$`\Pi_q`$ 是压缩映射；Weyl 不等式说明每个奇异值的移动不超过算子范数。
- (c) 拆成 $`M=\hat M+(M-\hat M)`$，并利用 $`\lVert\hat M(I-\hat\Pi_r)\rVert_{\rm op}=\hat\sigma_{r+1}`$。
- (d) 把估计的中心化行展开到估计模式上，余项用 Cauchy–Schwarz；因为 $`\sum_j\lvert P_j-Q_j\rvert\le2`$，把 $`G`$ 换成 $`\hat G`$ 要付 $`2\varepsilon_n`$。
- (e) 来自输出向量 $`u_k`$ 的正交性与均值估计的无偏性；不需要不同模式之间独立。

**用到.**

- [Definition 4.8](04-08_response_operator_ZH.md#s4-8) — response operator
- [Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) — 谱充分性与截断接口

**被用于.**

- [Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) — 在其他 checkpoint 上复用锚点估计
- [Proposition 8.15](08-15_two_fidelity_response_estimation_ZH.md#s8-15) — 更便宜的逐格半径
- [Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16) — 组合保证中的锚点误差
- [Remark 8.17](08-17_open_problems_ZH.md#s8-17) — 跨格复用

## 相关工作

- (b)–(c) 是 Eckart–Young 最优性（[Eckart & Young, 1936](#ref-eckart1936approximation)）在矩阵扰动下的推论，配合逐格 Hoeffding 界（[Hoeffding, 1963](#ref-hoeffding1963probability)）。反事实 response 矩阵本身与 datamodels（[Ilyas et al., 2022](#ref-ilyas2022datamodels)）很接近，后者把模型输出对训练子集指示变量做回归；这里每一列是某个完整 learner state 上冻结的 feedback law，每个元素是一次完整续训。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").
