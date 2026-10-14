# Remark 8.17 · 未解决的问题

*实验*

[← 框架总图](../README_ZH.md) · [陈述](../statements/08-17_open_problems_ZH.md) · [实验](../experiments/08-17_open_problems_ZH.md) · [English](08-17_open_problems.md)

**本页检验什么.** 本结果由下面摘要中的检验一并检验。每条摘要说明该检验对本结果检验什么、设置与关键数字；完整描述在标题所链接的页面上。前两条关于 Theorem C 的范围与校准问题；其余几条在真实模型与合成实例上测量：持续训练链条中的严格证书离有信息量还有多远，以及双保真估计量需要什么。

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

*配对替换分数*（[Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6)）.

- 在产生它的历史条件下，固定第 $`t`$ 轮的一个被 audit 状态 $`z`$（complete learner state，含 optimizer state）。候选为 $`j=0,1,\dots,M`$，$`M\ge1`$，其中 $`j=0`$ 是参照。候选 $`j`$ 的消息律为 $`P^j=P^j_t(\cdot\mid z)`$，消息 $`Y`$ 产生增量 $`\Delta^j=\Delta_t(z,G_t(z,Y,\xi))`$（含更新随机性）。
- **surrogate 尾段** $`T_\omega`$ 用随机性 $`\omega`$ 运行一段声明的（可以截短的）参照尾段，把一个状态映到最终状态。**surrogate 目标** $`\tilde F`$ 给最终状态打分（例如留出成功轨迹的 teacher-forced log-likelihood，映到 $`[0,1]`$）。记

```math
\varphi_\omega(u)=\tilde F\big(T_\omega(z+u)\big),\qquad\tilde L(P^j)=\mathbb E\,\varphi_\omega(\Delta^j),
```

其中 $`\omega`$ 与 $`\Delta^j`$ 独立。$`\tilde L(P^j)`$ 是"这一轮用候选 $`j`$、之后走 surrogate 尾段"的 surrogate 结果。

- **副本** $`\ell=1,\dots,r`$，$`r\ge2`$：抽一个新的 $`\omega_\ell`$，并为每个候选抽一条消息，得到增量 $`\Delta^j_\ell`$。同一副本内所有候选共用 $`\omega_\ell`$（**common random numbers**）；它们的消息也可以用共同的随机数耦合，只要每个 $`\Delta^j_\ell`$ 的边缘分布正确。不同副本相互独立。
- 若候选的消息律支撑有限，可以用支撑上的精确加权平均代替抽样，此时随机性只来自 $`\omega`$。
- 有序对 $`(j,k)`$ 的分数为

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_{\ell=1}^rZ^{jk}_\ell .
```

- 任务一侧，$`L_F(P^j)=\mathbb E\,F_E\big(T^{\rm full}(z+\Delta^j)\big)`$ 是"这一轮用候选 $`j`$、之后走完整参照尾段、按任务目标打分"的结果。用设定中的记号，$`L_F(P^j)=(K^j_tV^0_{t+1})(z)`$，所以 $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$。

*它的假设*（[Assumption 8.7](../statements/08-07_certificate_assumptions_ZH.md#s8-7)）.

- **(I) 副本独立.** 在产生 $`z`$ 的历史条件下，向量 $`(Z^{jk}_\ell)_{j,k}`$（$`\ell=1,\dots,r`$）独立同分布；$`r`$ 在观察分数之前固定。
- **(B) 分数有界.** 对每一对，$`Z^{jk}_\ell`$ 几乎必然落在一个已知的、长度为 $`B`$ 的区间里。例如 $`\tilde F\in[0,1]`$ 时 $`B\le2`$。
- **(A) surrogate 对齐.** 存在 $`\alpha>0`$ 与 $`\varepsilon_{\rm sur}\ge0`$，使对所有候选对，

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ 同时吸收了 surrogate 与任务目标之间的差别，以及尾段截短带来的差别。

**被检验的陈述（[Remark 8.17](../statements/08-17_open_problems_ZH.md#s8-17)）.**

- **可证明的取值范围.** [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8) 在先验范围下成立（$`\tilde F\in[0,1]`$ 时 $`B\le2`$）。这时范围项的量级为 $`7B\log(4M^2/\delta)/(3r)`$，对任何负担得起的 $`r`$，都可能远大于相邻候选之间的差距。分数的观测范围不能代替它。缺的是一个可证明的、远小于先验范围的界，例如 [Proposition 8.10(c)](../statements/08-10_cost_and_crn_ZH.md#s8-10) 中的 pathwise Lipschitz 界，配上一个对真实训练尾段被证明的 $`\bar\lambda`$。
- **校准的迁移.** 需要一个可检验的条件，使在校准对上拟合的 $`(\alpha,\varepsilon_{\rm sur})`$ 在新的候选对与之后的状态上仍然有效。
- 在这两点解决之前，Theorem 8.8 在实践中是一个无偏、低方差的排序分数；只有当 $`B`$ 与 $`(\alpha,\varepsilon_{\rm sur})`$ 被独立论证时，它才是严格的证书。
- **每条续训分支的信息量，以及闭合的识别.** 维数为 $`d`$ 的闭合预测测试空间并不需要 $`d`$ 个续训协议：一条分支可以输出多个指标，所以每条分支有 $`p`$ 个指标时，张成它至少需要 $`\lceil d/p\rceil`$ 个协议，而识别闭合需要扩展测试（[Proposition 4.12](../statements/04-12_response_predictive_state_factorization_ZH.md#s4-12)）。尚未解决：一条完整续训分支关于 response operator 及其闭合能提供多少独立信息，以及以最小成本识别闭合需要哪些干预。
- **跨格复用.** [Proposition 8.15](../statements/08-15_two_fidelity_response_estimation_ZH.md#s8-15) 减少一格之内的复现，但每格（候选 × checkpoint × 协议）仍至少要一次完整续训。在格子之间共享校正或接口，需要矩阵补全那样的覆盖条件；只有低秩并不够（[Proposition 8.12](../statements/08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)）。
- **随效应缩放的证书.** [Propositions 8.12](../statements/08-12_finite_catalog_spectral_certificate_ZH.md#s8-12) 与 [8.14](../statements/08-14_anchor_reuse_drift_certificate_ZH.md#s8-14) 以及 [Corollary 8.16](../statements/08-16_composed_guarantee_ZH.md#s8-16) 的严格半径，是有界取值集中不等式给出的绝对精度，在实测量级下为空。尚未解决：在负担得起的续训次数下，构造半径随效应大小缩放的方差自适应或去偏配对证书。这里发现的"为空"是所述估计量的性质，并不证明这样的证书不存在。
- **真实学习器上的短续训对比完整续训.** [Proposition 8.15](../statements/08-15_two_fidelity_response_estimation_ZH.md#s8-15) 的节省条件在成本比为十分之一时，若每个便宜输出都计费，需要约 0.787（固定系数）或 0.575（最优系数）的配对相关系数；若每对中的便宜输出被复用，则为 0.766 或约 0.316。真实模型上还没有测过这个相关系数。

### [Check 8.14](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-14) · 受控面板上的有限 CRN 配对替换（Theorem 8.8、Proposition 8.10、Remarks 8.11 and 8.17）

- **它对 Remark 8.17 检验什么.** 先验区间是否就是严格证书的瓶颈，以及校准迁移是否确实是一个缺口。
- **设置.** 与主检验相同：1,728 个候选对，$`r\in\lbrace64,256\rbrace`$，与任务一致的 surrogate 与需校准的短尾段 surrogate；严格版本用先验区间 $`B=2`$，诊断版本用分数的观测区间。
- **关键数字.** 严格证书：与任务一致的 surrogate 认证 0 对，需校准的至多认证 8 对（0.46%），没有错误；$`r=256`$ 时 empirical Bernstein 半径（[Maurer & Pontil, 2009](#ref-maurer2009empirical)）的范围项为 0.156，而差距的中位数是 0.0037。观测区间诊断：与任务一致的 surrogate 认证 45%–57% 的候选对、没有错误，需校准的认证 23%–28%、有 22–28 个错误。
- **结论.** 两个未解决问题都是真实的：如果有一个与观测区间同样大小的可证明范围，证书就能下结论；而校准不会自动迁移。

### [Check 8.16](../experiments/08-09_first_order_version_margin_criterion_ZH.md#c8-16) · 实测的一阶证书及其上限（Corollary 8.9）

- **它对 Remark 8.17 检验什么.** 一阶认证失败是与范围问题不同的另一个原因：即使还没轮到范围，二阶余项就已经让一阶证书失效。
- **设置.** 同一面板；$`G=8`$ 个 CRN 组；有限差分步长 $`h\in\lbrace10^{-3},3\times10^{-3},10^{-2}\rbrace`$；有限差分项与余项都用实测曲率。
- **关键数字.** 1,584 个光滑候选对认证 0 对；仅余项一项就在 68.9% 的候选对上超过实测一阶分数；$`G=8`$ 时 empirical Bernstein 的范围项是观测区间的 2.85 倍。
- **结论.** Theorem 8.8 完全去掉了余项这一原因；剩下的是范围，即第一个未解决问题。

### [Check 8.17](../experiments/08-08_crn_paired_replacement_certificate_ZH.md#c8-17) · 真实模型上的证书：R1，P1 对 P3（Theorem 8.8）

- **对 Remark 8.17 检验什么.** 范围问题在真实配方上是否依然存在。
- **设置.** ALFWorld 配方，被审计状态 $`z_{10}`$，P1（重放 $`K=20`$ 条示范）对 P3（按奖励筛选的自生成轨迹），$`r=64`$ 个副本，$`T=10`$ 次共享尾段更新，在 40 条留出教师轨迹上的有界几何概率 surrogate。
- **关键数字.** $`\hat S=+0.001378`$（64 个副本中 57 个为正）；诊断半径 0.001206；$`B=2`$ 的严格半径 0.325。surrogate 校准尚未估计。
- **结论.** Theorem C 的两个未解决问题在真实模型上都存在：严格半径约为分数的 240 倍，而且还没有校准。

### [Check 8.18](../experiments/08-12_finite_catalog_spectral_certificate_ZH.md#c8-18) · 无需间隙的截断残差与目录半径（Proposition 8.12）

- **对 Remark 8.17 检验什么.** 在负担得起的复现数下严格目录证书有多大，以及目录的成本单位。
- **设置.** $`p=10`$ 个测试、$`m=20`$ 个候选、$`\delta=0.05`$ 时的逐格 Hoeffding 半径；$`m=20`$、$`n=4`$ 时的运行次数。
- **关键数字.** 每格 4、16、64、256 次复现时半径为 1.06、0.53、0.26、0.13；一个续训协议时 80 次运行，十个协议时 800 次。
- **结论.** 需要随效应缩放的证书（第三个新问题）；运行次数由不同的续训协议决定（第一个新问题）。

### [Check 8.21](../experiments/08-14_anchor_reuse_drift_certificate_ZH.md#c8-21) · 实测评估噪声下的漂移证书（Proposition 8.14(c)、(d)）

- **对 Remark 8.17 检验什么.** 用所述的估计量，能否便宜地认证跨 checkpoint 的复用。
- **设置.** 真实模型噪声参照（两台相同服务器上 300 个 dev 任务成功 241 与 242 个，15 个翻转），$`M=10`$ 对，$`\delta=0.05`$。
- **关键数字.** 10、40、100、1,000 对时半径为 0.718、0.507、0.404、0.227；要到 0.03 约需 $`4.9\times10^6`$ 对；平凡界 0.5。
- **结论.** 成立但为空：这是该估计量与该不等式的性质，正是第三个新问题要改进的地方。

### [Check 8.22](../experiments/08-15_two_fidelity_response_estimation_ZH.md#c8-22) · 无偏性、分配与成本门槛（Proposition 8.15）

- **对 Remark 8.17 检验什么.** 短续训需要多大的相关性才能省计算，以及"减复现、不减格子"的限制。
- **设置.** 便宜运行的成本为完整运行的十分之一；固定系数与嵌套最优系数。
- **关键数字.** 同口径门槛：每个便宜输出都计费时 0.787（固定系数）对 0.575（最优系数），每对中的便宜输出被复用时 0.766 对约 0.316；每格 4 次完整复现时至多省 4 倍。
- **结论.** 第四个新问题是框架尚未做的一项测量；即使它成功，第二个新问题依然存在。

### [Check 8.23](../experiments/08-16_composed_guarantee_ZH.md#c8-23) · 单步下限，以及有符号和对比逐项认证（Corollary 8.16）

- **对 Remark 8.17 检验什么.** 组合保证在实测量级下是否有信息量。
- **设置.** $`B=2`$，一个测试，20 个候选、每格 10 次复现时的 $`\varepsilon_n`$，10 对时的漂移半径；$`R=20`$ 的有符号和 Monte Carlo。
- **关键数字.** 单步下限 2.59；单位噪声、单位误差下，有符号和估计需要 400 次查询，逐项认证需要 8,000 次。
- **结论.** 在这个量级下组合界是一份误差预算；缺的是随效应缩放的证书。

## 复现

复现命令与结果文件见主检验所在的页面：[Theorem 8.8](../experiments/08-08_crn_paired_replacement_certificate_ZH.md), [Corollary 8.9](../experiments/08-09_first_order_version_margin_criterion_ZH.md), [Proposition 8.12](../experiments/08-12_finite_catalog_spectral_certificate_ZH.md), [Proposition 8.14](../experiments/08-14_anchor_reuse_drift_certificate_ZH.md), [Proposition 8.15](../experiments/08-15_two_fidelity_response_estimation_ZH.md), [Corollary 8.16](../experiments/08-16_composed_guarantee_ZH.md)。这些检验使用 NumPy（[Harris et al., 2020](#ref-harris2020array)）与 SciPy（[Virtanen et al., 2020](#ref-virtanen2020scipy)）。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-maurer2009empirical"></a>

**Andreas Maurer, Massimiliano Pontil (2009).** Empirical Bernstein bounds and sample variance penalization. *Proceedings of the 22nd Conference on Learning Theory (COLT)*. arXiv [0907.3740](https://arxiv.org/abs/0907.3740).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
