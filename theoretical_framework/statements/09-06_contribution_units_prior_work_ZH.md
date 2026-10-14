# Remark 9.6 · 贡献单元与最近的已有工作

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](09-06_contribution_units_prior_work_ZH.md) · [English](09-06_contribution_units_prior_work.md)

**一段话概括.** 这个框架里几乎每一件工具都是经典的：算子逼近、预测状态实现、performance-difference 恒等式、两点下界、最优设计、凸可实现性、控制变量。经典工具的组合仍然可以是原创贡献，但前提是：它提出了现有框架没有处理好的问题，并且推出能改变方法或实验做法的结论。本 remark 把框架的贡献写成三个单元，每一个都明确是组合层面的主张，而不是新数学；然后对每一条最近的已有工作，列出必须承认什么、框架还必须证明什么增量。它不声称原创性已经成立，并直接说明还缺什么。

**本页用到的设定与记号.**

- *贡献单元.* 审稿人可以与已有工作比较、推翻或验证的主张：说清问题、组合起来的结果，以及它改变的决策。
- *组合主张.* 主张已有结果之间的连接，在这个问题里推出了各部分单独给不出的可用结论。有针对性的文献检索不能证明不存在先例。
- *可检验的增量.* 由组合推出、并能在真实模型上与最近的已有方法在相同总成本下比较的预测或成本节省。

<a id="s9-6"></a>

**Remark 9.6（贡献单元与最近的已有工作）.**

**(a) 三个贡献单元.**

1. **问题与表示.** 经验充分性是相对于一个声明的测试类来定义的，这个测试类在未来的学习操作下保持闭合，而不只相对于当前的部署。检查点上的 response 经由下一步预测状态的变化而分解（[Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12)）。这样，消息压缩（[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)）与学习器记忆（[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)）落在同一个对象上。[Proposition 4.11](04-11_static_not_dynamic_closure_error_ZH.md#s4-11) 与 [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6) 的构造说明：只匹配当前行为或即时参数变化，可能丢掉后续训练才会显现的信息。
2. **实现与成本.** 对于跨检查点共享、而反馈差异依状态而定的接口，框架给出复用界（[Proposition 8.13](08-13_shared_features_across_checkpoints_ZH.md#s8-13)）、带可执行漂移检查的锚点迁移界（[Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14)）、有限目录的运行次数（[Proposition 8.12](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)），以及带精确成本条件的无偏两保真度估计量（[Proposition 8.15](08-15_two_fidelity_response_estimation_ZH.md#s8-15)）。它们合起来说明摊销何时划算、何时应当放弃。在真实模型上是否划算，尚未检验。
3. **获取与编译.** 测试能看见的任务后果的不确定性，与合法反馈的凸包，进入同一个半径；它的两个下界把信息缺口与可实现性缺口区分开（[Propositions 6.14](06-14_convex_realizability_legal_pool_ZH.md#s6-14) 与 [6.16](06-16_information_compilation_radius_ZH.md#s6-16)）。在逐测试校准、闭合与覆盖的条件下，这个半径与最终训练误差相连（[Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16)）。线性高斯设计（[Proposition 6.15](06-15_linear_gaussian_target_ZH.md#s6-15)）是一个可计算的特例，不是一般规则。

这三条都不主张新数学，也不主张整体的原创性已经成立。[Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2) 与 [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4) 的恒等式是记账工具，不算作贡献。

**(b) 最近的已有工作.**

| 工作方向 | 必须承认的已有内容 | 框架还必须证明的 |
|---|---|---|
| Machine teaching（[Zhu, 2015](#ref-zhu2015machine)；[Liu et al., 2017](#ref-liu2017iterative)；[Lessard, Zhang & Zhu, 2019](#ref-lessard2019optimal)） | 教学依赖学习器；按学习器的更新规则迭代挑选样本；顺序教学已经被写成最优控制问题 | 贡献不能是"首次按学习器选反馈"或"首次把训练看作受控系统"。必须说明：把未知环境的证据、后续训练的充分性、合法编译与计成本的维护组合起来，会改变决策 |
| Datamodels 与 TRAK（[Ilyas et al., 2022](#ref-ilyas2022datamodels)；[Park et al., 2023](#ref-park2023trak)） | 根据训练数据的变化反事实地预测模型输出；TRAK 处理了大规模下的计算可行性 | 反事实的 response 矩阵不是一类新对象。框架必须把一次性的数据归因，与依状态而定、多轮、在合法反馈上的接口区分开，并测量成本 |
| Datamodels 的调和分析（[Saunshi et al., 2023](#ref-saunshi2023understanding)） | 已经用 Fourier 系数与 noise stability 分析 datamodels，包括不拟合完整 datamodel 的残差估计 | 不是首次从谱的角度看数据影响。[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9) 的 response 增益谱与 Boolean Fourier 频率不同，估计方法需要具体比较 |
| 学习算法的 sketch（[Gunn, 2026](#ref-gunn2026sketch)） | 用训练算法的高阶导数，对删除数据的反事实做可压缩的预测，并给出有条件的计算保证 | 比较任务、访问模型与假设。它对高阶 Taylor 张量的稳定性条件在这里不作假设；它已经有成本理论，而这个框架还没有与之相当的结果 |
| 价值导向压缩、预测状态表示及其 TD 学习（[Poupart & Boutilier, 2002](#ref-poupart2002value)；[Littman, Sutton & Singh, 2001](#ref-littman2001predictive)；[Singh, James & Rudary, 2004](#ref-singh2004predictive)；[Boots & Gordon, 2010](#ref-boots2010predictive)） | 以测试为状态、与价值相关的压缩、动态闭合，都有先例 | 区别在于完整的学习过程，以及获取、编译与复用的约束。[Proposition 4.12](04-12_response_predictive_state_factorization_ZH.md#s4-12) 是统一这些约束的引理，不是重新发现预测状态 |
| 价值等价与任务充分的 world model（[Grimm et al., 2020](#ref-grimm2020value)；[Feng et al., 2026](#ref-feng2026learning)） | 相对于任务或函数类的模型充分性；为任务充分的 world model 做主动探索 | 说明学习更新与 optimizer 记忆改变了目标接口与获取选择；只把充分性改称"相对学习器"是不够的 |
| LESS、GRACE、Adam 的数据 Shapley 与 metagradient（[Xia et al., 2024](#ref-xia2024less)；[Panigrahi et al., 2025](#ref-panigrahi2025good)；[Ding et al., 2026](#ref-ding2026inrun)；[Engstrom et al., 2025](#ref-engstrom2025optimizing)） | 数据或 teacher 的价值可以依赖训练过程；Adam 已有专门的研究 | [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6) 的记忆构造只把"读取即时参数变化"的接口区分开。任何算法上的优势，都需要对每个方法做正确的适配，并在相同总成本下比较 |
| Performance difference 与 simulation lemma（[Kakade & Langford, 2002](#ref-kakade2002approximately)） | 通过价值递推与残差分解策略或模型的差异 | [Theorem 3.2](03-02_continuation_response_characterization_ZH.md#s3-2) 与 [Proposition 3.4](03-04_joint_decomposition_approximate_value_ZH.md#s3-4) 是基础工具，不是核心贡献 |
| Doubly robust 离策略评估（[Jiang & Li, 2016](#ref-jiang2016doubly)） | 价值模型加校正项，在部分情形下有匹配的下界 | 通向可估计组合保证的价值模型路线，必须与它比较；任何增量都必须来自学习系统的结构，并能降低多候选、多检查点决策的成本 |
| 多保真度与多层 Monte Carlo（[Peherstorfer, Willcox & Gunzburger, 2016](#ref-peherstorfer2016optimal)；[Giles, 2015](#ref-giles2015multilevel)） | 便宜的近似加昂贵的残差校正，以及最优分配 | 控制变量公式不是新的。可能的增量在于把它与 response 接口、完整学习器状态以及跨检查点复用结合起来 |

**(c) 还缺什么.**

- 没有任何定理表明在相同总成本下优于 LESS、TRAK 或 datamodels；[Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6) 是机制示意，不是这样的分离结果。
- 复用与组合保证的严格版本有效，但在实测量级下是空的（[Proposition 8.14](08-14_anchor_reuse_drift_certificate_ZH.md#s8-14)、[Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16)）。这是所用估计量与不等式的性质，不是"认证不可能"的证明。估计成本的理论必须把三件事分开：一对固定协议的收益、同时认证每个局部 response，以及对一类未来协议的充分性。
- 所以组合是否新颖，要看它能否给出可检验的增量：一个改变了真实决策的缺口诊断，暖检查点上可测的记忆效应（T1，[Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4)），或者真实模型上复用或两保真度带来的节省。在这样的结果出现之前，这三个单元是待检验的提案，不是已确立的贡献。

**说了什么.** 框架的贡献是把经典结果组合成一个关于后续训练充分性、其成本与其合法实现的问题。每一条最近的已有工作都已经占据了其中一部分，框架仍需对每一条证明可检验的增量。

**在框架中的作用.**

- 它固定框架主张什么是新的、承认什么不是，使后面的结果能对照正确的基线来读。
- 它把新颖性问题变成实验：每个单元都写明它应该改变的决策，[Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4) 记录目前的检验是否支持它。

**用到.**

- [Propositions 4.11–4.12](04-11_static_not_dynamic_closure_error_ZH.md#s4-11)、[Theorem 4.9](04-09_spectral_sufficiency_ZH.md#s4-9)、[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)、[Proposition 7.6](07-06_equal_immediate_parameters_different_continuation_ZH.md#s7-6)：单元 1。
- [Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate_ZH.md#s8-12)：单元 2。
- [Propositions 6.14–6.16](06-14_convex_realizability_legal_pool_ZH.md#s6-14)、[Corollary 8.16](08-16_composed_guarantee_ZH.md#s8-16)：单元 3。
- [Remark 9.5](09-05_assumption_audit_ZH.md#s9-5)：每个单元依赖的条件。

**被用于.**

- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：体现这三个单元的方法。

## 相关工作

- 上表中的比较就是本 remark 的相关工作；每一项引用都是被承认部分的原始来源。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-boots2010predictive"></a>

**Byron Boots, Geoffrey J. Gordon (2010).** Predictive state temporal difference learning. *Advances in Neural Information Processing Systems 23 (NIPS)*, Curran Associates. [proceedings.neurips.cc/paper/2010/hash/3473decccb0509fb264818a7512a8b9b](https://proceedings.neurips.cc/paper_files/paper/2010/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html).

<a id="ref-ding2026inrun"></a>

**Meng Ding, Zeqing Zhang, Di Wang, Lijie Hu (2026).** In-Run Data Shapley for Adam optimizer. arXiv preprint. arXiv [2602.00329](https://arxiv.org/abs/2602.00329).

<a id="ref-engstrom2025optimizing"></a>

**Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry (2025).** Optimizing ML training with metagradient descent. arXiv preprint. arXiv [2503.13751](https://arxiv.org/abs/2503.13751).

<a id="ref-feng2026learning"></a>

**Fan Feng, Yujia Zheng, Minghao Fu, Yongqiang Chen, Guangyi Chen, Kevin Patrick Murphy, Biwei Huang, Kun Zhang (2026).** Learning task-sufficient world models by synergizing agentic exploration and structured modeling. *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR 306:30616–30641. [proceedings.mlr.press/v306/feng26aa](https://proceedings.mlr.press/v306/feng26aa.html); arXiv [2607.04409](https://arxiv.org/abs/2607.04409).

<a id="ref-giles2015multilevel"></a>

**Michael B. Giles (2015).** Multilevel Monte Carlo methods. *Acta Numerica* 24, 259–328. DOI [10.1017/S096249291500001X](https://doi.org/10.1017/S096249291500001X).

<a id="ref-grimm2020value"></a>

**Christopher Grimm, André Barreto, Satinder Singh, David Silver (2020).** The value equivalence principle for model-based reinforcement learning. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 5541–5552. arXiv [2011.03506](https://arxiv.org/abs/2011.03506).

<a id="ref-gunn2026sketch"></a>

**Sam Gunn (2026).** How to sketch a learning algorithm. *arXiv preprint*. arXiv [2604.07328](https://arxiv.org/abs/2604.07328).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).

<a id="ref-lessard2019optimal"></a>

**Laurent Lessard, Xuezhou Zhang, Xiaojin Zhu (2019).** An optimal control approach to sequential machine teaching. *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 89:2495–2503. [proceedings.mlr.press/v89/lessard19a](https://proceedings.mlr.press/v89/lessard19a.html).

<a id="ref-littman2001predictive"></a>

**Michael L. Littman, Richard S. Sutton, Satinder Singh (2001).** Predictive representations of state. *Advances in Neural Information Processing Systems 14 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html).

<a id="ref-liu2017iterative"></a>

**Weiyang Liu, Bo Dai, Ahmad Humayun, Charlene Tay, Chen Yu, Linda B. Smith, James M. Rehg, Le Song (2017).** Iterative machine teaching. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:2149–2158. [proceedings.mlr.press/v70/liu17b](https://proceedings.mlr.press/v70/liu17b.html).

<a id="ref-panigrahi2025good"></a>

**Abhishek Panigrahi, Bingbin Liu, Sadhika Malladi, Sham Kakade, Surbhi Goel (2025).** In good GRACEs: Principled teacher selection for knowledge distillation. arXiv preprint. arXiv [2511.02833](https://arxiv.org/abs/2511.02833).

<a id="ref-park2023trak"></a>

**Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry (2023).** TRAK: Attributing model behavior at scale. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202:27074–27113. [proceedings.mlr.press/v202/park23c](https://proceedings.mlr.press/v202/park23c.html).

<a id="ref-peherstorfer2016optimal"></a>

**Benjamin Peherstorfer, Karen Willcox, Max Gunzburger (2016).** Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing* 38, no. 5 (section A, Methods and Algorithms for Scientific Computing). DOI [10.1137/15M1046472](https://doi.org/10.1137/15M1046472).

<a id="ref-poupart2002value"></a>

**Pascal Poupart, Craig Boutilier (2002).** Value-directed compression of POMDPs. *Advances in Neural Information Processing Systems 15 (NIPS)*, MIT Press. [proceedings.neurips.cc/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d](https://proceedings.neurips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html).

<a id="ref-saunshi2023understanding"></a>

**Nikunj Saunshi, Arushi Gupta, Mark Braverman, Sanjeev Arora (2023).** Understanding influence functions and datamodels via harmonic analysis. *11th International Conference on Learning Representations (ICLR)*. [openreview.net/forum?id=cxCEOSF99f](https://openreview.net/forum?id=cxCEOSF99f); arXiv [2210.01072](https://arxiv.org/abs/2210.01072).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).

<a id="ref-zhu2015machine"></a>

**Xiaojin Zhu (2015).** Machine teaching: An inverse problem to machine learning and an approach toward optimal education. *Proceedings of the 29th AAAI Conference on Artificial Intelligence (AAAI)* 29, no. 1. DOI [10.1609/aaai.v29i1.9761](https://doi.org/10.1609/aaai.v29i1.9761).
