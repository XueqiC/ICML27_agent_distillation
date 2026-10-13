# Remark 7.4 · 与已有工作的关系，以及裁剪的作用

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](07-04_relation_prior_work_role_clipping_ZH.md) · [English](07-04_relation_prior_work_role_clipping.md)

**一段话概括.** 本 remark 把 §7 关于 Adam 的结果放到已有工作中：一个 minibatch 扰动如何在 AdamW 的状态中传播，以及 Adam 下的 data attribution。它还点出幅度定律提示的一个机制：全局范数梯度裁剪限制了每条消息的幅度，也就限制了消息经由二阶矩造成的损害。这个机制作为可检验的假设提出，不作为结论。

**本页用到的设定与记号.**

- *幅度.* 对 warm Adam 的单个坐标，消息的幅度 $`\lambda`$ 是它沿平均梯度方向、以 running RMS $`\sqrt{v}`$ 为单位的梯度（[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)）。
- *二阶矩通道.* 消息一阶价值中经由二阶矩 $`v`$、即经由此后所有步长起作用的那一部分（[Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)）。
- *只看参数的分数与余弦分数.* [Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)(b) 中被证明为不一致排序器的两类影响分数：把 continuation 梯度与消息自己的参数步配对的分数，以及只依赖于消息梯度方向的分数。
- *全局范数裁剪*（[Pascanu et al., 2013](#ref-pascanu2013difficulty)）以阈值 $`c`$ 在 $`\lVert g\rVert>c`$ 时把梯度 $`g`$ 缩放为 $`c\,g/\lVert g\rVert`$。

<a id="s7-4"></a>

**Remark 7.4（与已有工作的关系，以及裁剪的作用）.**

- [Liu 与 Li（2026）](#ref-liu2026finite)把 AdamW 写成 input–state–output 系统，分析一个 minibatch 扰动在短 horizon（8–32 步）内如何通过动量与二阶矩状态传播。他们没有给出幅度定律、最优幅度、阈值，也没有讨论对数据选择的含义。[Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 在长 horizon 的平稳情形下补上了这些。
- Adam 版 In-Run Data Shapley（[Ding et al., 2026](#ref-ding2026inrun)）采用 fixed-state 近似，只给当前这一步的预条件更新打分，因此属于 [Proposition 7.3(b)](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 中只看参数的一类。用 Adam 步长计算的 TracIn 式分数（[Pruthi et al., 2020](#ref-pruthi2020estimating)）也属于这一类。LESS（[Xia et al., 2024](#ref-xia2024less)）按 Adam 预条件后的训练梯度与验证梯度的余弦排序，属于 [Proposition 7.3(b)](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3) 中的余弦一类。
- 全局范数裁剪限制了单条消息的幅度 $`\lambda`$，因此也限制了二阶矩通道能造成的损失（[Corollary 7.2(a),(c)](07-02_stationary_amplitude_law_ZH.md#s7-2)）。这为"裁剪为什么有帮助"提供了一个可检验的机制，但这里不把它作为结论。

**说了什么.**

- Corollary 7.2 的幅度定律、最优幅度与变号阈值，在关于 AdamW 状态动力学的最接近的已有分析中都没有出现。
- 三种常见的 attribution 分数落在 Proposition 7.3 证明为不一致的两类之中。
- 裁剪是"实际中大消息的危害为什么较小"的一个自然的候选解释。

**在框架中的作用.**

- 它划定 §7 中哪些是新结论、哪些不是。
- 它促使方法记录裁剪激活率并控制消息幅度（[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2) 第 4 步），也促成了真实模型检验 R3：比较裁剪开、关两种情况下的消息（[Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4)）。

**用到.**

- [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2)：幅度定律及其大幅度极限。
- [Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)：两类不一致的分数。

**被用于.**

- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：裁剪限制 $`\lambda`$（第 4 步）。
- [Remark 9.4](09-04_scope_of_evidence_ZH.md#s9-4)：对裁剪上限的真实模型检验 R3。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-ding2026inrun"></a>

**Meng Ding, Zeqing Zhang, Di Wang, Lijie Hu (2026).** In-Run Data Shapley for Adam optimizer. arXiv preprint. arXiv [2602.00329](https://arxiv.org/abs/2602.00329).

<a id="ref-liu2026finite"></a>

**Kang Liu, Suyan Li (2026).** Finite-horizon input–output dynamics of minibatch perturbations in AdamW. arXiv preprint. arXiv [2608.19762](https://arxiv.org/abs/2608.19762).

<a id="ref-pascanu2013difficulty"></a>

**Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013).** On the difficulty of training recurrent neural networks. *Proceedings of the 30th International Conference on Machine Learning (ICML)*, PMLR 28, no. 3, 1310–1318. [proceedings.mlr.press/v28/pascanu13](https://proceedings.mlr.press/v28/pascanu13.html).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
