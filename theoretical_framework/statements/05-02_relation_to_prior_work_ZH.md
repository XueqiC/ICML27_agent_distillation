# Remark 5.2 · 与 stochastic modified equations、TracIn、influence function 和 metagradient 的关系

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-02_relation_to_prior_work_ZH.md) · [English](05-02_relation_to_prior_work.md)

**一段话.** 这条 Remark 把 [Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1) 放到已有的训练一阶分析中间，说明它在完整学习器状态上的记账方式写清楚了什么，并解释为什么对 Adam 一类 optimizer，二阶余项在学习率上是一阶的。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

- **状态.** 完整 learner state 为 $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$。$`\theta`$ 是可训练参数（例如 LoRA adapter），$`o`$ 是 optimizer state（例如 Adam 的矩 $`m,v`$ 与步数计数器）。需要用到计数器时显式写出。
- **反馈与梯度.** 第 $`t`$ 轮，feedback law $`P_t(\cdot\mid z)`$ 给出消息 $`Y`$。学习器在 loss 归一化与累加之后计算 consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$，$`\xi`$ 是声明的更新随机性。若有裁剪，裁剪算作 optimizer 的一部分。
- **Optimizer.** optimizer 是一个可测的增量映射 $`z^+=z+\Delta_t(z,g)`$。被比较的 feedback law 共用同一个 optimizer 与同一个梯度映射，只在消息分布上不同。换一个 optimizer 就是换一个 $`\Delta`$，用同样的方式比较。
- **本节局部复用的字母.** 本节中 $`s`$ 表示 Adam 二阶矩已吸收的梯度计算次数（Proposition 5.8），与第 6 节的 agent 状态 $`s`$ 无关。裁剪阈值记为 $`c`$，weight decay 系数记为 $`\lambda_{\rm wd}`$。

分布 $`P`$ 的学习 kernel 为

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

对 $`(t,z)`$ 处的分布 $`P`$，消息层面的平均增量为 $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$，并记

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation value.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ 是协议 $`P`$ 的 continuation value：之后的更新都按 $`P`$ 进行，然后部署学生并用 $`F_E`$ 打分（[Definition 1.4](../README_ZH.md#s1-4)）。$`L(P)=\mathbb E\,F_E(Z_R^P)`$，$`d_t^Q`$ 是协议 $`Q`$ 下 $`Z_t`$ 的分布。
- **范数.** 在 $`\mathbb R^{p+q}`$ 上固定一个范数 $`\lVert\cdot\rVert`$，其对偶范数记为 $`\lVert\cdot\rVert_*`$。

<a id="s5-2"></a>

**Remark 5.2（与 stochastic modified equations、TracIn、influence function 和 metagradient 的关系）.**

- Proposition 5.1 是对完整状态增量做 Taylor 展开，再代入逐轮的 telescoping 求和。它与 weak approximation / stochastic modified equations（[Li, Tai & E, 2019](../references_ZH.md#li2019stochastic)）、TracIn（[Pruthi et al., 2020](../references_ZH.md#pruthi2020estimating)）、influence function（[Koh & Liang, 2017](../references_ZH.md#koh2017understanding)）、metagradient（[Engstrom et al., 2025](../references_ZH.md#engstrom2025optimizing)）以及 In-Run Data Shapley（[Wang et al., 2025](../references_ZH.md#wang2025data)）属于同一类计算。我们不主张它的新颖性；它是推导后面结果的工具。
- 这套记账显式写出了：
  - *对象.* 一阶统计量是完整状态平均增量在 $`U`$ 上的投影，包括 optimizer 的矩 $`m`$ 和 $`v`$；既不是平均梯度，也不是 loss。梯度分布的哪些特征会进入，完全由 optimizer 映射 $`\Delta`$ 决定（Theorem 5.10）。
  - *特例.* 对 SGD，(a) 取 $`k=2`$ 就是 [Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement_ZH.md#s4-7) 中的局部 SGD 展开；(c) 是 [Theorem 4.2(a)–(c)](../statements/04-02_sufficiency_duality_transfer_ZH.md#s4-2) 用于线性化的 response 类；(b) 是 [Theorem 3.2](../statements/03-02_continuation_response_characterization_ZH.md#s3-2) 的 telescoping 恒等式在每一轮插入局部展开。
- *余项衡量的是什么（量级上的启发式；界本身是 (b) 中的 $`M_2`$）.* 把增量分块。
  - *参数块.* $`\Delta^\theta=O(\eta)`$，二阶贡献 $`\tfrac12\nabla^2_{\theta\theta}V[\Delta^\theta,\Delta^\theta]`$ 为 $`O(\eta^2)`$。
  - *一阶矩块（momentum $`m`$）.* $`\Delta^m=O(1-\beta_1)`$。continuation value 通过之后的参数增量依赖 $`m`$，主阶是线性的，系数为 $`O(\eta/(1-\beta_1))`$。所以 $`\nabla^2_{mm}V=O(\eta^2/(1-\beta_1)^2)`$，这一块的贡献也是 $`O(\eta^2)`$。
  - *二阶矩块（Adam 的 $`v`$）.* $`\Delta^v=(1-\beta_2)(G^{\odot2}-v)`$。continuation value 通过之后各步的分母依赖 $`v`$，这些分母只带一个 $`\eta`$ 因子、不再有更高次幂。所以这一块的二阶贡献是 $`\eta`$ 的**一阶**量。它相对这一块一阶项的大小由 $`\Delta v_i/v_i^{\rm next}`$ 决定，$`v^{\rm next}`$ 是下一步用到的二阶矩：warm 状态下约为 warmth $`\kappa_i`$（Definition 5.4），cold 状态下是 1 的量级。
  - 因此在 $`(\mathrm S_1)`$ 成立处，(b) 的余项对 SGD、momentum 与 sign 类 optimizer 每轮是 $`O(\eta^2)`$；对 Adam 类 optimizer、多于一轮时，它是 $`\eta`$ 的一阶量，相对大小与 warmth 同阶。$`R=1`$ 时边界之后的 continuation value 是 $`F_E(\theta)`$，不依赖 $`o`$，对任何 optimizer 余项都是 $`O(\eta^2)`$。
- *实际中的必要性.* (c) 只有在一阶项超过余项界时才能察觉失配。余项为 $`O(\eta^2)`$ 时，只要 $`\eta`$ 足够小，任何固定的失配都能被察觉。对 Adam 类的多轮 response，余项也是 $`\eta`$ 的一阶量，察觉需要 [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion_ZH.md#s8-9) 的 margin 条件；在二阶矩通道占主导的地方，需要有限替换的测量（[Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate_ZH.md#s8-8)、[Remark 7.5](../statements/07-05_where_first_order_rankers_fail_ZH.md#s7-5)）。
- *代价.* $`\mu_t^P(z)`$ 只需要在状态 $`z`$ 上计算 consumed gradient，不需要训练尾段（[Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states_ZH.md#s8-4)）。常数 $`\Lambda_t`$ 与 $`M_{2,t}`$ 是 continuation value 的性质，不能靠对消息抽样来估计。

**说了什么.** Proposition 5.1 是代入 telescoping 求和的 Taylor 展开，与 stochastic modified equations、TracIn、influence functions、metagradient、In-Run Data Shapley 属于同一类计算。它在记账上多出来的是完整状态：Adam 的二阶矩块使余项在学习率上是一阶的。

**在框架中的作用.** 确定新颖性的边界（Proposition 5.1 不主张新颖），并说明一阶检测什么时候需要 margin 条件。

## 参考文献

- [Li, Q., Tai, C., & E, W. (2019). Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations. *Journal of Machine Learning Research*, vol. 20, no. 40.](../references_ZH.md#li2019stochastic)
- [Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020). Estimating training data influence by tracing gradient descent. *NeurIPS*.](../references_ZH.md#pruthi2020estimating)
- [Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. *ICML*.](../references_ZH.md#koh2017understanding)
- [Engstrom, L., Ilyas, A., Chen, B., Feldmann, A., Moses, W., & Madry, A. (2025). Optimizing ML training with metagradient descent. arXiv:2503.13751.](../references_ZH.md#engstrom2025optimizing)
- [Wang, J. T., Mittal, P., Song, D., & Jia, R. (2025). Data Shapley in one training run. *ICLR*.](../references_ZH.md#wang2025data)
