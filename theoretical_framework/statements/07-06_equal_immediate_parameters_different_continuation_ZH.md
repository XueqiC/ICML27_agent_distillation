# Proposition 7.6 · 即时参数相同并不决定后续

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](07-06_equal_immediate_parameters_different_continuation_ZH.md) · [证明](../proofs/07-06_equal_immediate_parameters_different_continuation_ZH.md) · [实验](../experiments/07-06_equal_immediate_parameters_different_continuation_ZH.md) · [English](07-06_equal_immediate_parameters_different_continuation.md)

**一段话概括.** 没有 optimizer 记忆的学习器不记得自己是怎样走到当前参数的：两条消息只要留下相同的参数，之后无论怎样训练，未来都一样。Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）会记得。本命题用标准超参数、从一个只需一步普通更新就能到达的状态，精确地说明这一点。动量不为零时，Adam 的步长是梯度的二对一函数，所以两个不同的梯度给出同一个参数更新；但它们留下的矩不同，下一次更新就已经分开。$`\epsilon>0`$ 时构造依然成立。这是冷启动附近的机制示意，并不度量 warm 训练中记忆效应有多大；(e) 给出 warm 状态下的精确恒等式，说明记忆在那里携带了什么。它与 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 合称框架中的 **Theorem B**。

**本页用到的设定与记号.**

- *complete learner state、continuation value 与 response.* $`z=(\theta,o)`$，参数为 $`\theta`$，optimizer state 为 $`o`$（[Definition 1.3](../README_ZH.md#s1-3)）；参照 continuation value 为 $`V^0_{t+1}`$，单边界 response 为 $`g_{t,z}(y)=\mathbb E_{\xi_t}V^0_{t+1}(U_t(z,y,\xi_t))`$（[Definition 1.4](../README_ZH.md#s1-4)）。
- *无记忆的学习器.* 参照 kernel $`K^0_k`$（$`k>t`$）把 $`(\theta,k)`$ 的函数映为 $`(\theta,k)`$ 的函数，且部署得分只依赖参数，$`F_E(z)=F_E(\theta)`$。学习率计划固定、没有动量、缓冲区或采样器记忆的普通 SGD 是无记忆的。带动量的 SGD 不是：它的动量缓冲就是 optimizer state。
- *Adam，单个坐标.* 超参数 $`\beta_1\in(0,1)`$、$`\beta_2\in(0,1)`$，学习率 $`\eta>0`$，$`\epsilon\ge0`$，做 bias correction，无 weight decay。步数计数为 $`s`$ 时，consumed gradient $`g`$ 给出

```math
m'=\beta_1m+(1-\beta_1)g,\qquad v'=\beta_2v+(1-\beta_2)g^2,\qquad \theta'=\theta-\eta\,q_{1,s}\,\frac{m'}{\sqrt{q_{2,s}v'}+\epsilon},
```

其中 $`q_{1,s}=(1-\beta_1^{\,s})^{-1}`$、$`q_{2,s}=(1-\beta_2^{\,s})^{-1}`$，与 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 相同。
- *步长大小.* 对 $`\epsilon=0`$ 与固定状态 $`(m,v,s)`$，记 $`a=\beta_1m`$、$`b=1-\beta_1`$、$`c=\beta_2v`$、$`d=1-\beta_2`$，以及

```math
S(g)=\frac{a+bg}{\sqrt{c+dg^2}},\qquad\text{于是}\qquad \theta'=\theta-\eta\,\frac{q_{1,s}}{\sqrt{q_{2,s}}}\,S(g).
```

- *可实现的梯度.* 下面用到的每个 consumed gradient 都是一维平方损失 $`\tfrac12(\theta-y)^2`$ 在当前参数处的梯度，标签 $`y`$ 相应选取。每个这样的损失 Hessian 都是 $`1`$。

<a id="s7-6"></a>

**Proposition 7.6（即时参数相同并不决定后续）.**

**(a) 无记忆的学习器.** 对无记忆的学习器，$`V^0_{t+1}(z)`$ 只通过 $`\theta`$ 依赖于 $`z`$。因此，第 $`t`$ 轮的两条消息 $`y_A,y_B`$ 只要更新后下一参数的分布相同，它们对每个声明的任务、每个无记忆的后续训练，单边界 response 都相等：$`g_{t,z}(y_A)=g_{t,z}(y_B)`$。

**(b) Adam 的步长是梯度的二对一函数.** 设 $`\epsilon=0`$、$`v>0`$、$`m>0`$（$`m<0`$ 的情形对称）。在 $`g>-a/b`$ 上，步长大小 $`S`$ 为正，在 $`(-a/b,g^\star]`$ 上严格递增，在 $`[g^\star,\infty)`$ 上严格递减，其中

```math
g^\star=\frac{bc}{ad}=\frac{(1-\beta_1)\beta_2v}{\beta_1(1-\beta_2)m},\qquad S(g^\star)=\sqrt{\frac{a^2}{c}+\frac{b^2}{d}},\qquad \lim_{g\to\infty}S(g)=\frac{b}{\sqrt d}.
```

严格介于 $`b/\sqrt d`$ 与 $`S(g^\star)`$ 之间的每个值，恰好由两个梯度 $`g_A<g^\star<g_B`$ 取到。这两条消息给出相同的下一参数 $`\theta'`$，一阶矩却不同，$`m'_A\ne m'_B`$。

**(c) 标准超参数下的一个精确实例.** 取 $`\beta_1=0.9`$、$`\beta_2=0.999`$、$`\eta=1`$、$`\epsilon=0`$，做 bias correction，无 weight decay。从 $`(\theta,m,v)=(0,0,0)`$ 出发，用 consumed gradient $`1`$ 走一步，到达 checkpoint $`z_1=(-1,\tfrac1{10},\tfrac1{1000})`$，此时步数计数为 $`1`$。在 $`z_1`$ 处，第二步的两个梯度 $`g_A=\tfrac12`$ 与 $`g_B=\tfrac{2103}{790}`$ 是 $`711g^2-2248.2\,g+946.35=0`$ 的两个根，也就是满足

```math
\frac{(0.9+g)^2}{0.999+g^2}=\frac{1960}{1249}.
```

因此：

- 即时参数完全相同，两边都是 $`\theta_2\approx-1.932179639`$；
- 矩不同：$`g_A`$ 之后 $`(m_2,v_2)=(\tfrac7{50},\tfrac{1249}{10^6})`$，$`g_B`$ 之后 $`(\tfrac{1407}{3950},\tfrac{50460849}{6241\cdot10^6})`$；
- 下一次更新不同：第三步用同一个损失，它在 $`\theta_2`$ 处的梯度为 $`1`$；两边第三步的平方步长 $`m_3^2/v_3`$ 是两个不同的有理数 $`\tfrac{51076000}{2247751}`$ 与 $`\tfrac{1103967076000}{56651388151}`$，$`\theta_3^A-\theta_3^B\approx-0.0712027`$。对有界得分 $`F(\theta)=(1+e^{-\theta})^{-1}`$，差为 $`\approx-0.00364994`$。

对任意 $`\eta>0`$，用相同的 consumed gradient，即时参数仍然相同，第三步之后的参数差是上面数值的 $`\eta`$ 倍。

**(d) $`\epsilon`$ 为正.** 对每个足够小的 $`\epsilon>0`$，存在一个关于 $`\epsilon`$ 连续、且 $`g_B(0)=\tfrac{2103}{790}`$ 的梯度 $`g_B(\epsilon)`$，使得每一步都用这个 $`\epsilon`$ 运行 (c) 的实例时，即时参数仍然相同，第三步之后的差仍不为零。$`\epsilon=10^{-8}`$ 时，$`g_B(\epsilon)\approx2.662025475`$，$`\theta_3^A-\theta_3^B\approx-0.0712027`$。

**(e) warm 状态下记忆携带了什么.** 在任何 $`v>0`$ 的状态下，consumed gradient $`g`$ 使二阶矩发生的相对变化精确为

```math
\frac{v'-v}{v}=(1-\beta_2)\Big(\frac{g^2}{v}-1\Big).
```

若 $`v`$ 保持不变、不做 bias correction、$`\epsilon=0`$，且尾段梯度不依赖 $`\theta`$，则消息梯度的扰动 $`\delta g`$ 在滞后 $`k=0,1,\dots`$ 时使参数移动 $`-\eta(1-\beta_1)\beta_1^{\,k}\,\delta g/\sqrt v`$。消息自己的更新只承担总一阶位移 $`-\eta\,\delta g/\sqrt v`$ 中 $`1-\beta_1`$ 的份额，前 $`K`$ 次更新合计承担 $`1-\beta_1^{\,K}`$。

**说了什么.**

- 没有 optimizer 记忆时，消息之后的参数概括了这条消息对此后训练的全部作用。Adam 下并非如此，即使这条消息的全部即时效应都精确已知。
- (c) 中的相等不是巧合。动量不为零时，Adam 的步长在峰值 $`g^\star`$ 两侧是梯度的二对一函数，而两个梯度留下的矩不同。
- 用 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)(b) 的语言说，(c) 中两条消息的参数增量完全相同，只在动量通道与二阶矩通道上不同。
- warm 状态下，记忆通过两条途径起作用。动量把一个梯度一阶位移的大部分放在消息自己这一步之后才兑现：$`\beta_1=0.9`$ 时，当步只兑现 $`1-\beta_1=0.1`$，十步之内约兑现 $`0.65`$。当消息的梯度相对 $`v`$ 很大时，二阶矩的变化可以远大于 $`1-\beta_2`$。
- *适用范围.*
  - (c) 是冷启动附近的构造。步数计数为 $`2`$ 时，$`v`$ 的 bias correction 为 $`1/(1-\beta_2^{\,2})\approx500`$，单个梯度就主导了 $`v`$。
  - 它区分的是只读即时参数变化的接口；在一维情形下，所有方向（余弦）分数也平凡地打平。它并不是与 LESS（[Xia et al., 2024](#ref-xia2024less)）、TRAK（[Park et al., 2023](#ref-park2023trak)）或 datamodels（[Ilyas et al., 2022](#ref-ilyas2022datamodels)）的分离结果，这些方法读取的是别的信息。
  - 在 (e) 的前提下，总一阶位移是即时位移的常数倍，所以按即时位移排序在那里保持不变。排序可以因曲率而改变（即使 Hessian 固定，动量对一次脉冲的 response 也随各方向的曲率而不同），也可以因 $`v`$ 而改变。

**直觉.**

- Adam 用 running RMS 去除动量，而新梯度同时也会抬高这个 RMS。梯度小时两者都只增加一点；梯度极大时两者都增加很多，比值饱和。中间某处步长达到峰值，于是一个中等的梯度和一个很大的梯度可以给出同样的步长。
- 参数分不清这两条消息，矩分得清。大的梯度留下更大的动量和更大的 $`v`$，下一次更新两者都会读到。

**在框架中的作用.**

- 它与 [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1) 合称 Theorem B。Theorem 7.1 算出一条消息经参数、动量、二阶矩三个通道的精确一阶价值；本命题用一个精确的例子说明，动量通道与二阶矩通道不能从接口中丢掉。两者都不用 [Corollary 7.2](07-02_stationary_amplitude_law_ZH.md#s7-2) 的平稳性、线性 response 或 mean-field 前提。
- 正因如此，方法记录并据以决策的是 complete learner state（[Definition 1.3](../README_ZH.md#s1-3)），而不只是参数（[Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)）。
- 它在真实模型上的对应是预先注册的状态移植检验（[预注册文档](../../docs/direction/2026-10-14_T1_optimizer_state_transplant_prereg_ZH.md)）。在一个 warm checkpoint 上，用相同的参数分别接上参照分支或备选分支的动量与二阶矩继续训练，按 2×2 设计。结果尚未得到。
- 对 $`\theta=f(q)`$ 这样的参数化（例如 LoRA），局部可达的参数方向是 $`f`$ 的 Jacobian 的值域。这是局部陈述，并不限制多次更新之后 response 的秩。

**推导要点.**

- (a) 反向归纳：$`V^0_R=F_E`$ 只依赖 $`\theta`$，而只读 $`(\theta,k)`$ 的 kernel 把这样的函数映为这样的函数。
- (b) $`S'(g)`$ 的符号就是 $`bc-adg`$ 的符号；Cauchy–Schwarz 不等式给出峰值及其取到的位置。
- (c) 精确的有理数运算：第二步的平方步长 $`m_2^2/v_2`$ 相同，第三步的平方步长 $`m_3^2/v_3`$ 不同。
- (d) 在 $`(g_B,0)`$ 处用隐函数定理（那里 $`S'(g_B)<0`$），再用第三步之差的连续性。
- (e) 关于 $`v`$ 只需一行代数；动量递推是线性的。

**用到.**

- [Theorem 7.1](07-01_state_channel_decomposition_ZH.md#s7-1)：带 bias correction 的 Adam/AdamW 更新与三个通道。
- [Proposition 5.3](05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)：冷启动第一步的见证；本命题把它推广到之后的步与 $`\epsilon>0`$。

**被用于.**

- [Algorithm 9.2](09-02_ced_algorithm_ZH.md#s9-2)：以 complete learner state 作为记录的状态。

## 相关工作

- [Proposition 5.3](05-03_cold_start_adam_sign_quotient_ZH.md#s5-3)(e) 在冷启动的第一步、$`\epsilon=0`$、$`\beta_1=\beta_2=\tfrac12`$ 时给出一个见证：那时 Adam 的步长是符号函数，只有动量还记得梯度的大小。$`\epsilon>0`$ 时冷启动的第一步是梯度的双射，所以那个见证需要 $`\epsilon=0`$。本命题说明，同样的失效也出现在之后的步上，而且用的是标准超参数、带 bias correction、$`\epsilon>0`$，原因是在任何动量不为零的状态下，Adam 的步长都是梯度的二对一函数。
- 以 running RMS 为单位，(b) 中的步长大小 $`S`$ 就是 [Proposition 7.3](07-03_inconsistency_two_proxy_classes_ZH.md#s7-3)(b) 里只看参数的分数 $`h(\lambda)`$，其峰值 $`g^\star`$ 就是那里的 $`\lambda_{\rm PO}`$。Proposition 7.3 用这个分数在 $`\lambda_{\rm PO}`$ 以下递增来说明 Corollary 7.2 模型中的排错；这里则用峰值两侧的先升后降，不借助任何模型，得到即时参数的精确打平。
- 只按当前 Adam 状态下所引起的更新来评价训练样本的分数，读的正是 (c) 固定住的那个量。In-Run Data Shapley for Adam（[Ding et al., 2026](#ref-ding2026inrun)）只为当前的预条件更新打分（[Remark 7.4](07-04_relation_prior_work_role_clipping_ZH.md#s7-4)），所以它给 (c) 中两条消息相同的分数，尽管两者的后续不同。LESS（[Xia et al., 2024](#ref-xia2024less)）比较的是 Adam 预条件梯度的方向，在一维情形下不含任何信息；在高维情形下它读到的不只是即时更新，(c) 对那种情形没有结论。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-ding2026inrun"></a>

**Meng Ding, Zeqing Zhang, Di Wang, Lijie Hu (2026).** In-Run Data Shapley for Adam optimizer. arXiv preprint. arXiv [2602.00329](https://arxiv.org/abs/2602.00329).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-park2023trak"></a>

**Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry (2023).** TRAK: Attributing model behavior at scale. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202:27074–27113. [proceedings.mlr.press/v202/park23c](https://proceedings.mlr.press/v202/park23c.html).

<a id="ref-xia2024less"></a>

**Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen (2024).** LESS: Selecting influential data for targeted instruction tuning. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:54104–54132. [proceedings.mlr.press/v235/xia24c](https://proceedings.mlr.press/v235/xia24c.html); arXiv [2402.04333](https://arxiv.org/abs/2402.04333).
