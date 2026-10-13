# Proposition 5.5 · Adam 的精确一阶统计量与冻结分母误差

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-05_exact_first_order_statistic_adam_ZH.md) · [证明](../proofs/05-05_exact_first_order_statistic_adam_ZH.md) · [实验](../experiments/05-05_exact_first_order_statistic_adam_ZH.md) · [English](05-05_exact_first_order_statistic_adam.md)

**一段话.** 对 Adam，[Proposition 5.1](05-01_optimizer_transformed_expansion_ZH.md#s5-1) 说反馈律的一阶统计量是 Adam 增量的均值。这个命题把它说精确：平均增量是梯度分布的非线性泛函，不由梯度的均值与二阶矩决定；冻结分母后，参数步变成带 preconditioner 的 momentum SGD，逐坐标误差由 warmth 显式控制。因此常见的"warm Adam 只需匹配 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$"依赖两个不同的近似。

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

**Adam 与 AdamW.** Adam（[Kingma & Ba, 2015](#ref-kingma2015adam)）带 bias correction，$`\epsilon`$ 在根号外，可选 decoupled weight decay（AdamW；[Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)），不用 AMSGrad。计数器为 $`t\ge1`$ 的一次更新作用在 $`z=(\theta,m,v,t-1)`$ 与 consumed gradient $`G\in\mathbb R^p`$ 上。逐坐标地，

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- 超参数满足 $`\beta_1\in[0,1)`$，$`\beta_2\in(0,1)`$，$`\eta>0`$，$`\lambda_{\rm wd}\ge0`$，$`\epsilon\ge0`$。运算都是精确实数运算。
- 下一个完整状态为 $`z^+=(\theta^+,m^+,v^+,t)`$。
- *cold* 状态指 $`m=v=0`$ 且 $`t=1`$；*warm* 状态指对每个坐标 $`i`$ 有 $`v_i>0`$。

*Warm 状态的记号（[Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate_ZH.md#s5-4)）.*

考虑一次计数器为 $`t\ge1`$、作用在 $`z=(\theta,m,v,t-1)`$ 上的 Adam 更新，并设 $`v_i>0`$，或在某些 $`v_i=0`$ 时 $`\epsilon>0`$。

- *保留的分母、warmth、相对步长.* 对 consumed gradient $`G`$，

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- 当 $`v_i>0`$ 时，$`d_i=a_i\sqrt{1+\kappa_i}`$，$`v_i^+=\beta_2v_i\,(1+\kappa_i)`$。warmth $`\kappa_i`$ 就是新梯度对保留下来的二阶矩造成的相对扰动。
- *Frozen-denominator surrogate.* 取固定的对角 preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$，

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ 是带固定 preconditioner $`\Phi`$ 的一步 momentum SGD。$`\tilde\Delta`$ 是 *frozen-denominator 增量*：它的 $`(\theta,m)`$ 块关于 $`G`$ 仿射，$`v`$ 块关于 $`G^{\odot2}`$ 仿射。$`\tilde z^+`$ 还把二阶矩冻结在 $`\beta_2v`$，是 Proposition 5.6 中用来比较的状态。对 $`g\in\mathbb R^p`$，$`\tilde z^+(g)`$ 表示同一映射在 consumed gradient 为 $`g`$ 时的值。

<a id="s5-5"></a>

**Proposition 5.5（Adam 的精确一阶统计量与冻结分母误差）.** 在 Definition 5.4 的设定下：

- **(a) 精确统计量.** Adam 增量为 $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$，其中 $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$，$`\Delta m=(1-\beta_1)(G-m)`$，$`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$。Proposition 5.1 中的一阶统计量就是精确均值 $`\mu=\mathbb E\,\Delta(z,G)`$。它的参数块是 $`G`$ 的分布的非线性泛函：若 $`\beta_1<1`$，且对某个坐标 $`i`$ 有 $`v_i>0`$ 或 $`\epsilon>0`$，则存在两个有限支撑的分布，$`\mathbb E\,G`$ 相同、$`\mathbb E\,G^{\odot2}`$ 也相同，但参数块 $`\mu_\theta`$ 不同。
- **(b) 冻结分母误差.** 对每个坐标与每个 $`G`$，

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) 两种近似要分开.** frozen-denominator 平均增量 $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ 由 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ 精确决定，且 $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$，$`m`$ 块与 $`v`$ 块则完全相同。所以"warm Adam 只需要 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$"这一说法依赖两个不同的近似：
  - Proposition 5.1 的一阶近似，余项由增量的二阶矩控制，分块结构见 Remark 5.2；
  - frozen-denominator 近似，误差由 (b) 控制，随 warmth 变小。

**说了什么.** 对 Adam，一阶统计量是 Adam 增量的精确均值，它不由梯度的均值与二阶矩决定。把分母冻结在保留值上，一步 Adam 就变成带 preconditioner 的 momentum SGD，每个坐标的误差至多 $`\eta u_i\min\lbrace\kappa_i/2,1\rbrace`$，$`\kappa_i`$ 是 warmth。

**直觉.** Adam 除以一个 running RMS，而新梯度本身会扰动这个 RMS。扰动小（warmth 小）时，这一步对梯度近似线性。

**在框架中的作用.** 把"warm Adam 只需要 $`(\mathbb E G,\mathbb E G^{\odot2})`$"里藏着的两个近似分开：一阶展开与冻结分母。

**推导要点.**

- 由 $`v^+=\beta_2v(1+\kappa)`$，真实分母为 $`a\sqrt{1+\kappa}`$；比较 $`1/(a\sqrt{1+\kappa}+\epsilon)`$ 与 $`1/(a+\epsilon)`$ 得到因子 $`1-(1+\kappa)^{-1/2}\le\min\lbrace\kappa/2,1\rbrace`$。
- Adam 的一步是每个梯度坐标的有界非常数函数，因此不是二次多项式；矩映射在四个点上的核向量给出前两阶矩相同、平均步长不同的两个分布。

**用到.**

- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion_ZH.md#s5-1) — optimizer 变换后的展开

**被用于.**

- [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine_ZH.md#s5-6) — warm Adam 关于 consumed gradient 一阶仿射
- [Proposition 5.8](../statements/05-08_evolution_warmth_ZH.md#s5-8) — warmth 的演化
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner_ZH.md#s5-10) — 各学习器的一阶统计量
- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam_ZH.md#s6-11) — SGD 与 warm Adam 的学习器权重

## 相关工作

- [Balles & Hennig, 2018](#ref-balles2018dissecting) 把 Adam 的方向解读为平均梯度的符号乘上一个方差自适应因子，于是它通过 $`\mathbb E\,G`$ 与 $`\mathbb E\,G^{\odot2}`$ 依赖梯度分布。Proposition 5.5 说明，对 warm 步的这种描述依赖两个彼此独立的近似：一阶展开与 frozen denominator，并用 warmth 给出第二个近似的误差界。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-balles2018dissecting"></a>

**Lukas Balles, Philipp Hennig (2018).** Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:404–413. [proceedings.mlr.press/v80/balles18a](https://proceedings.mlr.press/v80/balles18a.html).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).
