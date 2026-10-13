# Proposition 5.5 · Adam 的精确一阶统计量与冻结分母误差

*证明*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-05_exact_first_order_statistic_adam_ZH.md) · [证明](../proofs/05-05_exact_first_order_statistic_adam_ZH.md) · [实验](../experiments/05-05_exact_first_order_statistic_adam_ZH.md) · [English](05-05_exact_first_order_statistic_adam.md)

**本页证明什么.** (a) Adam 的增量，以及一对构造出来的分布：均值与二阶矩都相同，参数块却不同；(b) 逐坐标的冻结分母误差界；(c) 冻结分母下的平均增量是 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ 的仿射函数，由此把两个近似分开。

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

**陈述（[Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam_ZH.md#s5-5)）.** 在 Definition 5.4 的设定下：

- **(a) 精确统计量.** Adam 增量为 $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$，其中 $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$，$`\Delta m=(1-\beta_1)(G-m)`$，$`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$。Proposition 5.1 中的一阶统计量就是精确均值 $`\mu=\mathbb E\,\Delta(z,G)`$。它的参数块是 $`G`$ 的分布的非线性泛函：若 $`\beta_1<1`$，且对某个坐标 $`i`$ 有 $`v_i>0`$ 或 $`\epsilon>0`$，则存在两个有限支撑的分布，$`\mathbb E\,G`$ 相同、$`\mathbb E\,G^{\odot2}`$ 也相同，但参数块 $`\mu_\theta`$ 不同。
- **(b) 冻结分母误差.** 对每个坐标与每个 $`G`$，

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) 两种近似要分开.** frozen-denominator 平均增量 $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ 由 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$ 精确决定，且 $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$，$`m`$ 块与 $`v`$ 块则完全相同。所以"warm Adam 只需要 $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$"这一说法依赖两个不同的近似：
  - Proposition 5.1 的一阶近似，余项由增量的二阶矩控制，分块结构见 Remark 5.2；
  - frozen-denominator 近似，误差由 (b) 控制，随 warmth 变小。

*(a) 的证明.* 增量就是 AdamW 的定义。对第二个结论，固定坐标 $`i`$，令 $`G=g\,e_i`$（$`g`$ 为标量，其余坐标为零）。此时坐标 $`i`$ 的参数块为 $`\mathbb E f(g)`$，其中

```math
f(g)=-\eta\,\frac{\big(\beta_1m_i+(1-\beta_1)g\big)/(1-\beta_1^{\,t})}{\sqrt{\big(\beta_2v_i+(1-\beta_2)g^2\big)/(1-\beta_2^{\,t})}+\epsilon}.
```

- $`f`$ 在 $`\mathbb R`$ 上有界：$`v_i>0`$ 或 $`\epsilon>0`$ 时分母有正的下界；当 $`g\to\pm\infty`$ 时，$`f(g)`$ 趋于 $`\mp\eta(1-\beta_1)\sqrt{1-\beta_2^{\,t}}/\big((1-\beta_1^{\,t})\sqrt{1-\beta_2}\big)`$。因为 $`\beta_1<1`$，两个极限不同，所以 $`f`$ 不是常数。
- 在 $`\mathbb R`$ 上有界的不超过二次的多项式只能是常数，所以 $`f`$ 不是这样的多项式。于是存在四个点 $`g_1<g_2<g_3<g_4`$，$`f`$ 在其上与任何不超过二次的多项式都不一致（过图像上三点的二次多项式会错过某个第四点）。
- 线性映射 $`w\mapsto(\sum_kw_k,\sum_kw_kg_k,\sum_kw_kg_k^2)`$（从 $`\mathbb R^4`$ 到 $`\mathbb R^3`$）有非零核向量 $`w^*`$。它的行空间是 $`\mathrm{span}\lbrace1,g,g^2\rbrace`$ 在这四点上的限制，所以 $`\sum_kw^*_kf(g_k)\ne0`$，否则 $`f`$ 的限制就会落在这个张成空间里。
- 对小的 $`\tau>0`$，$`P=\tfrac14\mathbf 1+\tau w^*`$ 与 $`Q=\tfrac14\mathbf 1-\tau w^*`$ 是这四点上的概率向量（因为 $`\sum_kw^*_k=0`$）。它们的 $`\mathbb E\,g`$ 与 $`\mathbb E\,g^2`$ 相同，因此 $`\mathbb E\,G`$ 与 $`\mathbb E\,G^{\odot2}`$ 相同，而 $`\mathbb E_Pf-\mathbb E_Qf=2\tau\sum_kw^*_kf(g_k)\ne0`$。$`\square`$

*(b) 的证明.*

- *$`v_i>0`$ 的情形.* $`\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\lvert n_i\rvert\,\big\lvert\tfrac1{a_i\sqrt{1+\kappa_i}+\epsilon}-\tfrac1{a_i+\epsilon}\big\rvert`$。通分得到下面的恒等式。由 $`\epsilon\ge0`$，最后一个因子不超过 $`(\sqrt{1+\kappa_i}-1)/\sqrt{1+\kappa_i}=1-(1+\kappa_i)^{-1/2}`$；它小于 $`1`$，又因 $`\sqrt{1+\kappa}-1\le\kappa/2`$ 与 $`\sqrt{1+\kappa}\ge1`$ 而不超过 $`\kappa_i/2`$。所以该因子不超过 $`\min\lbrace \kappa_i/2,1\rbrace`$。

```math
\lvert\theta_i^+-\tilde\theta_i^+\rvert=\eta\,u_i\cdot\frac{a_i(\sqrt{1+\kappa_i}-1)}{a_i\sqrt{1+\kappa_i}+\epsilon}.
```

- *$`v_i=0`$ 且 $`\epsilon>0`$ 的情形.* 此时 $`a_i=0`$，$`\kappa_i=\infty`$，$`u_i=\lvert n_i\rvert/\epsilon`$。两个倒数 $`1/(d_i+\epsilon)`$ 与 $`1/\epsilon`$ 都在 $`[0,1/\epsilon]`$ 中，差不超过 $`1/\epsilon`$，所以左边不超过 $`\eta u_i=\eta u_i\min\lbrace \infty,1\rbrace`$。$`\square`$

*(c) 的证明.* 由 Definition 5.4，$`\tilde\theta^+-\theta=-\eta\lambda_{\rm wd}\theta-\eta\Phi\beta_1m/(1-\beta_1^{\,t})-\eta\Phi(1-\beta_1)G/(1-\beta_1^{\,t})`$ 与 $`m^+-m=(1-\beta_1)(G-m)`$ 关于 $`G`$ 仿射，系数由状态固定；$`v^+-v=(1-\beta_2)(G^{\odot2}-v)`$ 关于 $`G^{\odot2}`$ 仿射。取期望后 $`\tilde\mu`$ 是 $`(\mathbb E\,G,\mathbb E\,G^{\odot2})`$ 的仿射函数。$`\Delta`$ 与 $`\tilde\Delta`$ 的 $`m`$ 块、$`v`$ 块相同，$`\theta`$ 块相差 $`\theta^+-\tilde\theta^+`$，其期望逐坐标由 (b) 控制。$`\square`$

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).
