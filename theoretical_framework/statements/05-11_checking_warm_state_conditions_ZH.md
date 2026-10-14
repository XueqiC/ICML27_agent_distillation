# Remark 5.11 · 在真实模型上检验 warm 状态的条件

*陈述*

[← 框架总图](../README_ZH.md) · [陈述](../statements/05-11_checking_warm_state_conditions_ZH.md) · [实验](../experiments/05-11_checking_warm_state_conditions_ZH.md) · [English](05-11_checking_warm_state_conditions.md)

**一段话.** [Proposition 5.6](05-06_warm_adam_first_order_affine_ZH.md#s5-6) 的 warm 状态误差，是按敏感度加权的逐坐标求和，外加一个由最大 warmth 决定的项。这条 Remark 说明在真实模型上要测什么才能核对这个误差，以及为什么 warmth 的分位数核对不了它。

**本页用到的设定.**

下面重述本页用到的对象，以便单独阅读本页；一般记号见 [Definition 1.3–1.4](../README_ZH.md#s1-3)。

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

<a id="s5-11"></a>

**Remark 5.11（在真实模型上检验 warm 状态的条件）.**

- Proposition 5.6 的 warm 状态误差，是按敏感度加权的和 $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$，再加上 $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$。关于 $`\kappa`$ 分位数的陈述（例如"大多数坐标的 warmth 很小"）对这两项都不构成验证：少数 $`L_iu_i\kappa_i`$ 很大的坐标就能主导这个和，而第二项取决于最大值。
- 所以在真实模型上应当报告加权和 $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ 及其最大的几项，以及 $`\max_i\kappa_i`$。敏感度 $`L_i`$ 是 continuation value 的性质，单看更新是观察不到的；任何替代量 $`\hat L_i`$（例如留出验证梯度的幅度）都是 proxy，必须作为 proxy 报告。
- 测得的更新映射非可加性（$`\Delta`$ 关于消息离仿射有多远）是 optimizer 映射的性质，不是任务 response 的性质。response 是否仿射必须在 response 本身上测，例如用 [Definition 8.6](../statements/08-06_crn_paired_replacement_score_ZH.md#s8-6) 的配对替换分数。
- 由 Proposition 5.8(a)，训练后期典型坐标 warmth 的参照值是平台 $`(1-\beta_2)/\beta_2`$，而不是零。

**说了什么.** 在真实模型上核对 warm 状态条件，应报告加权和 $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$、其中最大的几项以及 $`\max_i\kappa_i`$，而不是 $`\kappa`$ 的分位数。敏感度 $`\hat L_i`$ 只是 proxy；update map 的非可加性不等于 response 的非可加性；训练后期 warmth 的参照值是平台 $`(1-\beta_2)/\beta_2`$。

**在框架中的作用.** 把 warm 状态的结论变成真实模型上的测量方案。在一个真实模型上（[Shridhar et al., 2021](#ref-shridhar2021alfworld) 的 ALFWorld 配方，12B 学生），预注册的 warm 判据在全部六个测量状态上都不成立，因为它的分母把敏感度的绝对值乘上带符号的参数步长，彼此抵消；与带符号一阶项的事后比较（明确标为事后）以及对 response 非可加性的直接测量，见 [Check 5.19](../experiments/05-11_checking_warm_state_conditions_ZH.md#c5-19) 与 [Check 5.20](../experiments/05-11_checking_warm_state_conditions_ZH.md#c5-20)。

## 参考文献

本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献](../references_ZH.md)。

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).
