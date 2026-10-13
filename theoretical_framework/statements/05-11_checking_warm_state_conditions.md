# Remark 5.11 · Checking the warm-state conditions on a real model

*Statement*

[← Framework](../README.md) · [Statement](../statements/05-11_checking_warm_state_conditions.md) · [Experiments](../experiments/05-11_checking_warm_state_conditions.md) · [中文](05-11_checking_warm_state_conditions_ZH.md)

**In one paragraph.** The warm-state error of [Proposition 5.6](05-06_warm_adam_first_order_affine.md#s5-6) is a sensitivity-weighted sum over coordinates plus a term set by the largest warmth. This remark states what must be measured on a real model to check that error, and why quantiles of the warmth do not check it.

**Setting used on this page.**

The objects below are restated so that this page can be read on its own; the general notation is that of [Definitions 1.3–1.4](../README.md#s1-3).

**Adam and AdamW.** Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) is bias-corrected, with $`\epsilon`$ outside the square root, optional decoupled weight decay (AdamW; [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)) and no AMSGrad. An update with counter $`t\ge1`$ acts on $`z=(\theta,m,v,t-1)`$ and a consumed gradient $`G\in\mathbb R^p`$. Coordinatewise,

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- The hyperparameters satisfy $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, $`\eta>0`$, $`\lambda_{\rm wd}\ge0`$, $`\epsilon\ge0`$. Arithmetic is exact real arithmetic.
- The next complete state is $`z^+=(\theta^+,m^+,v^+,t)`$.
- A *cold* state has $`m=v=0`$ and $`t=1`$. A *warm* state has $`v_i>0`$ for every coordinate $`i`$.

*Warm-state notation ([Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate.md#s5-4)).*

Consider an Adam update with counter $`t\ge1`$ at $`z=(\theta,m,v,t-1)`$, and assume $`v_i>0`$, or $`\epsilon>0`$ whenever some $`v_i=0`$.

- *Retained denominator, warmth, relative step.* For a consumed gradient $`G`$,

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- When $`v_i>0`$, $`d_i=a_i\sqrt{1+\kappa_i}`$ and $`v_i^+=\beta_2v_i\,(1+\kappa_i)`$. The warmth $`\kappa_i`$ is the relative perturbation that the new gradient makes to the retained second moment.
- *Frozen-denominator surrogate.* With the fixed diagonal preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$,

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ is one momentum-SGD step with the fixed preconditioner $`\Phi`$. $`\tilde\Delta`$ is the *frozen-denominator increment*: its $`(\theta,m)`$ blocks are affine in $`G`$ and its $`v`$ block is affine in $`G^{\odot2}`$. $`\tilde z^+`$ additionally freezes the second moment at $`\beta_2v`$; it is the comparison state of Proposition 5.6. For $`g\in\mathbb R^p`$, $`\tilde z^+(g)`$ is the same map evaluated at consumed gradient $`g`$.

<a id="s5-11"></a>

**Remark 5.11 (Checking the warm-state conditions on a real model).**

- The warm-state error of Proposition 5.6 is the sensitivity-weighted sum $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ plus the term $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$. A statement about quantiles of $`\kappa`$, such as "most coordinates have small warmth", does not verify either term: a few coordinates with large $`L_iu_i\kappa_i`$ can dominate the sum, and the second term depends on the maximum.
- What should be reported on a real model is therefore the weighted sum $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ and its largest terms, together with $`\max_i\kappa_i`$. The sensitivities $`L_i`$ are properties of the continuation value and are not observable from the update alone; any substitute $`\hat L_i`$, for example the magnitude of a held-out validation gradient, is a proxy and must be reported as one.
- A measured non-additivity of the update map (how far $`\Delta`$ is from affine in the message) is a property of the optimizer map, not of the task response. Whether the response is affine must be measured on the response itself, for example by the paired-replacement scores of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6).
- By Proposition 5.8(a), the relevant reference value for the warmth of a typical coordinate late in training is the plateau $`(1-\beta_2)/\beta_2`$, not zero.

**What it says.** To check the warm-state condition on a real model, report the weighted sum $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$, its largest terms and $`\max_i\kappa_i`$, not quantiles of $`\kappa`$. Sensitivities $`\hat L_i`$ are proxies; non-additivity of the update map is not non-additivity of the response; late in training the reference warmth is the plateau $`(1-\beta_2)/\beta_2`$.

**Role in the framework.** Turns the warm-state results into a measurement protocol for real models.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).
