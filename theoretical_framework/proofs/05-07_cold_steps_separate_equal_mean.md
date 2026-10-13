# Proposition 5.7 · Cold steps separate equal-mean feedback at first order

*Proof*

[← Framework](../README.md) · [Statement](../statements/05-07_cold_steps_separate_equal_mean.md) · [Proof](../proofs/05-07_cold_steps_separate_equal_mean.md) · [Experiments](../experiments/05-07_cold_steps_separate_equal_mean.md) · [中文](05-07_cold_steps_separate_equal_mean_ZH.md)

**What is proved here.** (a) and (b) by computing the expected sign of the consumed gradient under each law; the warm counterpart follows from Proposition 5.6.

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

**Sign map.** For $`\epsilon>0`$, $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$. For $`\epsilon=0`$, the *ideal sign update* is $`h_0(g)_i=\mathrm{sign}(g_i)`$, with $`\mathrm{sign}(0)=0`$. This convention is part of the definition; a literal floating-point Adam with $`\epsilon=0`$ would compute $`0/0`$.

**Utilities.** An *immediate-parameter utility* is a bounded measurable function of $`\theta_1`$. A *complete-state utility* is a bounded measurable function of the next complete state $`(\theta_1,m_1,v_1,1)`$. Consumed-gradient laws $`P,Q`$ are probability laws on $`\mathbb R^p`$. In the finite statements they are supported on a finite alphabet $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$. Put $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$.

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

**Statement ([Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7)).** Take a cold state ($`m=v=0`$, $`t=1`$), the ideal sign map $`h_0`$ (Proposition 5.3), a scalar parameter and the immediate utility $`V(\theta)=\tfrac12+w\theta`$, restricted to a neighbourhood where $`V\in[0,1]`$.

- **(a) Equal means.** Let $`P_0(G=1)=\tfrac14`$, $`P_0(G=-\tfrac13)=\tfrac34`$ and $`P_1(G=0)=1`$. Both laws have mean $`0`$, and $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$.
- **(b) Equal moments to any fixed order.** For $`N\ge1`$, let $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$ for even $`i`$ and $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$ for odd $`i`$, $`0\le i\le N`$ (the laws of [Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order.md#s4-4)), and $`G=q-Z`$ with $`q\in(0,1/N)`$. The two gradient laws agree on all moments of order below $`N`$, and $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$.
- In both cases the separation is first order in $`\eta`$. At a warm state with the same utility, Proposition 5.6 bounds the gap of equal-mean laws by $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$.

*Proof.*

- *(a)* By Proposition 5.3(a) with $`h_0=\mathrm{sign}`$, $`\theta_1=\theta_0-\eta\,\mathrm{sign}(G)`$. Under $`P_0`$, $`\mathbb E\,\mathrm{sign}\,G=\tfrac14-\tfrac34=-\tfrac12`$, so $`\mathbb E\theta_1=\theta_0+\eta/2`$. Under $`P_1`$, $`\mathrm{sign}\,0=0`$, so $`\mathbb E\theta_1=\theta_0`$. The utility is affine, so the response gap is $`\lvert w\rvert\eta/2`$. The means agree: $`\tfrac14\cdot1+\tfrac34\cdot(-\tfrac13)=0`$.
- *(b)* Both $`P_N`$ and $`Q_N`$ are probability laws because $`\sum_{i\ \rm even}\binom Ni=\sum_{i\ \rm odd}\binom Ni=2^{N-1}`$. For $`\ell<N`$, $`\sum_i(-1)^i\binom Ni\,i^\ell=0`$, so their moments of order below $`N`$ agree, and so do those of the affine image $`G=q-Z`$. With $`q\in(0,1/N)`$: under $`P_N`$, $`G>0`$ exactly when $`i=0`$, which has probability $`2^{1-N}`$, and otherwise $`Z\ge2/N>q`$ and $`G<0`$, so $`\mathbb E_{P_N}\mathrm{sign}\,G=2^{2-N}-1`$. Under $`Q_N`$, $`Z\ge1/N>q`$ always, so $`\mathbb E_{Q_N}\mathrm{sign}\,G=-1`$. The expected first steps differ by $`\eta\,2^{2-N}`$, and the affine utility gives the gap $`\lvert w\rvert\eta\,2^{2-N}`$.
- *Warm counterpart.* For $`V(\theta)=\tfrac12+w\theta`$ without a tail, $`F`$ is affine in $`g`$, so $`H_G=0`$; $`V`$ does not depend on $`(m,v)`$, so $`L_v=0`$; and $`L=\lvert w\rvert`$. With equal means, Proposition 5.6 gives the displayed bound. $`\square`$
