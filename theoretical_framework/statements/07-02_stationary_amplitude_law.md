# Corollary 7.2 · Stationary amplitude law (restricted mean-field model)

*Statement*

[← Framework](../README.md) · [Statement](07-02_stationary_amplitude_law.md) · [Proof](../proofs/07-02_stationary_amplitude_law.md) · [Experiments](../experiments/07-02_stationary_amplitude_law.md) · [中文](07-02_stationary_amplitude_law_ZH.md)

**In one paragraph.** Under plain SGD, a larger gradient in a useful direction is always better. Under Adam it is not: a large message also inflates the running second moment $`v`$, which shrinks the following steps. This corollary evaluates the state-channel decomposition of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) in the simplest regime where everything can be computed in closed form: one coordinate, a warm stationary optimizer state, tail gradients that do not depend on the parameters. It gives the exact response to a message of any amplitude, an optimal amplitude, a sign change, and a criterion for very large amplitudes. It is a restricted model of the second-moment channel, not one of the framework's general claims: its premises are strong, and its real-model prediction was not supported ([Check 7.6](../experiments/07-02_stationary_amplitude_law.md#c7-6)).

**Setting used on this page.**

- *Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) without bias correction, one coordinate $`i`$.* Moments $`m,v`$, $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, learning rate $`\eta`$, $`\epsilon=0`$, no weight decay. A consumed gradient $`g`$ updates $`m'=\beta_1m+(1-\beta_1)g`$, $`v'=\beta_2v+(1-\beta_2)g^2`$ and moves the parameter by $`-\eta m'/\sqrt{v'}`$.
- *Message and tail.* At round $`t`$ a message with consumed gradient $`g_i`$ is applied; then $`T=R-1-t`$ tail updates follow. The final score is $`F`$, and $`(\nabla F)_i`$ is its derivative in coordinate $`i`$.
- *Stationary linear-response regime.* The tail gradients $`g_{k,i}`$ do not depend on $`\theta`$, are stationary with mean $`\bar g_i\ne0`$ and mean square $`\bar v_i>0`$, and $`F`$ is affine in $`\theta_i`$.
- *Warm state.* The message arrives at $`m_{t,i}=\bar g_i`$, $`v_{t,i}=\bar v_i`$.
- *Normalized quantities.*
  - $`\alpha_i=-(\nabla F)_i\,\mathrm{sign}(\bar g_i)`$, so $`\alpha_i>0`$ when moving against $`\bar g_i`$ improves the outcome;
  - $`\lambda=g_i\,\mathrm{sign}(\bar g_i)/\sqrt{\bar v_i}`$, the same-direction amplitude of the message in units of the running RMS $`\sqrt{\bar v_i}`$;
  - $`\rho_i=\lvert\bar g_i\rvert/\sqrt{\bar v_i}\in(0,1]`$, the signal-to-noise ratio of the coordinate;
  - $`b_j=(1-\beta_1)\beta_1^{\,j}`$, $`c_j=(1-\beta_2)\beta_2^{\,j}`$ for $`j=0,\dots,T`$ ($`j=0`$ is the message update), and $`\omega_T=1-\beta_2^{\,T+1}=\sum_{j\le T}c_j`$.
- *Mean-field response.* $`\Delta F_i(\lambda)`$ is the change of $`F`$ when a message of amplitude $`\lambda`$ replaces a zero-gradient message, computed with each step $`m_k/\sqrt{v_k}`$ replaced by $`\mathbb Em_k/\sqrt{\mathbb Ev_k}`$, the expectations being over the tail gradients.
- *First-order value.* $`\mathrm{Val}_i(\lambda)`$ is the first-order value of the message in coordinate $`i`$ ([Theorem 7.1](07-01_state_channel_decomposition.md#s7-1)).
- $`\kappa_{\rm iso}=(1-\beta_1)/\big(\sqrt{1-\beta_2}\,(1-\beta_1/\sqrt{\beta_2})\big)`$ is the isolated-visit constant of [Proposition 6.11(c)](06-11_learner_weights_sgd_warm_adam.md#s6-11).

<a id="s7-2"></a>

**Corollary 7.2 (Stationary amplitude law; restricted mean-field model).** Assume, for a coordinate $`i`$:

- the tail gradients $`g_{k,i}`$ do not depend on $`\theta`$ (linear-response regime, $`J_k=0`$), are stationary with mean $`\bar g_i\ne0`$ and mean square $`\bar v_i>0`$, and $`F`$ is affine in $`\theta_i`$;
- $`\epsilon=0`$, no bias correction, $`\lambda_{\rm wd}=0`$;
- the message arrives at a warm state $`m_{t,i}=\bar g_i`$, $`v_{t,i}=\bar v_i`$; there are $`T=R-1-t`$ tail updates after the message update.

With $`\alpha_i,\lambda,\rho_i,b_j,c_j,\omega_T`$ and the mean-field response $`\Delta F_i(\lambda)`$ as defined in the setting above:

**(a) Exact mean-field response, every $`\lambda`$.**

```math
\Delta F_i(\lambda)=\alpha_i\eta\sum_{j=0}^{T}\Big[\frac{\rho_i+b_j(\lambda-\rho_i)}{\sqrt{1+c_j(\lambda^2-1)}}-\frac{\rho_i(1-b_j)}{\sqrt{1-c_j}}\Big].
```

**(b) Small increments: the quadratic law.** Let $`Q_i(\lambda)=(1-\beta_1^{\,T+1})\lambda-\tfrac12\rho_i\omega_T\lambda^2`$ and $`S_i(\lambda)=\lvert\lambda\rvert(1-\beta_1^{\,T+1})+\tfrac12\rho_i\omega_T\lambda^2`$. If $`\beta_2\ge\tfrac12`$, then for every $`\lambda`$

```math
\big\lvert\Delta F_i(\lambda)-\alpha_i\eta\,Q_i(\lambda)\big\rvert\le11\,(1-\beta_2)\max\lbrace1,\lambda^2\rbrace\;\lvert\alpha_i\rvert\eta\,S_i(\lambda).
```

The same bound holds for the mean-field first-order value difference $`\mathrm{Val}_i(\lambda)-\mathrm{Val}_i(0)`$. Consequently:

- $`Q_i`$ is concave and is maximized at $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho_i\omega_T)`$.
- Let $`u=11(1-\beta_2)\max\lbrace1,9\lambda^{\star2}\rbrace`$ and $`\alpha_i>0`$. If $`u\le1/8`$, then $`\Delta F_i>0`$ on $`[\lambda^\star,\,2\lambda^\star(1-4u)]`$ and $`\Delta F_i<0`$ on $`[2\lambda^\star(1+4u),\,3\lambda^\star]`$. So the sign change lies within relative distance $`4u`$ of $`2\lambda^\star`$, and a same-direction message just above it is worse than a zero-gradient message.
- The threshold $`2\lambda^\star`$ is therefore usable only when $`(1-\beta_2)(2\lambda^\star)^2\ll1`$. Otherwise the sign change, if any, must be computed from (a) and (c).
- Short remaining horizon, $`T(1-\beta_2)\ll1`$. Here $`\omega_T\approx(T+1)(1-\beta_2)`$, so $`\lambda^\star`$ is large and $`(1-\beta_2)(2\lambda^\star)^2\approx4(1-\beta_1^{\,T+1})^2/\big(\rho_i^2(T+1)^2(1-\beta_2)\big)`$. Unless this is small, $`2\lambda^\star`$ lies outside the range of (b).
- Long remaining horizon, $`T(1-\beta_2)\gg1`$. Here $`\omega_T\to1`$, $`\lambda^\star\to(1-\beta_1^{\,T+1})/\rho_i\approx1/\rho_i`$, and $`(1-\beta_2)(2\lambda^\star)^2\approx4(1-\beta_2)/\rho_i^2`$, which is small unless the coordinate is very noisy.

**(c) Large amplitudes.** As $`\lambda\to\infty`$,

```math
\Delta F_i(\lambda)\to\alpha_i\eta\big[\kappa_{\rm iso}(T)-\rho_i\,\Lambda(T)\big],\qquad \kappa_{\rm iso}(T)=\sum_{j=0}^{T}\frac{b_j}{\sqrt{c_j}},\qquad \Lambda(T)=\sum_{j=0}^{T}\frac{1-b_j}{\sqrt{1-c_j}} .
```

- Let $`\alpha_i>0`$. If $`\rho_i\Lambda(T)>\kappa_{\rm iso}(T)`$, every sufficiently large same-direction message is worse than a zero-gradient message, and $`\Delta F_i`$ has a positive zero. If $`\rho_i\Lambda(T)<\kappa_{\rm iso}(T)`$, every sufficiently large same-direction message is better. The exact critical signal-to-noise ratio is $`\kappa_{\rm iso}(T)/\Lambda(T)`$.
- Since $`\Lambda(T)=(T+1)\big(1+O(\tfrac1{T+1}+(1-\beta_2))\big)`$, the criterion is approximately $`\rho_i(T+1)>\kappa_{\rm iso}(T)`$.
- $`\kappa_{\rm iso}(T)`$ is the finite-horizon version of $`\kappa_{\rm iso}`$, to which it converges as $`T\to\infty`$ when $`\beta_1<\sqrt{\beta_2}`$.
- Reading. A very large message sets its own denominator, so its direct effect saturates near $`\eta\kappa_{\rm iso}(T)`$. Meanwhile it depresses every later step: at lag $`j`$ the comparator's step is about $`\rho_i`$ and the message's step is about $`\rho_i/\sqrt{1+c_j\lambda^2}`$. This lasts while $`c_j\lambda^2\gtrsim1`$, i.e. for about $`\log((1-\beta_2)\lambda^2)/(1-\beta_2)`$ steps, capped at $`T+1`$.

**(d) Comparison with SGD.** For SGD in the same regime, $`\Delta F_i(\lambda)=\alpha_i\eta\sqrt{\bar v_i}\,\lambda`$ is linear: a larger same-direction message is always better.

**What it says.**

- The response of the final outcome to one message under warm Adam has an exact closed form in the message amplitude $`\lambda`$, valid for every $`\lambda`$ under the mean-field substitution.
- For small increments the response is a concave quadratic with an explicit error bound: it peaks at $`\lambda^\star`$ and changes sign near $`2\lambda^\star`$.
- For very large amplitudes the response has an explicit limit; its sign decides whether huge messages are harmful.

**Intuition.**

- A big gradient inflates $`v`$ and so shrinks many following steps. Beyond some size the shrinkage costs more than the big step gains.
- The effect is strongest for coordinates with a high signal-to-noise ratio $`\rho_i`$, whose later steps are worth the most.

**Role in the framework.**

- It is a mechanism hypothesis that is exact inside its model and nowhere else. The model is one coordinate at a warm stationary state, linear response (tail gradients that do not depend on $`\theta`$, $`F`$ affine), $`\epsilon=0`$, no bias correction, and the mean-field substitution of $`\mathbb Em/\sqrt{\mathbb Ev}`$ for $`\mathbb E[m/\sqrt v]`$. Each of these is a restriction, not a property of training in general.
- Its real-model prediction was not supported. At the state used by the real-model test, one batch was applied at five magnitudes: the measured responses were positive and inside their diagnostic radii at every magnitude, while the prediction obtained by plugging the saved optimizer state into (a) was negative and 28 to 339 times larger in absolute value ([Check 7.6](../experiments/07-02_stationary_amplitude_law.md#c7-6)). The most likely failed premise is linear response; that is a hypothesis, not a tested cause.
- What does not depend on it: the exact three-channel decomposition of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1) and the exact memory separation of [Proposition 7.6](07-06_equal_immediate_parameters_different_continuation.md#s7-6), which together form Theorem B and use none of these premises.
- Inside its model it is the first quantitative statement in the framework that, under Adam, the value of a message is non-monotone in its magnitude, and it is the setting in which [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3) proves that two families of influence scores misrank. Where the method flags messages that are large relative to the running second moment ([Algorithm 9.2](09-02_ced_algorithm.md#s9-2), step 4), the flag is a screening heuristic motivated by this model; ranking is left to finite measurement ([Theorem 8.8](08-08_crn_paired_replacement_certificate.md#s8-8)).

**Derivation outline.**

- Under the mean-field substitution the message's deviation decays geometrically: $`\mathbb Em_{t+j}=\bar g+b_j(g-\bar g)`$ and $`\mathbb Ev_{t+j}=\bar v(1+c_j(g^2/\bar v-1))`$. Summing the $`T+1`$ steps $`\mathbb Em/\sqrt{\mathbb Ev}`$, with and without the message, gives (a).
- (b) expands $`(1+c_j(\lambda^2-1))^{-1/2}`$ with explicit mean-value bounds on the first- and second-order remainders. The cross terms $`\sum_jb_jc_j`$ are absorbed into the constant.
- (c) is the termwise limit: the first fraction tends to $`b_j/\sqrt{c_j}`$. A negative limit together with $`\Delta F'(0)>0`$ forces a positive zero.

**Illustration (analytic, from the formula).** Take $`\beta_1=0.9`$, $`\beta_2=0.99`$ and $`\rho=0.5`$.

- With $`T=1000`$ remaining updates, $`\lambda^\star=2`$. The exact formula peaks at $`\lambda\approx1.96`$ and changes sign at $`\lambda\approx3.98`$, next to $`2\lambda^\star=4`$. A message twice the running RMS is best; one four times larger than the RMS is worth less than no message.
- With $`T=20`$, $`2\lambda^\star\approx18.7`$ but the exact sign change is at $`\lambda\approx112`$, where the quadratic law no longer applies. The large-amplitude limit is negative, but only slightly ($`\rho\Lambda\approx10.1`$ against $`\kappa_{\rm iso}\approx9.2`$).

**Uses.**

- [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1): the three-channel decomposition and its linear-response costates.
- [Proposition 6.11](06-11_learner_weights_sgd_warm_adam.md#s6-11): the isolated-visit constant $`\kappa_{\rm iso}`$.

**Used by.**

- [Proposition 7.3](07-03_inconsistency_two_proxy_classes.md#s7-3): where parameter-only and cosine scores misrank.
- [Remark 7.4](07-04_relation_prior_work_role_clipping.md#s7-4): clipping as a cap on $`\lambda`$.
- [Remark 7.5](07-05_where_first_order_rankers_fail.md#s7-5): the saturating regime where first-order rankers fail.
- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): magnitude control (step 4).

## Related work

- Adam's step in one coordinate is governed by that coordinate's signal-to-noise ratio; this is the basis of the reading of Adam as a sign step scaled by a variance-adaptation factor ([Balles & Hennig, 2018](#ref-balles2018dissecting)). Here the same ratio $`\rho_i`$ decides whether a large message helps or harms the later steps.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-balles2018dissecting"></a>

**Lukas Balles, Philipp Hennig (2018).** Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:404–413. [proceedings.mlr.press/v80/balles18a](https://proceedings.mlr.press/v80/balles18a.html).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).
