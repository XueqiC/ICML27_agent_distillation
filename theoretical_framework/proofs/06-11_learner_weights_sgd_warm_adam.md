# Proposition 6.11 · Learner weights of SGD and warm Adam

*Proof*

[← Framework](../README.md) · [Statement](../statements/06-11_learner_weights_sgd_warm_adam.md) · [Proof](../proofs/06-11_learner_weights_sgd_warm_adam.md) · [Experiments](../experiments/06-11_learner_weights_sgd_warm_adam.md) · [中文](06-11_learner_weights_sgd_warm_adam_ZH.md)

**What is proved here.** The influence functions and weights $`W_s`$ of SGD and of warm Adam in the frequent and isolated regimes, and the resulting ratio between learners.

**Setting used on this page.**

- **Agent states** $`s\in\mathcal S`$, a finite set; on this page $`s`$ denotes an agent state ([Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9)).
- **Per-state oracle laws.** At state $`s`$ the oracle message is $`Y\sim P^*_s`$ (for example the teacher's action or correction there), on a finite alphabet; the laws vary independently across states, so nothing is assumed to generalize between states.
- **Influence function** $`f_s(y)`$ of $`\Psi`$ at $`s`$: the derivative of $`\Psi`$ when $`P^*_s`$ is moved towards the point mass at $`y`$; it has mean zero under $`P^*_s`$.
- **Learner-relative weight** $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$: how much one more oracle sample at $`s`$ is worth for estimating $`\Psi`$.
- **Counts.** State $`s`$ receives $`N_s=k_s+n_s`$ oracle samples: $`k_s`$ from demonstrations (in expectation $`KH\,d_T(s)`$, with $`H`$ the trajectory length and $`d_T`$ the teacher's state-visitation distribution) and $`n_s`$ from queries, $`\sum_sn_s=n`$.
- **State-owned coordinates.** Each agent state $`s`$ has its own scalar parameter $`\theta_s`$; each training step visits one state $`s_k\sim d`$ i.i.d., where $`d(s)`$ is the student's visitation probability, and consumes one oracle message there.
- **Gradients.** The consumed gradient $`g=G(s,Y)`$ does not depend on the parameters (linear-response regime); at state $`s`$, $`m_1=\mathbb Eg`$ and $`m_2=\mathbb Eg^2`$.
- **Outcome** $`F=\langle\zeta,\theta_R\rangle`$ after $`R`$ steps with learning rate $`\eta`$; $`\Psi`$ is the expected outcome change.
- **Adam** with momentum $`\beta_1`$, second-moment decay $`\beta_2`$, $`\beta_1<\sqrt{\beta_2}`$ and negligible $`\epsilon`$ ([Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5) gives the exact Adam increment).

**Statement ([Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11)).** Besides [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9), assume:

- every agent state owns its own parameter coordinate (state-owned coordinates);
- each training step draws one state $`s_k\sim d`$ i.i.d. and one oracle message there;
- the consumed gradient does not depend on the parameters (the linear-response regime): $`g=G(s,Y)`$; write $`m_1=\mathbb Eg`$ and $`m_2=\mathbb Eg^2`$ at state $`s`$;
- the outcome is $`F=\langle\zeta,\theta_R\rangle`$, and $`\Psi`$ is the expected outcome change $`\mathbb E F(\theta_R)-F(\theta_0)`$, a functional of $`(P^*_s)_s`$;
- Adam has $`\epsilon`$ negligible relative to $`\sqrt v`$ and $`\beta_1<\sqrt{\beta_2}`$.

Then, to leading order as $`R\to\infty`$ in each stated regime:

- **(a) SGD.** $`f_s^{\rm SGD}(y)=-\eta R\,d(s)\,\zeta_s\,(g-m_1)`$, so $`W_s^{\rm SGD}=\eta^2R^2\zeta_s^2\,d(s)^2\,\mathrm{Var}(g)`$.
- **(b) Warm Adam, frequent regime.** Assume many visits within the effective averaging window of the second moment: $`d(s)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$ with bias correction, and in addition $`t\gg1/(1-\beta_2)`$ without it. Then

```math
f_s^{\rm Adam}(y)=-\eta R\sqrt{d(s)}\,\zeta_s\Big[\frac{g-m_1}{\sqrt{m_2}}-\frac{m_1(g^2-m_2)}{2m_2^{3/2}}\Big],\qquad W_s^{\rm Adam}=\eta^2R^2\zeta_s^2\,d(s)\,\mathrm{Var}\Big(\frac{g}{\sqrt{m_2}}-\frac{m_1g^2}{2m_2^{3/2}}\Big).
```

  The second term in the bracket is the second-moment channel: a message that raises $`m_2`$ lowers every later step size.

- **(c) Warm Adam, isolated regime.** Assume $`d(s)\ll\min\lbrace1-\beta_1,1-\beta_2\rbrace`$, and either no bias correction or $`t\gg1/(1-\beta_2)`$. Each visit then meets $`m\approx0`$ and $`v\approx0`$ and moves the coordinate by $`-\eta\,\kappa_{\rm iso}\,\mathrm{sign}(g)`$, with

```math
\kappa_{\rm iso}=\frac{1-\beta_1}{\sqrt{1-\beta_2}\,\big(1-\beta_1/\sqrt{\beta_2}\big)} .
```

  Hence $`f_s(y)=-\eta R\,d(s)\,\zeta_s\,\kappa_{\rm iso}\,(\mathrm{sign}\,g-\mathbb E\,\mathrm{sign}\,g)`$ and $`W_s\propto d(s)^2\,\mathrm{Var}(\mathrm{sign}\,g)`$: the same scaling in $`d`$ as SGD, but the message enters only through its sign. With bias correction, a visit at step $`t\gg1/(1-\beta_1)`$ to a coordinate whose $`m`$ and $`v`$ are still zero (for example its first visit) moves it by $`-\eta\,\kappa_{\rm iso}\sqrt{1-\beta_2^{\,t}}\,\mathrm{sign}(g)`$; the constant changes, the scaling in $`d`$ does not.

- **(d) Consequence.** Let $`s`$ be a frequent and $`s'`$ a rarer state, $`d(s)>d(s')`$, both in the frequent regime and with the same variance factors. The weight ratio $`W_{s'}/W_s`$ is $`(d(s')/d(s))^2`$ under SGD and $`d(s')/d(s)`$ under warm Adam. Relative to the frequent state, warm Adam therefore weights the rarer state $`d(s)/d(s')`$ times more heavily than SGD does. Demonstrations that cover the frequent states and miss the student's long tail can suffice for SGD and not for warm Adam.

**Results used in the proof.**

- [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9): independent per-state laws and influence functions.
- [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10): parts (c) and (d), for the consequence.

**Classical tools.** The law of large numbers and the delta method ([van der Vaart, 1998](#ref-vandervaart1998asymptotic)). The Adam recursion, with bias correction, is that of [Kingma & Ba, 2015](#ref-kingma2015adam).

**Proof.** Coordinates do not interact, so $`\Psi=\sum_s\zeta_s\,\mathbb E[\theta_{s,R}-\theta_{s,0}]`$, and it suffices to compute, for each $`s`$, the Gateaux derivative of the expected displacement with respect to $`P^*_s`$. Write $`\delta m_1(y)=g(y)-m_1`$ and $`\delta m_2(y)=g(y)^2-m_2`$ for the derivatives of $`m_1,m_2`$ along $`\delta_y-P^*_s`$.

(a) The SGD displacement is $`-\eta\sum_k1\lbrace s_k=s\rbrace g_k`$ with mean $`-\eta Rd(s)m_1`$, linear in $`P^*_s`$; its derivative is $`-\eta Rd(s)\,\delta m_1`$.

(b) Visits form a Bernoulli($`d(s)`$) sequence. The second moment is a weighted average of the per-step squared gradients $`g_k^21\lbrace s_k=s\rbrace`$ over an effective window of $`\min\lbrace t,1/(1-\beta_2)\rbrace`$ steps (after bias correction). In the frequent regime the window contains many visits, so by the law of large numbers $`v_k=d(s)m_2(1+o(1))`$; a single visit changes $`v`$ by the relative amount $`(1-\beta_2)g^2/(d(s)m_2)=o(1)`$, so the correlation between $`m_k`$ and $`v_k`$ is of lower order. The momentum is linear in the gradients and $`\sum_km_k=\sum_jg_j1\lbrace s_j=s\rbrace(1-\beta_1^{R-j})`$, whose mean is $`Rd(s)m_1(1+o(1))`$. Hence the expected displacement is

```math
\Phi(m_1,m_2)=-\eta R\sqrt{d(s)}\,\frac{m_1}{\sqrt{m_2}}\,\big(1+o(1)\big).
```

$`\Phi`$ is a smooth function of $`(m_1,m_2)`$, so by the chain rule (delta method) its influence function is $`\partial_{m_1}\Phi\,\delta m_1+\partial_{m_2}\Phi\,\delta m_2`$ with $`\partial_{m_1}\Phi=-\eta R\sqrt{d}/\sqrt{m_2}`$ and $`\partial_{m_2}\Phi=\tfrac12\eta R\sqrt d\,m_1/m_2^{3/2}`$. Multiplying by $`\zeta_s`$ gives the stated $`f_s`$; the term from $`\partial_{m_2}\Phi`$ is the second-moment channel.

(c) Between visits, $`m`$ and $`v`$ are multiplied by $`\beta_1`$ and $`\beta_2`$ per step; a gap is about $`1/d(s)`$ steps, so $`\beta_1^{1/d}`$ and $`\beta_2^{1/d}`$ are negligible and each visit meets $`m\approx0`$, $`v\approx0`$. At the $`j`$-th step after a visit with gradient $`g`$, $`m=(1-\beta_1)\beta_1^{\,j}g`$ and $`v=(1-\beta_2)\beta_2^{\,j}g^2`$, so the displacement is

```math
-\eta\,\frac{(1-\beta_1)\beta_1^{\,j}g}{\sqrt{(1-\beta_2)\beta_2^{\,j}}\,\lvert g\rvert}=-\frac{\eta(1-\beta_1)}{\sqrt{1-\beta_2}}\Big(\frac{\beta_1}{\sqrt{\beta_2}}\Big)^{j}\mathrm{sign}(g).
```

The geometric series converges because $`\beta_1<\sqrt{\beta_2}`$, and its terms decay within $`O(1/(1-\beta_1/\sqrt{\beta_2}))`$ steps, much less than the gap; summing over $`j\ge0`$ gives $`-\eta\kappa_{\rm iso}\mathrm{sign}(g)`$. The total displacement is $`-\eta\kappa_{\rm iso}\sum_k1\lbrace s_k=s\rbrace\mathrm{sign}(g_k)`$, with mean $`-\eta Rd(s)\kappa_{\rm iso}\mathbb E\,\mathrm{sign}(g)`$, linear in $`P^*_s`$, which gives $`f_s`$. With bias correction, the $`j`$-th step after a visit at step $`t`$ is divided by $`(1-\beta_1^{\,t+j})`$ and multiplied by $`\sqrt{1-\beta_2^{\,t+j}}`$; for $`t\gg1/(1-\beta_1)`$ and $`j`$ within the decay range, these factors are $`1+o(1)`$ and $`\sqrt{1-\beta_2^{\,t}}\,(1+o(1))`$.

(d) Divide the weights of (a) and of (b), then use [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)(c), (d). $`\square`$

**Comments.**

- The two regimes meet near $`d(s)\approx1-\beta_2`$, where a visit's effect on $`v`$ neither averages out nor decays completely. Neither closed form applies there.
- In the isolated regime the displacement depends on the message only through $`\mathrm{sign}(g)`$: the sign-descent aspect of Adam analysed by [Balles & Hennig, 2018](#ref-balles2018dissecting).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-balles2018dissecting"></a>

**Lukas Balles, Philipp Hennig (2018).** Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:404–413. [proceedings.mlr.press/v80/balles18a](https://proceedings.mlr.press/v80/balles18a.html).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-vandervaart1998asymptotic"></a>

**A. W. van der Vaart (1998).** Asymptotic Statistics. Cambridge University Press, Cambridge (Cambridge Series in Statistical and Probabilistic Mathematics 3). DOI [10.1017/CBO9780511802256](https://doi.org/10.1017/CBO9780511802256).
