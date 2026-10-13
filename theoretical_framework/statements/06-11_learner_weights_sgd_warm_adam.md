# Proposition 6.11 · Learner weights of SGD and warm Adam

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-11_learner_weights_sgd_warm_adam.md) · [Proof](../proofs/06-11_learner_weights_sgd_warm_adam.md) · [Experiments](../experiments/06-11_learner_weights_sgd_warm_adam.md) · [中文](06-11_learner_weights_sgd_warm_adam_ZH.md)

**In one paragraph.** This proposition computes the learner-relative weights $`W_s`$ for SGD and warm Adam when each agent state owns a parameter coordinate. SGD weights a state by the square of its visitation; warm Adam in the frequent-visit regime by the visitation itself, with a second-moment term; for very rarely visited states Adam returns to the squared scaling but sees only the sign of the feedback.

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

<a id="s6-11"></a>

**Proposition 6.11 (Learner weights of SGD and warm Adam).** Besides [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9), assume:

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

**What it says.**

- For SGD the weight of a state grows like the square of its visitation; for warm Adam in the frequent regime it grows only linearly and includes a second-moment term; for very rare states Adam returns to the squared scaling but sees only the sign of the feedback.

**Intuition.**

- Adam normalizes by the running second moment, so rarely visited states take steps as large as common ones; their feedback therefore matters more than under SGD.

**Role in the framework.**

- It shows that the optimizer changes where information must be collected, and it supplies the constant $`\kappa_{\rm iso}`$ whose finite-horizon version appears in [Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2).

**Derivation outline.**

- SGD's expected displacement is linear in the state's message law, so its influence function is proportional to $`d(s)(g-m_1)`$.
- In the frequent regime the second moment averages to $`d(s)m_2`$, so the expected displacement is $`-\eta R\sqrt{d(s)}\,m_1/\sqrt{m_2}`$; differentiating with respect to $`m_1`$ and $`m_2`$ (delta method) gives the influence function, the $`m_2`$-derivative being the second-moment channel.
- In the isolated regime both moments have decayed before each visit; the visit's own momentum and second moment then decay geometrically and their ratio sums to $`\kappa_{\rm iso}\,\mathrm{sign}(g)`$.
- Near $`d(s)\approx1-\beta_2`$ neither closed form applies.

**Illustration.** [Example 6.13](../statements/06-13_long_tail_example.md#s6-13): one common state visited with probability $`1-q`$ and $`N`$ tail states each visited with probability $`q/N`$, all in the frequent regime. Relative to the common state, the tail's share of $`\sum_sW_s/N_s`$ is larger under warm Adam than under SGD by the factor $`N(1-q)/q`$, for instance $`80`$ when $`N=20`$ and $`q=0.2`$.

**Uses.**

- [Assumption 6.9](../statements/06-09_state_coverage_model.md#s6-9): state-coverage model.
- [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10): weights enter the information formula.
- [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5): exact Adam increment.
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10): first-order statistic by learner.

**Used by.**

- [Corollary 6.12](../statements/06-12_learner_separations.md#s6-12): long-tail separation.
- [Example 6.13](../statements/06-13_long_tail_example.md#s6-13): worked example.
- [Corollary 7.2](../statements/07-02_stationary_amplitude_law.md#s7-2): finite-horizon isolated constant.
- [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2): learner-relative allocation in CED.

**Status.** proved to leading order in each stated regime.

## Related work

- Adam, with momentum, second-moment decay and bias correction, is as defined in [Kingma & Ba, 2015](#ref-kingma2015adam).
- In the isolated regime each visit moves the coordinate by a fixed multiple of the gradient's sign: the sign-descent aspect of Adam analysed by [Balles & Hennig, 2018](#ref-balles2018dissecting), related to signSGD ([Bernstein et al., 2018](#ref-bernstein2018signsgd)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-balles2018dissecting"></a>

**Lukas Balles, Philipp Hennig (2018).** Dissecting Adam: The sign, magnitude and variance of stochastic gradients. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:404–413. [proceedings.mlr.press/v80/balles18a](https://proceedings.mlr.press/v80/balles18a.html).

<a id="ref-bernstein2018signsgd"></a>

**Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, Animashree Anandkumar (2018).** signSGD: Compressed optimisation for non-convex problems. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:560–569. [proceedings.mlr.press/v80/bernstein18a](https://proceedings.mlr.press/v80/bernstein18a.html).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).
