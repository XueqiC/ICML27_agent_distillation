# §6 Theorem A: the learner-relative information bound — Statements

[← Framework](../README.md) · [Proofs](../proofs/06_learner_relative_information.md) · [Experiments](../experiments/06_learner_relative_information.md) · [中文](06_learner_relative_information_ZH.md)

### §6 · Theorem A: the learner-relative information bound — exact statements

[Proofs](../proofs/06_learner_relative_information.md) · [Experiments](../experiments/06_learner_relative_information.md)

**What this section does.**

- It measures how much $`K`$ demonstrations and $`n`$ queries reveal about the oracle feedback in the one direction that matters for the actual learner and the declared task, and shows that this learner-relative information norm sets the best achievable first-order fidelity of any distiller, up to a factor between $`1/4`$ and $`1`$ (Theorem A: [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3) and [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5)).
- It identifies the optimal queries as a c-optimal design whose target vector is set by the optimizer ([Lemma 6.4](../proofs/06_learner_relative_information.md#s6-4)).
- It derives aliasing, the zero value of replay and a demonstration–query exchange rate in learner-relative form ([Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7), [Corollary 6.8](../proofs/06_learner_relative_information.md#s6-8)).
- For the state-coverage structure of agent distillation it gives explicit formulas, the water-filling query allocation and the learner weights of SGD and warm Adam, and shows that the same demonstrations can be sufficient for one learner and not for another ([Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10)–[Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12)).

**Setting ([Definition 6.1](../proofs/06_learner_relative_information.md#s6-1), [Definition 6.2](../proofs/06_learner_relative_information.md#s6-2)).** Environments $`E_\theta`$, $`\theta\in\Theta\subseteq\mathbb R^p`$, have oracle feedback laws $`P^*_\theta`$. The data are $`K`$ i.i.d. demonstrations from $`P^D_\theta`$ and $`n`$ possibly adaptive queries with outcomes from $`P^Q_\theta(\cdot\mid x)`$, $`x\in\mathcal X_Q`$. A distiller is any map from the transcript and $`\theta`$-free private randomness to a feedback law $`Q`$; replay, rewriting and synthesis are distillers. At fixed audited complete learner states $`z_1,\dots,z_T`$ the oracle mean increment is $`\mu^*_t(\theta)=\mu^{P^*_\theta}_t(z_t)`$, the learner Jacobian is $`J_t=D_\theta\mu^*_t`$, and the declared task fixes the continuation-value gradients $`g_t=\nabla V_{t+1}(z_t)`$. Then

```math
\Psi(\theta)=\sum_t\langle g_t,\mu^*_t(\theta)\rangle,\qquad c=\nabla\Psi=\sum_tJ_t^*g_t,\qquad \mathrm{Err}_\theta(Q)=\Big\lvert\Psi(\theta)-\sum_t\langle g_t,\mu^Q_t(z_t)\rangle\Big\rvert,
```

```math
\mathbb I_{K,n}(\theta;h)=K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h,\qquad \mathfrak n_{K,n}(\theta)=\sup_{h\neq0}\frac{\lvert c^\top h\rvert}{\sqrt{\mathbb I_{K,n}(\theta;h)}} .
```

$`\mathrm{Err}`$ is the first-order part of the outcome gap between $`Q`$ and the oracle feedback at the audited states ([Proposition 5.1](../proofs/05_optimizer_state.md#s5-1)); $`\mathfrak n_{K,n}`$ is the learner-relative information norm. Some asymptotic statements assume the growth condition (G): $`aN\le\mathbb I_{K,n}(\theta_0;h)\le bN`$ for unit $`h`$, with $`N\to\infty`$.

#### Theorem 6.3 · Learner-relative lower bound

**(a)** For all $`\theta_0,\theta_1`$ and every distiller,

```math
\max_{i}\mathbb E_{\theta_i}\mathrm{Err}_{\theta_i}(Q)\ge\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert\Big(1-\sqrt{\tfrac12\big(K\,\mathrm{KL}_D+n\sup_x\mathrm{KL}_Q(x)\big)}\Big).
```

**(b)** Under local regularity (differentiability in quadratic mean, uniform KL expansions), for a unit direction $`h`$ with $`c^\top h\neq0`$ and $`\tau=\mathbb I_{K,n}(\theta_0;h)^{-1/2}\to0`$,

```math
\inf_Q\max_{\theta\in\lbrace\theta_0,\theta_0+\tau h\rbrace}\mathbb E_\theta\mathrm{Err}_\theta(Q)\ge\frac{\lvert c^\top h\rvert}{4\sqrt{\mathbb I_{K,n}(\theta_0;h)}}\,(1-o(1)).
```

**(c)** Under (G), the local minimax lower bound is $`\tfrac14\mathfrak n_{K,n}(\theta_0)(1-o(1))`$.

*Derivation outline.*

- The distiller's own contribution $`\sum_t\langle g_t,\mu^Q_t(z_t)\rangle`$ does not depend on $`\theta`$, so by the triangle inequality the errors in the two worlds sum to at least $`\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$.
- Le Cam's two-point argument: on the overlap of the two transcript laws no distiller can be right in both worlds, so the sum of expected errors is at least the gap times $`1-\mathrm{TV}`$.
- Pinsker's inequality and the chain rule for KL along the adaptive transcript: demonstrations contribute $`K\,\mathrm{KL}_D`$, each query at most $`\sup_x\mathrm{KL}_Q(x)`$, and the query choices contribute nothing because they follow the same rule in both worlds.
- Locally, KL is quadratic in the displacement with the Fisher informations, and the choice $`\tau=\mathbb I^{-1/2}`$ keeps the total variation at most $`1/2`$, which yields the constant $`1/4`$.

*Illustration.* With one parameter, $`\Psi(\theta)=c\theta`$, demonstrations of Fisher information $`i_D`$ each and queries of information $`i_Q`$ each, $`\mathfrak n_{K,n}=\lvert c\rvert/\sqrt{Ki_D+ni_Q}`$. A learner with a larger sensitivity $`\lvert c\rvert`$ needs proportionally more information for the same error.

*Uses* [Definition 6.2](../proofs/06_learner_relative_information.md#s6-2), [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1). *Used by* [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5), [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7), [Corollary 6.8](../proofs/06_learner_relative_information.md#s6-8), [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/06_learner_relative_information.md#s6-3) · [Experiments](../experiments/06_learner_relative_information.md#c6-2) · proved

#### Lemma 6.4 · Design minimax equality

For a compact query space with continuous $`I_Q(\cdot)`$, approximate designs $`\xi`$, $`M(\xi)=\int I_Q(x)\,\xi(\mathrm dx)`$ and $`A(\xi)=KI_D+nM(\xi)\succ0`$,

```math
\inf_\xi c^\top A(\xi)^{-1}c=\sup_{h\neq0}\frac{(c^\top h)^2}{K\,h^\top I_Dh+n\sup_xh^\top I_Q(x)h}=\mathfrak n_{K,n}^2 ,
```

and the infimum is attained by the c-optimal design $`\xi^*`$ for the target vector $`c=\sum_tJ_t^*g_t`$. With $`n=0`$ and singular $`I_D`$ both sides are $`c^\top(KI_D)^+c`$ if $`c\in\mathrm{range}(I_D)`$ and $`+\infty`$ otherwise.

*Derivation outline.*

- For a fixed design, Cauchy–Schwarz writes $`c^\top A^{-1}c`$ as a supremum over directions $`h`$ of $`(c^\top h)^2/h^\top Ah`$.
- For a fixed direction, the best design puts all its mass on the most informative query point, which produces the $`\sup_x`$ term.
- The function $`h^\top A(\xi)h`$ is linear in the design and convex in the direction, and the designs form a compact convex set, so Sion's minimax theorem exchanges the two optimizations.

*Uses* [Definition 6.2](../proofs/06_learner_relative_information.md#s6-2). *Used by* [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5), [Remark 6.6](../proofs/06_learner_relative_information.md#s6-6).

[Proof](../proofs/06_learner_relative_information.md#s6-4) · [Experiments](../experiments/06_learner_relative_information.md#c6-1) · proved

#### Theorem 6.5 · Achievability; tight to constants in the scalar case

Assume (G), the regularity of [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3)(b), realizability (R) of the oracle law at any $`\theta'`$, identifiability (I) by some design, smoothness (S) of $`\Psi`$, and efficient two-stage estimation (E). Query with the regularized c-optimal design $`\xi_n=(1-\varepsilon_n)\xi^*+\varepsilon_n\xi_0`$, $`\varepsilon_n\to0`$, $`\varepsilon_n\sqrt n\to\infty`$, and output $`Q=P^*_{\hat\theta}`$. Then

```math
\mathbb E_\theta\mathrm{Err}_\theta(Q)\le\mathfrak n_{K,n}(\theta)\,(1+o(1))+o(N^{-1/2}),
```

and the last term is $`o(\mathfrak n_{K,n})`$ when $`c\neq0`$. With [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3)(c), the smallest achievable first-order error lies between $`\tfrac14\mathfrak n_{K,n}`$ and $`\mathfrak n_{K,n}`$ up to factors $`1\pm o(1)`$.

*Derivation outline.*

- The error of the plug-in output is $`\lvert\Psi(\hat\theta)-\Psi(\theta)\rvert`$, bounded by the linear term $`\lvert c^\top(\hat\theta-\theta)\rvert`$ plus a quadratic remainder.
- The regularized design loses only a factor $`1/(1-\varepsilon_n)`$ in $`c^\top A^{-1}c`$, so the linear term is $`\mathfrak n_{K,n}(1+o(1))`$ by Jensen's inequality and [Lemma 6.4](../proofs/06_learner_relative_information.md#s6-4).
- The small nondegenerate component makes $`\hat\theta`$ consistent in every direction, so the remainder is $`o(N^{-1/2})`$; c-optimal designs alone are often singular and would not guarantee this.

*Illustration.* In a Gaussian linear experiment the plug-in error is exactly $`\sqrt{2/\pi}\,\mathfrak n_{K,n}\approx0.80\,\mathfrak n_{K,n}`$ at the optimal design, inside the band $`[1/4,1]`$.

*Uses* [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3), [Lemma 6.4](../proofs/06_learner_relative_information.md#s6-4). *Used by* [Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/06_learner_relative_information.md#s6-5) · [Experiments](../experiments/06_learner_relative_information.md#c6-2) · proved

#### Remark 6.6 · Class-uniform fidelity

When fidelity is required uniformly over a declared class $`\mathcal V`$ of continuation values, the lower bound becomes $`\tfrac14\sup_{V\in\mathcal V}\mathfrak n_{K,n}(c_V)`$, and a plug-in distiller with a mixture of c-optimal designs reaches $`d^{3/2}\sup_V\mathfrak n_{K,n}(c_V)(1+o(1))`$, where $`d=\dim\mathrm{span}\lbrace c_V\rbrace\le\dim\mathrm{span}\,\mathcal G`$. The two bounds differ by at most $`4d^{3/2}`$.

*Derivation outline.* The triangle inequality holds for the supremum over the class, which gives the lower bound. For the upper bound, John's theorem sandwiches the class seminorm between an ellipsoidal norm and $`\sqrt d`$ times it; a uniform mixture of the c-optimal designs for the ellipsoid's $`d`$ axes loses at most a factor $`d`$ in each variance.

[Proof](../proofs/06_learner_relative_information.md#s6-6) · no separate check · proved (with the factor $`4d^{3/2}`$)

#### Corollary 6.7 · Aliasing and the zero value of replay

If $`P^D_{\theta_0}=P^D_{\theta_1}`$, $`\Psi(\theta_0)\neq\Psi(\theta_1)`$, and there are no queries (or the query laws also coincide), then for every $`K`$ every distiller has worst expected error at least $`\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$. Replay, rewriting and synthesis are distillers and cannot lower this bound. Whether demonstrations suffice depends on whether the directions they cannot see have a component along this learner's $`c`$.

*Derivation outline.* Identical transcript laws give total variation zero in [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3)(a).

*Uses* [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3). Learner-relative form of [Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6)(a) and [Proposition 2.4](../proofs/02_feasibility_and_acquisition.md#s2-4). *Used by* [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10), [Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/06_learner_relative_information.md#s6-7) · [Experiments](../experiments/06_learner_relative_information.md#c6-3) · proved

#### Corollary 6.8 · Exchange rate between demonstrations and queries

With $`e_D(h)=h^\top I_Dh`$ and $`e_Q(h)=\sup_xh^\top I_Q(x)h`$, a distiller with local worst-case first-order error at most $`\varepsilon`$ must have, for every direction $`h`$,

```math
K\,e_D(h)+n\,e_Q(h)\ge\frac{(c^\top h)^2}{16\,\varepsilon^2}\,(1-o(1)).
```

At the binding direction $`h^\star`$ one demonstration is worth $`e_D(h^\star)/e_Q(h^\star)`$ queries; $`h^\star`$, and hence the exchange rate, depends on the learner.

*Derivation outline.* Rearrange [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3)(b) direction by direction.

*Uses* [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3). Learner-relative form of [Proposition 2.11](../proofs/02_feasibility_and_acquisition.md#s2-11). *Used by* [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/06_learner_relative_information.md#s6-8) · no separate check (a rearrangement of Theorem 6.3) · proved

#### Assumption 6.9, Proposition 6.10 · State coverage and water-filling

**[Assumption 6.9](../proofs/06_learner_relative_information.md#s6-9).** Finitely many agent states $`s`$; at state $`s`$ the oracle message has law $`P^*_s`$, the laws vary independently across states, and $`\Psi`$ is a differentiable functional of $`(P^*_s)_s`$ with influence function $`f_s`$. State $`s`$ receives $`N_s=k_s+n_s`$ oracle samples: $`k_s`$ from demonstrations (in expectation $`KH\,d_T(s)`$, with $`d_T`$ the teacher's state visitation) and $`n_s`$ from queries.

**[Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10).** Let $`W_s=\mathrm{Var}_{P^*_s}f_s(Y)`$, the learner-relative weight of state $`s`$.

- **(a)** Every distiller has local minimax first-order error at least $`\tfrac14\big(\sum_sW_s/N_s\big)^{1/2}(1-o(1))`$.
- **(b)** Replaying the acquired messages at each state (the per-state empirical law) achieves $`\big(\sum_sW_s/N_s\big)^{1/2}(1+o(1))`$.
- **(c)** The optimal allocation of $`n`$ queries is water-filling:

```math
n_s^\star=\Big(\sqrt{W_s/\lambda}-k_s\Big)_+,\qquad \textstyle\sum_sn_s^\star=n .
```

- **(d)** A state with $`W_s>0`$ and $`k_s=n_s=0`$ leaves an error floor that no $`K`$ removes.

*Derivation outline.*

- (a) A tilting submodel $`(1+\tau h_sf_s)P^*_s`$ has derivative $`\sum_sh_sW_s`$ and Fisher information $`\sum_sN_sh_s^2W_s`$; [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3) and Cauchy–Schwarz, with the optimal tilt $`h_s=1/N_s`$, give the bound.
- (b) The plug-in error is a sum of independent per-state sample means of the influence function plus a remainder of order $`1/N`$.
- (c) The objective $`\sum_sW_s/(k_s+n_s)`$ is convex; the KKT conditions equalize the marginal gain $`W_s/(k_s+n_s)^2`$ across queried states.
- (d) Nothing observed depends on $`P^*_s`$, so [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7) applies.

*Illustration.* Two states with equal teacher coverage, one with weight four times the other: water-filling gives the heavier state twice as many total samples ($`\sqrt4=2`$), not four times as many.

*Uses* [Theorem 6.3](../proofs/06_learner_relative_information.md#s6-3), [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7). *Used by* [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11), [Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/06_learner_relative_information.md#s6-10) · [Experiments](../experiments/06_learner_relative_information.md#c6-5) · proved

#### Proposition 6.11 · Learner weights of SGD and warm Adam

Assume state-owned parameter coordinates, one state $`s_k\sim d`$ visited per step, gradients $`g=G(s,Y)`$ that do not depend on the parameters (linear-response regime), outcome $`F=\langle\zeta,\theta_R\rangle`$, and $`m_1=\mathbb Eg`$, $`m_2=\mathbb Eg^2`$. To leading order:

- **(a) SGD.** $`W_s^{\rm SGD}=\eta^2R^2\zeta_s^2\,d(s)^2\,\mathrm{Var}(g)`$.
- **(b) Warm Adam, frequent regime** (many visits per second-moment window, $`d(s)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$):

```math
W_s^{\rm Adam}=\eta^2R^2\zeta_s^2\,d(s)\,\mathrm{Var}\Big(\frac{g}{\sqrt{m_2}}-\frac{m_1g^2}{2m_2^{3/2}}\Big),
```

  where the second term is the second-moment channel.

- **(c) Warm Adam, isolated regime** ($`d(s)\ll\min\lbrace1-\beta_1,1-\beta_2\rbrace`$): each visit moves the coordinate by $`-\eta\kappa_{\rm iso}\mathrm{sign}(g)`$ with $`\kappa_{\rm iso}=(1-\beta_1)/\big(\sqrt{1-\beta_2}(1-\beta_1/\sqrt{\beta_2})\big)`$, so $`W_s\propto d(s)^2\mathrm{Var}(\mathrm{sign}\,g)`$. With bias correction a first visit at step $`t`$ carries the extra factor $`\sqrt{1-\beta_2^{\,t}}`$.
- **(d)** For two states in the frequent regime, warm Adam weights the rarer state, relative to the frequent one, $`d(s)/d(s')`$ times more heavily than SGD does.

*Derivation outline.*

- SGD's expected displacement is linear in the state's message law, so its influence function is proportional to $`d(s)(g-m_1)`$.
- In the frequent regime the second moment averages to $`d(s)m_2`$, so the expected displacement is $`-\eta R\sqrt{d(s)}\,m_1/\sqrt{m_2}`$; differentiating with respect to $`m_1`$ and $`m_2`$ (delta method) gives the influence function, the $`m_2`$-derivative being the second-moment channel.
- In the isolated regime both moments have decayed before each visit; the visit's own momentum and second moment then decay geometrically and their ratio sums to $`\kappa_{\rm iso}\,\mathrm{sign}(g)`$.
- Near $`d(s)\approx1-\beta_2`$ neither closed form applies.

*Illustration.* [Example 6.13](../proofs/06_learner_relative_information.md#s6-13): one common state visited with probability $`1-q`$ and $`N`$ tail states each visited with probability $`q/N`$, all in the frequent regime. Relative to the common state, the tail's share of $`\sum_sW_s/N_s`$ is larger under warm Adam than under SGD by the factor $`N(1-q)/q`$, for instance $`80`$ when $`N=20`$ and $`q=0.2`$.

*Uses* [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10), [Proposition 5.5](../proofs/05_warm_adam.md#s5-5), [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10). *Used by* [Corollary 6.12](../proofs/06_learner_relative_information.md#s6-12), [Corollary 7.2](../proofs/07_second_moment_channel.md#s7-2), [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2).

[Proof](../proofs/06_learner_relative_information.md#s6-11) · [Experiments](../experiments/06_learner_relative_information.md#c6-4) · proved to leading order in the stated regime

#### Corollary 6.12 · Learner separations

- **(a)** Let the oracle consumed gradient be $`b\,G_\alpha(Y)`$ with an unknown scale $`b>0`$, and let the demonstrations depend on $`\alpha`$ only. For always-active clipped SGD (or normalized SGD) the oracle mean increment does not depend on $`b`$, so the error tends to zero as $`K\to\infty`$. For SGD the learner-relative functional depends on $`b`$, the demonstrations do not, and no $`K`$ suffices. The same demonstrations are sufficient for one learner and insufficient for the other.
- **(b)** When the student's state distribution has many low-frequency states, in the frequent regime of warm Adam, that the teacher rarely reaches, those states carry a larger share of the information requirement under warm Adam than under SGD.

*Derivation outline.* (a) Always-active clipping removes the scale, so $`c`$ has no component along $`b`$; for SGD the scale moves $`\Psi`$ while the demonstration law stays fixed, which is [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7). With clipping active only with probability $`p_{\rm act}`$, [Proposition 5.9](../proofs/05_learner_statistics.md#s5-9) bounds the remaining scale dependence by $`c_{\rm clip}(1-p_{\rm act})`$. (b) Combine [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10) with the weights of [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11).

*Uses* [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5), [Corollary 6.7](../proofs/06_learner_relative_information.md#s6-7), [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10), [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11), [Proposition 5.9](../proofs/05_learner_statistics.md#s5-9). *Used by* [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3).

[Proof](../proofs/06_learner_relative_information.md#s6-12) · [Experiments](../experiments/06_learner_relative_information.md#c6-6) · proved; (b) to leading order in the stated regime
