# §2 Feasibility and acquisition — Proofs

[← Framework](../README.md) · [Experiments](../experiments/02_feasibility_and_acquisition.md) · [中文](02_feasibility_and_acquisition_ZH.md)

**Summary.**

- [Theorem 2.2](#s2-2) splits the student's expected regret exactly into a representation term, an information term and an algorithm term. All three are nonnegative. The information term vanishes if and only if every environment consistent with the data shares an optimal policy.
- [Proposition 2.3](#s2-3) gives the minimax dual of the robust version of the information term. [Proposition 2.4](#s2-4) shows that no environment-independent post-processing of the data can lower it: replay, rewriting and synthesis can only lower the algorithm term.
- [Theorem 2.6](#s2-6) gives a family in which the teacher is optimal and its demonstrations have the same law in both worlds, yet the student's optimal policies differ. Every passive learner then loses at least $`\Delta`$. With reset queries, order $`\Delta^{-2}\log(1/\delta)`$ queries are necessary and sufficient. [Proposition 2.7](#s2-7) shows that compiling the acquired fact into the student is a separate cost.
- [Theorem 2.10](#s2-10) makes the number $`K`$ of demonstrations explicit in a hidden-bit model: the information floor after $`K`$ demonstrations, and the optimal use of a hard query budget. [Proposition 2.11](#s2-11) shows that demonstrations and adaptive queries draw on one information budget.
- **Uses.** The setting of [Definitions 1.1–1.2](../README.md#s1-1) (environment, student and reference classes, common initial information $`I_0`$, transcript $`D`$).
- **Used by.** [Proposition 3.1](03_learning_consequence.md#s3-1) (the gaps there live in what the learner extracts); the learner-relative versions [Corollary 6.7](06_learner_relative_information.md#s6-7) and [Corollary 6.8](06_learner_relative_information.md#s6-8) ([Remark 2.12](#s2-12)); the acquisition step of the method ([Algorithm 9.2](09_mechanism_and_method.md#s9-2), [Remark 9.3](09_mechanism_and_method.md#s9-3)).

## The regret decomposition

<a id="s2-1"></a>

**Definition 2.1 (Bayes value and the three regret terms).** Consider a finite decision problem.

- $`\mathcal E`$ is a finite set of environments with prior $`\mu`$. The prior is conditional on the common initial information $`I_0`$: the pretrained model, the known rules and the $`K`$ trusted demonstrations.
- $`\Pi\subseteq\Pi_{\rm ref}`$ are finite policy classes with values $`J_E(\pi)\in[0,1]`$. $`\Pi`$ is the student's deployment class and $`\Pi_{\rm ref}`$ the reference class.
- The transcript $`D`$ takes values in a finite alphabet $`\mathcal D`$ with law $`P(D\mid E)`$. Every experiment the learner may run, adaptive or not, is part of $`D`$.
- A learner is a Markov kernel $`\lambda(\cdot\mid D)\in\Delta(\Pi)`$ with output $`\widehat\pi\sim\lambda(\cdot\mid D)`$. Its private randomness is conditionally independent of $`E`$ given $`D`$.
- $`\mu(E\mid D)`$ is the posterior.

Define

```math
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi),\qquad B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D],
```

and the representation, information and algorithm regrets

```math
\mathcal R_{\rm rep}=\mathbb E[J_E^{\rm ref}-J_E^\Pi],\qquad \mathcal R_{\rm info}=\mathbb E J_E^\Pi-\mathbb E_D B(D),\qquad \mathcal R_{\rm alg}=\mathbb E_D\big[B(D)-\mathbb E[J_E(\widehat\pi)\mid D]\big].
```

All maxima are attained because the classes are finite. $`B(D)`$ is the Bayes value: the best expected return a student in $`\Pi`$ can reach given the transcript.

<a id="s2-2"></a>

**Theorem 2.2 (Regret decomposition and decision ambiguity).** Under Definition 2.1:

**(a) Exact decomposition.** The expected regret against the reference optimum is

```math
\mathbb E[J_E^{\rm ref}-J_E(\widehat\pi)]=\mathcal R_{\rm rep}+\mathcal R_{\rm info}+\mathcal R_{\rm alg},\qquad \mathcal R_{\rm rep},\ \mathcal R_{\rm info},\ \mathcal R_{\rm alg}\ge0. \qquad(2.1)
```

**(b) Decision ambiguity.** $`\mathcal R_{\rm info}=0`$ if and only if, for every $`D`$ with $`\Pr(D)>0`$,

```math
\bigcap_{E:\,\mu(E\mid D)>0}\arg\max_{\pi\in\Pi}J_E(\pi)\neq\varnothing. \qquad(2.2)
```

*Proof.* **(a)** The identity follows by adding and subtracting $`\mathbb EJ_E^\Pi`$ and $`\mathbb E_DB(D)`$. It remains to show that each term is nonnegative.

- $`\mathcal R_{\rm rep}\ge0`$: since $`\Pi\subseteq\Pi_{\rm ref}`$, $`J_E^\Pi\le J_E^{\rm ref}`$ for every $`E`$.
- $`\mathcal R_{\rm info}\ge0`$: fix $`D`$ with $`\Pr(D)>0`$. For every $`\pi\in\Pi`$, $`\mathbb E[J_E(\pi)\mid D]\le\mathbb E[J_E^\Pi\mid D]`$, because $`J_E(\pi)\le J_E^\Pi`$ pointwise. Maximizing over $`\pi`$ gives $`B(D)\le\mathbb E[J_E^\Pi\mid D]`$. Taking expectations over $`D`$ and applying the tower property gives $`\mathbb E_DB(D)\le\mathbb EJ_E^\Pi`$.
- $`\mathcal R_{\rm alg}\ge0`$: conditional independence of $`\widehat\pi`$ and $`E`$ given $`D`$ gives

```math
\mathbb E[J_E(\widehat\pi)\mid D]=\sum_{\pi\in\Pi}\lambda(\pi\mid D)\,\mathbb E[J_E(\pi)\mid D]\le\max_\pi\mathbb E[J_E(\pi)\mid D]=B(D),
```

because a convex combination cannot exceed its largest term.

**(b)** By the tower property,

```math
\mathcal R_{\rm info}=\sum_{D:\Pr(D)>0}\Pr(D)\,\big(\mathbb E[J_E^\Pi\mid D]-B(D)\big).
```

Each bracket is nonnegative by (a), so $`\mathcal R_{\rm info}=0`$ if and only if every bracket vanishes. Fix such a $`D`$ and let $`\pi_D\in\arg\max_\pi\mathbb E[J_E(\pi)\mid D]`$. Then

```math
\mathbb E[J_E^\Pi\mid D]-B(D)=\sum_{E}\mu(E\mid D)\big(J_E^\Pi-J_E(\pi_D)\big),
```

a sum of nonnegative terms.

- ($`\Leftarrow`$) If $`\pi^\circ`$ lies in the intersection (2.2), then $`\mathbb E[J_E(\pi^\circ)\mid D]=\mathbb E[J_E^\Pi\mid D]\ge B(D)\ge\mathbb E[J_E(\pi^\circ)\mid D]`$, so the bracket is zero.
- ($`\Rightarrow`$) If the bracket is zero, every summand with $`\mu(E\mid D)>0`$ vanishes. Then $`\pi_D`$ is optimal in every environment of positive posterior mass, so it lies in the intersection. $`\square`$

<a id="s2-3"></a>

**Proposition 2.3 (Robust ambiguity duality).** For a finite ambiguity class $`C\subseteq\mathcal E`$ let

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{E\in C}\big[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\big]. \qquad(2.3)
```

Then

```math
r(C)=\max_{\mu\in\Delta(C)}\Big[\sum_E\mu_EJ_E^\Pi-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)\Big]. \qquad(2.4)
```

Moreover:

- **(a)** If every allowed experiment has the same law under every $`E\in C`$, every learner has worst-case expected regret at least $`r(C)`$ on $`C`$. If $`C`$ is known, the minimax mixture attains $`r(C)`$.
- **(b)** $`r(C)=0`$ if and only if some $`\pi\in\Pi`$ is optimal in every $`E\in C`$.

*Proof.* Define the regret matrix $`M_{E,\pi}=J_E^\Pi-J_E(\pi)\in[0,1]`$. For a mixture $`\rho`$ the map $`\mu\mapsto\mu^\top M\rho`$ is linear, so its maximum over $`\Delta(C)`$ is attained at a vertex, and

```math
r(C)=\min_{\rho\in\Delta(\Pi)}\max_{\mu\in\Delta(C)}\mu^\top M\rho .
```

This is the value of a finite two-player zero-sum game. Von Neumann's minimax theorem exchanges min and max:

```math
r(C)=\max_{\mu\in\Delta(C)}\min_{\rho\in\Delta(\Pi)}\mu^\top M\rho=\max_\mu\min_{\pi\in\Pi}(\mu^\top M)_\pi .
```

The inner minimum over the simplex is attained at a pure policy, and $`(\mu^\top M)_\pi=\sum_E\mu_EJ_E^\Pi-\sum_E\mu_EJ_E(\pi)`$, which gives (2.4).

- **(a)** If no experiment distinguishes the members of $`C`$, the law of $`D`$ is the same under every $`E\in C`$, and so is the output law $`\bar\rho=\sum_D\Pr(D)\lambda(\cdot\mid D)`$. The learner's expected regret in $`E`$ is $`(M\bar\rho)_E`$, so its worst case over $`C`$ is at least $`\min_\rho\max_E(M\rho)_E=r(C)`$. If $`C`$ is known, output a minimizer $`\rho^\star`$, which exists because $`\Delta(\Pi)`$ is compact and the objective is continuous.
- **(b)** If $`\pi^\circ`$ is optimal in every $`E\in C`$, the point mass at $`\pi^\circ`$ has worst-case regret 0. Conversely, if $`r(C)=0`$ is attained by $`\rho^\star`$, then $`\sum_\pi\rho^\star(\pi)M_{E,\pi}=0`$ for every $`E\in C`$. Since $`M\ge0`$, every $`\pi`$ in the support of $`\rho^\star`$ has $`M_{E,\pi}=0`$ for all $`E\in C`$, so it is a common optimum. $`\square`$

<a id="s2-4"></a>

**Proposition 2.4 (Post-processing cannot reduce information regret).** Let $`D'\sim\kappa(\cdot\mid D)`$ be a randomized post-processing whose kernel $`\kappa`$ does not depend on $`E`$, so that $`E\to D\to D'`$ is a Markov chain. Then

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(2.5)
```

that is, $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$. If $`Z\perp E\mid(D,I_0)`$, then $`B(D,Z)=B(D)`$, so adding $`Z`$ to the transcript leaves $`\mathcal R_{\rm info}`$ unchanged.

*Proof.* Let $`h_\pi(D)=\mathbb E[J_E(\pi)\mid D]`$. The Markov property ($`E\perp D'\mid D`$) and the tower property give

```math
\mathbb E[J_E(\pi)\mid D']=\mathbb E\big[\mathbb E[J_E(\pi)\mid D,D']\,\big|\,D'\big]=\mathbb E[h_\pi(D)\mid D'] .
```

A maximum of conditional expectations is at most the conditional expectation of the maximum:

```math
\max_\pi\mathbb E[h_\pi(D)\mid D']\le\mathbb E\big[\max_\pi h_\pi(D)\,\big|\,D'\big]=\mathbb E[B(D)\mid D'] .
```

Taking expectations over $`D'`$ gives (2.5). Since $`\mathbb EJ_E^\Pi`$ does not depend on the transcript, (2.5) is equivalent to $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$. For the augmentation, $`Z\perp E\mid D`$ gives $`\mu(E\mid D,Z)=\mu(E\mid D)`$, hence $`\mathbb E[J_E(\pi)\mid D,Z]=h_\pi(D)`$ for every $`\pi`$ and $`B(D,Z)=B(D)`$. $`\square`$

**Comments on Theorem 2.2 and Propositions 2.3–2.4.**

- **What each term isolates.**
  - $`\mathcal R_{\rm rep}`$ is what no signal can remove: the student cannot execute the reference optimum.
  - $`\mathcal R_{\rm info}`$ is what remains when the transcript does not determine a common optimal decision.
  - $`\mathcal R_{\rm alg}`$ is the learner's failure to use information it has. One learner's failure is therefore not an information lower bound, and additional information need not help a fixed learner, because it can only lower $`\mathcal R_{\rm info}`$.
- **Generated data.** By Proposition 2.4, synthetic rewrites, paraphrases and replays of $`D`$ cannot lower $`\mathcal R_{\rm info}`$. They can lower $`\mathcal R_{\rm alg}`$ by making existing information easier to learn. Knowledge already inside a pretrained teacher belongs to $`I_0`$; new external queries enlarge $`D`$.
- **Probability semantics of (2.3).** Suppose a confidence set $`C_D`$ contains the true environment with probability $`1-\delta`$ and a mixture has worst-case gap at most $`\varepsilon`$ on $`C_D`$. This bounds the mixture's *expected* regret on that data event, not the regret of one sampled policy.
- **Equality is Bayes-additive only.** Maximizing the three terms separately, for instance taking worst cases term by term, does not preserve (2.1).
- **Scope.** Finiteness guarantees attained maxima and well-defined conditional expectations; general measurable spaces need measurable-selection and integrability conditions. The decomposition is a classical decision-theoretic identity. Its role here is to separate the three error sources that the rest of the framework treats separately.

## Passive impossibility and active acquisition

<a id="s2-5"></a>

**Definition 2.5 (Two-world aliased family).**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects the world. A gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform.
- The student has one aliased context, so it cannot see $`s`$, and makes one decision: action $`A`$ with probability $`p`$, otherwise $`B`$.
- The teacher observes $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$.
- Success probabilities:

| State | Success of $`A`$ | Success of $`B`$ |
|---|---:|---:|
| $`s=0`$ | $`1`$ | $`\tfrac12-2v\Delta`$ |
| $`s=1`$ | $`\tfrac12+2v\Delta`$ | $`1`$ |

Two interfaces are considered.

- **Passive interface.** The learner sees only teacher demonstrations, possibly with $`s`$ revealed, and randomized post-processing of them.
- **Query interface.** In addition, the learner may reset to $`s=1`$, execute $`A`$ and observe the Bernoulli outcome. Queries may be chosen adaptively. The number of queries $`N`$ is a stopping time with $`\mathbb E_vN<\infty`$. The learner's algorithm and private randomness are the same in both worlds.

Write $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ for the binary KL divergence.

*Values in this family.* Averaging over $`s`$,

```math
J_v(p)=\tfrac12\big[p+(1-p)(\tfrac12-2v\Delta)\big]+\tfrac12\big[p(\tfrac12+2v\Delta)+(1-p)\big]=\tfrac34+v\Delta(2p-1).
```

- $`J_v`$ is linear in $`p`$, so $`J_v^\Pi=\tfrac34+\Delta`$, attained at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$.
- The teacher always plays an action with success probability 1. Every other state–action success probability equals $`\tfrac12\pm2\Delta\in[\tfrac14,\tfrac34]`$ because $`2\Delta\le\tfrac14`$. The teacher is therefore strictly optimal in both worlds.

<a id="s2-6"></a>

**Theorem 2.6 (Passive impossibility, active attainability).** In the family of Definition 2.5:

- **(a) Passive impossibility.** Under the passive interface every learner satisfies $`\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta`$.
- **(b) Active lower bound.** Under the query interface, let $`0<\delta<1/2`$, and suppose that in each world, with probability at least $`1-\delta`$, the learner outputs a student with regret at most $`\Delta/2`$. Then for each $`v`$,

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(2.6)
```

- **(c) Active upper bound.** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ independent queries suffice to output the optimal endpoint $`p\in\lbrace 0,1\rbrace`$ with probability at least $`1-\delta`$ in each world.

*Proof.* **(a)** A demonstration records $`(s,\text{action},\text{outcome})`$, equal to $`(0,A,1)`$ or $`(1,B,1)`$ with probability $`\tfrac12`$ each, in both worlds. The demonstration law does not depend on $`v`$, and neither does the law of any randomized post-processing of it, including the learner's output $`\widehat p`$. Let $`m=\mathbb E\widehat p`$, the same in both worlds. Because $`J_v`$ is affine in $`p`$, $`\mathbb EJ_v(\widehat p)=J_v(m)`$, and the two expected regrets are

```math
J_{+1}^\Pi-J_{+1}(m)=2\Delta(1-m),\qquad J_{-1}^\Pi-J_{-1}(m)=2\Delta m .
```

Their maximum is at least their average, $`\Delta`$. Equality holds at $`m=\tfrac12`$, for example with $`\widehat p\equiv\tfrac12`$.

**(b)** *Step 1 (reduction to a binary test).* In world $`+1`$, regret $`2\Delta(1-\widehat p)\le\Delta/2`$ holds exactly when $`\widehat p\ge3/4`$. In world $`-1`$, $`2\Delta\widehat p\le\Delta/2`$ holds exactly when $`\widehat p\le1/4`$. Let $`A=\lbrace \widehat p>1/2\rbrace`$. Then $`P_{+1}(A)\ge1-\delta`$ and $`P_{-1}(A)\le\delta`$.

*Step 2 (KL of the stopped transcript).* Let $`T`$ be the full transcript: demonstrations, query outcomes $`Y_1,\dots,Y_N`$ and the learner's private randomness, with law $`P_v^T`$.

- The demonstrations and the private randomness have the same law in both worlds. Each queried outcome is Bernoulli $`(\tfrac12+2v\Delta)`$ given the past. Write $`d_\Delta=\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)`$.
- Let $`\ell(y)=\log\frac{P_{+1}(Y=y)}{P_{-1}(Y=y)}`$ be the per-query log-likelihood ratio. It is bounded: $`\lvert\ell\rvert\le\log\frac{1/2+2\Delta}{1/2-2\Delta}=:c`$.
- The demonstration and randomization factors cancel, so the log-likelihood ratio of the transcript is $`\Lambda(T)=\sum_{t=1}^{N}\ell(Y_t)`$.
- Let $`\mathcal F_{t-1}`$ be the history before query $`t`$. The event $`\lbrace t\le N\rbrace`$ is $`\mathcal F_{t-1}`$-measurable, and $`\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]=d_\Delta`$ on that event. Moreover $`\sum_t\mathbb E_{+1}\lvert1\lbrace t\le N\rbrace \ell(Y_t)\rvert\le c\,\mathbb E_{+1}N<\infty`$, so Fubini's theorem applies (Wald's identity):

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})=\mathbb E_{+1}\Lambda(T)=\sum_{t\ge1}\mathbb E_{+1}\big[1\lbrace t\le N\rbrace \,\mathbb E_{+1}[\ell(Y_t)\mid\mathcal F_{t-1}]\big]=\mathbb E_{+1}[N]\,d_\Delta .
```

*Step 3 (data processing).* The event $`A`$ is a function of $`T`$, so

```math
\mathrm{KL}(P^T_{+1}\Vert P^T_{-1})\ge\mathrm{kl}(P_{+1}(A),P_{-1}(A))\ge\mathrm{kl}(1-\delta,\delta).
```

The second inequality holds because, for $`y\le x`$, $`\mathrm{kl}(x,y)`$ is nondecreasing in $`x`$ and nonincreasing in $`y`$; on the rectangle $`x\ge1-\delta>\tfrac12>\delta\ge y`$ the minimum is at $`(1-\delta,\delta)`$. Steps 2 and 3 give the first inequality in (2.6) for $`v=+1`$. For $`v=-1`$ the same argument applies, since $`\mathrm{kl}(\tfrac12-2\Delta,\tfrac12+2\Delta)=d_\Delta`$.

*Step 4 (the constant).* For $`0<u\le\tfrac14`$,

```math
\mathrm{kl}(\tfrac12+u,\tfrac12-u)=2u\log\frac{1+2u}{1-2u}=4u\,\mathrm{artanh}(2u)=8u^2\sum_{j\ge0}\frac{(4u^2)^j}{2j+1}.
```

The series has positive coefficients and is increasing in $`u`$; at $`u=\tfrac14`$ it equals $`2\,\mathrm{artanh}(\tfrac12)=\log3`$. Hence $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le8\log3\cdot u^2\le16u^2`$. With $`u=2\Delta\le\tfrac14`$ this gives $`d_\Delta\le64\Delta^2`$, the second inequality in (2.6).

**(c)** Draw $`n`$ independent query outcomes and set $`\widehat v=+1`$ if $`\bar Y>\tfrac12`$ and $`\widehat v=-1`$ otherwise (a tie counts as an error in the analysis). Output $`\widehat p=1\lbrace \widehat v=+1\rbrace`$. In world $`+1`$, $`\mathbb E\bar Y=\tfrac12+2\Delta`$, and the one-sided Hoeffding inequality gives

```math
P_{+1}(\bar Y\le\tfrac12)\le\exp(-2n(2\Delta)^2)=\exp(-8n\Delta^2)\le\delta
```

whenever $`n\ge\log(1/\delta)/(8\Delta^2)`$. World $`-1`$ is symmetric. On the success event the output is the optimal endpoint, with regret 0. $`\square`$

<a id="s2-7"></a>

**Proposition 2.7 (Compilation cost).** In the family of Definition 2.5, after the sign is estimated as $`\widehat v`$:

- choose the target $`q=7/8`$ if $`\widehat v=+1`$ and $`q=1/8`$ otherwise;
- generate $`m`$ Bernoulli $`(q)`$ target labels, and let the student be their empirical frequency $`\widehat p`$.

If $`m\ge\lceil32\log(2/\delta_L)\rceil`$, then $`\lvert\widehat p-q\rvert\le1/8`$ with probability at least $`1-\delta_L`$. If $`\widehat v`$ is correct with probability at least $`1-\delta`$ (Theorem 2.6(c)), the compiled student has regret at most $`\Delta/2`$ with probability at least $`1-\delta-\delta_L`$.

*Proof.* The two-sided Hoeffding inequality gives

```math
\Pr(\lvert\widehat p-q\rvert>\tfrac18)\le2\exp(-2m/64)=2e^{-m/32}\le\delta_L
```

when $`m\ge32\log(2/\delta_L)`$. Suppose $`\widehat v=v=+1`$ and $`\lvert\widehat p-q\rvert\le\tfrac18`$. Then $`\widehat p\ge\tfrac34`$ and the regret is $`2\Delta(1-\widehat p)\le\Delta/2`$. The case $`v=-1`$ is symmetric, with $`\widehat p\le\tfrac14`$. A union bound over the identification failure ($`\le\delta`$) and the compilation failure ($`\le\delta_L`$) gives the stated probability. $`\square`$

**Comments on Theorem 2.6 and Proposition 2.7.**

- **Perfect demonstrations can be information-insufficient.** The teacher succeeds with probability 1 in both worlds, yet $`\mathcal R_{\rm info}\ge\Delta`$ for every passive learner (Theorem 2.2). The missing fact, the sign of $`v`$, is not in the demonstrations even when they reveal $`s`$: a demonstration shows what the teacher did, not what a student with different abilities should do.
- **Acquisition and compilation are separate costs.**
  - Identification costs $`\Theta(\Delta^{-2}\log(1/\delta))`$ authentic environment queries; (2.6) and Theorem 2.6(c) match up to constants.
  - Compilation then costs $`O(\log(1/\delta_L))`$ synthetic target labels, independent of $`\Delta`$. These labels are compiled, not new teacher executions, and are accounted for separately.
- **Tightness.** The exact computations of [Check 2.3](../experiments/02_feasibility_and_acquisition.md#c2-3) show that an optimally tuned SPRT attains (2.6) in some configurations, so its constant cannot be improved in general.
- **Assumptions used.** Knowledge of $`\Delta`$ enters the stopping rule of (c). Part (b) needs the declared query interface, a world-independent algorithm and $`\mathbb E_vN<\infty`$. The argument is a standard change of measure applied to a student–teacher information mismatch; it is not a new testing inequality.

## The number of demonstrations

<a id="s2-8"></a>

**Definition 2.8 (Hidden-bit model).** Every probability is conditional on a fixed, declared pretrained information state $`I_{\rm pre}`$. Extra teacher feedback is never counted as part of the $`K`$ demonstrations.

- **Bits and deployment.** The unknown bits $`\Theta=(\Theta_1,\dots,\Theta_d)`$ are independent and fair. Context $`j`$ occurs at deployment with weight $`w_j\ge0`$, $`\sum_jw_j=1`$, and the correct decision at context $`j`$ is $`\Theta_j`$. The student class contains every context-to-action table, so $`\mathcal R_{\rm rep}=0`$. The utility is $`J_\Theta(\pi)=\sum_jw_j\Pr_\pi(A_j=\Theta_j)`$, so the informed optimum is 1.
- **Non-leaking demonstrations.** Trajectory $`k`$ reveals a random subset $`S_k\subseteq[d]`$ together with the labels $`\Theta_{S_k}`$. The subsets $`S_1,\dots,S_K`$ are IID with law $`L`$ and independent of $`\Theta`$; dependence among the coordinates revealed within one trajectory is allowed. Any further transcript content has a conditional law that depends only on the revealed labels and independent randomness. Write $`q_j=\Pr(j\in S)`$.
- **Hard query budget.** After the demonstrations, the learner may make at most $`n`$ noiseless unit-cost queries, each revealing one chosen bit. The choices may depend on everything observed so far. $`n`$ is a per-transcript hard budget, not an expected budget.
- $`U_K`$ is the set of bits revealed by none of the $`K`$ trajectories, and $`M_K=\lvert U_K\rvert`$.

<a id="s2-9"></a>

**Lemma 2.9 (Demonstrations leak nothing about unrevealed bits).** In the model of Definition 2.8, conditional on the full transcript $`T`$, the unseen bits $`(\Theta_j)_{j\in U_K}`$ are IID fair and independent of $`T`$ and of the learner's private randomness.

*Proof.* The joint law factorizes as

```math
\Pr(S_{1:K},\Theta,X)=\prod_kL(S_k)\cdot2^{-d}\cdot\Pr(X\mid S_{1:K},\Theta_{\cup_kS_k}),
```

where $`X`$ is the additional transcript content. Fix the observed $`(S_{1:K},\Theta_{\cup S_k},X)`$. The only factor that depends on $`\Theta_{U_K}`$ is $`2^{-d}`$, which is constant. Hence $`\Theta_{U_K}`$ is uniform on $`\lbrace 0,1\rbrace ^{\lvert U_K\rvert}`$ given $`T`$. The learner's randomness is independent of everything else. $`\square`$

<a id="s2-10"></a>

**Theorem 2.10 (Few-shot information floor and optimal hard-budget querying).** In the model of Definition 2.8:

**(a) Information floor.** Without queries, the minimum achievable expected regret after $`K`$ trajectories is

```math
R_{\rm info}(K)=\frac12\sum_{j=1}^dw_j(1-q_j)^K, \qquad(2.7)
```

with the convention $`(1-q)^0=1`$. The marginal value of one more trajectory is

```math
R_{\rm info}(K)-R_{\rm info}(K+1)=\frac12\sum_jw_jq_j(1-q_j)^K. \qquad(2.8)
```

**(b) Optimal queries.** With weights of the unseen bits sorted as $`w_{(1)}\ge\dots\ge w_{(M_K)}`$,

```math
R_{\rm info}(K,n)=\frac12\,\mathbb E\Big[\sum_{r=n+1}^{M_K}w_{(r)}\Big], \qquad(2.9)
```

and querying the $`n`$ heaviest unseen bits is optimal; adapting the queries to the revealed values cannot help. In the uniform case ($`w_j=1/d`$, each trajectory reveals one uniformly drawn context),

```math
R_{\rm info}(K,n)=\frac{\mathbb E[(M_K-n)_+]}{2d},\qquad R_{\rm info}(K,0)=\frac12(1-1/d)^K. \qquad(2.10)
```

In general $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$, and the inequality can be strict.

*Proof.* **(a)**

- *Lower bound.* Fix any learner. For $`j\in U_K`$, Lemma 2.9 gives $`\Pr(A_j=\Theta_j\mid T)=\tfrac12`$ for every possibly randomized action $`A_j`$. For $`j\notin U_K`$ the success probability is at most 1. Hence the conditional regret is at least $`\tfrac12\sum_{j\in U_K}w_j`$.
- *Achievability.* The table that copies every revealed label and guesses the rest attains this bound, and it lies in the student class.
- *Taking expectations.* The $`S_k`$ are IID, so $`\Pr(j\in U_K)=\Pr(j\notin S_1)^K=(1-q_j)^K`$ and $`\mathbb E\sum_{j\in U_K}w_j=\sum_jw_j(1-q_j)^K`$, which is (2.7).
- *Marginal value.* $`(1-q_j)^K-(1-q_j)^{K+1}=q_j(1-q_j)^K`$ gives (2.8).

**(b)** Condition on the demonstration transcript and let $`V(U,b)`$ be the optimal conditional regret with unseen set $`U`$ and remaining budget $`b`$. We show by induction on $`b`$ that

```math
V(U,b)=\tfrac12\Big(\sum_{j\in U}w_j-\mathrm{top}_b(U)\Big),
```

where $`\mathrm{top}_b(U)`$ is the sum of the $`\min(b,\lvert U\rvert)`$ largest weights in $`U`$.

- *Base case.* For $`b=0`$ the claim is (a), conditionally.
- *Inductive step.* With $`b\ge1`$ the learner may stop, query a known bit, or query some $`j\in U`$.
  - Stopping gives $`V(U,0)`$ and querying a known bit gives $`V(U,b-1)`$. Both are at least the right-hand side by the induction hypothesis, since $`\mathrm{top}_b`$ is monotone in $`b`$.
  - Querying $`j\in U`$ reveals $`\Theta_j`$. By Lemma 2.9 the remaining unseen bits are still IID fair and independent of the revealed value, so the continuation problem is $`V(U\setminus\lbrace j\rbrace ,b-1)`$ whatever value is observed; value-adaptive branching gains nothing.

Hence

```math
V(U,b)=\min_{j\in U}\tfrac12\Big(\sum_{U\setminus\lbrace j\rbrace }w-\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\Big)=\tfrac12\Big(\sum_Uw-\max_{j\in U}\big[w_j+\mathrm{top}_{b-1}(U\setminus\lbrace j\rbrace )\big]\Big).
```

The maximum equals $`\mathrm{top}_b(U)`$, attained by taking $`j`$ of largest weight. This completes the induction and shows that the greedy rule is optimal. Averaging $`V(U_K,n)`$ over the demonstrations gives (2.9).

- *Uniform case.* With weights $`1/d`$, $`\sum_{r>n}w_{(r)}=(M_K-n)_+/d`$, which gives (2.10); for $`n=0`$, (a) with $`q_j=1/d`$ gives $`R_{\rm info}(K,0)=\tfrac12(1-1/d)^K`$.
- *Jensen.* Since $`x\mapsto(x-n)_+`$ is convex, $`\mathbb E[(M_K-n)_+]\ge(\mathbb EM_K-n)_+`$.
- *Strictness.* Take $`d=3`$, $`K=3`$, $`n=1`$. Then $`\Pr(M_3=0)=6/27`$, $`\Pr(M_3=1)=18/27`$, $`\Pr(M_3=2)=3/27`$ and $`\mathbb EM_3=8/9<1`$. The right-hand side is $`0`$, while $`R_{\rm info}(3,1)=\Pr(M_3=2)/(2\cdot3)=1/54`$. $`\square`$

<a id="s2-11"></a>

**Proposition 2.11 (Demonstrations and queries share one information budget).** Consider two-world adaptive acquisition.

- The world is $`v\in\lbrace 0,1\rbrace`$, and the two worlds require different final decisions.
- Each of $`K`$ IID demonstrations satisfies $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})\le\kappa_D^{(v)}<\infty`$.
- A common history-dependent rule selects queries and stops at $`N\le N_{\max}<\infty`$. The rule's selection, stopping and private randomization kernels do not depend on $`v`$.
- For every allowed query $`x`$ and every preceding history $`h`$ on the relevant support, $`\mathrm{KL}(P_v(\cdot\mid h,x)\Vert P_{1-v}(\cdot\mid h,x))\le\kappa_X^{(v)}<\infty`$.
- The final decision errs with probability at most $`\delta`$ in each world, where $`0<\delta<1/2`$.

Then

```math
K\kappa_D^{(v)}+\kappa_X^{(v)}\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta). \qquad(2.11)
```

If $`0<\kappa_X^{(v)}<\infty`$,

```math
\mathbb E_vN\ge\frac{[\mathrm{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}{\kappa_X^{(v)}}. \qquad(2.12)
```

If $`\kappa_X^{(v)}=0`$ and $`K\kappa_D^{(v)}<\mathrm{kl}(1-\delta,\delta)`$, no query budget achieves the stated errors.

*Proof.* *Padding.* Pad the transcript after stopping with a fixed null symbol to length $`N_{\max}`$, and write it as $`T=(O_1,\dots,O_K,(X_1,Y_1),\dots,(X_N,Y_N))`$ together with the stop indicators and the private randomness.

*Likelihood ratio.* The selection, stopping and randomization kernels are the same in both worlds, so they cancel:

```math
\log\frac{P_v(T)}{P_{1-v}(T)}=\sum_{k=1}^K\log\frac{P^{\rm demo}_v(O_k)}{P^{\rm demo}_{1-v}(O_k)}+\sum_{t=1}^{N_{\max}}1\lbrace t\le N\rbrace \log\frac{P_v(Y_t\mid H_{t-1},X_t)}{P_{1-v}(Y_t\mid H_{t-1},X_t)}.
```

The finite KL bounds imply absolute continuity on the relevant support, so every term is finite almost surely.

*Taking $`\mathbb E_v`$.*

- The first sum has expectation $`K\,\mathrm{KL}(P^{\rm demo}_v\Vert P^{\rm demo}_{1-v})\le K\kappa_D^{(v)}`$.
- In the second sum, $`\lbrace t\le N\rbrace`$ and $`X_t`$ are measurable with respect to $`(H_{t-1},\text{randomness})`$. Conditioning term by term, each summand has expectation $`\mathbb E_v[1\lbrace t\le N\rbrace \,\mathrm{KL}(P_v(\cdot\mid H_{t-1},X_t)\Vert P_{1-v}(\cdot\mid H_{t-1},X_t))]\le\kappa_X^{(v)}\Pr_v(t\le N)`$.
- The sum is finite because $`N_{\max}<\infty`$. Summing over $`t`$ gives $`\kappa_X^{(v)}\mathbb E_vN`$.

Hence $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\le K\kappa_D^{(v)}+\kappa_X^{(v)}\mathbb E_vN`$.

*Data processing.* Apply data processing to the indicator that the final decision is correct for $`v`$. Its probabilities are $`x\ge1-\delta`$ under $`v`$ and $`y\le\delta`$ under $`1-v`$, so $`\mathrm{KL}(P_v^T\Vert P_{1-v}^T)\ge\mathrm{kl}(x,y)\ge\mathrm{kl}(1-\delta,\delta)`$ by the monotonicity of the binary kl on the rectangle $`x\ge1-\delta>\tfrac12>\delta\ge y`$. Combining the two bounds gives (2.11), and rearranging gives (2.12).

*The case $`\kappa_X^{(v)}=0`$.* The left side of (2.11) equals $`K\kappa_D^{(v)}`$. If this is below $`\mathrm{kl}(1-\delta,\delta)`$, (2.11) fails for every $`N`$, so no such algorithm exists. $`\square`$

**Comments on Theorem 2.10 and Proposition 2.11.**

- **Replay adds zero.** Re-presenting already revealed labels leaves $`U_K`$ unchanged, so it does not change $`R_{\rm info}`$. This is Proposition 2.4 specialized to this model. Replay can still reduce the learner's own error $`\mathcal R_{\rm alg}`$.
- **No universal rate in $`K`$.** A coordinate with $`q_j=0`$ leaves an irreducible passive error $`\tfrac12w_j`$, and a small $`q_j>0`$ makes convergence arbitrarily slow. Any rate in $`K`$ needs a coverage assumption on $`(q_j)`$.
- **What each assumption does.**
  - Independence of the bits makes queries non-interacting. With correlated bits one query informs another, and the greedy result changes.
  - The hard budget is essential: an expected-budget constraint allows allocation across transcripts.
  - The non-leaking reveal mechanism is essential: reveal sets chosen based on the bits would themselves leak information.
- **Scope of (2.11).** It is a binary-identification bound. A policy-learning version needs disjoint near-optimal output sets. A zero right-hand side in (2.12) does not establish achievability.
- **Floor, not achieved risk.** $`R_{\rm info}(K)`$ is a Bayes information floor. An actual learner's risk can be higher; the excess is its algorithm term $`\mathcal R_{\rm alg}`$ of Theorem 2.2.

<a id="s2-12"></a>

**Remark 2.12 (Learner-relative versions).** §6 restates the results of this section for the quantity that matters for distillation: the first-order consequence of a feedback law for one declared learner.

- [Corollary 6.7](06_learner_relative_information.md#s6-7) generalizes Theorem 2.6(a) and Proposition 2.4. If the demonstrations have the same law in two environments that differ on the learner-relative functional $`\Psi`$, then every distiller, including replay, rewriting and synthesis, has worst-case error at least $`\tfrac12\lvert\Psi(\theta_1)-\Psi(\theta_0)\rvert`$, for every $`K`$.
- [Corollary 6.8](06_learner_relative_information.md#s6-8) generalizes Proposition 2.11. Asymptotically, $`K\,e_D(h)+n\,e_Q(h)\ge(c^\top h)^2/(16\varepsilon^2)`$ for every direction $`h`$, where $`c`$ is determined by the learner. The rate at which demonstrations substitute for queries therefore depends on the learner, not only on the data.
- The difference in object: this section measures the decision regret of the student's policy; §6 measures the fidelity of a feedback law relative to a learner's update. Both show that missing information can only be supplied by new experiments.

## Related work

- The decomposition of Theorem 2.2 and the duality of Proposition 2.3 are classical decision theory (Bayes risk and von Neumann's minimax theorem). Proposition 2.4 is the data-processing property of Bayes risk.
- Theorem 2.6(b) and Proposition 2.11 use the standard change-of-measure argument for adaptive sampling (Wald's identity and the data-processing inequality for KL), as in lower bounds for sequential testing and best-arm identification. The SPRT is Wald's sequential probability ratio test.
- The new content of this section is the setting: a student–teacher information mismatch in which perfect demonstrations are insufficient, the separation between acquisition and compilation, and the explicit dependence on the number $`K`$ of demonstrations.
