# Theorem 2.6 · Passive impossibility, active attainability

*Statement*

[← Framework](../README.md) · [Statement](02-06_passive_impossibility_active_attainability.md) · [Proof](../proofs/02-06_passive_impossibility_active_attainability.md) · [Experiments](../experiments/02-06_passive_impossibility_active_attainability.md) · [中文](02-06_passive_impossibility_active_attainability_ZH.md)

**In one paragraph.** Can perfect demonstrations still be insufficient? In the two-world family the teacher is optimal and its demonstrations have the same law in both worlds, yet the aliased student's optimal action is opposite. This theorem shows that every learner that sees only demonstrations loses at least $`\Delta`$ in one world, and that reset queries close the gap with a number of queries of order $`\Delta^{-2}\log(1/\delta)`$, which is also necessary.

**Setting used on this page.**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects one of two worlds; the gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform. The teacher sees $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$; both succeed with probability 1, so the teacher is optimal in both worlds.
- The student is aliased (it cannot see $`s`$) and plays $`A`$ with probability $`p`$. The other two state–action pairs succeed with probability $`\tfrac12\mp2v\Delta`$, so the student's value is $`J_v(p)=\tfrac34+v\Delta(2p-1)`$ and its best value is $`J_v^\Pi=\tfrac34+\Delta`$, at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$.
- *Passive interface:* the learner sees teacher demonstrations (possibly with $`s`$) and post-processing of them. *Query interface:* it may also reset to $`s=1`$, play $`A`$ and observe the Bernoulli $`(\tfrac12+2v\Delta)`$ outcome, adaptively, with a stopping time $`N`$.
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ is the binary KL divergence.
- Full definition: [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5).

<a id="s2-6"></a>

**Theorem 2.6 (Passive impossibility, active attainability).** In the family of [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5):

- **(a) Passive impossibility.** Under the passive interface every learner satisfies $`\max_{v}\mathbb E\big[J_v^\Pi-J_v(\widehat p)\big]\ge\Delta`$.
- **(b) Active lower bound.** Under the query interface, let $`0<\delta<1/2`$, and suppose that in each world, with probability at least $`1-\delta`$, the learner outputs a student with regret at most $`\Delta/2`$. Then for each $`v`$,

```math
\mathbb E_vN\ge\frac{\mathrm{kl}(1-\delta,\delta)}{\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)}\ge\frac{\mathrm{kl}(1-\delta,\delta)}{64\Delta^2}. \qquad(2.6)
```

- **(c) Active upper bound.** $`n_{\rm query}=\lceil\log(1/\delta)/(8\Delta^2)\rceil`$ independent queries suffice to output the optimal endpoint $`p\in\lbrace 0,1\rbrace`$ with probability at least $`1-\delta`$ in each world.

**What it says.**

- Passive learners, however they process the demonstrations, lose at least $`\Delta`$ in one of the worlds.
- With reset queries, about $`\log(1/\delta)/\Delta^2`$ queries are necessary and sufficient, up to a constant factor, to output a near-optimal student with probability $`1-\delta`$ in both worlds.

**Intuition.**

- A perfect demonstration shows what the teacher did, not what a student with different abilities should do; the student has to try.

**Role in the framework.**

- It shows why the method needs an active acquisition step and why demonstration quality is not information sufficiency. Its change-of-measure argument is reused in [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11), and its passive part is generalized by [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7).

**Derivation outline.**

- (a) In both worlds a demonstration is, with probability one half each, "choose $`A`$ at $`s=0`$ and succeed" or "choose $`B`$ at $`s=1`$ and succeed", so the law of the demonstrations and of the learner's output does not depend on $`v`$. The value is affine in $`p`$; the two regrets are $`2\Delta(1-m)`$ and $`2\Delta m`$ with $`m`$ the mean output, and their maximum is at least their average $`\Delta`$.
- (b) Reduce "$`\Delta/2`$-optimal in both worlds" to a binary test of the event $`\lbrace\widehat p>1/2\rbrace`$. With a stopping time, Wald's identity makes the KL divergence of the whole transcript equal to the expected number of queries times the KL of one query, and data processing makes it at least $`\mathrm{kl}(1-\delta,\delta)`$. The constant 64 comes from $`\mathrm{kl}(\tfrac12+u,\tfrac12-u)\le16u^2`$.
- (c) Average independent query outcomes and apply the one-sided Hoeffding inequality.

**Comments.**

- **Perfect demonstrations can be information-insufficient.** The teacher succeeds with probability 1 in both worlds, yet $`\mathcal R_{\rm info}\ge\Delta`$ for every passive learner ([Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)). The missing fact, the sign of $`v`$, is not in the demonstrations even when they reveal $`s`$: a demonstration shows what the teacher did, not what a student with different abilities should do.
- **Tightness.** The exact computations of [Check 2.3](../experiments/02-06_passive_impossibility_active_attainability.md#c2-3) show that an optimally tuned SPRT attains (2.6) in some configurations, so its constant cannot be improved in general.
- **Assumptions used.** Knowledge of $`\Delta`$ enters the stopping rule of (c). Part (b) needs the declared query interface, a world-independent algorithm and $`\mathbb E_vN<\infty`$. The argument is a standard change of measure applied to a student–teacher information mismatch; it is not a new testing inequality.

**Uses.**

- [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2): information term of the regret.
- [Proposition 2.3](../statements/02-03_robust_ambiguity_duality.md#s2-3): hedging value of indistinguishable worlds.
- [Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4): post-processing adds no information.
- [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5): the two-world family.

**Used by.**

- [Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7): compiling the acquired sign.
- [Theorem 2.10](../statements/02-10_few_shot_information_floor.md#s2-10): the information floor in K.
- [Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11): the same change-of-measure argument.
- [Corollary 6.7](../statements/06-07_aliasing_zero_value_replay.md#s6-7): learner-relative aliasing bound.
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3): acquire before compiling.
