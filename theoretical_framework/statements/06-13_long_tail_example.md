# Example 6.13 · One common state and N long-tail states

*Statement*

[← Framework](../README.md) · [Statement](../statements/06-13_long_tail_example.md) · [中文](06-13_long_tail_example_ZH.md)

**In one paragraph.** A worked example of [Corollary 6.12](../statements/06-12_learner_separations.md#s6-12)(b): one common state and $`N`$ tail states, all in Adam's frequent regime. Relative to the common state, the tail's share of the information requirement is larger under warm Adam than under SGD by the factor $`N(1-q)/q`$.

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

<a id="s6-13"></a>

**Example 6.13 (One common state and $`N`$ long-tail states).** The student visits a common state with probability $`1-q`$ and each of $`N`$ tail states with probability $`q/N`$. The teacher covers the common state with probability $`1-q_T`$ and each tail state with probability $`q_T/N`$, so $`N_s\propto KH\,d_T(s)`$. All states have the same variance factor and deployment weight, and the tail states are in the frequent regime of warm Adam, $`(q/N)\min\lbrace t,1/(1-\beta_2)\rbrace\gg1`$.

- **Contributions to $`\sum_sW_s/N_s`$.** Up to the common factor:
  - tail states: $`N(q/N)^2/(q_T/N)=q^2/q_T`$ under SGD, and $`N(q/N)/(q_T/N)=Nq/q_T`$ under warm Adam;
  - common state: $`(1-q)^2/(1-q_T)`$ under SGD, and $`(1-q)/(1-q_T)`$ under warm Adam.
- **Amplification.** Relative to the common state, the tail term is larger under warm Adam than under SGD by the factor $`N(1-q)/q`$. For instance, $`N=20`$ and $`q=0.2`$ give a factor $`80`$; with $`\beta_2=0.999`$ each tail state then has $`(q/N)/(1-\beta_2)=10`$ expected visits per second-moment window, inside the frequent regime.
- **Allocation.** The water-filling rule of [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10)(c) therefore sends more of the query budget to the tail states under warm Adam than under SGD.

**What it says.**

- One common state and $`N`$ tail states: relative to the common state, the tail's share of the information requirement is larger under warm Adam than under SGD by $`N(1-q)/q`$ (for example $`80`$ when $`N=20`$, $`q=0.2`$).

**Role in the framework.**

- It shows how large the learner-relative amplification can be in a realistic visitation profile, and why water-filling sends more queries to the tail under Adam.

**Uses.**

- [Proposition 6.10](../statements/06-10_water_filling_allocation.md#s6-10): information formula and water-filling.
- [Proposition 6.11](../statements/06-11_learner_weights_sgd_warm_adam.md#s6-11): learner weights.

**Used by.**

- [Corollary 6.12](../statements/06-12_learner_separations.md#s6-12): illustrates part (b).

**Status.** example (an exact computation from Propositions 6.10 and 6.11).
