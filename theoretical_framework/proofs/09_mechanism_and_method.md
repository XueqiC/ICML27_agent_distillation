# §9 Mechanism and method — Proofs

[← Framework](../README.md) · [Experiments](../experiments/09_mechanism_and_method.md) · [中文](09_mechanism_and_method_ZH.md)

**Summary.**

- [Proposition 9.1](#s9-1) is a mechanism with a strict null. Fix the marginal law of each feedback unit and change only how units are grouped into joint training packets. The learning outcome is unchanged for every regrouping if and only if the joint-packet response is additive. One SGD step at a common parameter is exactly invariant to regrouping; sequential steps are not.
- [Algorithm 9.2](#s9-2) states the method that follows from §2–§8, Consequence-Calibrated Experience Distillation (CED), as a numbered procedure. Each step cites the statement that justifies it, and the algorithm lists the guarantee each output carries.
- [Remark 9.3](#s9-3) derives the six design rules behind the algorithm, each as a consequence of earlier statements.
- [Remark 9.4](#s9-4) separates what the controlled evidence covers from the planned real-model tests of Theorems A, B and C.
- Uses: the response of [Theorem 3.2](03_learning_consequence.md#s3-2); the information results of §2 and §6; the learner statistics of §5 and §7; the audits and certificates of §8.

## Setting

The learner, feedback laws, increment maps and continuation values are those of [Definitions 1.3–1.4](../README.md#s1-3). For the mechanism of Proposition 9.1:

- $`\mathcal Y_1=\lbrace 1,\dots,m\rbrace`$ and $`\mathcal Y_2=\lbrace 1,\dots,n\rbrace`$ are finite sets of valid feedback units, for example the units that fill two slots of a training packet.
- $`\mu\in\Delta(\mathcal Y_1)`$ and $`\nu\in\Delta(\mathcal Y_2)`$ are the slot marginals, assumed strictly positive. With zero entries, everything below holds on the positive-support product.
- Every pair $`(i,j)`$ is a valid packet unless stated otherwise.
- The **coupling polytope** is

```math
\mathcal C(\mu,\nu)=\lbrace \Gamma\in\mathbb R^{m\times n}_{\ge0}:\ \Gamma\mathbf 1=\mu,\ \Gamma^\top\mathbf 1=\nu\rbrace .
```

- At a complete learner state $`z`$, $`H_z(i,j)`$ is the expected final task return after the actual update on the joint packet $`(i,j)`$, followed by the reference tail. It is a one-boundary response in the sense of [Theorem 3.2](03_learning_consequence.md#s3-2), with the packet as the message.
- The **coupling width** is

```math
w(H_z)=\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle .
```

It is the largest change in expected learning value that can be obtained by reorganizing which units are packed together, while both slot marginals, and hence the per-unit content, stay fixed.

For the method, a **candidate feedback law** (also called a provider) $`j\in\lbrace 0,1,\dots,M\rbrace`$ is an executable rule that, at a complete state $`z`$ and round $`t`$, emits messages $`Y\sim P^j_t(\cdot\mid z)`$: training units, packets or targets. Candidate $`0`$ is the reference (for example the raw teacher demonstrations). Its mean response and its signed advantage over the reference are

```math
\bar g_t(z,j)=\mathbb E_{Y\sim P^j_t(\cdot\mid z)}\,g_{t,z}(Y),\qquad A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0),
```

with $`g_{t,z}`$ the one-boundary response of [Definition 1.4](../README.md#s1-4).

## A mechanism with a strict null

<a id="s9-1"></a>

**Proposition 9.1 (Coupling width).** In the setting above:

- **(a) Zero width.** $`w(H_z)=0`$ if and only if $`H_z(i,j)=a_i+b_j`$ for some $`a\in\mathbb R^m`$ and $`b\in\mathbb R^n`$.
- **(b) Bound by the additive residual.** For all $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$,

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty .
```

- The inequality in (b) can be strict.
- **(c) Restricted pairings.** If some pairs are forbidden, so that the feasible couplings form a proper face of $`\mathcal C(\mu,\nu)`$, zero feasible width does not imply additivity.
- **(d) Strict additive null.** One SGD step at a common parameter, on the union of all packets with identical per-example weights, is invariant to how the examples are grouped into packets. Two or more sequential steps are not invariant in general; for quadratic losses the difference is of order $`\eta^2`$ and is given explicitly in the proof.

*Proof.*

**(a), "if".** For $`H=a\mathbf 1^\top+\mathbf 1b^\top`$,

```math
\langle\Gamma,H\rangle=\sum_ia_i\sum_j\Gamma_{ij}+\sum_jb_j\sum_i\Gamma_{ij}=\mu^\top a+\nu^\top b,
```

which is the same for every $`\Gamma\in\mathcal C(\mu,\nu)`$.

**(a), "only if".**

- $`\Gamma_0=\mu\nu^\top`$ lies in $`\mathcal C(\mu,\nu)`$ and is strictly positive.
- Fix rows $`i\ne i'`$ and columns $`j\ne j'`$, and let $`E=e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$. $`E`$ has zero row sums and zero column sums, so $`\Gamma_0\pm sE\in\mathcal C(\mu,\nu)`$ for $`0<s\le\min_{k,l}(\Gamma_0)_{kl}`$.
- Zero width means that $`\langle\Gamma,H\rangle`$ is constant on $`\mathcal C(\mu,\nu)`$. Hence $`\langle\Gamma_0+sE,H\rangle=\langle\Gamma_0-sE,H\rangle`$, that is $`\langle E,H\rangle=0`$:

```math
H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}=0\quad\text{for all rectangles}.
```

- Put $`a_i=H_{i1}`$ and $`b_j=H_{1j}-H_{11}`$. The rectangle $`(i,1,j,1)`$ gives $`H_{ij}=H_{i1}+H_{1j}-H_{11}=a_i+b_j`$; this is trivial when $`i=1`$ or $`j=1`$.

**(b).** $`\Delta=\Gamma-\Gamma'`$ has zero row and column sums, so $`\langle\Delta,a\mathbf 1^\top+\mathbf 1b^\top\rangle=0`$ for all $`a,b`$. Moreover $`\lVert\Delta\rVert_1\le\lVert\Gamma\rVert_1+\lVert\Gamma'\rVert_1=2`$. Hence, for every $`a,b`$,

```math
\lvert\langle\Delta,H\rangle\rvert=\lvert\langle\Delta,H-a\mathbf 1^\top-\mathbf 1b^\top\rangle\rvert\le\lVert\Delta\rVert_1\,\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty\le2\lVert H-a\mathbf 1^\top-\mathbf 1b^\top\rVert_\infty .
```

Take the infimum over $`(a,b)`$.

*Strictness.* Let $`\mu=\nu=(0.9,0.1)`$ and $`H=e_{11}`$.

- Every coupling has the form $`\Gamma_{11}=x`$, $`\Gamma_{12}=\Gamma_{21}=0.9-x`$, $`\Gamma_{22}=x-0.8`$ with $`x\in[0.8,0.9]`$. So $`\langle\Gamma,H\rangle=x`$ and the width is $`0.1`$.
- Every additive table has rectangle sum $`0`$, while $`H`$ has rectangle sum $`1`$. So the residual $`R=H-a\mathbf 1^\top-\mathbf 1b^\top`$ has rectangle sum $`1`$, and some entry satisfies $`\lvert R_{kl}\rvert\ge1/4`$.
- The checkerboard $`R=\tfrac14\begin{pmatrix}1&-1\cr -1&1\end{pmatrix}`$ attains this, with $`a=(\tfrac34,\tfrac14)`$ and $`b=(0,-\tfrac12)`$.
- The right side of (b) is therefore $`2\cdot\tfrac14=0.5>0.1`$.

Couplings form a restricted feasible set, unlike the unrestricted laws of [Theorem 4.2](04_complexity_of_sufficiency.md#s4-2), so (b) is an upper bound, not a duality equality.

**(c).** Let $`\mu=\nu=(\tfrac12,\tfrac12)`$, forbid the pair $`(2,2)`$, and let $`H=e_{22}`$, the table $`\begin{pmatrix}0&0\cr 0&1\end{pmatrix}`$. Every feasible coupling has $`\Gamma_{22}=0`$, so $`\langle\Gamma,H\rangle=0`$ and the feasible width is $`0`$. But $`H`$ has rectangle sum $`1`$ and is not additive.

**(d).** Let the examples be $`\lbrace (x_k,w_k)\rbrace _{k\in I}`$ with per-example losses $`\ell_k`$, and let a grouping be a partition of $`I`$ into packets.

*One step at a common parameter.* The update is

```math
\theta^+=\theta-\eta\sum_{k\in I}w_k\nabla\ell_k(\theta).
```

It depends only on the multiset of examples and weights, not on the partition. Regrouping therefore changes nothing, so this is a strict null: any measured difference indicates an implementation or randomness difference.

*Two sequential steps.* Packet $`P_1`$ is applied first, then $`P_2`$. Take quadratic losses $`\ell_k(\theta)=\tfrac12(x_k^\top\theta-y_k)^2`$ and write $`g_P(\theta)=\sum_{k\in P}w_k\nabla\ell_k(\theta)`$ and $`H_P=\sum_{k\in P}w_kx_kx_k^\top`$. Then

```math
\theta_1=\theta_0-\eta\,g_{P_1}(\theta_0),\qquad g_{P_2}(\theta_1)=g_{P_2}(\theta_0)-\eta H_{P_2}g_{P_1}(\theta_0),
```

so

```math
\theta_2=\theta_0-\eta\big(g_{P_1}(\theta_0)+g_{P_2}(\theta_0)\big)+\eta^2H_{P_2}\,g_{P_1}(\theta_0).
```

- The first-order term depends only on $`P_1\cup P_2`$.
- The second-order term $`H_{P_2}g_{P_1}(\theta_0)`$ depends on which examples are in which packet, so it generally changes under regrouping.
- Nonlinear networks and Adam add further interactions. $`\square`$

*Reading Proposition 9.1.*

- **What a nonzero width means.** Fixed-marginal reorganization of authentic units can change the learning outcome only through the non-additive part of the joint-packet response $`H_z`$. This is an effect on the learning consequence, separate from per-unit quality, gradient variance or action-level cost.
- **What $`H_z`$ must be.** It is the full-tail response to the whole packet under the declared learning experiment. A table of token-level losses is not a table of learning consequences.
- **Scope.** The result is coupling linear algebra; no new duality is claimed. It turns the question "does packet organization matter for this learner?" into a measurable quantity with an exact null.

## The method

<a id="s9-2"></a>

**Algorithm 9.2 (Consequence-Calibrated Experience Distillation, CED).**

*Input.*

- A student: initial complete learner state $`z_0=(\theta_0,o_0)`$; a declared recipe (update $`U_t`$, optimizer, horizon $`R`$, reference tail); declared deployment tasks with score $`F_E`$.
- Evidence: $`K`$ trusted demonstrations per task family, and a budget of $`n`$ permitted experiments (teacher queries at states the student visits, environment branches, resets).
- Candidate generators that produce feedback laws $`P^1,\dots,P^M`$ (for example replay of the demonstrations, teacher corrections at student states, self-generated trajectories filtered by the environment reward, teacher rewrites of the demonstrations), and the reference law $`P^0`$.
- An audit budget: checkpoints $`t_1<\dots<t_A`$, $`r`$ replicas per candidate and checkpoint, a surrogate objective $`\tilde F`$, a small number of full-tail task-evaluation paired audits for calibration, and a confidence level $`\delta`$.

*Output.*

- The distilled student $`z_R`$.
- An evidence ledger that separates new evidence from replays, rewrites and synthetic descendants.
- At each checkpoint: the paired-replacement scores $`\hat S^{jk}`$ with their radii, the selected law, and whether the selection is certified.
- A frozen selector, and separate method-transfer and policy-transfer reports.

*Procedure.*

1. **Record the complete learner state.** At every checkpoint keep a restorable complete state $`z_t=(\theta_t,o_t)`$, including $`m`$, $`v`$, the step counter, the sampler and the random state. Record warmth as in [Remark 5.11](05_optimizer_state.md#s5-11): the sensitivity-weighted warmth and its maximum term.

   *Justification:* the statistic a feedback law must carry depends on the optimizer state, not on $`\theta`$ alone ([Proposition 5.3](05_optimizer_state.md#s5-3), [Propositions 5.6–5.7](05_optimizer_state.md#s5-6), [Theorem 5.10](05_optimizer_state.md#s5-10)); coverage and transport are statements about complete states ([Theorem 8.2](08_verification.md#s8-2)).
2. **Acquire before compiling.**
   - Run a pilot to estimate, at the states the student visits, the learner-relative weights $`W_s`$ and the teacher coverage $`k_s=KH\,d_T(s)`$. Use the closed forms of [Proposition 6.11](06_learner_relative_information.md#s6-11) when parameter coordinates are state-owned; otherwise measure $`W_s`$ on state clusters with grouped CRN replacement scores ([Definition 8.6](08_verification.md#s8-6)).
   - Allocate the remaining queries by water-filling, $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ ([Proposition 6.10](06_learner_relative_information.md#s6-10)); in the general model this is the c-optimal design with target vector $`c=\sum_tJ_t^*g_t`$ ([Theorem 6.5](06_learner_relative_information.md#s6-5)).
   - Keep the ledger. Only new evidence counts toward the information budget.

   *Justification:* post-processing cannot reduce information regret ([Proposition 2.4](02_feasibility_and_acquisition.md#s2-4), [Corollary 6.7](06_learner_relative_information.md#s6-7)); passive data can leave a gap that only queries close ([Theorem 2.6](02_feasibility_and_acquisition.md#s2-6)); demonstrations and queries share one budget at a learner-relative exchange rate ([Proposition 2.11](02_feasibility_and_acquisition.md#s2-11), [Corollary 6.8](06_learner_relative_information.md#s6-8)); the minimal error is of order $`\mathfrak n_{K,n}`$ and is attained by the c-optimal design ([Theorems 6.3 and 6.5](06_learner_relative_information.md#s6-3)).
3. **Compile candidates.** Turn the evidence into a finite menu of candidate feedback laws $`P^0,\dots,P^M`$. Each candidate is specified by the packets it actually emits: unit content and grouping.

   *Justification:* grouping changes the outcome exactly through the non-additive part of the joint-packet response ([Proposition 9.1](#s9-1)), so a candidate is its packet law, not a bag of units.
4. **Record each candidate's learner statistic and control its magnitude.**
   - At $`z_t`$, record for each candidate the quantities that determine its first-order effect under this optimizer: the mean optimizer increment $`\mathbb E\,\Delta`$ ([Theorem 5.10](05_optimizer_state.md#s5-10); for AdamW the mean Adam increment of [Proposition 5.5](05_optimizer_state.md#s5-5)), the clipping activation rate ([Proposition 5.9](05_optimizer_state.md#s5-9)), and per coordinate the magnitude ratio $`\lambda`$ relative to the running RMS and the signal-to-noise ratio $`\rho`$ read from the optimizer state ([Corollary 7.2](07_second_moment_channel.md#s7-2)).
   - Flag a candidate whose mass on high-$`\rho`$ coordinates lies beyond the value peak $`\lambda^\star=(1-\beta_1^{\,T+1})/(\rho\,\omega_T)`$, or in the large-$`\lambda`$ regime of Corollary 7.2(c). A rescaled or clipped variant of a candidate is a new candidate and enters the menu of step 5.

   *Justification:* candidates with the same mean gradient can differ at first order ([Theorem 5.10](05_optimizer_state.md#s5-10)); under warm Adam a larger same-direction message can be worth less ([Corollary 7.2](07_second_moment_channel.md#s7-2)); parameter-only and cosine scores misrank or tie there ([Proposition 7.3](07_second_moment_channel.md#s7-3)); and first-order scores are unreliable where the second-moment channel dominates ([Remark 7.5](07_second_moment_channel.md#s7-5)). Flagged candidates are therefore ranked only by the finite measurement of step 5. Global-norm clipping caps $`\lambda`$ ([Remark 7.4](07_second_moment_channel.md#s7-4)).
5. **Measure consequence at the actual states.**
   - At $`z_t`$, draw $`r`$ replicas with common random numbers and compute the paired-replacement scores $`\hat S^{jk}`$ for all pairs, including every candidate against the reference ([Definition 8.6](08_verification.md#s8-6)), with radii $`e_{\rm stat}^{jk}`$ ([Theorem 8.8](08_verification.md#s8-8)).
   - Calibrate $`(\alpha,\varepsilon_{\rm sur})`$ on a few candidate pairs against full-tail task-evaluation paired audits ([Theorem 8.3](08_verification.md#s8-3), [Remark 8.11](08_verification.md#s8-11)).
   - First-order scores may serve only as a screen, on pairs whose first-order margin exceeds the second-order remainder ([Corollary 8.9](08_verification.md#s8-9)) and that step 4 did not flag.
   - If a response model on features is fitted, fit it under the bounded-coefficient objective of [Theorem 8.1](08_verification.md#s8-1).
6. **Select, train, re-audit, freeze, test transfer.**
   - Select $`j_t\in\arg\max_j\big[\hat S^{j0}-e_{\rm stat}^{j0}-\varepsilon_{\rm sur}\big]`$. If no candidate has a positive lower bound against the reference, keep the reference; a selection made from point estimates is reported as uncertified.
   - Train with $`P^{j_t}`$ until the next checkpoint. Using the selection at states that were not audited requires the transport and exit terms of [Theorem 8.2](08_verification.md#s8-2), or a re-audit at the next checkpoint that checks whether the ordering still holds ([Remark 8.11](08_verification.md#s8-11)).
   - Freeze the candidate menu, features, cells, surrogate, calibration and selector before the evaluation data. Evaluate method transfer (the frozen protocol rerun on a new task family with its own $`K`$ demonstrations) and policy transfer (the distilled student deployed on the new family with no further update) separately, on held-out families, recipes and states.

*Guarantees.* Each guarantee is the cited theorem applied to the output of the corresponding step; no further argument is needed.

- **(G1) Per checkpoint.** Under [Assumption 8.7](08_verification.md#s8-7), with probability at least $`1-\delta`$, simultaneously for all pairs at $`z_t`$: if step 6 selects $`j_t\ne0`$ because its lower bound is positive, then replacing one round of the reference by $`P^{j_t}`$ at $`z_t`$, followed by the reference tail, does not lower the task value ([Theorem 8.8](08_verification.md#s8-8)).
- **(G2) Whole run.** If the audits of step 5 are full-tail paired audits at the actual states and step 6 selects by their lower confidence bound, then $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$ ([Theorem 8.3](08_verification.md#s8-3)).
- **(G3) Information.** No step that only post-processes the transcript lowers the information regret ([Proposition 2.4](02_feasibility_and_acquisition.md#s2-4)). Under the assumptions of [Theorem 6.5](06_learner_relative_information.md#s6-5), the acquisition of step 2 attains the learner-relative information bound up to the constant factors stated there.
- **(G4) Otherwise empirical.** A selection made from point estimates, a calibration whose transfer is not verified, or a selection used at states that were not audited without transport terms carries no certificate. The claim is then about the mean outcome of the frozen method over a declared population of tasks and seeds. It is supported by whole-run randomized comparisons and is labelled empirical.
- **Known limitation.** With the a-priori range of [Assumption 8.7](08_verification.md#s8-7), the radius of Theorem 8.8 can exceed the typical gaps between candidates; this is an open problem ([Remark 8.12](08_verification.md#s8-12)). In that regime step 6 falls under (G4).

## Derivation of the design rules

<a id="s9-3"></a>

**Remark 9.3 (Derivation of the design rules).** Algorithm 9.2 implements six rules. Each rule states what a method that claims the corresponding guarantee must do, followed by the statements that force it. The rules are necessary conditions for those guarantees; they do not by themselves show that an instance of CED outperforms a given baseline.

**Rule 1 (Acquire before compiling).** Decision-relevant facts that the transcript does not determine can be supplied only by new permitted experiments; how much to acquire, and where, is set by the learner.

- Let $`D'=C(D)`$ be any randomized post-processing of the transcript: replay, rewriting, paraphrase or synthetic descendants. Then $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$ ([Proposition 2.4](02_feasibility_and_acquisition.md#s2-4)). By the decomposition of [Theorem 2.2](02_feasibility_and_acquisition.md#s2-2), compilation can lower only the algorithm term.
- Passive data can leave a gap even when the teacher is optimal: in the two-world family the worst-case regret is at least $`\Delta`$ ([Theorem 2.6](02_feasibility_and_acquisition.md#s2-6)(a)). More generally, if two environments have the same demonstration law and different learner-relative functionals, every distiller, replay included, has worst error at least half their difference, for every $`K`$ ([Corollary 6.7](06_learner_relative_information.md#s6-7)).
- The amount to acquire is bounded below by [Theorem 2.6](02_feasibility_and_acquisition.md#s2-6)(b), [Theorem 2.10](02_feasibility_and_acquisition.md#s2-10) and [Proposition 2.11](02_feasibility_and_acquisition.md#s2-11) in the explicit models, and by $`\tfrac14\mathfrak n_{K,n}`$ in general ([Theorem 6.3](06_learner_relative_information.md#s6-3)). It is attained by the c-optimal design ([Theorem 6.5](06_learner_relative_information.md#s6-5)), which in the state-coverage model is water-filling ([Proposition 6.10](06_learner_relative_information.md#s6-10)).
- The target vector $`c=\sum_tJ_t^*g_t`$ depends on the optimizer, so the same demonstrations can be sufficient for one learner and insufficient for another ([Corollary 6.12](06_learner_relative_information.md#s6-12)). Acquisition must therefore be planned for the learner actually used.
- Compiling an acquired fact into the student has its own cost ([Proposition 2.7](02_feasibility_and_acquisition.md#s2-7)), accounted separately from acquisition.

**Rule 2 (Select by measured consequence).** On a declared family, a selection rule is consistent with the learning-value ordering only if it is a function of the mean responses $`\lbrace\bar g_t(z,j)\rbrace_j`$, or of a statistic proved sufficient for them on that family.

- *The response is the invariant.* Two message laws are interchangeable for every one-boundary replacement followed by the declared tail if and only if they have equal means under every response of the declared class ([Theorem 3.2](03_learning_consequence.md#s3-2)(b)). The value of replacing the reference by candidate $`j`$ at $`z`$ is exactly $`A_t(z,j)`$, and the total effect of a candidate policy is $`\sum_t\mathbb E_{d^S_t}A_t`$ ([Theorem 8.3](08_verification.md#s8-3)(b)).
- *Static value and mean gradients do not identify the ordering.* [Proposition 3.1](03_learning_consequence.md#s3-1)(a) gives two sources that agree on the teacher value and on the static policy value of every aliased student, yet distill students of different value. A rule that depends only on these statistics makes the same choice for the instance and its relabelling, so it selects the worse source in one of them. [Proposition 3.1](03_learning_consequence.md#s3-1)(b) gives two packets with equal label marginals and equal mean gradient whose learning-value difference changes sign between two tasks; a rule depending only on these statistics is wrong for one of the tasks.
- *Two proxy classes are inconsistent.* Task-blind statistics cannot rank correctly for every task, and parameter-only first-order scores and cosine scores misrank or tie in the region described by [Corollary 7.2](07_second_moment_channel.md#s7-2) ([Proposition 7.3](07_second_moment_channel.md#s7-3)). For other static proxies no inconsistency is proved; the static proxies tested in the controlled learner did not reliably select ([experiments, Check 9.4](../experiments/09_mechanism_and_method.md#c9-4)).
- *Packet organization is part of the candidate.* Regrouping units with fixed marginals changes the learning value exactly through the non-additive part of $`H_z`$ ([Proposition 9.1](#s9-1)). A measured response must therefore be the response to the packets actually emitted, not a sum of per-unit scores.
- *No universal small statistic.* For unrestricted learners no fixed family of fewer than $`m-1`$ statistics determines all response means ([Theorem 4.2](04_complexity_of_sufficiency.md#s4-2)(d)). Small sufficient statistics exist only for restricted families, for example joint moments up to order $`N`$ for affine learners ([Theorem 4.4](04_complexity_of_sufficiency.md#s4-4)).

**Rule 3 (Match the learner's statistic, not the raw gradient).** What a candidate must match, to first order, is the mean increment of the complete learner state in the directions of the continuation-value gradients. Responses and their statistics are therefore indexed by complete states and their warmth.

- The first-order representation expresses the outcome difference through the mean increment of the complete state ([Proposition 5.1](05_optimizer_state.md#s5-1)). [Theorem 5.10](05_optimizer_state.md#s5-10) computes that statistic for each learner: the mean gradient for SGD; the clipped mean under clipping, and the mean direction only up to $`c(1-p_{\rm act})`$ ([Proposition 5.9](05_optimizer_state.md#s5-9)); the sign law at a cold Adam step, with magnitudes retained in $`m`$ ([Proposition 5.3](05_optimizer_state.md#s5-3)); the mean Adam increment at a warm state, which reduces to $`(\mathbb EG,\mathbb E\,G\odot G)`$ only under the frozen-denominator approximation ([Proposition 5.5](05_optimizer_state.md#s5-5), [Proposition 5.6](05_optimizer_state.md#s5-6)).
- At a cold state, laws with equal means separate at first order ([Proposition 5.7](05_optimizer_state.md#s5-7)). With $`\beta_2`$ fixed, warmth levels off at $`(1-\beta_2)/\beta_2`$ instead of vanishing ([Proposition 5.8](05_optimizer_state.md#s5-8)), so the frozen-denominator description keeps a per-step error floor and warmth must be measured, not assumed small.
- Under warm Adam, a message also acts through the second-moment channel ([Theorem 7.1](07_second_moment_channel.md#s7-1)); its value is concave in the magnitude relative to the running RMS and becomes negative beyond an explicit threshold in the stated regime ([Corollary 7.2](07_second_moment_channel.md#s7-2)). The magnitude relative to $`\sqrt v`$ is therefore part of the statistic.
- Two states with identical parameters but different optimizer moments emit different next states and can have different responses, so transport conditions are stated on complete states ([Theorem 8.2](08_verification.md#s8-2)).

**Rule 4 (Fit stable response models).** A response model $`c+b^\top\phi`$ on features measured to tolerances $`\epsilon_j`$ is fitted by minimizing $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$.

- [Theorem 8.1](08_verification.md#s8-1) identifies this objective as the exact certified unresolved discrepancy; with a response estimate within $`\eta`$ in sup norm, the certified bound is its value plus $`2\eta`$. For each fixed $`b`$ the bound is the bracket evaluated at $`b`$, so the minimizing $`b`$ gives the best certificate.
- A model with a smaller residual but larger $`\sum_j\epsilon_j\lvert b_j\rvert`$ gives a strictly worse certificate whenever the sum is larger. A zero residual can coexist with a vacuous certificate when the features are small multiples of the response ([Theorem 8.1](08_verification.md#s8-1)).

**Rule 5 (Account for coverage; certify by paired replacement; use first-order audits only where the margin dominates).** A guarantee on the learning value of the selected candidate takes one of four forms:

- (i) the frozen-audit certificate with measured transport and exit probability ([Theorem 8.2](08_verification.md#s8-2)(b));
- (ii) the actual-state lower-confidence-bound guarantee with full-tail paired audits ([Theorem 8.3](08_verification.md#s8-3)(c));
- (iii) the CRN paired-replacement certificate at audited states with a calibrated surrogate ([Theorem 8.8](08_verification.md#s8-8)), extended along training only by transport terms or by re-auditing ([Remark 8.11](08_verification.md#s8-11));
- (iv) whole-run randomized comparisons of the frozen method, labelled empirical.

Derivation:

- Dropping the transport term is invalid: an audit can report zero discrepancy while the learning gap is $`1`$ ([Theorem 8.2](08_verification.md#s8-2)(d)).
- Full-tail audits at the actual states guarantee a loss of at most $`\delta`$, but cost on the order of a thousand paired replicas per candidate ([Theorem 8.3](08_verification.md#s8-3)(c)–(d)).
- The paired-replacement score is exact in expectation, needs no continuation-value gradient and no curvature, and its radius has only a statistical term and a calibration term ([Theorem 8.8](08_verification.md#s8-8)). Under common random numbers its variance is governed by the difference of the two candidates' increments ([Proposition 8.10](08_verification.md#s8-10)).
- A first-order score certifies a pair only if the pair's first-order term exceeds the second-order remainder bound. Pairs whose mean increments agree along the surrogate gradient are therefore never certified at first order, whatever their true gap ([Corollary 8.9](08_verification.md#s8-9)); Theorem 8.8 can certify them.
- Point-estimate selection and transport-free extrapolation carry none of the guarantees (i)–(iii).

**Rule 6 (Freeze, then test transfer).** Selectors, features, cells, surrogates and candidates are frozen before the data that evaluates them; transfer is guaranteed only through a verified decomposition of the target responses and is otherwise measured.

- *Freezing.* The audit event of [Theorem 8.2](08_verification.md#s8-2)(a) is simultaneous over a candidate set fixed before the audit. Reusing audit samples to construct or train a candidate changes its law, and the certificate then needs a new independent audit. The same holds for any selector tuned on its evaluation data.
- *Transfer.* If every target-family response satisfies $`g=c+b^\top\phi+r`$ with $`\lVert b\rVert_1\le B`$ and $`\lVert r\rVert_\infty\le\tau`$, then $`\sup_g\lvert(P-Q)g\rvert\le B\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau`$ ([Theorem 4.2](04_complexity_of_sufficiency.md#s4-2)(c)). Conversely, if some target response lies outside the span of the features, there are feature-matched laws that it separates ([Theorem 4.2](04_complexity_of_sufficiency.md#s4-2)(a)), and no audit on the training family alone certifies the target.
- *Two kinds of transfer.* Method transfer evaluates $`L_{E'}(S)`$ for the frozen protocol $`S`$ applied to a new family $`E'`$ with its own $`K`$ demonstrations; its test class is the response class of $`E'`$, defined through updates. Policy transfer evaluates $`J_{E'}(\pi_z)`$ for the distilled student $`z`$ with no further update; its test functions are the deployment scores of $`E'`$. These are different functionals with different test classes, so a bound on one does not bound the other, and they are reported separately.

## Scope of the evidence

<a id="s9-4"></a>

**Remark 9.4 (What the controlled evidence covers, and the planned real-model tests).**

- **Covered.** Proposition 9.1 is checked by exact finite computations and in a controlled small network; the selection rules are probed on frozen held-out panels of that network ([experiments](../experiments/09_mechanism_and_method.md)). The checks of Theorems A, B and C are on the experiment pages of [§6](../experiments/06_learner_relative_information.md), [§7](../experiments/07_second_moment_channel.md) and [§8](../experiments/08_verification.md). All of them are exact finite computations or controlled small networks.
- **Not covered.** No statement on this page is supported by a real-model or benchmark result. The real-model tests below are designed and not yet run.
- **Planned real-model tests** (ALFWorld; a LoRA student trained by AdamW with gradient clipping, at the main-table recipe). Each configuration runs with a single seed during exploration, and every ranking is pre-registered before the evaluation it predicts.
  - *R0 (re-analysis of the existing pilot).* Report the sensitivity-weighted warmth and its maximum term ([Remark 5.11](05_optimizer_state.md#s5-11)), and measure the non-additivity of the **response** with CRN surrogate tails, alongside the non-additivity of the update map already measured.
  - *R1 (Theorem C: same-state, same-budget ranking).* From a saved complete optimizer state, compare four candidate feedback laws: replay of the $`K`$ demonstrations, teacher corrections at student states (queries), self-generated trajectories filtered by the environment reward, and teacher rewrites of the demonstrations. Rank them by the CRN paired-replacement score with a calibrated surrogate ([Theorem 8.8](08_verification.md#s8-8)), pre-register the ranking, then continue training with each candidate and evaluate. Comparison rankers: LESS, GRACE, gradient norm and static proxies. The criteria are agreement of the certified pairs with the realized ordering, rank correlation, and a separate test of calibration transfer.
  - *R2 (Theorem A: query allocation).* With a fixed query budget, compare no queries, uniform allocation over student states, an SGD-style allocation, and learner-relative water-filling with $`W_s`$ measured by grouped CRN scores ([Proposition 6.10](06_learner_relative_information.md#s6-10)). The predicted order and magnitudes are written down beforehand from the variance formula of Proposition 6.10.
  - *R3 (Theorem B: the second-moment channel).* Build message batches with equal LESS cosine scores but different magnitudes $`\lambda`$ on high signal-to-noise coordinates. The prediction is that pairs with $`\lambda_1+\lambda_2>2\lambda^\star`$ are misranked by cosine and parameter-only scores ([Proposition 7.3](07_second_moment_channel.md#s7-3)); the CRN certificate of R1 measures them.
  - Baselines are run only from complete official code.

## Related work

- **Data and teacher selection.** LESS (Xia et al., 2024) ranks training data by gradient cosine similarity to a target; GRACE (Panigrahi et al., 2025) scores teachers for distillation. Algorithm 9.2 instead ranks candidate feedback laws by their measured consequence at the actual optimizer state, and [Proposition 7.3](07_second_moment_channel.md#s7-3) shows where cosine-type scores are inconsistent.
- **Active acquisition.** DAgger (Ross et al., 2011) queries the teacher at the student's states. Step 2 allocates such queries by learner-relative weights rather than uniformly ([Proposition 6.10](06_learner_relative_information.md#s6-10)).
- **Influence and metagradients.** TracIn (Pruthi et al., 2020), influence functions (Koh & Liang, 2017), metagradients (Engstrom et al., 2025) and In-Run Data Shapley (Wang et al., 2025) are first-order or fixed-state attributions. In Algorithm 9.2 they can serve only as the screen of step 5, under [Corollary 8.9](08_verification.md#s8-9).
- **Packet organization.** Proposition 9.1 is standard coupling linear algebra; its use here is as a mechanism test with an exact null.
