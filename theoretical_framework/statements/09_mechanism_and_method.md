# §9 Mechanism and method — Statements

[← Framework](../README.md) · [Proofs](../proofs/09_mechanism_and_method.md) · [Experiments](../experiments/09_mechanism_and_method.md) · [中文](09_mechanism_and_method_ZH.md)

### §9 · Mechanism and method — exact statements

[Proofs](../proofs/09_mechanism_and_method.md) · [Experiments](../experiments/09_mechanism_and_method.md)

**What this section does.**

- It isolates one learning-consequence mechanism with an exact null: the joint organization of feedback units whose marginals are fixed ([Proposition 9.1](../proofs/09_mechanism_and_method.md#s9-1)).
- It states the method that follows from §2–§8, Consequence-Calibrated Experience Distillation (CED), as a numbered procedure in which every step cites the statement that justifies it and every output carries a stated guarantee or is labelled empirical ([Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2)).
- It derives the six design rules behind the procedure ([Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3)) and separates the controlled evidence from the planned real-model tests ([Remark 9.4](../proofs/09_mechanism_and_method.md#s9-4)).

**Setting.**

- Two slots of a training packet are filled from finite unit sets $`\mathcal Y_1,\mathcal Y_2`$ with strictly positive marginals $`\mu,\nu`$. The coupling polytope $`\mathcal C(\mu,\nu)`$ is the set of nonnegative tables with row sums $`\mu`$ and column sums $`\nu`$.
- $`H_z(i,j)`$ is the expected final return after the actual update on the joint packet $`(i,j)`$ at complete state $`z`$, followed by the reference tail: a one-boundary response with the packet as the message. The coupling width is $`w(H_z)=\max_\Gamma\langle\Gamma,H_z\rangle-\min_\Gamma\langle\Gamma,H_z\rangle`$ over $`\Gamma\in\mathcal C(\mu,\nu)`$.
- A candidate feedback law (provider) $`j\in\lbrace0,\dots,M\rbrace`$ emits messages $`Y\sim P^j_t(\cdot\mid z)`$; candidate $`0`$ is the reference. Its mean response is $`\bar g_t(z,j)=\mathbb E_{P^j_t}g_{t,z}(Y)`$ and its signed advantage is $`A_t(z,j)=\bar g_t(z,j)-\bar g_t(z,0)`$.

#### Proposition 9.1 · Coupling width

**(a)** $`w(H_z)=0`$ if and only if $`H_z(i,j)=a_i+b_j`$ for some vectors $`a,b`$.

**(b)** For all couplings $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$,

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty ,
```

and the inequality can be strict.

**(c)** With forbidden pairings, zero feasible width does not imply additivity.

**(d)** One SGD step at a common parameter, on the union of all packets with identical per-example weights, is invariant to regrouping. Sequential steps are not; for quadratic losses the difference is $`\eta^2H_{P_2}g_{P_1}(\theta_0)`$.

*Derivation outline.*

- (a) An additive table has the same inner product $`\mu^\top a+\nu^\top b`$ with every coupling. Conversely, the product coupling $`\mu\nu^\top`$ is interior, so every rectangle move $`e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$ is feasible in both directions; zero width forces every rectangle sum of $`H_z`$ to vanish, which is additivity.
- (b) The difference of two couplings has zero row and column sums, so it annihilates additive tables, and its $`\ell_1`$ norm is at most 2; Hölder's inequality gives the bound.
- (c) Forbidding one pair makes the indicator of that pair invisible to every feasible coupling, although it is not additive.
- (d) One step sums the per-example gradients at the same parameter, which does not depend on the partition. In two steps the second packet's gradient is evaluated after the first update, which produces the cross term $`\eta^2H_{P_2}g_{P_1}`$.

*Illustration.* With $`\mu=\nu=(0.9,0.1)`$ and $`H=e_{11}`$, every coupling has $`\Gamma_{11}=x\in[0.8,0.9]`$, so the width is $`0.1`$; the best additive approximation leaves a checkerboard residual of size $`1/4`$, so the bound (b) is $`0.5`$. The bound is valid but loose.

*Uses* [Definition 1.4](../README.md#s1-4), [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2). *Used by* [Algorithm 9.2](../proofs/09_mechanism_and_method.md#s9-2) (step 3), [Remark 9.3](../proofs/09_mechanism_and_method.md#s9-3) (Rule 2).

[Proof](../proofs/09_mechanism_and_method.md#s9-1) · [Experiments](../experiments/09_mechanism_and_method.md#c9-1) · proved

#### Algorithm 9.2 · Consequence-Calibrated Experience Distillation (CED)

*Input:* an initial complete learner state, a declared recipe and deployment tasks, $`K`$ trusted demonstrations per task family, a budget of $`n`$ permitted experiments, candidate generators with a reference law $`P^0`$, and an audit budget (checkpoints, $`r`$ CRN replicas, a surrogate $`\tilde F`$, a few full-tail calibration audits, confidence $`\delta`$). *Output:* the distilled student, an evidence ledger, per-checkpoint scores with radii and the selected law (certified or not), a frozen selector, and separate method-transfer and policy-transfer reports.

1. **Record the complete learner state** at each checkpoint, with its warmth ([Remark 5.11](../proofs/05_learner_statistics.md#s5-11)); the learner's statistic depends on the optimizer state ([Theorem 5.10](../proofs/05_learner_statistics.md#s5-10)).
2. **Acquire before compiling.** Estimate learner-relative weights $`W_s`$ and teacher coverage $`k_s`$ in a pilot, allocate the remaining queries by water-filling $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$ ([Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10)), the c-optimal design of [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5), and keep a ledger in which only new evidence counts ([Proposition 2.4](../proofs/02_feasibility_and_acquisition.md#s2-4)).
3. **Compile candidates** as packet laws, since grouping is part of the response ([Proposition 9.1](../proofs/09_mechanism_and_method.md#s9-1)).
4. **Record each candidate's learner statistic and control its magnitude:** the mean optimizer increment, the clipping activation rate, and per-coordinate $`\lambda`$ and $`\rho`$; flag candidates beyond the value peak $`\lambda^\star`$ of [Corollary 7.2](../proofs/07_second_moment_channel.md#s7-2), whose ranking then relies only on step 5 ([Proposition 7.3](../proofs/07_second_moment_channel.md#s7-3), [Remark 7.5](../proofs/07_second_moment_channel.md#s7-5)).
5. **Measure consequence at the actual states** by CRN paired-replacement scores with radii ([Theorem 8.8](../proofs/08_certificate.md#s8-8)), calibrated against full-tail paired audits ([Theorem 8.3](../proofs/08_verification.md#s8-3), [Remark 8.11](../proofs/08_certificate.md#s8-11)); first-order scores serve only as a screen above the margin ([Corollary 8.9](../proofs/08_certificate.md#s8-9)).
6. **Select by lower confidence bound, train, re-audit, freeze, test transfer.** Keep the reference if no lower bound is positive; account for transport when using a selection at unaudited states ([Theorem 8.2](../proofs/08_verification.md#s8-2)); evaluate method transfer and policy transfer separately.

*Guarantees.* (G1) a certified selection does not lower the task value of the one-round replacement at the audited state, with probability at least $`1-\delta`$ ([Theorem 8.8](../proofs/08_certificate.md#s8-8)); (G2) with full-tail audits at the actual states, $`L_E({\rm adaptive})-L_E(0)\ge-\delta`$ over the run ([Theorem 8.3](../proofs/08_verification.md#s8-3)); (G3) acquisition attains the learner-relative information bound up to the constants of [Theorem 6.5](../proofs/06_learner_relative_information.md#s6-5); (G4) everything else is empirical and labelled so. With the a-priori score range the certificate radius can exceed typical gaps; that is open ([Remark 8.12](../proofs/08_certificate.md#s8-12)).

*Illustration.* Suppose the reference replays the $`K`$ demonstrations and a second candidate adds teacher corrections at long-tail student states. For states visited often enough to be in the frequent regime of [Proposition 6.11](../proofs/06_learner_relative_information.md#s6-11), the learner-relative weight under warm Adam scales like the visitation rate $`d(s)`$, not like $`d(s)^2`$ as under SGD, so relative to the common states the long-tail states weigh more, and step 2 spends more queries there than an SGD-style allocation would; step 5 then decides by measured consequence whether the corrections, compiled as packets, actually beat replay at the current state.

*Uses* §2–§8 as cited in each step. *Used by* the planned real-model tests ([Remark 9.4](../proofs/09_mechanism_and_method.md#s9-4)).

[Proof](../proofs/09_mechanism_and_method.md#s9-2) · [Experiments](../experiments/09_mechanism_and_method.md#c9-4) · derived design rule

#### Remark 9.3 · The six design rules

1. **Acquire before compiling.** Post-processing cannot lower the information regret, passive data can leave a gap even with an optimal teacher, and the amount and placement of acquisition are set by the learner ([Proposition 2.4](../proofs/02_feasibility_and_acquisition.md#s2-4), [Theorem 2.6](../proofs/02_feasibility_and_acquisition.md#s2-6), [Theorems 6.3](../proofs/06_learner_relative_information.md#s6-3) and [6.5](../proofs/06_learner_relative_information.md#s6-5), [Proposition 6.10](../proofs/06_learner_relative_information.md#s6-10)).
2. **Select by measured consequence.** The response is the invariant ([Theorem 3.2](../proofs/03_learning_consequence.md#s3-2)); static value and mean gradients do not identify the ordering ([Proposition 3.1](../proofs/03_learning_consequence.md#s3-1)); task-blind statistics and parameter-only or cosine scores are inconsistent ([Proposition 7.3](../proofs/07_second_moment_channel.md#s7-3)); the other static proxies tested did not reliably select in the controlled learner.
3. **Match the learner's statistic, not the raw gradient.** The first-order statistic is the mean increment of the complete state ([Proposition 5.1](../proofs/05_optimizer_state.md#s5-1), [Theorem 5.10](../proofs/05_learner_statistics.md#s5-10)); under warm Adam the magnitude relative to the running RMS is part of it ([Theorem 7.1](../proofs/07_second_moment_channel.md#s7-1), [Corollary 7.2](../proofs/07_second_moment_channel.md#s7-2)).
4. **Fit stable response models.** Measurement error on a feature costs the size of its coefficient ([Theorem 8.1](../proofs/08_verification.md#s8-1)).
5. **Account for coverage; certify by paired replacement; first-order audits only where the margin dominates** ([Theorems 8.2](../proofs/08_verification.md#s8-2), [8.3](../proofs/08_verification.md#s8-3), [8.8](../proofs/08_certificate.md#s8-8), [Corollary 8.9](../proofs/08_certificate.md#s8-9)).
6. **Freeze, then test transfer.** Audit events are simultaneous only over candidates fixed beforehand; transfer is guaranteed only through a verified decomposition of target responses ([Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)(c)); method transfer and policy transfer are different functionals.

The rules are necessary conditions for the stated guarantees; they do not by themselves show that an instance of CED beats a given baseline.

[Proof](../proofs/09_mechanism_and_method.md#s9-3) · [Experiments](../experiments/09_mechanism_and_method.md#c9-4) · derived design rule

#### Remark 9.4 · Scope of the evidence

- The evidence for Proposition 9.1 and for the selection rules consists of exact finite computations and a controlled small network. No real-model claim is made.
- The planned real-model tests are R0 (re-analysis of the pilot: sensitivity-weighted warmth and response non-additivity), R1 (Theorem C: same-state, same-budget ranking of four candidate feedback laws by the CRN certificate, pre-registered and then trained out), R2 (Theorem A: query allocation by learner-relative water-filling against uniform and SGD-style allocations) and R3 (Theorem B: message batches with equal cosine scores but different magnitudes).

[Proof](../proofs/09_mechanism_and_method.md#s9-4) · [Experiments](../experiments/09_mechanism_and_method.md) · open

---
