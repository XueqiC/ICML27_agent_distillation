# §3 Learning consequence — Experiments

[← Framework](../README.md) · [Proofs](../proofs/03_learning_consequence.md) · [中文](03_learning_consequence_ZH.md)

**Verdict.**

- [Proposition 3.1](../proofs/03_learning_consequence.md#s3-1): PASS. Both separations hold as exact identities (75 assertions, residual 0): identical static values $`\tfrac12+p/20`$ with frequency-learner returns exactly $`\tfrac{43}{80}`$ and $`\tfrac{41}{80}`$ for every $`m=1,\dots,30`$; equal packet means with a gap of exactly $`\eta^2/8`$ that reverses to $`-\eta^2/8`$ under $`1-p^2`$.
- The same separation appears in a trained controlled network: in all 24 exactly enumerated configurations ($`R\le4`$, SGD and Adam) the two statically equivalent sources give different trained students, with the sign the proposition predicts.
- [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2): PASS. On 45 random finite learning processes with actual update maps and update randomness (1,867 assertions, residual 0) the telescoping identity (3.2) holds exactly, adversarial choices of certified sets never break the stopped bound (3.5), and one-boundary values differ exactly by $`(P-Q)g`$.
- The stopped composition stays valid and non-vacuous when local discrepancies are certified at one audited cell and transported to visited states: 400 of 400 package instances, tightest at 47% of the bound.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 3.1](#c3-1) | [Proposition 3.1](../proofs/03_learning_consequence.md#s3-1) | `v014:C3` | PASS |
| [Check 3.2](#c3-2) | [Proposition 3.1](../proofs/03_learning_consequence.md#s3-1) in a trained network | `v014:C23` (constructions (a)–(b)) | controlled finding |
| [Check 3.3](#c3-3) | [Theorem 3.2](../proofs/03_learning_consequence.md#s3-2) | `v014:C4` | PASS |
| [Check 3.4](#c3-4) | [Theorem 3.2(c)](../proofs/03_learning_consequence.md#s3-2) with audited transport | `pkg:T12` | PASS |

<a id="c3-1"></a>

## Check 3.1 · Identical static values and identical mean updates, different learning values (Proposition 3.1)

**Verdict.** PASS: 60 instances, 75 assertions, worst residual 0. The gap of Proposition 3.1(a) is exactly $`1/40`$ for every $`m=1,\dots,30`$, and the gap of Proposition 3.1(b) is exactly $`\eta^2/8`$ (and $`-\eta^2/8`$ under $`1-p^2`$).

**Why.**

- Proposition 3.1(a) asserts an *exact* equality of static values for all $`p`$ together with an exact learning-value gap for all $`m`$. An arithmetic slip in the table, a non-optimal teacher or an $`m`$-dependent gap would falsify it.
- Proposition 3.1(b) asserts equal means but a curvature-dependent gap whose sign depends on the task.

**Setup.**

- SymPy symbolic computation in $`p`$ and $`\eta`$.
- Exact binomial expectations of the frequency learner in `fractions.Fraction`, for $`m=1,\dots,30`$ and both sources.
- Statewise teacher optimality is checked at all four states.
- Seed `SeedSequence([20261010, 3])`; the check is deterministic.

**Procedure.**

1. Expand $`J_\rho(\pi_p)`$ for both state laws and assert equality with $`\tfrac12+p/20`$.
2. Assert that the teacher's value is $`\tfrac35`$ under both laws and that its action is optimal at every state.
3. For each $`m`$, compute $`\mathbb E\,J(\pi_{\widehat p})`$ exactly as a binomial sum and assert $`\tfrac{43}{80}`$ and $`\tfrac{41}{80}`$.
4. Enumerate the four equally likely label pairs; assert $`\mathbb Ep^+=\tfrac12`$, the gap $`\eta^2/8`$ for $`p^2`$ and the gap $`-\eta^2/8`$ for $`1-p^2`$.

**Result.**

| Quantity | Value | JSON key (entry `v014:C3`) |
|---|---|---|
| Instances / assertions / worst residual | 60 / 75 / 0 | `n_instances`, `n_assertions`, `worst_residual` |
| Gap of Proposition 3.1(a) (every $`m=1,\dots,30`$) | $`1/40`$ | `details.prop5_gap` |
| Gap of Proposition 3.1(b) | $`\eta^2/8`$ (and $`-\eta^2/8`$ under $`1-p^2`$) | `details.prop6_gap` |
| Status | PASS | `status` |

**Why it supports the statement.** Both separations are exact identities, not approximations. Two sources can agree on every static quantity a selector might check for the student class (every aliased student's value, the teacher's value, the label marginals and the mean gradient) and still produce different learned students.

<a id="c3-2"></a>

## Check 3.2 · The same separation in a trained network (Proposition 3.1, controlled network)

**Verdict.** Controlled finding: in all 24 exactly enumerated configurations ($`R\le4`$) the two statically equivalent sources of each construction give different trained students, with the sign predicted by Proposition 3.1. Scope: a controlled small network, not a real model and not a benchmark.

**Why.** Proposition 3.1 uses idealized learners (frequency distillation, one SGD step). The question is whether a nonlinear network trained by gradient methods also distinguishes the statically equivalent sources. A zero gap would mean the separation is an artifact of the idealized learner.

**Setup.**

- Network: one-hot context → 4 tanh units → sigmoid action probability, fixed initialization, cross-entropy on label blocks.
- Optimizers: SGD with $`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$; Adam ($`\beta=(0.9,0.999)`$, $`\epsilon=10^{-8}`$, bias correction) with $`\eta\in\lbrace 0.01,0.1\rbrace`$.
- $`R\in\lbrace 1,4,40\rbrace`$ updates, using the source laws of Constructions (a) and (b).
- For $`R\le4`$ all training histories are enumerated exactly. For $`R=40`$, 2,400 independent training runs per source; SE counts training runs.
- Seed `SeedSequence([20261010, 23])`.

**Result (gap = reference − alternative for (a), independent − paired for (b); exact enumeration unless an SE is given).**

| Construction | Optimizer, $`\eta`$ | $`R=1`$ | $`R=4`$ | $`R=40`$ (SE) |
|---|---|---:|---:|---:|
| (a) | SGD 0.01 | 1.21e-4 | 4.81e-4 | 4.38e-3 (1.7e-5) |
| (a) | SGD 0.1 | 1.21e-3 | 4.48e-3 | 2.10e-2 (9.2e-5) |
| (a) | SGD 1.0 | 1.10e-2 | 1.88e-2 | 2.44e-2 (2.0e-4) |
| (a) | Adam 0.01 | 5.04e-4 | 2.00e-3 | 1.54e-2 (6.9e-5) |
| (a) | Adam 0.1 | 4.91e-3 | 1.54e-2 | 2.40e-2 (1.9e-4) |
| (b) | SGD 0.01 | 3.47e-6 | 1.36e-5 | 3.7e-5 (2.0e-4) |
| (b) | SGD 0.1 | 3.46e-4 | 1.15e-3 | 2.0e-3 (1.1e-3) |
| (b) | SGD 1.0 | 2.86e-2 | 1.86e-2 | 1.61e-2 (2.7e-3) |
| (b) | Adam 0.01 | 9.19e-5 | 6.33e-4 | 1.23e-3 (9.1e-4) |
| (b) | Adam 0.1 | 8.75e-3 | 2.76e-2 | 1.23e-2 (2.4e-3) |

Values are read from `controlled_neural.details.configurations[task∈{Prop5,Prop6}].neural_gap` and `.SE`; the rows with $`\eta=3`$ are in the JSON. At small SGD steps the measured gap matches the network's local tangent prediction: for (a) with SGD $`\eta=0.01`$, $`R=1`$, 1.21170e-4 measured against 1.21171e-4 predicted (`.local_affine_SGD_prediction`).

![Figure 3.1: equal evaluation, different learned students](../figures/fig3_1_static_vs_learning.png)

*Figure 3.1.* (a) Construction (a): every aliased student has the same value $`\tfrac12+p/20`$ under both sources, and the teacher has value $`\tfrac35`$ under both, yet frequency distillation yields $`\tfrac{43}{80}`$ against $`\tfrac{41}{80}`$. (b) Construction (b): identical label marginals, mean gradient and mean update, but a learning-value gap of $`+\eta^2/8`$ under $`J=p^2`$ and $`-\eta^2/8`$ under $`J=1-p^2`$. (c) In the controlled tanh network, the statically equivalent sources still give different trained students (exact enumeration, $`R\le4`$; log scale). [PDF](../figures/fig3_1_static_vs_learning.pdf)

**Why it supports the statement.**

- In a trained network, as in the proposition, statically equivalent feedback sources yield different students. Every exactly enumerated gap is positive, matching the predicted direction: the reference source is better in (a), and independent packets are better under $`p^2`$ in (b).
- The gap grows with the step size, as a learning-consequence effect should; in (a) it also grows with the number of updates for $`\eta\le1`$.
- At $`R=40`$ (Monte Carlo) three gaps of construction (b) are within two standard errors of zero (SGD $`\eta=0.01`$ and $`0.1`$, Adam $`\eta=0.01`$); the exact rows carry the evidence.
- The magnitude depends on the optimizer: at equal $`\eta`$ the Adam gaps exceed the SGD gaps. §5 explains this dependence ([Proposition 5.3](../proofs/05_optimizer_state.md#s5-3), cold-start Adam).

<a id="c3-3"></a>

## Check 3.3 · Exact composition, stopped composition and one-boundary characterization (Theorem 3.2)

**Verdict.** PASS: 45 random finite learning processes, 1,867 assertions, worst residual 0. The adversarial search over certified sets never made $`\lvert\text{gap}\rvert-(\text{pre-exit drift}+\rho)`$ positive.

**Why.**

- (3.2) is an identity. A mistake in orientation (for instance using the reference occupancy instead of $`d_t^S`$) or in the continuation recursion would give a nonzero residual on random processes.
- (3.5) claims that the exit penalty is $`\rho`$, not $`R\rho`$. A certified-set choice that makes $`\lvert\text{gap}\rvert`$ exceed the bound would falsify it.
- (3.4) rests on one-boundary values differing exactly by $`(P-Q)g`$.

**Setup.**

- 45 random processes with $`\lvert\mathcal Z\rvert\in[4,20]`$ states and $`R\in[1,8]`$ rounds.
- Message alphabets of size 2–4; update randomness $`\xi`$ takes two values with probabilities $`\tfrac13,\tfrac23`$.
- A random deterministic update table $`U_t(z,y,\xi)\in\mathcal Z`$ and random reference and alternative message laws at every $`(t,z)`$; the kernels $`K_t^0,K_t^S`$ are built from these, as in the theorem.
- Rational terminal scores $`F\in[0,1]`$, a rational initial law, exact `fractions.Fraction` arithmetic. Seed `SeedSequence([20261010, 4])`.

**Procedure.**

1. Compute $`V_t^0`$ backward and $`d_t^S`$ forward. Assert $`L(0)-L(S)=\sum_t d_t^S(K_t^0-K_t^S)V_{t+1}^0`$ exactly and $`\lvert L(0)-L(S)\rvert\le\sum_td_t^S\lvert(K_t^0-K_t^S)V_{t+1}^0\rvert`$.
2. Stopped bound. Certified masks $`C_t`$ are all-certified on even instances and random on odd ones. A 35-step hill-climb flips single $`(t,z)`$ entries to *maximize* $`\lvert\text{gap}\rvert-(\text{pre-exit drift}+\rho)`$, with $`\rho`$ the exact first-exit probability; at every step the check asserts $`\lvert\text{gap}\rvert\le\text{pre-exit drift}+\rho`$.
3. At a random snapshot $`(t,z)`$ per round, build the actual response vector $`g(y)=\sum_\xi\Pr(\xi)V_{t+1}^0(U_t(z,y,\xi))`$ and assert that the one-boundary value difference equals $`(P-Q)g`$ exactly.
4. Witnesses:
   - *Separating response.* $`P=(\tfrac12,0,\tfrac12)`$ and $`Q=(0,1,0)`$ agree on $`g_1\equiv1`$ and $`g_2=(0,\tfrac12,1)`$ but differ on $`g_3=(0,0,1)`$.
   - *Cancellation.* Two states, $`K^0`$ the identity, $`K^S`$ the swap, two rounds, $`F=(0,1)`$, start at state 0.

**Result.**

| Quantity | Value | JSON key (entry `v014:C4`) |
|---|---|---|
| Processes / assertions / worst residual | 45 / 1,867 / 0 | `n_instances`, `n_assertions`, `worst_residual` |
| Largest $`\lvert\text{gap}\rvert-(\text{pre-exit drift}+\rho)`$ found by adversarial search | $`0.0`$ (never positive) | `details.adversarial_exit_best_lhs_minus_rhs` |
| Cancellation witness: final gap / $`\sum_t\mathbb E\lvert\text{drift}\rvert`$ | $`0`$ / $`2`$ | `details.cancellation_absolute_drift` |
| Separating-response witness | $`P,Q`$ equal on $`g_1,g_2`$, unequal on $`g_3`$ | asserted in `check_C4` |
| Status | PASS | `status` |

![Figure 3.2: telescoping, one-boundary equivalence and cancellation](../figures/fig3_2_response.png)

*Figure 3.2.* (a) On an explicit three-state process, the learning-value gap equals the sum over rounds of the occupancy-weighted response discrepancies (3.2), and is bounded by the sum of their absolute values (3.3). (b) The laws $`P=(\tfrac12,0,\tfrac12)`$ and $`Q=(0,1,0)`$ agree on two responses and differ on a third: two laws are interchangeable for one-boundary replacements exactly when they agree on every response (3.4). (c) Per-round discrepancies can cancel (final gap 0, total absolute drift 2), so matching one final mean does not require pointwise matching. [PDF](../figures/fig3_2_response.pdf)

**Why it supports the statement.**

- The learning-value difference is *exactly* the occupancy-weighted sum of one-boundary response discrepancies, with the alternative's occupancy, as in (3.2).
- An adversary choosing which states to certify could not break the stopped bound: one exit costs at most $`\rho`$ (3.5).
- One-boundary values differ exactly by $`(P-Q)g`$, so the response class is the right invariant (3.4).
- The cancellation witness confirms that pointwise matching is not necessary for a single final mean, which is why Theorem 3.2(b) quantifies over all one-boundary replacements.

<a id="c3-4"></a>

## Check 3.4 · Stopped composition with audited transport (Theorem 3.2(c))

**Verdict.** PASS: 400 random Markov instances; every bound is valid, every bound is below 1, and the largest gap-to-bound ratio is 0.470.

**Why.** An independent implementation of the stopped bound in the form in which [Theorem 8.2](../proofs/08_verification.md#s8-2) uses it: responses written as indicator features with a constant baseline, discrepancies measured at one audited anchor state and transported to the visited states.

**Setup.**

- 400 random 4-state, 3-round Markov processes. Reference rows $`P_t(z,\cdot)`$ are drawn from $`\mathrm{Dirichlet}(1)`$; the alternative is $`Q=0.97P+0.03\,\mathrm{Dirichlet}(1)`$; the initial state is fixed.
- In half the trials state 3 is uncertified, and the mass that reaches it counts toward $`\rho`$.
- Bound per round: $`\sum_z\Pr(Z_t=z,\text{alive})\,B_t\,(D_t+a_t(z))`$, where
  - $`b=V_{t+1}(0{:}3)-V_{t+1}(3)`$ and $`B_t=\lVert b\rVert_1`$;
  - $`D_t`$ is the anchor discrepancy at state 0;
  - $`a_t(z)`$ is the sup-norm transport error from state $`z`$ to the anchor.

  The total bound adds $`\rho`$.
- Package seed 15019.

**Result.**

| Quantity | Value | JSON key (`checks.T12_stopped_occupancy_composition`) |
|---|---|---|
| Instances | 400 | `instances` |
| Gap ≤ bound | 400 / 400 (asserted) | — |
| Bounds strictly below 1 (non-vacuous) | 400 / 400 | `bounds_below_one` |
| Largest gap / bound | 0.470 | `maximum_gap_over_bound` |

**Why it supports the statement.** The stopped composition (3.5) stays valid and non-vacuous when the local discrepancies $`\varepsilon_t`$ are certified at an audited cell and transported to the visited states. This is the form on which the frozen-audit theorem ([Theorem 8.2](../proofs/08_verification.md#s8-2)) builds.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C3,C4,C23 --output theory/independent_checks/subset.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: the entries of `v014:C3` and `v014:C4`; `controlled_neural.details.configurations` (filter `task` = `Prop5` or `Prop6`).
- `theory/independent_checks/check_v014_independent.py`: `check_C3`, `check_C4` (with `process`, `continuations`, `process_stats`), `check_C23`.
- `docs/direction/v015_revision/results_v015.json`: `checks.T12_stopped_occupancy_composition`; code `docs/direction/v015_revision/verify_v015_extensions.py` (`stopped_learning`).
- Figures: `figures/fig3_1_static_vs_learning.{png,pdf}`, `figures/fig3_2_response.{png,pdf}`; specs in `figures/specs/fig3_1.md`, `fig3_2.md`.
