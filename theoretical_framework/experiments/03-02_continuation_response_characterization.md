# Theorem 3.2 · Continuation-response characterization

*Experiments*

[← Framework](../README.md) · [Statement](../statements/03-02_continuation_response_characterization.md) · [Proof](../proofs/03-02_continuation_response_characterization.md) · [Experiments](03-02_continuation_response_characterization.md) · [中文](03-02_continuation_response_characterization_ZH.md)

**What is tested here.** [Check 3.3](../experiments/03-02_continuation_response_characterization.md#c3-3) tests all three parts on 45 random finite learning processes with actual update maps and update randomness (1,867 assertions, residual 0): the telescoping identity (3.2) holds exactly, one-boundary values differ exactly by $`(P-Q)g`$, and an adversarial search over certified sets never breaks the stopped bound (3.5) (PASS). [Check 3.4](../experiments/03-02_continuation_response_characterization.md#c3-4) confirms the stopped composition when discrepancies are certified at one audited state and transported to the visited states: 400 of 400 instances valid and non-vacuous, tightest at 47% of the bound (PASS).

**Setting used on this page.**

- The complete learner state $`z\in\mathcal Z`$: parameters, optimizer state, buffers, sampler state, randomness and history ([Definition 1.3](../README.md#s1-3)).
- At round $`t<R`$ a feedback law $`P_t(\cdot\mid z)`$ emits a message $`Y_t`$ and the actual update gives $`Z_{t+1}=U_t(Z_t,Y_t,\xi_t)`$, with update randomness $`\xi_t`$. Integrating $`Y_t`$ and $`\xi_t`$ gives the learning kernel $`K_t`$, acting on bounded functions by $`(K_tf)(z)=\mathbb E\,f(U_t(z,Y,\xi))`$.
- $`K_t^0`$ is the reference protocol and $`K_t^S`$ an alternative protocol $`S`$; both start from the same initial law $`\nu`$. $`Z_t^S`$ is the chain driven by $`K^S`$, and its law $`d_t^S`$ is the occupancy at round $`t`$.
- $`F_E(z)\in[0,1]`$ is the real-task return of the student deployed from state $`z`$; $`L_E(S)=\mathbb E\,F_E(Z_R^S)`$ is the learning value of protocol $`S`$, and $`L_E(0)`$ that of the reference.
- The reference continuation value is $`V_R^0=F_E`$, $`V_t^0=K_t^0V_{t+1}^0`$: the expected final score when training continues from a state under the reference protocol ([Definition 1.4](../README.md#s1-4)).
- The one-boundary response at a snapshot $`(t,z)`$ is $`g_{t,z}(y)=\mathbb E_{\xi_t}V_{t+1}^0(U_t(z,y,\xi_t))`$: the expected final return after message $`y`$, the actual update and the declared reference tail. The response class $`\mathcal G`$ collects $`g_{t,z}`$ over a declared family $`\mathcal I`$ of snapshots.
- A one-boundary replacement $`I(t,z,P)`$: from state $`z`$ at round $`t`$, draw $`Y\sim P`$, apply the actual update, follow the declared reference tail, score $`F_E`$.
- Certified sets $`C_t\subseteq\mathcal Z`$, the first exit time $`\tau`$ and the exit probability $`\rho=\Pr(\tau<R)`$ are defined in part (c).

**Statement under test ([Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2)).** In this setting:

**(a) Exact composition.**

```math
L_E(0)-L_E(S)=\sum_{t=0}^{R-1}\mathbb E_{d_t^S}\big[(K_t^0-K_t^S)V_{t+1}^0(Z_t)\big]. \qquad(3.2)
```

Consequently, if $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$ for $`d_t^S`$-almost every $`z`$,

```math
\lvert L_E(0)-L_E(S)\rvert\le\sum_t\mathbb E_{d_t^S}\varepsilon_t(Z_t). \qquad(3.3)
```

**(b) Operational characterization.** Two message laws $`P,Q`$ yield equal value for every one-boundary replacement in the declared family if and only if

```math
\mathbb E_Pg=\mathbb E_Qg\quad\text{for every }g\in\mathcal G. \qquad(3.4)
```

**(c) Stopped composition.** Let $`C_t\subseteq\mathcal Z`$ be certified sets, $`\tau=\min\lbrace t<R:Z_t^S\notin C_t\rbrace`$ (with $`\tau=R`$ if there is no such $`t`$), and $`\rho=\Pr(\tau<R)`$. If $`\lvert(K_t^0-K_t^S)V_{t+1}^0(z)\rvert\le\varepsilon_t(z)`$ for all $`z\in C_t`$, then

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau}\varepsilon_t(Z_t)+\rho. \qquad(3.5)
```

<a id="c3-3"></a>

## Check 3.3 · Exact composition, stopped composition and one-boundary characterization (Theorem 3.2)

**Verdict.** PASS: 45 random finite learning processes, 1,867 assertions, worst residual 0. The adversarial search over certified sets never made $`\lvert\text{gap}\rvert-(\text{pre-exit drift}+\rho)`$ positive.

**Why.**

- (3.2) is an identity of performance-difference type ([Kakade & Langford, 2002](#ref-kakade2002approximately)). A mistake in orientation (for instance using the reference occupancy instead of $`d_t^S`$) or in the continuation recursion would give a nonzero residual on random processes.
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

**Why.** An independent implementation of the stopped bound in the form in which [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) uses it: responses written as indicator features with a constant baseline, discrepancies measured at one audited anchor state and transported to the visited states.

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

**Why it supports the statement.** The stopped composition (3.5) stays valid and non-vacuous when the local discrepancies $`\varepsilon_t`$ are certified at an audited cell and transported to the visited states. This is the form on which the frozen-audit theorem ([Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)) builds.

## Reproduce

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C4 --output theory/independent_checks/subset.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: the entry of `v014:C4`; code `theory/independent_checks/check_v014_independent.py` (`check_C4` with `process`, `continuations`, `process_stats`).
- `docs/direction/v015_revision/results_v015.json`: `checks.T12_stopped_occupancy_composition`; function `stopped_learning`; code `docs/direction/v015_revision/verify_v015_extensions.py`.
- Figures: `figures/fig3_2_response.{png,pdf}`; specs `figures/specs/fig3_2.md`.
