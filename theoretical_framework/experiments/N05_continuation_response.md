# N05 · Continuation-response characterization — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N05_continuation_response.md) · [中文](N05_continuation_response_ZH.md)

**Verdict.** N05 holds. Check C4 verified all three theorems exactly, with 1,867 assertions and worst residual 0, on 45 random finite learning processes with *actual* deterministic update maps and update randomness:

- **(a)** the telescoping identity [N05(a)](../proofs/N05_continuation_response.md) exactly in rationals, and the bound [N05(a)](../proofs/N05_continuation_response.md);
- **(c)** the stopped bound [N05(c)](../proofs/N05_continuation_response.md), under adversarial hill-climbing over the certified sets, which never pushed $`\lvert\text{gap}\rvert-(\text{pre-exit drift}+\rho)`$ above $`0`$;
- **(b)** the one-boundary characterization [N05(b)](../proofs/N05_continuation_response.md) at random snapshots.

Constructed witnesses confirm two consequences: a separating response exists whenever the laws differ on some $`g`$, and drifts across rounds can cancel (final gap 0, total absolute drift 2). Package check T12 confirms the stopped composition on 400 further Markov instances: every bound is valid, every bound is below 1, and the tightest instance reaches 47% of its bound.
![Telescoping, one-boundary equivalence and cancellation](../figures/N05_response.png)

[PDF](../figures/N05_response.pdf)

<a id="c4--n05ac-equations-n05an05c"></a>

### C4 — N05(a)–(c)

**Why.**

- [N05(a)](../proofs/N05_continuation_response.md) is an identity. Any mistake in orientation, such as using the reference occupancy instead of $`d_t^S`$, or in the continuation recursion would give a nonzero residual on random processes.
- [N05(c)](../proofs/N05_continuation_response.md) claims that the exit penalty is $`\rho`$, not $`R\rho`$. A certified-set choice that makes $`\lvert\text{gap}\rvert`$ exceed the bound would falsify it.
- [N05(b)](../proofs/N05_continuation_response.md) claims that one-boundary values differ exactly by $`(P-Q)g`$.

**Setup.**

- 45 random processes with $`\lvert\mathcal Z\rvert\in[4,20]`$ states and $`R\in[1,8]`$ rounds.
- Message alphabets of size 2–4, with update randomness $`\xi`$ taking two values with probabilities $`\tfrac13,\tfrac23`$.
- A random deterministic update table $`U_t(z,y,\xi)\in\mathcal Z`$, and random reference and alternative message laws at every $`(t,z)`$. The kernels $`K_t^0,K_t^S`$ are built from these, as in the theory.
- Rational terminal scores $`F\in[0,1]`$, a rational initial law, and exact `fractions.Fraction` arithmetic.
- Seed `SeedSequence([20261010, 4])`.

**Procedure.**

1. Compute $`V_t^0`$ backward and $`d_t^S`$ forward. Assert $`L(0)-L(S)=\sum_t d_t^S(K_t^0-K_t^S)V_{t+1}^0`$ exactly, and $`\lvert L(0)-L(S)\rvert\le\sum_td_t^S\lvert(K_t^0-K_t^S)V_{t+1}^0\rvert`$.
2. Stopped bound. Certified masks $`C_t`$ are all-certified on even instances and random on odd ones. A 35-step hill-climb flips single $`(t,z)`$ entries to *maximize* $`\lvert\text{gap}\rvert-(\text{pre-exit drift}+\rho)`$, where $`\rho`$ is the exact first-exit probability. At every step the check asserts $`\lvert\text{gap}\rvert\le\text{pre-exit drift}+\rho`$.
3. At a random snapshot $`(t,z)`$ per round, build the actual response vector $`g(y)=\sum_\xi\Pr(\xi)V_{t+1}^0(U_t(z,y,\xi))`$. Assert that the one-boundary value difference equals $`(P-Q)g`$ exactly.
4. Witnesses:
   - Separating response. $`P=(\tfrac12,0,\tfrac12)`$ and $`Q=(0,1,0)`$ agree on $`g_1\equiv1`$ and $`g_2=(0,\tfrac12,1)`$ but differ on $`g_3=(0,0,1)`$.
   - Cancellation. Two states, $`K^0=`$ identity, $`K^S=`$ swap, two rounds, $`F=(0,1)`$, start at state 0.

**Result.**

| Quantity | Value | JSON key (`claims[id=C4]`) |
|---|---|---|
| Processes / assertions / worst residual | 45 / 1,867 / 0 | `n_instances`, `n_assertions`, `worst_residual` |
| Largest $`\lvert\text{gap}\rvert-(\text{pre-exit drift}+\rho)`$ found by adversarial search | $`0.0`$ (never positive) | `details.adversarial_exit_best_lhs_minus_rhs` |
| Cancellation witness: final gap / $`\sum_t\mathbb E\lvert\text{drift}\rvert`$ | $`0`$ / $`2`$ | `details.cancellation_absolute_drift` |
| Separating-response witness | $`P,Q`$ equal on $`g_1,g_2`$, unequal on $`g_3`$ | asserted in `check_C4` |
| Status | PASS | `status` |

**What this shows.**

- The learning-value difference is *exactly* the occupancy-weighted sum of one-boundary response discrepancies, with the alternative's occupancy, as in [N05(a)](../proofs/N05_continuation_response.md).
- An adversary choosing which states to certify could not break the stopped bound. One exit costs at most $`\rho`$.
- One-boundary values differ exactly by $`(P-Q)g`$, so the response class is the right invariant.
- The cancellation witness confirms that pointwise matching is not necessary for a single final mean, which is why [N05(b)](../proofs/N05_continuation_response.md) quantifies over all one-boundary replacements.

### T12 (package) — stopped composition with audited transport

**Why.** This is an independent implementation of the stopped bound in a form close to how it is used later (N14). Responses are written as indicator features with constant baseline. Discrepancies are measured at one audited anchor state and transported to the visited states.

**Setup.**

- 400 random 4-state, 3-round Markov processes. The reference rows $`P_t(z,\cdot)`$ are Dirichlet[N01(a)](../proofs/N01_decomposition_and_ambiguity.md); the alternative is $`Q=0.97P+0.03\,\mathrm{Dirichlet}(1)`$; the initial state is fixed at 1.
- In half the trials state 3 is uncertified, and its mass counts toward $`\rho`$.
- Bound per round: $`\sum_z\Pr(Z_t=z,\text{alive})\,B_t\,(D_t+a_t(z))`$, where:
  - $`b=V_{t+1}(0{:}3)-V_{t+1}(3)`$ and $`B_t=\lVert b\rVert_1`$;
  - $`D_t`$ is the anchor discrepancy at state 0;
  - $`a_t(z)`$ is the sup-norm transport error from state $`z`$ to the anchor.
  The total bound adds $`\rho`$.
- Package seed 15019.

**Result.**

| Quantity | Value | JSON key (`checks.T12_stopped_occupancy_composition`) |
|---|---|---|
| Instances | 400 | `instances` |
| Gap $`\le`$ bound | 400 / 400 (asserted) | — |
| Bounds strictly below 1 (non-vacuous) | 400 / 400 | `bounds_below_one` |
| Largest gap / bound | 0.470 | `maximum_gap_over_bound` |

**What this shows.** The stopped composition stays valid and non-vacuous when the local discrepancies are certified at an audited cell and transported to visited states. This is the form that the frozen-audit theorem (N14) builds on.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C4 --output theory/independent_checks/subset.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: `claims[id=C4]`.
- `theory/independent_checks/check_v014_independent.py`: `process`, `continuations`, `process_stats`, `check_C4`.
- `docs/direction/v015_revision/results_v015.json`: `checks.T12_stopped_occupancy_composition`; code `docs/direction/v015_revision/verify_v015_extensions.py` (`stopped_learning`).
