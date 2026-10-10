# N01 · Regret decomposition and decision ambiguity — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N01_decomposition_and_ambiguity.md) · [中文](N01_decomposition_and_ambiguity_ZH.md)

**Verdict.** N01 holds. Independent check C1 verified all four propositions in exact rational arithmetic on 500 random finite decision problems, with 5,810 assertions and no violation:

- the decomposition identity (1) and the nonnegativity of all three terms;
- the zero-information criterion (2);
- primal = dual for the robust ambiguity value (3)=(4), solved as two separate LPs on 220 instances;
- data processing (5), searched adversarially over deterministic and randomized garblings, where the largest observed increase of $B$ was exactly $0$.

The package check `robust_ambiguity_minimax` reproduces (3)=(4) on the canonical two-world instance, with primal and dual both equal to $0.1$.

![Regret decomposition, data processing and minimax duality](../figures/N01_decomposition.png)

[PDF](../figures/N01_decomposition.pdf)

### C1 — Propositions 1–3, equations (1)–(5)

**Why.** The decomposition is an identity, so any algebraic error in how $B(D)$, the posterior or the learner kernel is defined would show up as a nonzero residual. The criterion (2) is an "if and only if"; a counterexample in either direction would falsify it. The duality (3)=(4) and the data-processing inequality (5) are inequalities that a randomly generated or adversarially chosen instance could violate if a hypothesis were missing.

**Setup.**

- 500 random instances, with 3–6 environments and a random rational prior, and 2–5 policies in $\Pi$ plus two extra policies in $\Pi_{\rm ref}$.
- Values $J_E(\pi)\in[0,1]$ are random rationals with denominator 12.
- The transcript alphabet has size $\lvert\mathcal D\rvert=4$, with a random channel $P(D\mid E)$ and a random learner kernel $\lambda(\cdot\mid D)$.
- One instance in four is forced to have a common optimum with ties ($J_E(\pi_1)=J_E(\pi_2)=1$ for all $E$). One in four is forced to have environment-specific optima.
- All arithmetic is exact (`fractions.Fraction`). LPs are solved with HiGHS at tolerance $10^{-8}$.
- Seed `SeedSequence([20261010, 1])`.

**Procedure.** For each instance:

1. Compute $\mathcal R_{\rm rep},\mathcal R_{\rm info},\mathcal R_{\rm alg}$ and assert that they sum exactly to the total regret and are each $\ge0$.
2. Decide whether a common posterior-optimal policy exists for every $D$, and assert that this holds exactly when $\mathcal R_{\rm info}=0$.
3. Garble $D$ through 3-symbol channels: one random randomized channel, and on the first 30 instances all $3^4=81$ deterministic channels, which are the extreme points of the convex objective. Assert $B(D')\le B(D)$.
4. Augment $D$ by an independent $Z$ and assert $B(D,Z)=B(D)$ exactly.
5. On 220 instances, solve the primal min-over-mixtures LP (3) and the dual max-over-priors LP (4) separately. Assert that they are equal and that the value is $0$ exactly when a common optimum exists.

**Result.**

| Quantity | Value | JSON key |
|---|---|---|
| Instances / assertions | 500 / 5,810 | `claims[C1].n_instances`, `.n_assertions` |
| Worst residual (identity, LP equality) | $1.33\times10^{-15}$ | `claims[C1].worst_residual` |
| Worst LP feasibility residual | $1.33\times10^{-15}$ | `claims[C1].worst_lp_feasibility_residual` |
| Instances with separate primal/dual LPs | 220 | `claims[C1].details.minimax_instances` |
| Largest $B(D')-B(D)$ over all garblings | $0.0$ | `claims[C1].details.adversarial_garbling_best_lhs_minus_rhs` |
| Status | PASS | `claims[C1].status` |

**What this shows.** The decomposition is exact and each term is nonnegative on every instance. The zero-information criterion matched the direct computation in both directions, including tie cases. No garbling, even an extreme-point one chosen adversarially, increased the posterior-optimal value. The dual LP always attained the primal value. The residuals are at the level of floating-point LP solves; every rational identity has residual exactly zero. All of this is consistent with Propositions 1.1–1.4 holding without additional hypotheses on finite problems.

### Package check `robust_ambiguity_minimax` — the canonical two-world instance

**Why.** This gives a hand-checkable instance of (3)=(4) whose optimal mixtures are known in closed form.

**Setup.** Two environments prefer opposite policies, with regret matrix $M=\begin{pmatrix}0&0.2\cr 0.2&0\end{pmatrix}$. The primal and dual LPs are solved with SciPy HiGHS. Package v0.14 verifier, seed 2026101014.

**Result.**

| Quantity | Value | JSON key |
|---|---|---|
| Primal $r(C)$ | 0.1 | `robust_ambiguity_minimax.primal_regret` |
| Dual value | 0.1 | `robust_ambiguity_minimax.dual_regret` |
| Minimax policy mixture | $(0.5,0.5)$ | `.policy_mixture` |
| Least-favourable prior | $(0.5,0.5)$ | `.environment_mixture` |

**What this shows.** When the two worlds cannot be told apart, every learner's output law is common to both worlds, and the best achievable worst-case expected regret is $r(C)=0.1>0$. This is half of the $0.2$ regret of committing to either policy. The value is a *mixture-averaged* regret, as Remark 3 of the proof states; a sampled policy has regret $0$ or $0.2$.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C1 --output theory/independent_checks/subset.json
theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py --output theory/v014_repro/results_v014_rerun.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: entry `claims[id=C1]`.
- `theory/independent_checks/SPEC.md` (C1) and `theory/independent_checks/check_v014_independent.py` (`check_C1`).
- `theory/v014_repro/results_v014_rerun.json`: key `robust_ambiguity_minimax`.
