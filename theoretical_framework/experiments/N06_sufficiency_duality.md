# N06 · Sufficiency duality and transfer — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N06_sufficiency_duality.md) · [中文](N06_sufficiency_duality_ZH.md)

**Verdict.** N06 holds. Check C5 tests all three parts of Thm. 8 and Cor. 9 on 300 random finite instances (6,014 assertions, status PASS):

- **Span criterion (a).** It agrees with an independent LP separation search on every instance.
- **Duality (b).** The direct worst-pair LP and twice the best uniform approximation error, computed by a separate LP, agree to within $2.2\times10^{-14}$.
- **Transfer (c).** Adversarial maximization of the left side minus the right side of the transfer inequality never exceeds $1.1\times10^{-16}$, which is zero within floating-point error.
- **Corollary 9.** The event supremum equals $\gamma\,\mathrm{TV}$ exactly in rational arithmetic, and the response dimension equals $m-1$, for every $m=2,\dots,8$.
- **Independent cross-check.** The feature subspaces chosen in C19 also satisfy discrepancy $=2\times$ (uniform error) within $1.6\times10^{-15}$, on all 21 subspaces.

![Thm. 8(b) in one picture and across all tested subspaces](../figures/N06_duality.png)

[PDF](../figures/N06_duality.pdf)

### C5 — Theorem 8 (a)–(c) and Corollary 9

**Why.** Thm. 8 converts "which distinctions between feedback laws matter" into linear algebra.

- Part (a) claims that the span criterion is *exactly* the right test.
- Part (b) claims that the worst-case unresolved discrepancy *equals* twice an approximation distance. This is an equality, not merely a bound.
- Part (c) claims a transfer bound with explicit coefficient and residual terms.

Each claim has a falsifier:

- a matched pair that separates a response lying in the span;
- an in-span response that the span test misses;
- a feasible pair whose discrepancy exceeds $2\operatorname{dist}_\infty$, or no pair attaining it;
- a pair violating (21).

C5 searches for each of these independently.

**Setup.**

- 300 instances. The alphabet size is $m\sim\mathrm U\lbrace 4,\dots,12\rbrace $ and the feature count is $d\sim\mathrm U\lbrace 1,\dots,m-2\rbrace $.
- Feature tables $\phi\in\mathbb Q^{d\times m}$ and three responses $\mathcal G\subset\mathbb Q^m$ are random rationals.
- Each instance also tests $g=\phi_1$, an in-span control whose expected discrepancy is 0.
- Transfer instances use random $b$ and $r$, rescaled so that $g=\tfrac12+b^\top\phi+r\in[0,1]^m$.
- Corollary 9 uses random rational laws $P,Q$ on $m=2,\dots,8$, with $a=\tfrac15$ and $\gamma=\tfrac35$.
- Seed: `SeedSequence([20261010, 5])`.

**Procedure.**

1. *Exact ranks.* SymPy computes the rank of $[\mathbf 1;\phi]$ and of $[\mathbf 1;\phi;g]$ to decide $g\in H_\phi$. It also computes $d_0=\operatorname{rank}[\mathbf 1;\mathcal G]-1$.
2. *Primal LP.* HiGHS solves $\max(P-Q)g$ over feature-matched probability pairs, directly in the variables $(P,Q)$.
3. *Approximation LP.* A separate LP computes $\min_{h\in H_\phi}\lVert g-h\rVert_\infty$.
4. *Assertions for (a) and (b).* The check asserts that $\text{width}=2\cdot\text{error}$, and that $\text{width}\le10^{-8}$ exactly when $g\in H_\phi$.
5. *Transfer LP.* An LP maximizes $(P-Q)g-B\lVert\Delta\phi\rVert_\infty-2\tau$ over all pairs, and the check asserts that the optimum is $\le0$.
6. *Corollary 9.* All $2^m$ events are enumerated in exact rationals, and the check compares $\max_A\lvert(P-Q)(a+\gamma\mathbf 1_A)\rvert$ with $\gamma\cdot\tfrac12\lVert P-Q\rVert_1$. The rank of $[\mathbf 1;\lbrace \mathbf 1_A\rbrace ]$ is checked to be $m$.

**Result.**

| Quantity | Value | Source (JSON key) |
|---|---:|---|
| Instances / assertions | 300 / 6,014 | `claims[id=C5].n_instances`, `.n_assertions` |
| Status | PASS | `claims[id=C5].status` |
| $\max\lvert\text{primal width}-2\cdot\text{approx. error}\rvert$ | $2.23\times10^{-14}$ | `claims[id=C5].details.max_duality_residual` |
| Best adversarial transfer score (must be $\le0$) | $1.11\times10^{-16}$ | `claims[id=C5].details.adversarial_transfer_best_lhs_minus_rhs` |
| Worst residual over all assertions | $5.76\times10^{-14}$ | `claims[id=C5].worst_residual` |
| Cor. 9: event supremum $=\gamma\,\mathrm{TV}$, dimension $=m-1$ | exact for $m=2,\dots,8$ | assertions `all-event total variation`, `event response dimension` (PASS) |

All values are from `theory/independent_checks/results_independent_v014.json`.

**What this shows.**

- The span criterion and the LP separation search never disagree, so part (a) is the exact test.
- The two LPs are solved independently: one over pairs of laws, one over approximating functions. Their optima coincide to about $10^{-14}$, the solver tolerance, so the duality in (b) is an equality and not just a bound.
- Even an adversary optimizing against (21) cannot push the left side above the right, so the transfer inequality holds.

### C19 cross-check — sharp duality on the selected subspaces

**Why.** C19 (main page: [N09](N09_approximate_complexity.md)) independently re-applies part (b). It does so on structured response families rather than random ones.

**Setup and procedure.**

- Three response families have the same maximum degree 3 on the 125-point grid $\lbrace 0,\tfrac14,\dots,1\rbrace ^3$.
- For each feature budget $d=1,\dots,7$, a subspace is selected greedily.
- C19 then computes (i) the best uniform error for that subspace by the approximation LP, and (ii) the worst-pair discrepancy for that subspace by the primal LP. It asserts that (ii) equals twice (i).

**Result.** Over all 21 (family, $d$) subspaces, $\max\lvert\text{discrepancy}-2\cdot\text{uniform error}\rvert=1.6\times10^{-15}$. This is computed from `claims[id=C19].details.families[*].curve[*]`.

**What this shows.** The equality (20) holds on structured, non-random responses as well, including the multivariate parity family, where the discrepancy remains large.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C5,C19 \
  --output theory/independent_checks/subset.json
```

`--only` preserves each claim's random stream (`SeedSequence([20261010, claim number])`), so the subset run reproduces the full-run numbers.

## Artifacts

- Results: `theory/independent_checks/results_independent_v014.json` (entries `C5`, `C19`)
- Code: `theory/independent_checks/check_v014_independent.py` (`check_C5`, `check_C19`)
- Specification: `theory/independent_checks/SPEC.md` (C5)
- Figure spec: `theoretical_framework/figures/specs/N06.md`
