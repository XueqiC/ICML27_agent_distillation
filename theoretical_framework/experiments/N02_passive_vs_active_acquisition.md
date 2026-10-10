# N02 · Passive impossibility, active attainability — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N02_passive_vs_active_acquisition.md) · [中文](N02_passive_vs_active_acquisition_ZH.md)

**Verdict.** N02 holds. Independent check C2 verified every part of Theorems 2.1–2.3 and Proposition 2.4 exactly, with 540 assertions and no violation, on 25 configurations ($\Delta\in\lbrace \tfrac18,\tfrac1{10},\tfrac1{16},\tfrac1{32},\tfrac1{64}\rbrace \times\delta\in\lbrace 0.01,0.05,0.1,0.25,0.4\rbrace $):

- the payoff formula $J_v(p)=\tfrac34+v\Delta(2p-1)$ and the passive minimax value $\Delta$, both symbolic;
- the lower bound (8) against the smallest successful fixed-sample test and against an optimally tuned SPRT, computed by exact binomial sums and an exact absorbing-walk recursion;
- the sufficiency of $n=\lceil\log(1/\delta)/(8\Delta^2)\rceil$;
- the compilation tail.

Every actual strategy needs at least as many queries as (8) requires. The SPRT reaches the bound exactly (ratio $1.000$) at $(\Delta,\delta)=(\tfrac18,\tfrac1{10})$ and $(\tfrac18,\tfrac14)$, so (8) cannot be improved in general.

![Query complexity: lower bound, optimal tests and sufficient n](../figures/N02_queries.png)

[PDF](../figures/N02_queries.pdf)

### C2 — Theorem 4 and the compilation paragraph

**Why.**

- Theorem 2.2 is a lower bound for *every* adaptive strategy. A strategy with error $\le\delta$ in both worlds and fewer expected queries would falsify it.
- Theorem 2.3 and Proposition 2.4 are sufficiency claims. An exact error above $\delta$ (or $\delta_L$) at the stated sample size would falsify them.
- The constant $64$ rests on $\operatorname{kl}(\tfrac12+u,\tfrac12-u)\le16u^2$, which is checked analytically and numerically.

**Setup.**

- Exact rational binomial tails use integer-numerator summation, not a normal approximation.
- The SPRT expected sample size and error are computed by exact tridiagonal elimination of the absorbing random walk with boundaries $\pm h$, where $h=\lceil\log\frac{1-\delta}{\delta}/\log\frac{1/2+2\Delta}{1/2-2\Delta}\rceil$.
- The payoff and the passive minimax are computed in SymPy.
- The KL inequality is checked on a dense grid of 20,001 points on $(0,\tfrac14]$, together with a bounded scalar optimizer that maximizes the ratio adversarially.
- Seed `SeedSequence([20261010, 2])`; the check is deterministic.

**Procedure.**

1. Find the smallest odd $n$ for which the threshold test succeeds with probability $\ge1-\delta$ in both worlds, counting ties as failures. Check that $n-1$ and $n-2$ fail.
2. Compute the SPRT's exact expected stopping time and its exact error.
3. Assert that both stopping times are $\ge$ the bound (8), and that (8) is $\ge$ its loosened form $\operatorname{kl}(1-\delta,\delta)/(64\Delta^2)$.
4. Assert that the exact error at $n_9=\lceil\log(1/\delta)/(8\Delta^2)\rceil$ is $\le\delta$.
5. For compilation, compute the exact probability that $\lvert\widehat p-q\rvert>\tfrac18$ at $m=\lceil32\log(2/\delta_L)\rceil$, for $q\in\lbrace \tfrac18,\tfrac78\rbrace $. Assert that it is $\le\delta_L$, and that every within-tolerance outcome has regret $\le\Delta/2$.

**Result (selected rows; all 25 are in the JSON).**

| $\Delta$ | $\delta$ | Bound (8) | Optimal fixed $n$ | SPRT $\mathbb E N$ | SPRT / bound | $n_9$ | Exact error at $n_9$ |
|---|---|---:|---:|---:|---:|---:|---:|
| 1/8 | 0.01 | 8.198 | 19 | 9.918 | 1.210 | 37 | 5.1e-4 |
| 1/8 | 0.1 | 3.200 | 7 | 3.200 | **1.000** | 19 | 8.9e-3 |
| 1/8 | 0.25 | 1.000 | 1 | 1.000 | **1.000** | 12 | 5.4e-2 |
| 1/16 | 0.01 | 35.262 | 85 | 35.282 | 1.0005 | 148 | 1.3e-3 |
| 1/32 | 0.05 | 84.356 | 173 | 87.030 | 1.032 | 384 | 8.0e-3 |
| 1/64 | 0.01 | 575.661 | 1383 | 580.574 | 1.0085 | 2358 | 1.3e-3 |
| 1/64 | 0.1 | 224.703 | 421 | 233.224 | 1.038 | 1179 | 1.6e-2 |

| Summary quantity | Value | JSON key |
|---|---|---|
| Configurations / assertions | 25 / 540 | `claims[C2].n_instances`, `.n_assertions` |
| Minimum SPRT / bound ratio | 1.0 | `claims[C2].details.minimum_sprt_ratio` |
| Minimum fixed-$n$ / bound ratio | 1.0 (at $\Delta=\tfrac18,\delta=0.25$) | `details.configurations[*].fixed_ratio` |
| Analytic $\sup_u\operatorname{kl}(\tfrac12+u,\tfrac12-u)/u^2$ on $(0,\tfrac14]$ | $8\log3=8.789$ | `details.analytic_kl_sup` |
| Optimizer's largest ratio | 8.789 (< 16) | `details.optimizer_kl_ratio` |
| Compilation $m$ ($\delta_L=0.01,0.05,0.1,0.25,0.4$) | 170, 119, 96, 67, 52 | `details.compilation[*].m` |
| Exact compilation error at those $m$ | 4.2e-6, 1.2e-4, 2.4e-4, 3.0e-3, 3.9e-3 | `details.compilation[*].error` |
| Worst residual | 0 | `claims[C2].worst_residual` |

**What this shows.**

- *Passive impossibility (7).* The symbolic payoff is exactly $\tfrac34+v\Delta(2p-1)$, and the two passive regrets $2\Delta(1-m)$ and $2\Delta m$ intersect at value exactly $\Delta$.
- *Lower bound (8).* No exactly computed strategy beats the bound. The SPRT, which is near-optimal for this problem, sits on it at two configurations. At $\Delta=\tfrac1{64}$ and $\delta\le0.25$ it is within 5% (ratios 1.009–1.047), so the bound has the right constant, not just the right order. For $\delta\le0.25$, fixed-sample tests need $1.6$–$2.4\times$ the bound, apart from the trivial case $n=1$; this is expected for non-sequential designs.
- *Upper bound (9).* At $n_9$ the exact error is far below $\delta$ in every configuration; Hoeffding is conservative.
- *Compilation.* The number of compiled labels is $O(\log(1/\delta_L))$, independent of $\Delta$. The exact tail is two to four orders of magnitude below $\delta_L$.

The two costs, $\Theta(\Delta^{-2}\log(1/\delta))$ authentic queries and $O(\log(1/\delta_L))$ compiled labels, are thus separately confirmed.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C2 --output theory/independent_checks/subset.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: entry `claims[id=C2]`, keys `details.configurations`, `details.compilation`, `details.analytic_kl_sup`, `details.optimizer_kl_ratio`, `details.minimum_sprt_ratio`.
- `theory/independent_checks/check_v014_independent.py`: `check_C2`, `binomial_tail`, `walk_recursion`.
