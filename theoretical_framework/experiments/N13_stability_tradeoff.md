# N13 · Stability–approximation tradeoff — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N13_stability_tradeoff.md) · [中文](N13_stability_tradeoff_ZH.md)

**Verdict.** N13 holds. In every tested instance the tolerant discrepancy $`W_{\mathbf{\epsilon}}(g)`$, computed by maximizing directly over pairs of laws, equals the dual formula [N13](../proofs/N13_stability_tradeoff.md) to machine precision. Three checks establish this:

| Check | Instances | Assertions | Worst residual |
|---|---:|---:|---:|
| C14 | 220 random adversarial instances | 2,420 | $`9.8\times10^{-13}`$ (primal–dual gap $`8.9\times10^{-16}`$) |
| C22 | 28 stability configurations | 112 | $`1.4\times10^{-17}`$ |
| Package check | 6 explicit tolerances + 36 random LPs | — | $`8.9\times10^{-16}`$ |

C22 reproduces the closed form $`W_\epsilon=\min\lbrace 1,\epsilon/\alpha\rbrace`$ exactly at all 28 configurations. This makes the stability term visible: a feature that represents the response exactly, but at a small scale $`\alpha`$, certifies only $`\epsilon/\alpha`$.

![Tolerant discrepancy equals the stability–approximation dual](../figures/N13_stability.png)

[PDF](../figures/N13_stability.pdf)

## C14 · Primal–dual equality on random instances

**Why.** [N13](../proofs/N13_stability_tradeoff.md) claims an *equality*: the worst response difference among tolerance-feasible law pairs equals a minimum over feature coefficients. A wrong dual, or a missing constraint such as the zero-mass condition, would show up as a strict gap on some random instance.

**Setup.**

- Random finite instances with $`m\in[3,12]`$ messages and $`d\in[1,m-1]`$ features.
- Random responses $`g`$ and features $`\phi`$.
- Random tolerances $`\epsilon_j\ge0`$, including exact zeros.
- Seed `SeedSequence([20261010, 14])`.

**Procedure.** Three independent linear programs are solved with HiGHS:

1. the primal maximum over $`(P,Q)`$ written directly in probability variables;
2. the oscillation dual $`\min_b[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert]`$;
3. the centered sup-norm dual $`\min_{b,c}[2\lVert g-c-\Phi b\rVert_\infty+\sum_j\epsilon_j\lvert b_j\rvert]`$.

The checks also assert three properties: $`W\le\mathrm{osc}(g)`$; the reduction at $`\mathbf{\epsilon}=0`$ to [N06(b)](../proofs/N06_sufficiency_duality.md); and $`W\to\mathrm{osc}(g)`$ for large $`\mathbf{\epsilon}`$. The primal LP is a global maximizer for each instance, so the search is adversarial by construction.

**Result** (`theory/independent_checks/results_independent_v014.json`, `claims[id=C14]`):

| Quantity | Value |
|---|---:|
| status | PASS |
| `n_instances` | 220 |
| `n_assertions` | 2,420 |
| `worst_residual` | $`9.80\times10^{-13}`$ |
| `details.max_duality_residual` | $`8.88\times10^{-16}`$ |

**What this shows.** The two dual forms and the primal agree on every instance, including zero tolerances and rank-deficient features. Exact duality therefore holds as stated. It needs no interior-point or rank condition.

## C22 · Stability control with scaled features

The scaled feature is $`\phi=\alpha g`$.

**Why.** [N13(c)](../proofs/N13_stability_tradeoff.md) is the cleanest way to see the stability term. Let the feature equal the response up to a scale, $`\phi=\alpha g`$. The approximation residual is then zero, yet the certificate must deteriorate as $`\alpha\to0`$, because a tolerance $`\epsilon`$ on $`\phi`$ only constrains $`g`$ to within $`\epsilon/\alpha`$.

**Setup.**

- Binary response $`g`$, with the feature $`\phi=\alpha g`$.
- Scales $`\alpha\in\lbrace 1,\tfrac12,\tfrac1{10},\tfrac1{100}\rbrace`$.
- Tolerances $`\epsilon\in\lbrace 0,\tfrac1{1000},\tfrac1{100},\tfrac1{20},\tfrac15,1,2\rbrace`$, giving 28 configurations.

**Procedure.** Solve the probability-variable primal LP and the noisy dual LP independently. Compare both with the closed form $`\min\lbrace 1,\epsilon/\alpha\rbrace`$, kept as an exact rational.

**Result** (`claims[id=C22].details.configurations`; all 28 rows match the formula):

| $`\alpha`$ | $`\epsilon=10^{-3}`$ | $`\epsilon=10^{-2}`$ | $`\epsilon=1/20`$ | $`\epsilon=1/5`$ | $`\epsilon\ge1`$ |
|---|---:|---:|---:|---:|---:|
| 1 | 0.001 | 0.01 | 0.05 | 0.2 | 1 |
| 1/2 | 0.002 | 0.02 | 0.1 | 0.4 | 1 |
| 1/10 | 0.01 | 0.1 | 0.5 | 1 | 1 |
| 1/100 | 0.1 | 1 | 1 | 1 | 1 |

At $`\epsilon=0`$ the discrepancy is $`0`$ for every $`\alpha`$. The worst residual is $`1.39\times10^{-17}`$.

**What this shows.** With an exact representation, the certificate is still only as good as $`\epsilon\lvert b\rvert`$ with $`b=1/\alpha`$. At $`\alpha=1/100`$, a tolerance of 1% already leaves the response completely undetermined ($`W=1`$). A small approximation residual is therefore insufficient by itself. The coefficient size must enter the certificate, exactly as [N13](../proofs/N13_stability_tradeoff.md) states.

## Package check · Explicit tolerant duality example

**Why.** This is an independent implementation from the theory package, run on an example with a closed form.

**Setup.**

- $`\mathcal Y=\lbrace 0,\tfrac12,1\rbrace`$, $`\phi(y)=y`$ and $`g=1\lbrace y=1\rbrace`$.
- The closed form is $`W_\epsilon=\min\lbrace 1,\tfrac12+\epsilon\rbrace`$.

**Procedure.** Solve the primal and the dual LP at six tolerances, and record the witness laws and the optimal coefficient. Then solve 36 additional random LP cases.

**Result** (`theory/v014_repro/results_v014_rerun.json`, key `noisy_feature_duality`):

| tolerance | primal | dual | closed form | optimal $`b`$ | `uniform_residual` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.50 | 0.50 | 0.50 | 1 | 0.25 |
| 0.05 | 0.55 | 0.55 | 0.55 | 1 | 0.25 |
| 0.25 | 0.75 | 0.75 | 0.75 | 1 | 0.25 |
| 0.5 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |
| 0.8 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |
| 1.5 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |

`uniform_residual` is $`\min_c\lVert g-c-b\phi\rVert_\infty`$ at the optimal $`b`$. `random_LP_cases` = 36, `maximum_primal_dual_residual` = $`8.9\times10^{-16}`$.

**What this shows.** The optimizer switches from using the feature ($`b=1`$) to ignoring it ($`b=0`$) exactly when the measurement cost $`\epsilon`$ exceeds the approximation gain. That switch is the tradeoff that [N13](../proofs/N13_stability_tradeoff.md) makes exact.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C14,C22 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # package check
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: `claims[id=C14]` and `claims[id=C22]`.
- `theory/v014_repro/results_v014_rerun.json`: `noisy_feature_duality`.
- Specification: `theory/independent_checks/SPEC.md` (C14, C22).
- Figure: `../figures/N13_stability.pdf` and `.png` (spec `../figures/specs/N13.md`).
