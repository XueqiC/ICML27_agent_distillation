# Theorem 8.1 · Stability–approximation tradeoff

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-01_stability_approximation_tradeoff.md) · [Proof](../proofs/08-01_stability_approximation_tradeoff.md) · [Experiments](../experiments/08-01_stability_approximation_tradeoff.md) · [中文](08-01_stability_approximation_tradeoff_ZH.md)

**What is tested here.**

- **[Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1) (stability–approximation tradeoff) holds exactly.** Primal and dual agree on 220 random adversarial instances; the largest duality residual is $`8.9\times10^{-16}`$. The stability witness reproduces $`W_\epsilon=\min\lbrace1,\epsilon/\alpha\rbrace`$ at all 28 configurations.

**Setting used on this page.**

Let $`\mathcal Y`$ be a finite message alphabet with $`\lvert\mathcal Y\rvert=m\ge2`$.

- Functions on $`\mathcal Y`$ are vectors in $`\mathbb R^m`$, and probability laws are vectors in the simplex $`\Delta(\mathcal Y)`$. For $`h\in\mathbb R^m`$ write $`(P-Q)h=\sum_y(P(y)-Q(y))h(y)`$.
- $`g\in\mathbb R^m`$ is a response. $`\phi_1,\dots,\phi_d\in\mathbb R^m`$ are fixed features, $`\Phi\in\mathbb R^{m\times d}`$ has columns $`\phi_j`$, and $`\Phi b=\sum_jb_j\phi_j`$.
- $`\mathbf{\epsilon}=(\epsilon_1,\dots,\epsilon_d)`$ are tolerances with $`\epsilon_j\ge0`$. Zero tolerances and linearly dependent features are allowed.
- The oscillation of $`h`$ is $`\mathrm{osc}(h)=\max_yh(y)-\min_yh(y)`$.

The *tolerant discrepancy* of $`g`$ is

```math
W_{\mathbf{\epsilon}}(g)=\sup\Big\lbrace\lvert(P-Q)g\rvert:\ P,Q\in\Delta(\mathcal Y),\ \lvert(P-Q)\phi_j\rvert\le\epsilon_j\ \ (j=1,\dots,d)\Big\rbrace .
```

It is the largest response difference between two message laws whose feature means differ by at most the declared tolerances. This is the situation after a finite audit that has measured each feature mean to accuracy $`\epsilon_j`$.

**Statement under test ([Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1)).**

**(a) Exact tradeoff.** Both minima below are attained, and

```math
W_{\mathbf{\epsilon}}(g)=\min_{b\in\mathbb R^d,\ c\in\mathbb R}\Big[2\big\lVert g-c\mathbf 1-\Phi b\big\rVert_\infty+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big]=\min_{b\in\mathbb R^d}\Big[\mathrm{osc}(g-\Phi b)+\sum_{j=1}^d\epsilon_j\lvert b_j\rvert\Big].
```

In particular $`W_{\mathbf{\epsilon}}(g)\le\mathrm{osc}(g)`$, which is at most $`1`$ when $`g\in[0,1]^m`$.

- **(b) Exact case.** For $`\mathbf{\epsilon}=0`$, $`W_0(g)=2\min_{h\in H_\phi}\lVert g-h\rVert_\infty`$ with $`H_\phi=\mathrm{span}\lbrace\mathbf 1,\phi_1,\dots,\phi_d\rbrace`$. This is the single-response form of [Theorem 4.2(b)](../statements/04-02_sufficiency_duality_transfer.md#s4-2).
- **(c) Stability witness.** Let $`g\in\lbrace0,1\rbrace^m`$ be non-constant, $`d=1`$ and $`\phi_1=\alpha g`$ with $`\alpha>0`$. Then $`W_\epsilon(g)=\min\lbrace1,\epsilon/\alpha\rbrace`$.
- **(d) Finite measurement bridge.** Fix a simultaneous event on which every audited feature mean satisfies $`\lvert\widehat\mu_{P,j}-\mu_{P,j}\rvert\le e_{P,j}`$ and $`\lvert\widehat\mu_{Q,j}-\mu_{Q,j}\rvert\le e_{Q,j}`$. Set $`\epsilon_j=\lvert\widehat\mu_{P,j}-\widehat\mu_{Q,j}\rvert+e_{P,j}+e_{Q,j}`$. Then on that event $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(g)`$. If moreover $`\lVert\widehat g-g\rVert_\infty\le\eta`$, then $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$.

<a id="c8-1"></a>

## Check 8.1 · Primal–dual equality on random instances (Theorem 8.1)

**Verdict.** On 220 random adversarial instances the tolerant discrepancy computed over pairs of laws equals both dual forms. The largest duality residual is $`8.9\times10^{-16}`$.

**Why.** [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1) claims an *equality*: the worst response difference among tolerance-feasible law pairs equals a minimum over feature coefficients. A wrong dual, or a missing constraint such as the zero-mass condition, would show up as a strict gap on some random instance.

**Setup.**

- Random finite instances with $`m\in[3,12]`$ messages and $`d\in[1,m-1]`$ features.
- Random responses $`g`$, random features $`\phi`$, and random tolerances $`\epsilon_j\ge0`$, including exact zeros.
- Seed `SeedSequence([20261010, 14])`.
- Three independent linear programs, solved with HiGHS:
  - the primal maximum over $`(P,Q)`$, written directly in probability variables;
  - the oscillation dual $`\min_b[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert]`$;
  - the centered sup-norm dual $`\min_{b,c}[2\lVert g-c-\Phi b\rVert_\infty+\sum_j\epsilon_j\lvert b_j\rvert]`$.
- Further assertions: $`W\le\mathrm{osc}(g)`$; the reduction at $`\mathbf{\epsilon}=0`$ to the exact case, Theorem 8.1(b), which is the single-response form of [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2)(b); and $`W\to\mathrm{osc}(g)`$ for large $`\mathbf{\epsilon}`$.
- The primal LP is a global maximizer on each instance, so the search is adversarial by construction.

**Result** (`theory/independent_checks/results_independent_v014.json`, entry `v014:C14`):

| Quantity | Value |
|---|---:|
| status | PASS |
| `n_instances` | 220 |
| `n_assertions` | 2,420 |
| `worst_residual` | $`9.80\times10^{-13}`$ |
| `details.max_duality_residual` | $`8.88\times10^{-16}`$ |

**Why it supports the statement.**

- The primal and both dual forms agree on every instance, including zero tolerances and rank-deficient features.
- Exact duality therefore holds as stated, without any interior-point or rank condition.

<a id="c8-2"></a>

## Check 8.2 · Stability control with scaled features (Theorem 8.1(c))

**Verdict.** For the feature $`\phi=\alpha g`$ the discrepancy equals $`\min\lbrace1,\epsilon/\alpha\rbrace`$ at all 28 configurations; the worst residual is $`1.39\times10^{-17}`$.

**Why.** The stability witness, [Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1)(c), is the cleanest view of the stability term. Let the feature equal the response up to a scale, $`\phi=\alpha g`$. The approximation residual is then zero, yet the certificate must deteriorate as $`\alpha\to0`$: a tolerance $`\epsilon`$ on $`\phi`$ constrains $`g`$ only to within $`\epsilon/\alpha`$.

**Setup.**

- Binary response $`g`$ and feature $`\phi=\alpha g`$.
- Scales $`\alpha\in\lbrace 1,\tfrac12,\tfrac1{10},\tfrac1{100}\rbrace`$ and tolerances $`\epsilon\in\lbrace 0,\tfrac1{1000},\tfrac1{100},\tfrac1{20},\tfrac15,1,2\rbrace`$: 28 configurations.
- The probability-variable primal LP and the tolerant dual LP are solved independently and compared with the closed form, kept as an exact rational.

**Result** (entry `v014:C22`, `details.configurations`; all 28 rows match the formula):

| $`\alpha`$ | $`\epsilon=10^{-3}`$ | $`\epsilon=10^{-2}`$ | $`\epsilon=1/20`$ | $`\epsilon=1/5`$ | $`\epsilon\ge1`$ |
|---|---:|---:|---:|---:|---:|
| 1 | 0.001 | 0.01 | 0.05 | 0.2 | 1 |
| 1/2 | 0.002 | 0.02 | 0.1 | 0.4 | 1 |
| 1/10 | 0.01 | 0.1 | 0.5 | 1 | 1 |
| 1/100 | 0.1 | 1 | 1 | 1 | 1 |

At $`\epsilon=0`$ the discrepancy is $`0`$ for every $`\alpha`$.

**Why it supports the statement.**

- With an exact representation the certificate is still only as good as $`\epsilon\lvert b\rvert`$, with $`b=1/\alpha`$.
- At $`\alpha=1/100`$ a 1% tolerance already leaves the response completely undetermined ($`W=1`$).
- A small approximation residual is therefore not a certificate by itself; the coefficient size must enter, as Theorem 8.1 states.

<a id="c8-3"></a>

## Check 8.3 · Explicit tolerant duality example (Theorem 8.1)

**Verdict.** On an example with a closed form, primal, dual and closed form coincide at six tolerances, and on 36 further random LPs the largest primal–dual residual is $`8.9\times10^{-16}`$.

**Why.** This is an independent implementation, from the theory package, on an example where the optimal coefficient is known.

**Setup.**

- $`\mathcal Y=\lbrace 0,\tfrac12,1\rbrace`$, $`\phi(y)=y`$ and $`g=1\lbrace y=1\rbrace`$; the closed form is $`W_\epsilon=\min\lbrace 1,\tfrac12+\epsilon\rbrace`$.
- The primal and dual LPs are solved at six tolerances; witness laws and the optimal coefficient are recorded.

**Result** (`theory/v014_repro/results_v014_rerun.json`, key `noisy_feature_duality`):

| tolerance | primal | dual | closed form | optimal $`b`$ | `uniform_residual` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.50 | 0.50 | 0.50 | 1 | 0.25 |
| 0.05 | 0.55 | 0.55 | 0.55 | 1 | 0.25 |
| 0.25 | 0.75 | 0.75 | 0.75 | 1 | 0.25 |
| 0.5 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |
| 0.8 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |
| 1.5 | 1.00 | 1.00 | 1.00 | 0 | 0.5 |

`uniform_residual` is $`\min_c\lVert g-c-b\phi\rVert_\infty`$ at the optimal $`b`$. `random_LP_cases` = 36 and `maximum_primal_dual_residual` = $`8.9\times10^{-16}`$.

**Why it supports the statement.** The optimum switches from using the feature ($`b=1`$) to ignoring it ($`b=0`$) exactly when the measurement cost $`\epsilon`$ outweighs the approximation gain. That switch is the tradeoff that Theorem 8.1 makes exact.

![Figure 8.1: Stability–approximation tradeoff](../figures/fig8_1_stability.png)

*Figure 8.1.* Left: tolerant discrepancy for the scaled feature $`\phi=\alpha g`$ ([Check 8.2](../experiments/08-01_stability_approximation_tradeoff.md#c8-2)); the LP values lie on $`\min\lbrace1,\epsilon/\alpha\rbrace`$, so an exact but small-scale feature certifies only $`\epsilon/\alpha`$. Right: the explicit example of [Check 8.3](../experiments/08-01_stability_approximation_tradeoff.md#c8-3), where primal, dual and closed form coincide and the optimum stops using the feature once $`\epsilon\ge\tfrac12`$. Exact finite computations. [PDF](../figures/fig8_1_stability.pdf)

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C14,C22 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # noisy_feature_duality
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: entries `v014:C14`, `v014:C22`; specification `theory/independent_checks/SPEC.md`.
- `theory/v014_repro/results_v014_rerun.json`: `noisy_feature_duality`.
- Figure: `../figures/fig8_1_stability` (`.pdf`, `.png`; spec in `../figures/specs/`).
