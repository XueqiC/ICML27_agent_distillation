# Theorem 4.7 · Update composition amplifies the information requirement

*Experiments*

[← Framework](../README.md) · [Statement](../statements/04-07_update_composition_amplifies_information_requirement.md) · [Proof](../proofs/04-07_update_composition_amplifies_information_requirement.md) · [Experiments](04-07_update_composition_amplifies_information_requirement.md) · [中文](04-07_update_composition_amplifies_information_requirement_ZH.md)

**What is tested here.** The degree composition bound (a), its attainment (b), the threshold learner (c) and the remainder of the local SGD expansion (d) are computed exactly; the geometric error accumulation (e) is checked on exact rational recursions; and a controlled small network locates the regime in which the local affine description of SGD applies.

- **Verdict: PASS.** Block degrees equal the composition bound in all 12 random chains, the squaring witness attains it, the threshold learner separates two equal-mean laws, and the SGD expansion remainder never exceeds $`0.275`$ times its bound ([Check 4.11](#c4-11)).
- The geometric accumulation bound is attained with equality, and the linear-in-$`R`$ reading fails without stability ([Check 4.12](#c4-12)).
- In a controlled network, the tangent affine-SGD prediction agrees within the preset margin in the small-$`\eta R`$ regime and fails at large step sizes ([Check 4.13](#c4-13)). This is a regime map, not a bound.

**Setting used on this page.**

- *Polynomial updates.* Student parameter $`p\in[0,1]^k`$, common start $`p_0`$, independent blocks $`Z_1,\dots,Z_R`$ with laws $`P_i`$ (reference source) or $`Q_i`$ (alternative), updates $`p_i=U_i(p_{i-1},Z_i)`$ whose coordinates are polynomials of degree at most $`a_i`$ in $`p_{i-1}`$ and at most $`b_i`$ in $`Z_i`$. A deployment with at most $`N`$ decisions has a polynomial value $`J(p)`$ of total degree at most $`N`$ ([Theorem 4.4(a)](../statements/04-04_affine_learners_moment_order.md#s4-4)).
- *Scalar witness laws of order $`M`$*: binomial weights $`\binom Mi/2^{M-1}`$ on the even versus the odd points of $`\lbrace0,\tfrac1M,\dots,1\rbrace`$; they agree on all moments below $`M`$ and differ at order $`M`$ ([Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order.md#s4-4)).
- *Smooth local setting.* $`\theta\in\mathbb R^p`$, $`V`$ a continuation value: the expected final score when training continues from $`\theta`$ ([Definition 1.4](../README.md#s1-4)). One SGD step with step size $`\eta`$ and consumed-gradient law $`P`$ maps $`V`$ to $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$; $`\mu_P=\mathbb E_PG`$, $`M_P=\mathbb E_PGG^\top`$; $`M_3`$ bounds the third derivative of $`V`$ on the step segments.
- *Error recursion.* $`e_0=0`$, $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$.
- Code IDs: `v014:` = `theory/independent_checks` (results `results_independent_v014.json`); `v015:` = `theory/v015_checks` (results `results_v015_independent.json`).

**Statement under test ([Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement.md#s4-7)).**

- **(a) Degree composition.** $`J(p_R)`$ has total degree in block $`Z_i`$ at most

```math
D_i=N\,b_i\prod_{t=i+1}^Ra_t, \qquad(4.8)
```

- and matching the block moments through order $`D_i`$ for every $`i`$ makes the two learning values equal.
- **(b) Attainment.** For every $`R`$ there are updates and an $`N=1`$ deployment whose response is $`Z_1^{2^{R-1}}`$, and the scalar witness laws of order $`2^{R-1}`$ give different learning values; (4.8) cannot be improved in general.
- **(c) Threshold learner.** For $`p=\mathbf 1\lbrace Z>1/2\rbrace`$ and $`J(p)=p`$, the laws $`P(Z=0)=\tfrac13`$, $`P(Z=\tfrac34)=\tfrac23`$ and $`Q(Z=\tfrac12)=1`$ have equal mean $`\tfrac12`$ but learning values $`\tfrac23`$ and $`0`$.
- **(d) Local SGD expansion.** If $`V`$ is $`C^3`$ on the step segments with third-derivative bound $`M_3`$,

```math
(K^P-K^Q)V(\theta)=-\eta\nabla V(\theta)^\top(\mu_P-\mu_Q)+\frac{\eta^2}{2}\langle\nabla^2V(\theta),M_P-M_Q\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\frac{M_3\eta^3}{6}\big(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\big). \qquad(4.9)
```

- **(e) Error accumulation.** $`e_R\le C\eta^2\sum_{s=0}^{R-1}(1+\eta L)^s`$, with equality when the recursion holds with equality; the linear bound $`RC\eta^2`$ fails in general for $`L>0`$.

<a id="c4-11"></a>

## Check 4.11 · Degree composition, attainment, the threshold learner and the SGD expansion (Theorem 4.7(a)–(d))

**Verdict.** PASS (72 instances, 371 assertions, worst residual 0). In all 12 random polynomial update chains the actual block degrees equal the bound $`D_i=Nb_i\prod_{t>i}a_t`$ of (4.8), and blockwise moment matching through $`D_i`$ preserves the learning value exactly. The squaring witness attains the bound at degrees 1, 2, 4, 8. The threshold learner separates two equal-mean laws, $`2/3`$ against $`0`$. The SGD expansion remainder never exceeds 0.275 of its $`\eta^3`$ bound.

**Why.** [Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement.md#s4-7)(a) claims that the response degree in block $`i`$ is at most $`Nb_i\prod_{t>i}a_t`$ and that this many moments suffice; a wrong degree count would make the moment sufficiency false. Part (b) claims the bound is attained, so a loose count would be detected. Part (c) claims that no moment order suffices for a discontinuous update. Part (d), the local SGD expansion (4.9), is the first-order description of SGD that the optimizer-transformed expansion of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) generalizes, so its remainder bound must hold.

**Setup.**

- 12 random chains with $`R\in\lbrace 1,2,3\rbrace`$, $`N\in\lbrace 1,2\rbrace`$, degrees $`a_i,b_i\in\lbrace 1,2\rbrace`$, and $`p_i=\lambda_ip_{i-1}^{a_i}+(1-\lambda_i)Z_i^{b_i}`$ with random $`\lambda_i\in\lbrace \tfrac14,\tfrac12,\tfrac34\rbrace`$.
- Exact symbolic pullback (SymPy) and exact rational block moments; block $`i`$ uses the scalar witness laws of order $`D_i+1`$.
- The squaring witness for $`R=1,\dots,4`$, and the threshold laws of part (c).
- Part (d): 15 random cubic $`V`$ on $`\mathbb R^3`$ with random discrete gradient laws on 6 atoms, and the same with a logistic $`V(\theta)=\sigma(w^\top\theta)`$; step sizes $`\eta\in\lbrace 0.01,0.1,0.5,1\rbrace`$.

**Procedure.**

- Expand $`J(p_R)`$, read off each block degree, and compare it with $`D_i`$; evaluate both learning values with moments matched through $`D_i`$.
- For part (d), compute the exact difference $`(K^P-K^Q)V(\theta)`$ and the first two expansion terms, and compare the remainder with $`\tfrac{M_3\eta^3}6(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3)`$:
  - for cubics, $`M_3`$ is the Frobenius norm of the constant third-derivative tensor, which upper-bounds the operator norm;
  - for the logistic, $`M_3=\lVert w\rVert^3/8`$, because $`\lvert\sigma'''\rvert\le1/8`$ globally.

**Result.** Source: `theory/independent_checks/results_independent_v014.json`, `v014:C13`.

| Quantity | Value |
|---|---|
| Status, instances, assertions, worst residual | PASS, 72, 371, 0.0 |
| Declared vs actual block degrees (`details.degrees`, 12 chains) | equal in every chain, e.g. $`a=(2,2,2)`$, $`b=(2,2,1)`$, $`N=2`$: $`D=(16,8,2)`$, actual $`(16,8,2)`$ |
| Value equality under blockwise matching | exact in all 12 chains |
| Squaring witness (`details.amplification`) | $`R=1,2,3,4`$; degree $`1,2,4,8`$; gap $`-1`$, $`1/4`$, $`3/256`$, $`315/16777216`$ |
| Threshold learner (assertions in `check_C13`) | equal mean $`1/2`$; values $`2/3`$ vs $`0`$ |
| Logistic remainder/bound, max over 15 instances (`details.logistic`) | $`\eta=0.01`$: 0.274; $`0.1`$: 0.274; $`0.5`$: 0.270; $`1`$: 0.265 |
| Largest remainder/bound overall (`best_ratio`) | 0.2743 |

**Why it supports Theorem 4.7(a)–(d).**

- The degree bound is exact on random chains and attained by the squaring witness, so the order a source must match grows with the composition of updates.
- The threshold learner shows that no fixed moment order suffices once the update is discontinuous.
- The expansion of part (d) holds with room to spare at every tested step size.

<a id="c4-12"></a>

## Check 4.12 · Accumulation of per-step errors (Theorem 4.7(e))

**Verdict.** PASS (717 instances, 1,435 assertions, worst residual 0). The geometric bound holds in every instance and is attained with equality. The witness $`\eta=\tfrac12`$, $`L=C=1`$, $`R=4`$ has $`e_4=65/32`$, above the linear value $`RC\eta^2=1`$.

**Why.** A one-step $`O(\eta^2)`$ error does not by itself give an $`O(R\eta^2)`$ training error. [Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement.md#s4-7)(e) states a geometric bound and claims it is sharp, so that linear accumulation needs a stability condition. Both claims are checked exactly.

**Setup.** Exact rational recursions $`e_{t+1}=s\cdot((1+\eta L)e_t+C\eta^2)`$ with slack $`s\in(0,1]`$, random $`\eta\in(0,1]`$, $`L\in[0,1]`$, $`C\in(0,1]`$ and $`R\le40`$; fixed-seed random restarts and hill-climbing over $`(\eta,L,C,R,s)`$ supplement the random cases.

**Procedure.** Assert $`e_R\le C\eta^2\sum_{s<R}(1+\eta L)^s`$, with equality at $`s=1`$. Search for the largest ratio of the bound to the linear simplification $`RC\eta^2`$. Evaluate the witness $`\eta=\tfrac12`$, $`L=C=1`$, $`R=4`$.

**Result.** Source: `theory/v015_checks/results_v015_independent.json`, `v015:V7`.

| Quantity | Value |
|---|---|
| Status, instances, assertions, worst residual | PASS, 717, 1,435, 0.0 |
| Geometric bound attained (`data.geometric_adversarial.best.score`) | 1.0 (e.g. $`\eta=4/5`$, $`L=1/5`$, $`C=7/10`$, $`R=13`$) |
| Witness (`data.linear_failure`) | $`e_4=65/32`$, bound $`65/32`$, linear $`RC\eta^2=1`$ |
| Largest bound/linear ratio found by the search (`best_ratio`) | $`3.78\times10^{30}`$ ($`\eta=2`$, $`L=2`$, $`R=47`$) |

**Why it supports Theorem 4.7(e).**

- The geometric bound is exact: it holds everywhere and is attained.
- The linear-in-$`R`$ reading fails as soon as $`\eta LR`$ is not small, so accumulating local expansions along training requires the stated stability or short-horizon condition.

<a id="c4-13"></a>

## Check 4.13 · Where the local affine SGD description applies in a controlled network (Theorem 4.7(d),(e))

**Verdict.** Regime map, consistent with the theorem. The tangent affine-SGD prediction agrees with the actual network in all 35 SGD settings with $`\eta\le0.1`$ and at most 4 updates, or $`\eta=0.01`$ over 40 updates; the largest deviation there is $`2.5\times10^{-4}`$. At $`\eta\ge1`$ it departs in 35 of 42 settings, by up to 0.53.

**Why.** [Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement.md#s4-7)(d),(e) predicts that for SGD the response is governed by the mean consumed gradient up to $`O(\eta^2)`$ per step, with errors that accumulate geometrically. In a nonlinear network, a fixed initial-tangent affine-SGD prediction should therefore be accurate at small $`\eta R`$ and inaccurate at large $`\eta`$ or long horizons. This is a test of where the description applies, not of a bound.

**Setup.** Controlled small network, not a large model and not an external benchmark.

- Network: one-hot context, 4 tanh units, sigmoid output; binary cross-entropy; float64.
- Training: plain SGD per sampled block, $`\eta\in\lbrace 0.01,0.1,1,3\rbrace`$, $`R\in\lbrace 1,4,40\rbrace`$ updates.
- Tasks: the two source pairs of [Proposition 3.1](../statements/03-01_static_value_not_learning_value.md#s3-1) (pairs of feedback sources with equal static value but different learning value), moment laws $`N=2,3,4`$, parity laws $`N=2,3`$ (7 tasks).
- Estimation: complete finite-history enumeration where enumerable, otherwise 2,400 independent training runs per source.
- Comparator: the initial-tangent affine-SGD response of the actual policy, the affine recursion $`p_t=p_{t-1}+\eta A(Z_t-p_{t-1})`$ in which $`A`$ is the tangent kernel of the policy at its initial parameters (SGD on the network linearized at initialization).
- Agreement margin, fixed in code before execution: $`10^{-4}+0.1\lvert\text{prediction}\rvert+4\,\mathrm{SE}`$.

**Procedure.** For each setting, compute the actual final expected-return gap between the two sources and the tangent-affine prediction, and record agreement under the margin.

**Result.** Source: `theory/independent_checks/results_independent_v014.json → controlled_neural.details.configurations` (SGD rows; code ID `v014:C23`); cohort totals in `details.observed_affine_comparison_cohorts`.

| $`\eta`$ | $`R=1`$ | $`R=4`$ | $`R=40`$ |
|---|---|---|---|
| 0.01 | 7/7 agree (max dev. $`1.1\times10^{-6}`$) | 7/7 ($`4.1\times10^{-6}`$) | 7/7 ($`2.4\times10^{-5}`$) |
| 0.1 | 7/7 ($`1.1\times10^{-4}`$) | 7/7 ($`2.5\times10^{-4}`$) | 5/7 ($`1.9\times10^{-3}`$) |
| 1 | 3/7 (0.013) | 0/7 (0.053) | 3/7 (0.084) |
| 3 | 0/7 (0.52) | 0/7 (0.53) | 1/7 (0.51) |

- Cohort "SGD, step $`\le0.1`$, at most four updates": 28/28 agreements, largest absolute deviation $`2.52\times10^{-4}`$.
- The additive one-step null (identical examples and weights, regrouped) has parameter residual exactly $`0.0`$ (`details.additive_null_parameter_residual`).
- The Adam rows of the same experiment test the optimizer-state results of §5, starting from cold-start Adam ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)); they are reported on the §5 pages.

![Figure 4.5: regime map of the local SGD expansion and exact composition checks](../figures/fig4_5_regime.png)

*Figure 4.5.* The tangent affine-SGD prediction matches the actual controlled network for $`\eta\le0.1`$ and short horizons and fails at $`\eta\ge1`$ (this check). The third-order remainder of the SGD expansion stays below its bound ([Check 4.11](#c4-11)). Per-step errors accumulate geometrically, and the linear-in-$`R`$ reading fails without a stability condition ([Check 4.12](#c4-12)). [PDF](../figures/fig4_5_regime.pdf)

**Why it supports Theorem 4.7(d),(e).**

- The affine first-order description of SGD is accurate exactly where the expansion and the stability condition say it should be: small step size and few updates, or a very small step over many updates.
- The departure grows with $`\eta`$ and with the horizon, consistent with the curvature and accumulation terms.

**Limits.** One small architecture and seven tasks; this locates the regime boundary, it does not certify a real model.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python \
  theory/independent_checks/check_v014_independent.py --only C13,C23 \
  --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V7 \
  --output theory/v015_checks/quick/v7.json
```

`--only` preserves each claim's random stream (`SeedSequence([20261010, claim number])`), so a subset run reproduces the full-run numbers.

## Artifacts

- Results: `theory/independent_checks/results_independent_v014.json` (entry `v014:C13`; `controlled_neural` for `v014:C23`); `theory/v015_checks/results_v015_independent.json` (entry `v015:V7`).
- Code: `theory/independent_checks/check_v014_independent.py` (`check_C13`, `check_C23`); `theory/v015_checks/mathematics.py` (`check_V7`); specification `theory/independent_checks/SPEC.md`.
- Existing plot: `theory/independent_checks/figs/C23_controlled_neural.png`. Figure spec: `theoretical_framework/figures/specs/fig4_5.md`.
