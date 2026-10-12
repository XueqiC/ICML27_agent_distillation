# §8 Verification (Theorem C) — Experiments

[← Framework](../README.md) · [Proofs](../proofs/08_verification.md) · [中文](08_verification_ZH.md)

**Verdict.**

- **[Theorem 8.1](../proofs/08_verification.md#s8-1) (stability–approximation tradeoff) holds exactly.** Primal and dual agree on 220 random adversarial instances; the largest duality residual is $`8.9\times10^{-16}`$. The stability witness reproduces $`W_\epsilon=\min\lbrace1,\epsilon/\alpha\rbrace`$ at all 28 configurations.
- **[Theorem 8.2](../proofs/08_verification.md#s8-2) (frozen selection with transport) holds, and its transport term is necessary.**
  - No violation on 2,572 exact finite training processes; worst residual 0; the bound is attained (ratio 1).
  - On an uncovered trained state the audited discrepancy is 0 while the learning gap is 1.
  - In a controlled network the measured-transport local bound contains the true contrast at 40/40 reached states. Predictions that assume zero transport miss at 36/40.
- **[Theorem 8.3](../proofs/08_verification.md#s8-3) (actual-state adaptive audit) holds at its stated strength.**
  - In 5 configurations × 100,000 runs the gain stays above both guarantees.
  - The signed identity is exact on 50 rational instances.
  - The lower confidence bound and the unconditional form are both necessary: there are witnesses with exact expected gain $`-78/15625`$ and a conditioning bias of $`1/4\to119/494`$.
  - The sufficient Hoeffding cost is 1,016 paired replicas per candidate.
- **[Proposition 8.4](../proofs/08_certificate.md#s8-4) (measuring the mean increment) holds in all 144 cells, up to $`p=10^5`$.** The realized deviation quantile is at most 0.615 of the bound and does not grow with $`p`$ at a fixed increment bound $`b`$, as [Remark 8.5](../proofs/08_certificate.md#s8-5) states.
- **[Theorem 8.8](../proofs/08_certificate.md#s8-8) (CRN paired-replacement certificate, Theorem C).** Measured on the frozen controlled panel of 1,728 candidate pairs:
  - As a ranker, the point estimate $`\hat S`$ has the wrong sign on 48 of 1,687 nonzero-gold pairs (2.8%).
  - Its measurement uses 7.0% of the trajectories of a full-tail paired audit at $`r=64`$, and 28.0% at $`r=256`$.
  - Common random numbers cut the variance of the paired difference to 0.12–0.15 of independent tails (ratio of summed variances).
  - The strict certificate with the a-priori range $`B=2`$ certifies 0 pairs with the aligned surrogate, and 8 pairs with the calibrated surrogate at $`r=256`$; none is wrong. This is the gap recorded in [Remark 8.12](../proofs/08_certificate.md#s8-12).
- **[Corollary 8.9](../proofs/08_certificate.md#s8-9) (first-order version and its cap).**
  - With the oracle continuation gradient the first-order score has zero wrong signs on the 56 of 396 rows where its margin exceeds the remainder bound, as (a) states.
  - The measured first-order certificate (b) certifies 0 of 1,584 smooth pairs. The second-order remainder alone exceeds the measured first-order score on 69% of pairs and the true gap on 73%; by the cap (c) those pairs can never be certified at first order.
- **[Remark 8.12](../proofs/08_certificate.md#s8-12) (open problems) is supported by data.** With the observed range in place of $`B`$, which is a diagnostic and not a guarantee:
  - the aligned surrogate certifies 45–57% of pairs with no error;
  - the calibrated surrogate certifies 23–28% of pairs with 22–28 wrong certificates, so transfer of the calibration can fail.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 8.1](#c8-1) | [Theorem 8.1](../proofs/08_verification.md#s8-1) | `v014:C14` | PASS |
| [Check 8.2](#c8-2) | [Theorem 8.1](../proofs/08_verification.md#s8-1)(c) | `v014:C22` | PASS |
| [Check 8.3](#c8-3) | [Theorem 8.1](../proofs/08_verification.md#s8-1) | `pkg:noisy_feature_duality` | PASS |
| [Check 8.4](#c8-4) | [Theorem 8.2](../proofs/08_verification.md#s8-2) | `v015:V1` | PASS |
| [Check 8.5](#c8-5) | [Theorem 8.2](../proofs/08_verification.md#s8-2)(a), (f) | `v014:C15` | Monte Carlo PASS; coverage witness reported as FAIL by design |
| [Check 8.6](#c8-6) | [Theorem 8.2](../proofs/08_verification.md#s8-2) | `v014:C20` | PASS |
| [Check 8.7](#c8-7) | [Theorem 8.2](../proofs/08_verification.md#s8-2)(b), (e) | `v015:E4` | PASS |
| [Check 8.8](#c8-8) | [Theorem 8.2](../proofs/08_verification.md#s8-2)(e), (f) | `pkg:T01`, `pkg:T02` | PASS |
| [Check 8.9](#c8-9) | [Theorem 8.3](../proofs/08_verification.md#s8-3)(c) | `v014:C16` | PASS |
| [Check 8.10](#c8-10) | [Theorem 8.3](../proofs/08_verification.md#s8-3)(a), (b) | `v014:C21` | PASS |
| [Check 8.11](#c8-11) | [Theorem 8.3](../proofs/08_verification.md#s8-3)(c) | `pkg:adaptive_gate_expectations` | PASS |
| [Check 8.12](#c8-12) | [Theorem 8.3](../proofs/08_verification.md#s8-3)(d) | `pkg:T13`, `v015:E7` | PASS; the `v015:E7` replica count is a forecast |
| [Check 8.13](#c8-13) | [Proposition 8.4](../proofs/08_certificate.md#s8-4)(a), [Remark 8.5](../proofs/08_certificate.md#s8-5) | `v015:V9` | PASS |
| [Check 8.14](#c8-14) | [Theorem 8.8](../proofs/08_certificate.md#s8-8), [Proposition 8.10](../proofs/08_certificate.md#s8-10), [Remarks 8.11–8.12](../proofs/08_certificate.md#s8-11) | `v016:C6b`, `v016:C5b` | PASS (no wrong strict certificate); strict coverage 0 (aligned) |
| [Check 8.15](#c8-15) | [Corollary 8.9](../proofs/08_certificate.md#s8-9)(a), (d) | `v015:E12` | zero wrong signs inside the margin (oracle gradient) |
| [Check 8.16](#c8-16) | [Corollary 8.9](../proofs/08_certificate.md#s8-9)(b), (c) | `v016:C6` | PASS (implementation); 0 of 1,584 pairs certified |

All checks are exact finite computations, Monte Carlo on finite processes, or a controlled small network. None is a real-model or benchmark result.

<a id="c8-1"></a>

## Check 8.1 · Primal–dual equality on random instances (Theorem 8.1)

**Verdict.** On 220 random adversarial instances the tolerant discrepancy computed over pairs of laws equals both dual forms. The largest duality residual is $`8.9\times10^{-16}`$.

**Why.** [Theorem 8.1](../proofs/08_verification.md#s8-1) claims an *equality*: the worst response difference among tolerance-feasible law pairs equals a minimum over feature coefficients. A wrong dual, or a missing constraint such as the zero-mass condition, would show up as a strict gap on some random instance.

**Setup.**

- Random finite instances with $`m\in[3,12]`$ messages and $`d\in[1,m-1]`$ features.
- Random responses $`g`$, random features $`\phi`$, and random tolerances $`\epsilon_j\ge0`$, including exact zeros.
- Seed `SeedSequence([20261010, 14])`.
- Three independent linear programs, solved with HiGHS:
  - the primal maximum over $`(P,Q)`$, written directly in probability variables;
  - the oscillation dual $`\min_b[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert]`$;
  - the centered sup-norm dual $`\min_{b,c}[2\lVert g-c-\Phi b\rVert_\infty+\sum_j\epsilon_j\lvert b_j\rvert]`$.
- Further assertions: $`W\le\mathrm{osc}(g)`$; the reduction at $`\mathbf{\epsilon}=0`$ to the exact case, Theorem 8.1(b), which is the single-response form of [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)(b); and $`W\to\mathrm{osc}(g)`$ for large $`\mathbf{\epsilon}`$.
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

**Why.** The stability witness, [Theorem 8.1](../proofs/08_verification.md#s8-1)(c), is the cleanest view of the stability term. Let the feature equal the response up to a scale, $`\phi=\alpha g`$. The approximation residual is then zero, yet the certificate must deteriorate as $`\alpha\to0`$: a tolerance $`\epsilon`$ on $`\phi`$ constrains $`g`$ only to within $`\epsilon/\alpha`$.

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

*Figure 8.1.* Left: tolerant discrepancy for the scaled feature $`\phi=\alpha g`$ (Check 8.2); the LP values lie on $`\min\lbrace1,\epsilon/\alpha\rbrace`$, so an exact but small-scale feature certifies only $`\epsilon/\alpha`$. Right: the explicit example of Check 8.3, where primal, dual and closed form coincide and the optimum stops using the feature once $`\epsilon\ge\tfrac12`$. Exact finite computations. [PDF](../figures/fig8_1_stability.pdf)

<a id="c8-4"></a>

## Check 8.4 · Exact finite training processes (Theorem 8.2)

**Verdict.** On 2,572 exact finite Markov training processes the transport bound is never violated (worst residual 0) and is attained with equality on a sharp witness. Each term of the bound is shown to be necessary by a separate witness.

**Why.** The learning-value bound, [Theorem 8.2](../proofs/08_verification.md#s8-2)(c), combines the stopped telescoping identity, the local transport bound (b) and the audit event (a). A mistake in any of them would produce an instance whose exact learning gap exceeds the bound. Both sides are computed exactly in rational arithmetic, on random and adversarial instances.

**Setup.**

- Finite Markov training processes: 2–4 certified states plus an exit state, message alphabets of size 2–4, and $`R`$ rounds.
- Random reference and candidate kernels, random actual updates, random bounded terminal values.
- An explicit response decomposition $`c+b^\top\phi+r`$, an exact transport error $`a_{t,j}(z)`$, feature drift $`\alpha_{zx}`$ (Theorem 8.2(e)), and exit probability $`\rho`$.
- Seed `SeedSequence([150150, 1])`; all values are `Fraction`.
- Procedure:
  1. compute the exact gap $`\lvert L_E(0)-L_E(j)\rvert`$ by backward values;
  2. compute the stopped occupancies forward, independently;
  3. evaluate the bound of Theorem 8.2(c) and assert gap ≤ bound on 2,000 random instances;
  4. run a hill-climbing adversarial search over candidate kernels;
  5. add the fixed witnesses below and 120 random instances of the IID-affine specialization (Theorem 8.2(d)).

**Result** (`theory/v015_checks/results_v015_independent.json`, group `v015:V1`):

| Quantity | Value | Key |
|---|---:|---|
| status | PASS | `status` |
| instances / assertions | 2,572 / 28,182 | `n_instances`, `n_assertions` |
| worst residual (violations of gap ≤ bound) | 0 | `worst_residual` |
| largest gap/bound ratio over all instances | 1.0 (sharp witness) | `best_ratio`, `data.sharp_transport_witness` |
| largest ratio over 2,000 random instances | 0.736 (gap 0.294, bound 0.400) | `data.worst_random` |
| best adversarial ratio | 0.198 (gap 0.220, bound 1.114, $`\rho=74/75`$) | `data.adversarial.best` |
| IID affine (Theorem 8.2(d)): instances with gap ≤ bound | 120 / 120; max ratio 1.0; adversarial 0.565 | `data.affine_40`, `data.affine_adversarial` |
| exact audit-event probability (Theorem 8.2(a)), $`n=32`$, $`\delta=0.3`$ | 0.99948; worst selection excess −0.605 (negative means (a) holds) | `data.exact_finite_audit_37_38.best` |

Witnesses: each shows that one term of the bound cannot be removed.

| Witness | Exact values | What must be charged |
|---|---|---|
| Uncovered trained state (`data.regression`) | gap $`=1`$; bound with $`a=1`$: $`1.0296`$ (capped at $`1`$); with exit $`\rho=1`$: $`1`$; with the transport term deleted: $`0.0296`$ | transport $`a`$ or exit $`\rho`$ |
| Sharp transport (`data.sharp_transport_witness`) | $`D=0`$, $`a=1`$, gap $`=`$ bound $`=1`$ | the bound is tight |
| Reachable state omitted (`data.omission_witness`) | reach probability $`8/9`$; gap $`8/9`$; bound with that state's term zeroed: $`0`$; valid exit bound: $`8/9`$ | every pre-exit visited state, or $`\rho`$ |
| Complete state (`data.optimizer_state_witness`) | $`\theta=-1`$ in both states; $`(m,v)=(\tfrac12,\tfrac12)`$ versus $`(1,2)`$; next first moment $`-1/8`$ versus $`+1/8`$ | cells defined on the complete learner state |
| Coefficient size (`data.coefficient_stability`) | feature scale $`10^{-4}`$, $`B=10^4`$, $`D=10^{-4}`$, residual 0, gap 1 | $`B_t`$ multiplies $`D`$ ([Theorem 8.1](../proofs/08_verification.md#s8-1)) |

**Why it supports the statement.** The bound holds wherever its hypotheses hold, it is attained, and each hypothesis is needed:

- a reachable state must either be covered with its measured transport or count toward the exit probability;
- cells must be complete learner states, not parameter checkpoints;
- coefficients enter multiplicatively.

<a id="c8-5"></a>

## Check 8.5 · Audit concentration and the coverage witness (Theorem 8.2(a), (f))

**Verdict.** The simultaneous audit event occurs at least at the nominal rate in all four Monte Carlo configurations, and the selection inequality holds on it. On the coverage probe the audited discrepancy is 0 while the learning gap is 1. The suite reports this check as FAIL by design: the failing assertion is a transport-free bound evaluated on the probe, which is exactly the violation that Theorem 8.2(f) predicts.

**Why.** Two things are tested:

- the simultaneous Hoeffding event and the selection inequality of [Theorem 8.2](../proofs/08_verification.md#s8-2)(a), by Monte Carlo;
- an exact state-dependent provider that matches the reference at the audited cell but not at a visited state. This is the minimal instance behind the necessity statement (f).

**Setup.**

- Four audit configurations $`(M,L,d,n_0,n_S,\delta)`$, with 3,000 independent trials each.
- A finite learning instance where every response is exactly $`c+b^\top\phi_x`$ ($`\tau=0`$) and transport is exact.
- Coverage probe: audited cell 0, certified states $`\lbrace 0,1\rbrace`$, initial state 1, update $`U(z,y)=y`$, terminal values $`(0,1)`$.
- Procedure: draw audit samples; record whether the event occurs; compute $`\widehat j`$ and the excesses of the selection inequality (a) and of the exact-transport certificate (d). For the probe, compute the exact gap and the expression obtained by treating state 1 as covered without a transport term.

**Result** (`theory/independent_checks/results_independent_v014.json`, entry `v014:C15`, `details`):

| $`M,L,d`$ | $`n_0,n_S`$ | $`\delta`$ | event frequency | worst excess, (a) | worst excess, (d) |
|---|---|---:|---:|---:|---:|
| 2,2,2 | 40, 50 | 0.10 | 1.000 | −0.873 | −0.238 |
| 4,3,3 | 100, 80 | 0.05 | 1.000 | −0.573 | −0.500 |
| 3,1,1 | 1000, 1200 | 0.20 | 0.99967 (SE 0.00033) | −0.160 | −0.061 |
| 2,2,2 | 8, 8 | 0.40 | 1.000 | −1.626 | −0.477 |

A negative excess means the inequality holds with that margin.

Coverage probe (`details.coverage_probe`): audited empirical discrepancy $`0`$, audit-event probability $`1`$, true learning gap $`1`$. With the transport term deleted the expression is $`0.0296`$. The suite status is FAIL because one assertion evaluates the transport-free expression on the probe (`worst_residual` 0.970, the gap minus that expression). All Monte Carlo assertions of (a) and (d) pass, as the table shows.

**Why it supports the statement.**

- The audit event occurs at least at rate $`1-\delta`$ in every configuration, and (a) and the exact-transport certificate (d) hold on it.
- The probe is the smallest case where audited and trained states differ. Only a bound that charges transport or exit, as Theorem 8.2(c) does, is valid there; this is statement (f).

<a id="c8-6"></a>

## Check 8.6 · Correct features, an omitted feature, and coverage with an exit penalty (Theorem 8.2)

**Verdict.** With the right features and coverage the certificate holds; an omitted relevant feature hides a gap of $`1/2`$; an uncovered visited state must be charged through $`\rho`$.

**Why.** Three finite controls separate the three ingredients of the certificate: correct feature coverage, decision-relevant features, and state coverage.

**Setup.** An exact finite learning process with known responses, under three controls: (i) the correct feature set; (ii) one relevant feature removed; (iii) training that reaches states outside the audited set.

**Result** (entry `v014:C20`, `details`; PASS, 11 assertions, worst residual 0):

| Control | Values |
|---|---|
| (i) correct features, $`n=120`$ audit, $`\delta=0.05`$ | $`\widehat D=1/12`$, $`e_0+e_S=0.270`$. Exact-transport certificate (Theorem 8.2(d)): $`0.354`$ ≥ actual three-update gap $`1/12`$. Affine certificate with $`C_J=2`$ for the squared-return task: $`0.707`$ ≥ actual gap $`23/216=0.106`$ |
| (ii) omitted feature | two providers match every audited feature yet differ in learning value by $`1/2`$ |
| (iii) coverage | gap $`1`$; with $`\rho=1`$ the bound is $`1`$; without $`\rho`$ the coverage-free value is $`0`$ |

**Why it supports the statement.**

- With correct features and coverage the certificate is valid.
- A missing feature leaves a gap that no audit of the remaining features can see; this is the exact sufficiency condition of [Theorem 4.2](../proofs/04_complexity_of_sufficiency.md#s4-2)(a).
- A visited uncovered state must enter through $`\rho`$.

<a id="c8-7"></a>

## Check 8.7 · Transport at reached states of a controlled network (Theorem 8.2(b), (e))

**Verdict.** At 40 actually reached states of a small Adam-trained network, transport is nonzero (median 0.0195). The local bound with measured transport contains the gold full-tail contrast at all 40 states, while anchored predictions that assume zero transport miss at 36.

**Why.** In a neural learner, training reaches new states. The local bound of Theorem 8.2(b), with the feature drift of (e), requires the transport term there. This check measures its size and tests whether the local bound with measured transport and feature drift contains the actual contrast.

**Setup.**

- Network: one-hot context, 4 tanh units, sigmoid output, mean cross-entropy, float64, trained with Adam (the controlled network used throughout these pages).
- Anchors: three actual reference states, after 0, 2 and 5 updates, with audited discrepancies $`D_x=0.200, 0.169, 0.188`$.
- Reached states: 40 fresh states between 1 and 13 updates. Messages: the same four scalar labels at every state.
- Provider rule $`Q_z=(1-\lambda)Q_3+\lambda\delta_1`$ with $`\lambda=0.1+0.6\,p(\theta_z)`$, so the candidate law depends on the state.
- Full tail: 6 actual updates, enumerated exactly. Seed `SeedSequence([150150, 104])`.
- Procedure at each reached state:
  1. measure the transport $`a_{t,j}(z)`$ and the feature drift $`\alpha_{zx}`$ to its anchor;
  2. fit the response decomposition and record $`B`$ and $`\tau`$;
  3. compute the gold full-tail contrast;
  4. compare it with (a) the anchored prediction, which assumes zero transport, and (b) the local bound $`B(D_x+a+2\alpha)+2\tau`$.

**Result** (group `v015:E4`, `data`; quantiles are min / median / 90% / max):

| Quantity | Value |
|---|---|
| transport $`a_{t,j}(z)`$ over 40 states | min 0.0014, median 0.0195, max 0.0727 (`observations[*].a`) |
| feature drift $`\alpha_{zx}`$ | min 0.021, median 0.139, max 0.283 (`observations[*].alpha`) |
| states where the gold contrast lies within $`B(D_x+a+2\alpha)+2\tau`$ | **40 / 40** (asserted) |
| anchored-prediction error | 0.00048 / 0.0063 / 0.0328 / 0.0681 (`absolute_error_quantiles`) |
| anchored predictions with error $`>10^{-3}`$ | 36 / 40 (`failures`) |
| response residual $`\lvert r\rvert`$ | 0.0003 / 0.047 / 0.146 / 0.540 (`response_residual_quantiles`) |

**Why it supports the statement.**

- Transport is nonzero at actually reached states, with a median of 0.02 against audited discrepancies of about 0.2. A certificate must measure it.
- Once it is measured, the local bound contains the true contrast everywhere. Extrapolating an anchor audit with zero transport is wrong at 90% of states.
- The median local bound is 0.34 against a median gold contrast of 0.023. The bound is valid but loose; in a neural learner the useful information is in measuring $`a`$, not in its worst-case size.

<a id="c8-8"></a>

## Check 8.8 · Package checks `pkg:T01` and `pkg:T02` (Theorem 8.2(e), (f))

**Verdict.** An independent implementation confirms the uncovered-state witness and the drift-corrected local bound on 1,200 instances.

**Why.** These are independent implementations from the revision package.

**Result** (`docs/direction/v015_revision/results_v015.json`, `checks`):

| Check | Result |
|---|---|
| `pkg:T01` (uncovered-state witness) | PASS. Gap $`1.0`$; transport bound (capped) $`1`$; exit bound $`1`$; with the transport term deleted, $`0.0296`$ |
| `pkg:T02` (local transport and feature drift, Theorem 8.2(b), (e)) | PASS on 1,200 instances; maximum local gap/bound $`0.503`$ |

**Why it supports the statement.** The necessity witness (f) and the feature-drift form (e) of the local bound are reproduced by code that shares nothing with Checks 8.4–8.7.

![Figure 8.2: Frozen audit with transport](../figures/fig8_2_transport.png)

*Figure 8.2.* Exact gap against the transport-aware bound on 2,000 random finite training processes (Check 8.4): no point lies above the diagonal. In the controlled network (Check 8.7) the measured-transport local bound contains the gold full-tail contrast at all 40 reached states, while zero-transport anchored predictions miss 36. The last panel shows transport and feature drift at reached states and the uncovered-state witness, where deleting the transport term gives an invalid 0.0296 against a gap of 1. [PDF](../figures/fig8_2_transport.pdf)

<a id="c8-9"></a>

## Check 8.9 · Adaptive audit on a finite learning process (Theorem 8.3(c))

**Verdict.** In 5 configurations, each with 100,000 independent training runs, the measured gain of the adaptive audit stays above both guarantees, $`-\delta`$ and $`\mathbb E\sum_tB_t-2\delta`$. The selected drift exceeds $`\sum_tB_t`$ with frequency 0.990–0.999, above every nominal $`1-\delta`$.

**Why.** [Theorem 8.3](../proofs/08_verification.md#s8-3)(c) makes a high-probability drift statement and two unconditional expected-value bounds. A violation would appear as a measured gain below $`-\delta`$ or below $`\mathbb E\sum B-2\delta`$, or as a drift-success frequency below $`1-\delta`$.

**Setup.**

- A small finite learning process with 4 states, $`R\le6`$, $`M\in\lbrace 2,3\rbrace`$ candidate kernels per state, and known true advantages.
- Audits sample bounded paired contrasts from the true model: full reference tail plus Bernoulli deployment.
- Selection by the lower confidence bound with $`e_t=\sqrt{2\log(2M/\delta_t)/n_t}`$, followed by a fresh update.
- Stress settings: small $`n_t`$, large $`\delta_t\in[0.1,0.4]`$, and candidates with slightly negative mean and high variance.
- Seed `SeedSequence([20261010, 16])`.
- For each configuration, 100,000 complete trainings (audits plus updates); the gain is estimated with its standard error and the two bounds are checked with a 4-SE margin.

**Result** (`theory/independent_checks/results_independent_v014.json`, entry `v014:C16`, `details.configurations`):

| $`R`$ | $`M`$ | $`n`$ | $`\delta`$ | radius | gain (SE) | $`\mathbb E\sum B`$ | drift frequency | ≥ $`1-\delta`$ |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 2 | 8 | 0.4 | 0.759 | 0.00084 (0.00091) | 0.00090 | 0.99412 | ✓ |
| 2 | 2 | 12 | 0.4 | 0.707 | −0.00067 (0.00102) | 0.00088 | 0.99002 | ✓ |
| 3 | 3 | 20 | 0.2 | 0.671 | 0.00003 (0.00106) | 0.00054 | 0.99407 | ✓ |
| 6 | 3 | 64 | 0.1 | 0.429 | 0.00115 (0.00108) | 0.00026 | 0.99628 | ✓ |
| 2 | 3 | 1000 | 0.2 | 0.090 | **0.13638** (0.00097) | 0.05153 | 0.99897 | ✓ |

Status PASS, with 500,450 instances and 468 assertions. An adversarial search over 327 unique instances found a best margin (bound − gain) of $`-0.199`$, so the bound is always below the gain (`adversarial_best_bound_minus_gain`).

Necessity witnesses (entry `v014:C16`, `details`):

| Witness | Value | Meaning |
|---|---|---|
| `point_estimate_exact_gain` | $`-78/15625`$ | selecting by $`\widehat A`$ without $`-e_t`$ can lose in expectation, so the lower confidence bound is needed |
| `zero_drift_conditioning_witness` | conditional value $`119/494`$ against unconditional $`1/4`$ (audit failure at state 1: $`9/128`$) | conditioning on audit success biases the terminal value, so the guarantee must be unconditional |
| `conditioning_probe_found_45_violation` | false (5 configurations searched) | the unconditional bounds are not violated even though conditional values are biased |
| `replicas_required_per_candidate` | 1016 | Hoeffding allocation for $`M=4,\delta=0.05,e=0.1`$ |

**Why it supports the statement.** The guarantees hold at their stated strength and no stronger:

- with few replicas and large $`\delta`$, the audit almost never selects, and the gain stays within the $`-\delta`$ floor;
- with enough replicas ($`n=1000`$), it selects the beneficial candidate and gains $`0.136`$, well above $`\mathbb E\sum B-2\delta`$.

<a id="c8-10"></a>

## Check 8.10 · Signed identity and fidelity versus improvement (Theorem 8.3(a), (b))

**Verdict.** The signed telescoping identity is exact on 50 rational processes (residual 0), and the fidelity bound holds with equality in three examples, including one where a faithful copy of an ineffective reference loses $`0.11`$.

**Why.** Theorem 8.3(b) is an exact identity for history-dependent provider choices, and (a) separates fidelity from improvement. Both are checked exactly.

**Setup.**

- 50 rational finite processes with state-dependent kernel selection, plus an adversarial search on $`\mathrm{LHS}-\mathrm{RHS}`$.
- Three fidelity examples: a good reference ($`\gamma>0`$), an ineffective one ($`\gamma<0`$) and a neutral one ($`\gamma=0`$), each paired with a candidate of fidelity $`\varepsilon=1/100`$.

**Result** (entry `v014:C21`): PASS, 834 assertions, worst residual 0, `adversarial_best_lhs_minus_rhs` = 0.

| initial | reference | candidate | $`\gamma`$ | $`\varepsilon`$ | candidate gain | gain ≥ $`\gamma-\varepsilon`$ |
|---:|---:|---:|---:|---:|---:|---|
| 1/2 | 4/5 | 79/100 | 3/10 | 1/100 | 29/100 | ✓ (equality) |
| 1/2 | 2/5 | 39/100 | −1/10 | 1/100 | −11/100 | ✓ (equality) |
| 1/2 | 1/2 | 49/100 | 0 | 1/100 | −1/100 | ✓ (equality) |

**Why it supports the statement.**

- The signed identity is exact.
- A high-fidelity copy of an ineffective reference does not improve the initial student: its gain is $`-0.11`$. Fidelity inherits only the reference's own gain.

<a id="c8-11"></a>

## Check 8.11 · Exact adaptive-gate expectations (Theorem 8.3(c))

**Verdict.** Exact expectations, with no Monte Carlo, satisfy both bounds; a slightly inferior candidate passes an optimistic gate with probability 0.00458, so the expected change can be strictly negative.

**Why.** An independent exact computation of the adaptive gate, and of why the guarantee carries $`-\delta`$ rather than $`0`$.

**Result** (`theory/v014_repro/results_v014_rerun.json`, `adaptive_gate_expectations`; $`n=400`$ per candidate, $`\delta=0.05`$, radius $`0.136`$):

| Quantity | Value |
|---|---:|
| unconditional expected gain | 0.200 |
| $`\mathbb E\sum`$ selected lower confidence bound | 0.132 |
| lower bound $`-\delta`$ | −0.050 |
| lower bound $`\mathbb E\sum B-2\delta`$ | 0.032 |
| failure probability | $`1.5\times10^{-8}`$ |
| no exact non-degradation: false-selection probability / expected gain ($`n=20`$, $`\delta=0.9`$, candidate 0.49 vs reference 0.50) | 0.00458 / $`-4.6\times10^{-5}`$ |

**Why it supports the statement.** Both bounds hold with the exact expectations, and the counterexample shows that "no worse than the reference" can only hold up to $`\delta`$.

<a id="c8-12"></a>

## Check 8.12 · Audit cost (Theorem 8.3(d))

**Verdict.** The Hoeffding allocation of 1,016 paired replicas per candidate ($`M=4`$, $`\delta=0.05`$, $`e=0.1`$) gives radius 0.09995, as stated; it is sufficient, not necessary. In the controlled network, deployment noise is 77–89% of the paired variance, and an empirical-Bernstein radius is *forecast* to need 559 replicas.

**Why.** The theorem fixes a valid but expensive allocation, and its cost must be known before any real-model use. The paired-contrast variance is measured in the controlled network to see how much a variance-sensitive radius could save.

**Setup (`v015:E7`).**

- The controlled network, one predeclared audit round, and four candidate providers against a shared reference.
- One reference replica is reused across candidates, with the cross-candidate covariance retained.
- 2,400 independent paired pilot replicas with inverse-CDF coupling.
- The variance is decomposed exactly into block sampling, suffix training and deployment. Seed `SeedSequence([150150, 107])`.

**Result** (`docs/direction/v015_revision/results_v015.json` `checks.T13_*`; `theory/v015_checks/results_v015_independent.json` group `v015:E7`, `data`):

| Quantity | Value |
|---|---|
| `pkg:T13`: sufficient Hoeffding pairs ($`M=4,\delta=.05,e=.1`$) | 1016; radius at 1016 = 0.09995; `is_necessary_lower_bound` = false |
| `v015:E7` Hoeffding allocation | 1016 per candidate; 5,080 training branches with a shared reference (8,128 without reuse) |
| `v015:E7` empirical-Bernstein **forecast** | 559 per candidate; radius 0.0999 at $`n=559`$, from the maximum pilot variance 0.129 |

Variance decomposition of the paired contrast (`v015:E7`):

| candidate | mean | block | suffix | deployment | total | deployment share |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −0.084 | 0.0133 | 0.0041 | 0.0908 | 0.1082 | 84% |
| 1 | −0.042 | 0.0084 | 0.0020 | 0.0454 | 0.0559 | 81% |
| 2 | −0.143 | 0.0096 | 0.0060 | 0.1202 | 0.1358 | 89% |
| 3 | 0.057 | 0.0096 | 0.0027 | 0.0410 | 0.0533 | 77% |

**Why it supports the statement.**

- The Hoeffding allocation is sufficient, as the theorem states.
- Most of the paired variance comes from deployment noise, not from the training update. A variance-sensitive fixed-$`n`$ radius is therefore forecast to need about 45% fewer replicas.
- The 559 is a forecast. An actual audit must compute its own radius at a fixed $`n`$, without optional stopping. `v015:E7` also records projections for a 12B model from measured cluster timings (`data.forecast`); those are planning inputs, not measurements.

![Figure 8.3: Actual-state adaptive audit](../figures/fig8_3_adaptive_audit.png)

*Figure 8.3.* The measured adaptive-audit gain (±2 SE, 100,000 trainings per configuration, Check 8.9) never falls below $`-\delta`$ or $`\mathbb E\sum B-2\delta`$, and it is large only when enough replicas allow selection. The second panel shows replicas per candidate needed for radius $`e`$: Hoeffding's 1,016 at $`e=0.1`$ is sufficient, and the empirical-Bernstein curve is a forecast from pilot variance. The third panel decomposes the paired-contrast variance in the controlled network (Check 8.12); deployment noise dominates. [PDF](../figures/fig8_3_adaptive_audit.pdf)

<a id="c8-13"></a>

## Check 8.13 · Concentration of the mean increment (Proposition 8.4(a), Remark 8.5)

**Verdict.** In all 144 cells (20,000 trials each, 2,880,000 instances) the failure rate is far below $`\delta`$, and the realized deviation quantile never exceeds 0.615 of the bound. The ratio does not grow from $`p=1`$ to $`p=100{,}000`$ at a fixed increment bound $`b`$.

**Why.** [Proposition 8.4](../proofs/08_certificate.md#s8-4)(a) states that the empirical mean increment concentrates in the Hilbert norm at rate $`b(1+\sqrt{2\log(1/\delta)})/\sqrt n`$, with no explicit dimension. LoRA adapters have tens of millions of parameters. An implicit dimension dependence would show as a failure rate above $`\delta`$, or as a realized quantile that rises toward the bound as $`p`$ grows.

**Setup.**

- Dimensions $`p\in\lbrace1,10,1000,100000\rbrace`$, sample sizes $`n\in\lbrace8,32,128\rbrace`$, confidence $`\delta\in\lbrace0.05,0.2\rbrace`$, and one- and two-sample statistics.
- Three bounded laws with $`b=1`$ (`data.laws`):
  - *sign:* a biased scalar sign times a fixed unit sign vector, i.e. a normalized sign-type increment with $`\eta=1/\sqrt p`$, with different means under $`P`$ and $`Q`$;
  - *clipped Gaussian:* $`N(0,I/p)`$ clipped at norm $`1`$;
  - *sparse spikes:* a uniformly chosen coordinate with a fair sign, norm $`1`$.
- High-dimensional resultants are sampled exactly, by isotropy and a Beta-angle recursion, without forming $`p`$-vectors (`data.implicit_sampling`).
- 20,000 independent trials per cell; seed `SeedSequence([150150, 9])`.
- In each cell: the fraction of trials whose deviation exceeds the bound (asserted ≤ $`\delta`$ plus four binomial standard errors) and the empirical $`(1-\delta)`$ quantile of deviation / bound.

**Result** (`theory/v015_checks/results_v015_independent.json`, group `v015:V9`, `data.rows`):

| $`p`$ | cells | worst failure rate, $`\delta=0.05`$ | worst failure rate, $`\delta=0.2`$ | lowest coverage | largest quantile/bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 0.00095 | 0.00765 | 0.99235 | 0.615 |
| 10 | 36 | 0.00030 | 0.00315 | 0.99685 | 0.574 |
| 1,000 | 36 | 0.00050 | 0.00355 | 0.99645 | 0.574 |
| 100,000 | 36 | 0.00040 | 0.00335 | 0.99665 | 0.574 |

By law, the largest quantile/bound is 0.574 (sign), 0.408 (clipped Gaussian) and 0.615 (sparse spikes); the overall maximum is `data.max_quantile_over_bound` = 0.615. For the one-sample clipped Gaussian at $`\delta=0.05`$ the ratio falls with $`p`$, from 0.406 to 0.291, at every $`n`$.

**Why it supports the statement.**

- The bound is valid and conservative by a factor of at least 1.6 in every cell, and nothing in it deteriorates with dimension at fixed $`b`$.
- As [Remark 8.5](../proofs/08_certificate.md#s8-5) states, $`b`$ itself can grow with dimension: for unnormalized sign-type increments $`b=\eta\sqrt p`$. The first-order differences then have the same scale, so the relative accuracy is still set by $`1/\sqrt n`$ (`data.caveat`).
- The check covers the estimation of the mean increment only. Ranking also needs the continuation gradient and the remainder (Checks 8.15–8.16).

The concentration result is panel A of [Figure 8.4](#c8-15).

<a id="c8-14"></a>

## Check 8.14 · Finite CRN paired replacement on the controlled panel (Theorem 8.8, Proposition 8.10, Remarks 8.11–8.12)

**Verdict.**

- **No wrong strict certificate** at either replica count: the assertion of [Theorem 8.8](../proofs/08_certificate.md#s8-8) passes.
- **The strict certificate is nearly empty.** With the a-priori range $`B=2`$, the aligned surrogate certifies 0 of 1,728 pairs and the calibrated short surrogate 0 ($`r=64`$) and 8 ($`r=256`$). The range term of the empirical Bernstein radius is 0.156 at $`r=256`$, against a median true gap of 0.0037. The zero-error statement is therefore nearly vacuous; this is the open problem of [Remark 8.12](../proofs/08_certificate.md#s8-12).
- **As a ranker the measurement works.** The aligned point estimate $`\hat S`$ has the wrong sign on 48 of 1,687 nonzero-gold pairs (2.8%) at $`r=256`$, and on 65 (3.9%) at $`r=64`$.
- **It is cheap.** Measurement takes 7.0% ($`r=64`$) and 28.0% ($`r=256`$) of the trajectories of a full-tail paired audit of the same panel.
- **CRN pays.** Common random numbers reduce the variance of the paired difference to 0.12–0.15 of independent tails (ratio of summed variances); the median pair ratio is 0.0035.
- **Calibration transfer can fail.** With the observed range in place of $`B`$, which is a diagnostic only, the aligned surrogate certifies 45–57% of pairs with no error, but the calibrated surrogate certifies 23–28% with 22–28 wrong certificates.

**Why.** [Theorem 8.8](../proofs/08_certificate.md#s8-8) replaces the oracle continuation gradient by a measurable quantity: the effect of one finite replacement at the audited state, estimated with common random numbers. Its estimand is exact in expectation and needs no smoothness, so its only error terms are statistical ($`e_{\rm stat}`$) and surrogate calibration ($`\varepsilon_{\rm sur}`$). The check asks three questions:

- whether any strict certificate is wrong (it must not be);
- how much the a-priori range costs (Remark 8.12);
- whether the calibration of Remark 8.11 transfers from calibration states to held-out states.

**Setup.**

- **Panel.** The frozen controlled-network provider panel: 40 unique held-out cells, nine provider laws (one reference and $`M=8`$ candidates), 48 split/cell occurrences, 1,728 unordered candidate pairs (1,584 on smooth objectives), and 432 candidate-versus-reference rows (396 smooth, the rows on which the oracle-margin analysis of Check 8.15 was run).
- **Gold.** The exact eight-update outcome of each pair: one candidate boundary followed by all 128 seven-update reference histories. It is recomputed after measurement and agrees with the frozen records to $`1.1\times10^{-16}`$.
- **Replacements.** The actual one-round optimizer increment is executed for each of the ten support atoms. In replica $`\ell`$ one fresh seven-update reference tail $`\omega_\ell`$ is drawn and shared by all candidates; each updated atom state is continued under it, and terminal outcomes are weighted by each provider's exact law. The integration over atoms happens after the nonlinear tail, so the estimand is $`\tilde L(P^j)-\tilde L(P^k)`$ of [Definition 8.6](../proofs/08_certificate.md#s8-6), not a mean-update approximation.
- **Replicas.** $`r=256`$ independent tails; $`r=64`$ uses the first 64. Streams `SeedSequence([160160,66,warmup_step,recipe_id,0])`.
- **Surrogates.**
  - *Aligned:* the bounded task utility after the full seven-update reference tail; $`\alpha=1`$, $`\varepsilon_{\rm sur}=0`$.
  - *Short:* the same utility after one reference update. Following Remark 8.11, it is calibrated before any held-out measurement on eight separate calibration cells (288 pairs) that share no complete learner state with the held-out cells: a positive zero-intercept least-squares fit of exact gold differences on measured short differences gives $`\alpha`$, and $`\varepsilon_{\rm sur}`$ is the largest calibration residual divided by $`\alpha`$.
- **Radius.** With $`\delta=0.05`$ simultaneous over pairs at one state, $`\log(4M^2/\delta)=8.5409`$ and $`\hat V`$ the unbiased sample variance of $`Z`$,

```math
e_{\rm stat}=\sqrt{\frac{2\hat V\log(4M^2/\delta)}{r}}+\frac{7B\log(4M^2/\delta)}{3(r-1)},\qquad \text{certified if }\ \lvert\hat S\rvert>e_{\rm stat}+\varepsilon_{\rm sur}.
```

- **Range.** All utilities lie in $`[0,1]`$, so every difference lies in $`[-1,1]`$: the a-priori range $`B=2`$ holds for every draw, state and pair. As a diagnostic only, the same formula is also evaluated with the observed range $`B=\max Z-\min Z`$, which is not a bound on the population support.
- **Comparators.** Separately executed independent tails per candidate, for the CRN variance ratio. For cost, the full-tail paired audit at the Hoeffding rate of 1,016 replicas per candidate ($`M=4`$, $`\delta=0.05`$, $`e=0.1`$), scaled to all executed panel candidates: $`40\times9\times1016=365{,}760`$ trajectories. The panel actually has $`M=8`$, whose Hoeffding rate is 1,154, so this comparison favours the audit.
- **Eligibility.** All pairs are eligible, including the 144 threshold-task pairs with discontinuous return, because Theorem 8.8 needs no smoothness.

**Result — calibration** (`theory/v016_checks/results_v016.json`, check `v016:C6b`, `data[0].calibration_by_r`):

| $`r`$ | $`\alpha`$ | $`\varepsilon_{\rm sur}`$ | largest calibration residual |
|---:|---:|---:|---:|
| 64 | 0.881 | 0.00834 | 0.00735 |
| 256 | 0.886 | 0.00830 | 0.00735 |

**Result — certificates** (check `v016:C6b`, `data[1..8]`; 1,728 pairs; "observed" rows are diagnostics):

| $`r`$ | surrogate | range | certified | wrong certified | certified on the 396 smooth reference rows | large-gold pairs missed ($`\lvert{\rm gold}\rvert\ge0.01`$) |
|---:|---|---|---:|---:|---:|---:|
| 64 | aligned | a-priori $`B=2`$ | 0 | 0 | 0 | 374 / 374 |
| 256 | aligned | a-priori $`B=2`$ | 0 | 0 | 0 | 374 / 374 |
| 64 | short | a-priori $`B=2`$ | 0 | 0 | 0 | 374 / 374 |
| 256 | short | a-priori $`B=2`$ | 8 (0.46%) | 0 | 0 | 366 / 374 |
| 64 | aligned | observed (diagnostic) | 784 (45.4%) | 0 | 157 | 282 / 374 |
| 256 | aligned | observed (diagnostic) | 989 (57.2%) | 0 | 206 | 166 / 374 |
| 64 | short | observed (diagnostic) | 398 (23.0%) | 22 | 90 (20 wrong) | 150 / 374 |
| 256 | short | observed (diagnostic) | 488 (28.2%) | 28 | 102 (22 wrong) | 137 / 374 |

For comparison, the oracle first-order margin of Check 8.15 covers 56 of the same 396 rows, and the measured first-order certificate of Check 8.16 covers 0.

**Result — radius components** (medians over the 1,728 pairs, computed from the raw companion `results_v016_raw.json.gz`):

| $`r`$ | surrogate | $`\lvert{\rm gold}\rvert`$ | $`\lvert\hat S\rvert`$ | variance term | range term, $`B=2`$ | range term, observed $`B`$ | $`\varepsilon_{\rm sur}`$ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | aligned | 0.0037 | 0.0037 | 0.0011 | 0.633 | 0.0028 | 0 |
| 256 | aligned | 0.0037 | 0.0036 | 0.00055 | 0.156 | 0.00087 | 0 |
| 64 | short | 0.0037 | 0.0038 | 0.00035 | 0.633 | 0.00043 | 0.0083 |
| 256 | short | 0.0037 | 0.0037 | 0.00017 | 0.156 | 0.00011 | 0.0083 |

**Result — ranking by the point estimate** (check `v016:C5b`, `data[0].overall.paired_replacement`, aligned, $`r=256`$): 48 wrong signs among 1,687 nonzero-gold pairs (2.8%) and no ties. On the same pairs the first-order rankers are wrong 11.4%–16.4% of the time and the LESS-style cosine 28.8%; the breakdown by optimizer configuration and by second-moment-channel share is in [§7 Experiments](07_second_moment_channel.md). The short surrogate's point estimate is wrong on 192 (11.4%, $`r=64`$) and 182 (10.8%, $`r=256`$) of the 1,687 pairs.

**Result — variance reduction by CRN** (`variance`):

| $`r`$ | surrogate | Var(CRN)/Var(independent), summed | median pair ratio | pairs with reduced variance |
|---:|---|---:|---:|---:|
| 64 | aligned | 0.120 | 0.0035 | 99.3% of 1,691 |
| 256 | aligned | 0.143 | 0.0036 | 98.8% of 1,691 |
| 64 | short | 0.148 | 0.0154 | 90.3% of 1,584 |
| 256 | short | 0.147 | 0.0148 | 90.0% of 1,584 |

Pairs whose independent-tail variance is exactly zero have no defined ratio (37 aligned, 144 short) and are counted separately.

**Result — cost** (check `v016:C6b`, `data[0].costs.by_r_surrogate`; reference 365,760 full-tail trajectories):

| $`r`$ | surrogate | trajectories | fraction of full-tail audit | fraction in tail updates |
|---:|---|---:|---:|---:|
| 64 | aligned | 25,600 | 7.0% | 7.0% |
| 64 | short (incl. calibration) | 40,960 | 11.2% | 4.0% |
| 256 | aligned | 102,400 | 28.0% | 28.0% |
| 256 | short (incl. calibration) | 133,120 | 36.4% | 7.6% |

Support sharing executes 10 atom trajectories per replica for 9 candidate values (14 without sharing). The independent-tail variance comparator and the held-out gold audits are diagnostics and are excluded. The audit has a different accuracy criterion, so these ratios alone do not establish superiority.

**Why it supports the statement.**

- **Theorem 8.8.** No strict certificate is wrong at either $`r`$, for either surrogate, on unordered pairs or reference rows. The estimand is exact in expectation, by (a): the point estimate ranks 97.2% of nonzero-gold pairs correctly, with no gradient, curvature or smoothness assumption, including on the threshold task.
- **Proposition 8.10(b), (c).**
  - With $`M=8`$ and $`\delta=0.05`$ the range term is $`7B\cdot8.54/(3(r-1))`$: 0.633 at $`r=64`$ and 0.156 at $`r=256`$ for $`B=2`$. This is two orders of magnitude above the typical gap.
  - The variance term is already below the gap at $`r=64`$.
  - CRN reduces the variance by roughly an order of magnitude in aggregate, and by more than two orders of magnitude for the median aligned pair.
- **Remark 8.12, first problem.** The bottleneck of the strict certificate is the a-priori range, not the variance. With the observed range the aligned surrogate would certify 45–57% of pairs without error, and 157–206 of the 396 reference rows against the oracle margin's 56. The observed range is not a proven bound. A provable range of that size, for example from a pathwise Lipschitz bound, is what is missing.
- **Remark 8.12, second problem; Remark 8.11.** The calibrated short surrogate certifies wrongly 22–28 times under the observed range. Its calibration envelope, fitted on calibration states, does not transfer to every held-out state and recipe; half of the wrong certificates fall in the held-out learning-rate split (11 of 22 at $`r=64`$, 14 of 28 at $`r=256`$). Calibration transfer is therefore an assumption that must be tested, as Remark 8.11 states.

The certificate results are panels B–D of [Figure 8.5](#c8-16).

<a id="c8-15"></a>

## Check 8.15 · The first-order margin with the oracle continuation gradient (Corollary 8.9)

**Verdict.** On the 56 of 396 smooth rows where the first-order margin exceeds the remainder bound, the first-order score computed with the **oracle** continuation gradient has zero wrong signs. On the same rows every static proxy is wrong 35.7% of the time. Outside the margin the oracle score is wrong on 34.1% of nonzero-gold rows.

**Why.** [Corollary 8.9](../proofs/08_certificate.md#s8-9)(a) states that a first-order score ranks two feedback laws correctly whenever its margin exceeds the second-order remainder. Corollary 8.9(d) adds that a cheap score can rank correctly only if (i) it pairs the optimizer-transformed mean increment with the continuation gradient at the actual state and (ii) it is used where the margin holds. The check tests both conditions on the exact rows of the frozen held-out panel, without refitting anything. It uses the exact continuation gradient of each row's reference tail, which a real audit does not have; Checks 8.14 and 8.16 are the measured counterparts.

**Setup.**

- The held-out panels and frozen proxy models of [§9 Experiments](09_mechanism_and_method.md), read unchanged; their SHA-256 hashes match before and after (`data.frozen_sha256_before`, `_after`).
- Each row is one candidate update at a learner state followed by the declared seven reference updates, with the exact full-tail gold contrast.
- Oracle score $`\langle\nabla V^0(z),\mu^{\rm cand}-\mu^{\rm ref}\rangle`$, with $`\nabla V^0`$ the exact continuation gradient of that row's reference tail.
- The remainder bound uses sampled, unpadded segment curvature, so "inside the margin" is an empirical classification, not a certified one.
- The 36 rows of the threshold task have a discontinuous return; they violate the smoothness assumption and are ineligible.
- Seed `SeedSequence([150150, 112])`.
- Rows are split by the ratio |score| / remainder bound; wrong-sign rates are reported for the oracle, every frozen proxy, measured short tails and random selection. A zero prediction is not counted as a wrong sign, and only rows with nonzero gold enter the rates.

**Result — overall** (`theory/v015_checks/results_v015_independent.json`, group `v015:E12`, `data.summary[split=overall]`):

| Quantity | Value |
|---|---:|
| rows / eligible (smooth) rows | 432 / 396 |
| rows with margin $`>`$ remainder bound | 56 (14.1% of eligible) |
| oracle wrong-sign rate inside the margin | **0** |
| oracle wrong-sign rate outside the margin | 0.341 (101 of 296 nonzero-gold rows) |
| oracle wrong-sign rate, all eligible nonzero-gold rows | 0.287 |
| static proxies inside the margin (input moment, mean gradient, sign, $`\Delta\theta`$, $`\Delta\theta`$ with $`m,v`$) | 0.357 each |
| measured short tail of 1 / 4 updates, inside the margin | 0 / 0 |
| measured short tail of 1 / 4 updates, outside the margin | 0.203 / 0.030 |
| random selection, inside the margin | 0.536 |
| full-tail gold | 0 |

**Result — by ratio |oracle score| / remainder bound** (`data.bins`):

| ratio | rows | oracle | mean gradient | $`\Delta\theta`$ with $`m,v`$ | short tail 1 | short tail 4 | random |
|---|---:|---:|---:|---:|---:|---:|---:|
| $`[0,0.25)`$ | 316 | 0.371 | 0.371 | 0.485 | 0.221 | 0.033 | 0.533 |
| $`[0.25,0.5)`$ | 12 | 0 | 0.250 | 0.083 | 0 | 0 | 0.333 |
| $`[0.5,1)`$ | 12 | 0 | 0 | 0 | 0 | 0 | 0.667 |
| $`[1,2)`$ | 8 | 0 | 0 | 0 | 0 | 0 | 0.625 |
| $`[2,\infty)`$ | 48 | 0 | 0.417 | 0.417 | 0 | 0 | 0.521 |

**Result — by split** (`data.summary`):

| split | eligible rows | margin rows | oracle wrong signs inside / overall |
|---|---:|---:|---|
| held-out states | 72 | 16 | 0 / 0.234 |
| held-out $`\epsilon`$ | 72 | 16 | 0 / 0.281 |
| held-out learning rate | 72 | 0 | — / 0.313 |
| held-out $`\epsilon`$ and learning rate | 72 | 0 | — / 0.313 |
| fresh → warm | 72 | 16 | 0 / 0.234 |
| warm → fresh | 36 | 8 | 0 / 0.406 |
| held-out threshold task | 0 (36 ineligible) | — | — |

**Why it supports the statement.**

- **The margin criterion holds.** Wherever the first-order margin exceeds the remainder bound, the oracle score never ranks wrongly, in each of the four splits that contain margin rows. It is never wrong once the ratio passes 0.25, so the empirical remainder bound is conservative.
- **Inside the margin, static proxies fail for lack of the continuation gradient.** On the 48 rows with ratio ≥ 2, where first order decides, the static proxies are wrong 41.7% of the time and the oracle 0%. They use the same update statistics without the state- and task-specific continuation gradient, so they violate condition (i) of Corollary 8.9(d).
- **Outside the margin the tail decides.** On the 316 rows with ratio below 0.25 even the oracle is wrong 37.1% of the time, while measured four-update tails are wrong 3.3% of the time; condition (ii) fails there.
- **The margin is rare.** Only 14% of eligible rows meet it, and none in the held-out learning-rate splits, whose step sizes are ten times those of the pilot recipes (Adam $`\eta=0.1`$, SGD $`\eta=1`$; `theory/v015_checks/proxy_split_freeze.json`, `recipe_shift_axes`). This is the remainder scaling of [Proposition 5.1](../proofs/05_optimizer_state.md#s5-1).
- **Limit of the evidence.** The score uses the oracle gradient, so the check shows what a first-order score can achieve at best; as Corollary 8.9(d) states, it is not a measurable certificate.

![Figure 8.4: First-order measurement and the first-order margin](../figures/fig8_4_first_order_audit.png)

*Figure 8.4.* Left: the empirical deviation of the mean increment stays below 0.62 of the bound at every dimension from 1 to 100,000 (Check 8.13). Middle: on the frozen held-out panel, the first-order score computed with the **oracle** continuation gradient has no wrong signs once its margin exceeds a quarter of the remainder bound, while static proxies remain wrong 42% of the time at large margins (Check 8.15). Right: inside the margin first order decides; outside it only measured tails are reliable. Controlled small network. [PDF](../figures/fig8_4_first_order_audit.pdf)

<a id="c8-16"></a>

## Check 8.16 · The measured first-order certificate and its cap (Corollary 8.9)

**Verdict.** Measured without any oracle gradient, the first-order certificate of Corollary 8.9(b) certifies 0 of 1,584 smooth pairs, at every step size $`h`$, for both surrogates; it therefore makes no certification error, vacuously. The causes are the two predicted by Corollary 8.9(c) and Proposition 8.10(b):

- the second-order remainder $`e_2`$ alone exceeds the measured first-order score on 69% of pairs (and the true gap on 73%);
- with eight groups the statistical radius is about six times the typical gap.

The measurement also costs 8.3 times the full-tail paired audit, 95% of it in curvature search.

**Why.** Corollary 8.9(b) says that a first-order certificate must add a finite-difference term $`e_{\rm fd}`$ and the second-order remainder $`e_2`$ to the statistical radius. By the cap (c), a pair is certified only if its first-order term exceeds $`e_2+\varepsilon_{\rm sur}`$, whatever its true gap. Measuring every term on the same panel as Check 8.14 shows how much of the panel this cap leaves.

**Setup.**

- Same panel, gold and calibration cells as Check 8.14.
- Each provider has one or two support atoms, so the mean increment is integrated exactly; $`G=8`$ groups, each with one fresh seven-update reference tail shared by all candidates (CRN).
- Group score $`Z_g=[\tilde F(T_\omega(z+h\mu_j))-\tilde F(T_\omega(z+h\mu_k))]/h`$ for $`h\in\lbrace10^{-3},3\times10^{-3},10^{-2}\rbrace`$, with the $`h`$ of smallest task-unit radius chosen per pair without consulting gold.
- Radius $`e_{\rm stat}+e_{\rm fd}+\varepsilon_{\rm sur}+e_2/\alpha`$:
  - $`e_{\rm stat}`$ by empirical Bernstein with $`\log(4M^2/\delta)=8.5409`$ and the observed range of the eight group scores (this observed range is not a proven support bound);
  - $`e_{\rm fd}`$ from measured ray curvature of the CRN continuation (five-point stencil with Richardson extrapolation);
  - $`e_2=\tfrac12M_2(m_2^j+m_2^k)`$ from measured curvature search along the actual increment rays.
- Surrogates as in Check 8.14. Short-surrogate calibration: $`\alpha=0.902`$ and $`\varepsilon_{\rm sur}=0.0078`$ at every $`h`$.
- The 144 threshold pairs are ineligible (nonsmooth).

**Result — coverage** (`theory/v016_checks/results_v016.json`, check `v016:C6`, `data[1..2]`):

| surrogate | eligible pairs | certified (every $`h`$) | wrong certified | large-gold pairs missed | reference rows certified (of 396) |
|---|---:|---:|---:|---:|---:|
| aligned | 1,584 | 0 | 0 | 281 / 281 | 0 |
| short | 1,584 | 0 | 0 | 281 / 281 | 0 |

The chosen $`h`$ is $`10^{-3}`$ for 1,501 aligned pairs and $`10^{-2}`$ for 83.

**Result — radius components** (aligned surrogate, chosen $`h`$, medians over the 1,584 eligible pairs, from the raw companion):

| Quantity | Median | Remark |
|---|---:|---|
| $`\lvert{\rm gold}\rvert`$ | 0.0035 | true pair gap |
| $`\lvert\text{first-order score}\rvert`$ | 0.0035 | |
| $`e_{\rm stat}`$ | 0.0195 | exceeds the score on 94.8% of pairs |
| of which the range term $`7B\log(4M^2/\delta)/(3(G-1))`$ | 0.0164 | $`=2.85\times`$ the observed range $`B`$ when $`G=8`$ |
| $`e_{\rm fd}`$ | $`1.5\times10^{-5}`$ | but above 1 on 16.7% of pairs (curvature blow-up) |
| $`e_2`$ | 0.0045 | exceeds the measured score on 68.9% of pairs and the true gap on 73.0% |

**Result — cost** (check `v016:C6`, `data[0].costs`):

| Item | Trajectories |
|---|---:|
| held-out score and finite-difference endpoints (per surrogate) | 43,200 |
| remainder curvature search | 2,995,200 |
| calibration (short surrogate) | 18,880 |
| all executed tail trajectories | 3,151,680 |
| curvature search share | 95.0% (96.4% of tail updates) |
| measurement / full-tail paired audit (365,760) | 8.31 (aligned), 8.36 (short) |

**Why it supports the statement.**

- **The cap is binding.** By Corollary 8.9(c) a pair can be certified at first order only if its first-order term exceeds $`e_2`$. The measured score exceeds $`e_2`$ on only 31.1% of pairs, and $`e_2+e_{\rm fd}`$ on 28.0%; the rest are ruled out before any statistics, even with unlimited replicas.
- **The remaining pairs fail on statistics.** With $`G=8`$ the empirical Bernstein range term is 2.85 times the observed range, and $`e_{\rm stat}`$ exceeds the score on 95% of pairs. This is the small-sample regime of Proposition 8.10(b): bringing the range term below $`0.2B`$ at $`M=8`$, $`\delta=0.05`$ needs $`r\ge101`$ replicas, not 8 groups.
- **The curvature terms are expensive and unstable.** Curvature search takes 95% of the tail cost and gives very large $`e_{\rm fd}`$ on a sixth of the pairs.
- **The two failure causes are independent.** Theorem 8.8 removes the second-order cause entirely; Check 8.14 shows that the statistical cause then reduces to the choice of range $`B`$.
- **No certificate is wrong**, but with zero coverage this is vacuous; the PASS status certifies the implementation, not usefulness.

![Figure 8.5: Measurable certificates on the controlled panel](../figures/fig8_5_certificate.png)

*Figure 8.5.* First-order certificate (Check 8.16): the median radius components $`e_{\rm stat}`$, $`e_{\rm fd}`$ and $`e_2`$ against the median true gap; the remainder alone is comparable to the gap, so nothing is certified. Finite CRN paired replacement (Check 8.14): the certified fraction under the a-priori range $`B=2`$ (strict) and under the observed range (diagnostic only), with wrong certificates, for both surrogates and $`r\in\lbrace64,256\rbrace`$. The remaining panels show the distribution of the CRN variance ratio and the trajectory cost relative to the full-tail paired audit. Controlled small network. [PDF](../figures/fig8_5_certificate.pdf)

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C14,C15,C16,C20,C21,C22 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V1,E4,E7,V9,E12 --output theory/v015_checks/quick/s8.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C6,C6b,C5b --output theory/v016_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v014_handoff/verify_environment_information_v014.py   # noisy_feature_duality, adaptive_gate_expectations
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py              # T01, T02, T13
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: entries `v014:C14`, `v014:C15`, `v014:C16`, `v014:C20`, `v014:C21`, `v014:C22`; specification `theory/independent_checks/SPEC.md`.
- `theory/v015_checks/results_v015_independent.json`: groups `v015:V1`, `v015:E4`, `v015:E7`, `v015:V9`, `v015:E12`; specification `theory/v015_checks/SPEC.md`; frozen panel inputs `theory/v015_checks/proxy_split_freeze.json` and `proxy_models_frozen.json`.
- `theory/v016_checks/results_v016.json`: checks `v016:C6`, `v016:C6b`, `v016:C5b`; raw companion `results_v016_raw.json.gz` (per-pair scores, radius components, replica outcomes; not committed); specification `theory/v016_checks/SPEC.md`; code `c5c6.py`, `c5c6b.py`.
- `theory/v014_repro/results_v014_rerun.json`: `noisy_feature_duality`, `adaptive_gate_expectations`.
- `docs/direction/v015_revision/results_v015.json`: `checks.T01_*`, `checks.T02_*`, `checks.T13_*`.
- Figures: `../figures/fig8_1_stability`, `fig8_2_transport`, `fig8_3_adaptive_audit`, `fig8_4_first_order_audit`, `fig8_5_certificate` (`.pdf` and `.png`; specs in `../figures/specs/`).
