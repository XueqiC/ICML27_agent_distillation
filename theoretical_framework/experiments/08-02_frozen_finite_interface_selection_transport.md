# Theorem 8.2 · Frozen finite-interface selection with transport

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-02_frozen_finite_interface_selection_transport.md) · [Proof](../proofs/08-02_frozen_finite_interface_selection_transport.md) · [Experiments](../experiments/08-02_frozen_finite_interface_selection_transport.md) · [中文](08-02_frozen_finite_interface_selection_transport_ZH.md)

**What is tested here.**

- **[Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) (frozen selection with transport) holds, and its transport term is necessary.**
  - No violation on 2,572 exact finite training processes; worst residual 0; the bound is attained (ratio 1).
  - On an uncovered trained state the audited discrepancy is 0 while the learning gap is 1.
  - In a controlled network the measured-transport local bound contains the true contrast at 40/40 reached states. Predictions that assume zero transport miss at 36/40.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Rounds are $`t=0,\dots,R-1`$, and $`z`$ is a complete learner state. For a first-order optimizer the actual update is $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$.
- At state $`z`$ in round $`t`$ the reference emits $`Y\sim P^0_{t,z}`$, and candidate $`j\in\lbrace1,\dots,M\rbrace`$ emits $`Y\sim P^j_{t,z}`$. The kernel of candidate $`j`$ is $`K^j_t`$.
- The reference continuation is $`V^0_{t+1}`$, with $`V^0_R=F_E\in[0,1]`$. The response is $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$. Integrating the message gives the kernel identity

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- $`L_E(j)=\mathbb E\,F_E(Z^j_R)`$ for the process driven by candidate $`j`$. All processes start from the same initial law of $`Z_0`$.
- The *local advantage* of candidate $`j`$ at $`(t,z)`$ is

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- It is the effect on the final score of replacing one reference round by one round of candidate $`j`$, followed by the reference tail. By [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2), the whole-run difference is a sum of such terms over the states the alternative process reaches.

**Frozen audit.** Using only independent pilot data, freeze before the audit:

- $`M`$ executable candidate feedback laws;
- a finite set $`\mathcal U`$ of audit cells $`u`$, each specifying a complete learner state, a task, a query, the update and the reference tail (the letter $`u`$ denotes a cell only in this subsection);
- $`d`$ features $`\phi_{u,\ell}:\mathcal Y\to[0,1]`$ per cell.

$`P^0_u`$ and $`P^j_u`$ are the message laws of the reference and of candidate $`j`$ in cell $`u`$. The audit draws $`n_0`$ independent reference messages per cell and $`n_S`$ independent messages per candidate and cell. Let $`\mu_{j,u,\ell}=\mathbb E_{P^j_u}\phi_{u,\ell}`$, with empirical means $`\widehat\mu_{j,u,\ell}`$, and

```math
D_j=\max_{u,\ell}\lvert\mu_{0,u,\ell}-\mu_{j,u,\ell}\rvert,\qquad\widehat D_j=\max_{u,\ell}\lvert\widehat\mu_{0,u,\ell}-\widehat\mu_{j,u,\ell}\rvert,
```

```math
e_0=\sqrt{\frac{\log(4\lvert\mathcal U\rvert d/\delta)}{2n_0}},\qquad e_S=\sqrt{\frac{\log(4M\lvert\mathcal U\rvert d/\delta)}{2n_S}}.
```

The selected candidate $`\widehat j`$ minimizes $`\widehat D_j`$ up to a known optimization error $`\eta_{\rm opt}`$.

**Certified states.** For each round $`t`$ there is a certified set $`\mathcal C_t`$ of complete states. Each $`z\in\mathcal C_t`$ carries the following.

- An audited cell $`u=u(t,z)`$ whose features are expressed in the message coordinates at $`z`$.
- A *transport bound*: for every candidate that can be selected on the audit event,

```math
\Big\lVert(P^0_{t,z}-P^j_{t,z})\phi_u-(P^0_u-P^j_u)\phi_u\Big\rVert_\infty\le a_{t,j}(z).\qquad\text{(T)}
```

- A *response decomposition*

```math
g_{t,z}=c_{t,z}+b_{t,z}^\top\phi_u+r_{t,z},\qquad\lVert b_{t,z}\rVert_1\le B_t,\quad\lVert r_{t,z}\rVert_\infty\le\tau_t.\qquad\text{(D)}
```

Let $`\sigma=\inf\lbrace t<R:Z_t\notin\mathcal C_t\rbrace`$ ($`\sigma=\infty`$ if no such $`t`$), a stopping time of the training process. **Fresh training:** conditional on the audit result, the selected candidate is trained with randomness independent of the audit samples, and $`\Pr_{\widehat j}(\sigma<R\mid\text{audit result})\le\rho`$.

**Statement under test ([Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)).** All bounds below may be capped at $`1`$.

**(a) Audit concentration.** With probability at least $`1-\delta`$, simultaneously for all $`j`$,

```math
\lvert D_j-\widehat D_j\rvert\le e_0+e_S,\qquad\Longrightarrow\qquad D_{\widehat j}\le\min_jD_j+2(e_0+e_S)+\eta_{\rm opt}.
```

**(b) Local transport bound.** For $`z\in\mathcal C_t`$ and $`u=u(t,z)`$,

```math
\big\lvert(K^0_t-K^j_t)V^0_{t+1}(z)\big\rvert\le\min\Big\lbrace1,\ B_t\big(D_j+a_{t,j}(z)\big)+2\tau_t\Big\rbrace .
```

**(c) Learning-value bound.** On the event of (a),

```math
\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_{t=0}^{R-1}\mathbb E_{\widehat j}\Big[1\lbrace t<\sigma\rbrace\Big\lbrace B_t\big(\widehat D_{\widehat j}+e_0+e_S+a_{t,\widehat j}(Z_t)\big)+2\tau_t\Big\rbrace\Big]+\rho .
```

- **(d) Specializations.** With exact transport ($`a\equiv0`$), $`\lvert L_E(0)-L_E(\widehat j)\rvert\le\sum_tB_t(\widehat D_{\widehat j}+e_0+e_S)+2\sum_t\tau_t+\rho`$. Under the state-independent IID-block and convex-affine assumptions of [Theorem 4.5(a)](../statements/04-05_round_independent_bound_averaging_rates.md#s4-5), with the audited features equal to the moments $`Z^\alpha`$, $`1\le\lvert\alpha\rvert\le N`$, $`\lvert L_E(0)-L_E(\widehat j)\rvert\le C_J(\widehat D_{\widehat j}+e_0+e_S)`$.
- **(e) Feature drift.** If state-dependent features $`\phi_z`$ satisfy $`\sup_y\lVert\phi_z(y)-\phi_u(y)\rVert_\infty\le\alpha_{zu}`$, then $`\lVert(P^0_{t,z}-P^j_{t,z})\phi_z\rVert_\infty\le D_j+a_{t,j}(z)+2\alpha_{zu}`$, and (b)–(c) hold with $`\phi_z`$ in (D) and $`a`$ replaced by $`a+2\alpha_{zu}`$.
- **(f) Necessity of transport.** For every audit size there is an instance with $`D_j=0`$, an exact response decomposition ($`B=1`$, $`\tau=0`$) and learning gap $`1`$. In it the transport-free expression $`B(\widehat D+e_0+e_S)`$ is strictly below the gap, while the bound of (c) remains valid and is tight.

<a id="c8-4"></a>

## Check 8.4 · Exact finite training processes (Theorem 8.2)

**Verdict.** On 2,572 exact finite Markov training processes the transport bound is never violated (worst residual 0) and is attained with equality on a sharp witness. Each term of the bound is shown to be necessary by a separate witness.

**Why.** The learning-value bound, [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)(c), combines the stopped telescoping identity, the local transport bound (b) and the audit event (a). A mistake in any of them would produce an instance whose exact learning gap exceeds the bound. Both sides are computed exactly in rational arithmetic, on random and adversarial instances.

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
| Coefficient size (`data.coefficient_stability`) | feature scale $`10^{-4}`$, $`B=10^4`$, $`D=10^{-4}`$, residual 0, gap 1 | $`B_t`$ multiplies $`D`$ ([Theorem 8.1](../statements/08-01_stability_approximation_tradeoff.md#s8-1)) |

**Why it supports the statement.** The bound holds wherever its hypotheses hold, it is attained, and each hypothesis is needed:

- a reachable state must either be covered with its measured transport or count toward the exit probability;
- cells must be complete learner states, not parameter checkpoints;
- coefficients enter multiplicatively.

<a id="c8-5"></a>

## Check 8.5 · Audit concentration and the coverage witness (Theorem 8.2(a), (f))

**Verdict.** The simultaneous audit event occurs at least at the nominal rate in all four Monte Carlo configurations, and the selection inequality holds on it. On the coverage probe the audited discrepancy is 0 while the learning gap is 1. The suite reports this check as FAIL by design: the failing assertion is a transport-free bound evaluated on the probe, which is exactly the violation that Theorem 8.2(f) predicts.

**Why.** Two things are tested:

- the simultaneous Hoeffding event ([Hoeffding, 1963](#ref-hoeffding1963probability)) and the selection inequality of [Theorem 8.2](../statements/08-02_frozen_finite_interface_selection_transport.md#s8-2)(a), by Monte Carlo;
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
- A missing feature leaves a gap that no audit of the remaining features can see; this is the exact sufficiency condition of [Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2)(a).
- A visited uncovered state must enter through $`\rho`$.

<a id="c8-7"></a>

## Check 8.7 · Transport at reached states of a controlled network (Theorem 8.2(b), (e))

**Verdict.** At 40 actually reached states of a small Adam-trained network, transport is nonzero (median 0.0195). The local bound with measured transport contains the gold full-tail contrast at all 40 states, while anchored predictions that assume zero transport miss at 36.

**Why.** In a neural learner, training reaches new states. The local bound of Theorem 8.2(b), with the feature drift of (e), requires the transport term there. This check measures its size and tests whether the local bound with measured transport and feature drift contains the actual contrast.

**Setup.**

- Network: one-hot context, 4 tanh units, sigmoid output, mean cross-entropy, float64, trained with Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) (the controlled network used throughout these pages).
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

**Why it supports the statement.** The necessity witness (f) and the feature-drift form (e) of the local bound are reproduced by code that shares nothing with [Checks 8.4](../experiments/08-02_frozen_finite_interface_selection_transport.md#c8-4)–[8.7](../experiments/08-02_frozen_finite_interface_selection_transport.md#c8-7).

![Figure 8.2: Frozen audit with transport](../figures/fig8_2_transport.png)

*Figure 8.2.* Exact gap against the transport-aware bound on 2,000 random finite training processes ([Check 8.4](../experiments/08-02_frozen_finite_interface_selection_transport.md#c8-4)): no point lies above the diagonal. In the controlled network ([Check 8.7](../experiments/08-02_frozen_finite_interface_selection_transport.md#c8-7)) the measured-transport local bound contains the gold full-tail contrast at all 40 reached states, while zero-transport anchored predictions miss 36. The last panel shows transport and feature drift at reached states and the uncovered-state witness, where deleting the transport term gives an invalid 0.0296 against a gap of 1. [PDF](../figures/fig8_2_transport.pdf)

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C15,C20 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V1,E4 --output theory/v015_checks/quick/s8.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # T01, T02
```

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: entries `v014:C15`, `v014:C20`.
- `theory/v015_checks/results_v015_independent.json`: groups `v015:V1`, `v015:E4`; specification `theory/v015_checks/SPEC.md`.
- `docs/direction/v015_revision/results_v015.json`: `checks.T01_*`, `checks.T02_*`.
- Figure: `../figures/fig8_2_transport`.
