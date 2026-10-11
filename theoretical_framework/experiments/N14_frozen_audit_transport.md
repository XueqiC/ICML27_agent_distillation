# N14 · Frozen finite-interface selection with transport — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N14_frozen_audit_transport.md) · [中文](N14_frozen_audit_transport_ZH.md)

**Verdict.** N14 holds. The transport bound [N14(b)](../proofs/N14_frozen_audit_transport.md) is never violated on 2,572 exact finite Markov training processes (V1, worst residual 0), and it is attained with equality on a sharp witness (best ratio 1). The audit concentration [N14(a)](../proofs/N14_frozen_audit_transport.md) holds on all Monte Carlo configurations (C15), and the IID-affine specialization [N14(c)](../proofs/N14_frozen_audit_transport.md) holds on all 120 random instances (V1).

The transport term is necessary. When a trained state is not covered by an audited cell, the audited discrepancy is 0 but the learning gap is 1:

- **(b)** charges that state either $`a=1`$ or $`\rho=1`$, so it gives $`\ge1`$ and stays valid;
- deleting the term gives $`0.0296`$;
- zeroing a reachable state's contribution gives $`0`$, against a gap of $`8/9`$.

In a controlled small network the transport term is not negligible. Across 40 reached states (E4), the local bound with measured transport contains the true contrast at every state. Predictions that assume zero transport miss at 36 of the 40 states.

The supporting checks are C20, T01 and T02.

![Transport-aware certificate versus actual gaps](../figures/N14_transport.png)

[PDF](../figures/N14_transport.pdf)

## V1 · Exact finite training processes

**Why.** [N14(b)](../proofs/N14_frozen_audit_transport.md) combines the stopped telescoping identity, the local transport bound and the audit event. A mistake in any of them would produce an instance whose exact learning gap exceeds the bound. V1 computes both sides exactly in rational arithmetic, on random and adversarial instances.

**Setup.**

- Finite Markov training processes: 2–4 certified states plus an exit state, message alphabets of size 2–4, $`R`$ rounds.
- Random reference and candidate kernels, random actual updates, random bounded terminal values.
- An explicit response decomposition $`c+b^\top\phi+r`$, an exact transport error $`a_{t,j}(z)`$, feature drift $`\alpha_{zx}`$ [N14(b′)](../proofs/N14_frozen_audit_transport.md), and exit probability $`\rho`$.
- Seed `SeedSequence([150150, 1])`; all values are `Fraction`.

**Procedure.**

1. Compute the exact gap $`\lvert L_E(0)-L_E(j)\rvert`$ by backward values.
2. Compute the stopped occupancies forward, independently.
3. Evaluate [N14(b)](../proofs/N14_frozen_audit_transport.md) and assert gap ≤ bound on 2,000 random instances.
4. Run a hill-climbing adversarial search over candidate kernels.
5. Add fixed witnesses, listed below, and 120 random instances of the IID-affine setting [N14(c)](../proofs/N14_frozen_audit_transport.md).

**Result** (`theory/v015_checks/results_v015_independent.json`, `groups[id=V1]`):

| Quantity | Value | Key |
|---|---:|---|
| status | PASS | `status` |
| instances / assertions | 2,572 / 28,182 | `n_instances`, `n_assertions` |
| worst residual (bound − gap violations) | 0 | `worst_residual` |
| largest gap/bound ratio over all instances | 1.0 (sharp witness) | `best_ratio`, `data.sharp_transport_witness` |
| largest ratio over 2,000 random instances | 0.736 (gap 0.294, bound 0.400) | `data.worst_random` |
| best adversarial ratio | 0.198 (gap 0.220, bound 1.114, $`\rho=74/75`$) | `data.adversarial.best` |
| IID affine [N14(c)](../proofs/N14_frozen_audit_transport.md): instances with gap ≤ bound | 120 / 120; max ratio 1.0; adversarial 0.565 | `data.affine_40`, `data.affine_adversarial` |
| exact audit event probability [N14(a)](../proofs/N14_frozen_audit_transport.md), $`n=32`$, $`\delta=0.3`$ | 0.99948; worst selection excess −0.605 (negative = [N14(a)](../proofs/N14_frozen_audit_transport.md) holds) | `data.exact_finite_audit_37_38.best` |

**Witnesses of the hypotheses.** Each witness shows that one term of [N14(b)](../proofs/N14_frozen_audit_transport.md) cannot be removed.

| Witness | Exact values | What must be charged |
|---|---|---|
| Uncovered trained state (`data.regression`) | gap $`=1`$; [N14(b)](../proofs/N14_frozen_audit_transport.md) with $`a=1`$: $`1.0296`$ (capped at $`1`$); with exit $`\rho=1`$: $`1`$; with the transport term deleted: $`0.0296`$ | transport $`a`$ or exit $`\rho`$ |
| Sharp transport (`data.sharp_transport_witness`) | $`D=0`$, $`a=1`$, gap $`=`$ bound $`=1`$ | [N14(b)](../proofs/N14_frozen_audit_transport.md) is tight |
| Reachable state omitted (`data.omission_witness`) | reach probability $`8/9`$; gap $`8/9`$; bound with the state's term zeroed: $`0`$; valid exit bound: $`8/9`$ | every pre-exit visited state, or $`\rho`$ |
| Complete state (`data.optimizer_state_witness`) | $`\theta=-1`$ in both states; $`(m,v)=(\tfrac12,\tfrac12)`$ versus $`(1,2)`$; next first moment $`-1/8`$ versus $`+1/8`$ | cells defined on the complete state |
| Coefficient size (`data.coefficient_stability`) | feature scale $`10^{-4}`$, $`B=10^4`$, $`D=10^{-4}`$, residual 0, gap 1 | $`B_t`$ multiplies $`D`$ (N13) |

**What this shows.** The bound holds wherever its hypotheses hold, it is attained, and each hypothesis is needed:

- a reachable state must either be covered with its measured transport or count toward the exit probability;
- cells must be complete learner states, not parameter checkpoints;
- coefficients enter multiplicatively.

## C15 · Audit concentration and the coverage witness

**Why.** C15 checks two things. First, the simultaneous Hoeffding event [N14(a)](../proofs/N14_frozen_audit_transport.md) and the selection inequality [N14(a)](../proofs/N14_frozen_audit_transport.md) by Monte Carlo. Second, an exact state-dependent provider that matches the reference at the audited cell but not at a visited state.

**Setup.**

- Four audit configurations $`(M,L,d,n_0,n_S,\delta)`$, with 3,000 independent trials each.
- A finite learning instance where every response is exactly $`c+b^\top\phi_x`$ ($`\tau=0`$) and transport is exact.
- The coverage probe: audited cell 0, certified states $`\lbrace 0,1\rbrace`$, initial state 1, update $`U(z,y)=y`$, terminal values $`(0,1)`$.

**Procedure.**

- Draw audit samples, record whether event [N14(a)](../proofs/N14_frozen_audit_transport.md) occurs, and compute $`\widehat j`$ and the excesses of [N14(a)](../proofs/N14_frozen_audit_transport.md) and [N14(c)](../proofs/N14_frozen_audit_transport.md).
- For the probe, compute the exact gap. Compare it with the expression obtained when state 1 is treated as covered without a transport term.

**Result** (`theory/independent_checks/results_independent_v014.json`, `claims[id=C15].details`):

| $`M,L,d`$ | $`n_0,n_S`$ | $`\delta`$ | event frequency | worst [N14(a)](../proofs/N14_frozen_audit_transport.md) excess | worst [N14(c)](../proofs/N14_frozen_audit_transport.md) excess |
|---|---|---:|---:|---:|---:|
| 2,2,2 | 40, 50 | 0.10 | 1.000 | −0.873 | −0.238 |
| 4,3,3 | 100, 80 | 0.05 | 1.000 | −0.573 | −0.500 |
| 3,1,1 | 1000, 1200 | 0.20 | 0.99967 (SE 0.00033) | −0.160 | −0.061 |
| 2,2,2 | 8, 8 | 0.40 | 1.000 | −1.626 | −0.477 |

Negative excess means the inequality holds with that margin.

Coverage probe (`details.coverage_probe`): audited empirical discrepancy $`0`$, audit event probability $`1`$, true learning gap $`1`$. With the transport term deleted, the expression is $`0.0296`$.

The suite flags C15 as FAIL by design: that assertion checks that a transport-free bound is violated on this probe. All Monte Carlo assertions pass.

**What this shows.** The audit event occurs at least at the nominal rate $`1-\delta`$ in every configuration, and [N14(a)](../proofs/N14_frozen_audit_transport.md) and the exact-transport certificate hold on it. The probe is the minimal case where audited and trained states differ. Only a bound that charges transport or exit, as [N14(b)](../proofs/N14_frozen_audit_transport.md) does, is valid there.

## C20 · Correct features, an omitted feature, and coverage with an exit penalty

**Why.** Three finite controls separate the three ingredients of the certificate: correct feature coverage, decision-relevant features, and state coverage.

**Setup.** An exact finite learning process with known responses: (i) the correct feature set, (ii) one relevant feature removed, (iii) training that reaches states outside the audited set.

**Result** (`claims[id=C20].details`):

| Control | Values |
|---|---|
| (i) correct features, $`n=120`$ audit, $`\delta=0.05`$ | $`\widehat D=1/12`$, $`e_0+e_S=0.270`$. Certificate [N14(c)](../proofs/N14_frozen_audit_transport.md): $`0.354`$ ≥ actual three-update gap $`1/12`$. Affine certificate [N14(c)](../proofs/N14_frozen_audit_transport.md) with $`C_J=2`$ for the squared-return task: $`0.707`$ ≥ actual gap $`23/216=0.106`$ |
| (ii) omitted feature | two providers match every audited feature yet differ in learning value by $`1/2`$ |
| (iii) coverage | gap $`1`$; with $`\rho=1`$ the bound is $`1`$; without $`\rho`$, the coverage-free value is $`0`$ |

**What this shows.** With the right features and coverage the certificate holds. A missing feature leaves a gap that no audit of the remaining features can see; this is [N06(a)–(c)](../proofs/N06_sufficiency_duality.md) (N06). A visited uncovered state must enter through $`\rho`$.

## E4 · Transport at reached states of a controlled network

**Why.** In a neural learner the training process reaches new states. [N14(b)](../proofs/N14_frozen_audit_transport.md) requires the transport term there, and E4 measures how large it is. It also checks whether the local bound [N14(b)](../proofs/N14_frozen_audit_transport.md)+[N14(b′)](../proofs/N14_frozen_audit_transport.md) with measured transport contains the actual contrast.

**Setup.**

- Network: the [C23](N11_adam_cold_start.md) (the v0.14 controlled-network experiment in which Adam departed from the affine comparator) network (one-hot context, 4 tanh units, sigmoid, mean cross-entropy, float64), trained with Adam.
- Anchors: three actual reference states, after 0, 2 and 5 updates. Audited discrepancies are $`D_x=0.200, 0.169, 0.188`$.
- Reached states: 40 fresh states between 1 and 13 updates.
- Messages: the same four scalar labels at every state.
- Provider rule: $`Q_z=(1-\lambda)Q_3+\lambda\delta_1`$ with $`\lambda=0.1+0.6\,p(\theta_z)`$. The candidate law therefore depends on the state.
- Full tail: 6 actual updates, enumerated exactly.
- Seed `SeedSequence([150150, 104])`.

**Procedure.** At each reached state:

1. measure the transport $`a_{t,j}(z)`$ and the feature drift $`\alpha_{zx}`$ to its anchor;
2. fit the response decomposition and record $`B`$ and $`\tau`$;
3. compute the gold full-tail contrast;
4. compare it with (a) the anchored prediction, which assumes zero transport, and (b) the local bound $`B(D_x+a+2\alpha)+2\tau`$.

**Result** (`groups[id=E4].data`; quantiles are min / median / 90% / max):

| Quantity | Value |
|---|---|
| transport $`a_{t,j}(z)`$ over 40 states | min 0.0014, median 0.0195, max 0.0727 (`observations[*].a`) |
| feature drift $`\alpha_{zx}`$ | min 0.021, median 0.139, max 0.283 (`observations[*].alpha`) |
| states where $`\lvert`$gold $`\rvert\le B(D_x+a+2\alpha)+2\tau`$ | **40 / 40** (asserted) |
| anchored-prediction error | 0.00048 / 0.0063 / 0.0328 / 0.0681 (`absolute_error_quantiles`) |
| anchored predictions with error $`>10^{-3}`$ | 36 / 40 (`failures`) |
| response residual $`\lvert r\rvert`$ | 0.0003 / 0.047 / 0.146 / 0.540 (`response_residual_quantiles`) |

**What this shows.** Transport is nonzero at actually reached states, with a median of 0.02 against audited discrepancies of about 0.2. A certificate must measure it. Once it is measured, the local bound contains the true contrast everywhere. Extrapolating an anchor audit with zero transport is wrong at 90% of states. The median diagnostic bound is 0.34 against a median gold contrast of 0.023. This is a valid but loose bound, so in a neural learner the useful information is in measuring $`a`$, not in its worst-case size.

## Package checks T01 and T02

**Why.** These are independent implementations from the v0.15 package.

**Result** (`docs/direction/v015_revision/results_v015.json`, `checks`):

| Check | Result |
|---|---|
| T01 (uncovered-state witness) | PASS. Gap $`1.0`$; transport bound (capped) $`1`$; exit bound $`1`$; with the transport term deleted, $`0.0296`$ |
| T02 (local transport and feature drift, [N14(b)](../proofs/N14_frozen_audit_transport.md)–[N14(b′)](../proofs/N14_frozen_audit_transport.md)) | PASS on 1,200 instances; maximum local gap/bound $`0.503`$ |

**What this shows.** These checks independently confirm the necessity witness and the drift-corrected local bound [N14(b′)](../proofs/N14_frozen_audit_transport.md).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V1,E4 --output theory/v015_checks/quick/n14.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C15,C20 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # T01, T02
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: `groups[id=V1]` and `groups[id=E4]`.
- `theory/independent_checks/results_independent_v014.json`: `claims[id=C15]` and `claims[id=C20]`.
- `docs/direction/v015_revision/results_v015.json`: `checks.T01_*` and `checks.T02_*`.
- Specifications: `theory/v015_checks/SPEC.md` (V1, E4) and `theory/independent_checks/SPEC.md` (C15, C20).
- Figure: `../figures/N14_transport.pdf` and `.png` (spec `../figures/specs/N14.md`).
