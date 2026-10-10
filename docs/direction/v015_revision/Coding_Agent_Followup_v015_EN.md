# Coding-agent follow-up: coverage repair and optimizer-aware information tests

Version 0.15. Prepared on 2026-10-10 UTC.
Status: executable follow-up specification. The external C23 experiments reported by the user have not been rerun here. Preserve the v0.14 plan and configuration unchanged.

## 1. Starting evidence and immediate objective

The user reports that the 19 v0.14 file hashes reproduce; an independent suite contains 23 checks, with 22 passes and one failure concerning Theorem 19 state coverage; a four-unit tanh/sigmoid network agrees with the controlled prediction under small-step SGD; and an Adam experiment using an N=3 moment-matching construction has a reported gap near 0.077 after one update and 0.12 after four updates. Obtain the exact matched-order specification from its source code; N=3 does not by itself mean that all moments through order three match.

Treat these as **reported observations**, not independently reproduced results. The external source code, raw arrays, exact estimands, optimizer configuration, checkpoints, and statistical repetitions are not currently available here. A plot does not establish all of these details. Do not convert the reported gaps into percentage points or claim an exact replication without the reward definition and raw outputs.

The immediate task is to determine which environmental feedback is required by the **actual learner–task composition**. The next work should repair the coverage claim, identify why Adam distinguishes the feedback sources, and test whether measurable optimizer-aware responses predict useful learning. Packet coupling remains an optional branch, not the unifying theory or a project-wide gate.

Return the following small reproducibility bundle before interpreting the reported C23 outcomes:

- The 19-file hash manifest and the 23 check definitions, expected results, actual results, and code commit.
- The full specification of each feedback law, its support/probabilities, and which input, gradient, or update moments actually match.
- Network architecture, initialization, dtype, loss, reduction, clipping, optimizer code/version and all hyperparameters.
- Raw per-support gradients, parameter updates, optimizer states and task returns for the one-step and four-step cases.
- Repetition seeds, exact versus Monte Carlo computations, standard errors and the definition of the reported 0.077 and 0.12 quantities.
- The minimal failing state-coverage example and the precise version/wording of Theorem 19 it tested.

Missing artifacts must be named as blockers. Do not replace the unavailable external implementation with a similar implementation and call it a replication.

## 2. Priority A: isolate Adam's response before adding model scale

### 2.1 Record the update that was actually executed

For each coordinate, log the gradient **after** accumulation, loss scaling and clipping, but **before** it enters the optimizer. Record \(g_t,m_{t-1},v_{t-1},t,\theta_t\), then the computed moments, bias corrections, denominator, parameter update, and any decoupled weight decay. State whether AMSGrad, fused kernels, parameter-group-specific settings, gradient scaling or nonstandard optimizer code is enabled.

With fresh zero moments, bias correction, no clipping or weight decay, and epsilon outside the square root, the first Adam update is exactly

\[
\Delta\theta=-\eta\frac{g}{|g|+\epsilon}.
\]

If the implementation instead places epsilon inside the square root, its first update is

\[
\Delta\theta=-\eta\frac{g}{\sqrt{g^2+\epsilon}}.
\]

These are different algorithms. The sign update is only the \(\epsilon\to0\) limit on nonzero gradients. At an exact zero-gradient atom, the implemented update is zero for positive epsilon. Do not substitute a sign formula into a finite-epsilon experiment.

Warm moments and subsequent steps require the full recurrence. For standard bias-corrected Adam,

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad
\Delta\theta_t=-\eta_t
\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.
\]

Use the actual global step counter and actual optimizer state. A reset-moment ablation is a different learner and must be labelled accordingly.

### 2.2 First-step finite-support audit

For every support point under each reported feedback source:

1. Compute its actual gradient from the same restored network state.
2. Enumerate the exact expected gradient, relevant gradient moments, and probability of \(g=0\) or \(|g|\le\epsilon\).
3. Apply the **actual finite-epsilon Adam map** coordinatewise; enumerate its exact expected update.
4. Compare that result with the optimizer's measured first update and with the resulting real task response.
5. Separately enumerate the sign-limit response as a diagnostic, never as the finite-epsilon ground truth.

Input moment matching does not imply gradient moment matching when the network/loss maps input nonlinearly to gradients. Gradient moment matching does not imply matching a rational normalized update. Matching a mean parameter update does not itself imply matching nonlinear deployment utility. Report all three transitions rather than assigning every discrepancy to “feature drift.”

Run a small ablation matrix:

| Factor | Values | What it isolates |
|---|---|---|
| Epsilon placement | Outside versus inside square root, explicitly implemented | Denominator semantics |
| Epsilon magnitude | Original value plus a small logarithmic grid | Finite-epsilon dependence |
| Bias correction | Original setting versus explicitly disabled control | Step-counter/correction effect |
| Moment state | Fresh zero state versus a genuinely warm checkpoint | Optimizer memory |
| Clipping | Disabled control versus original rule | Additional nonlinear transformation |
| Loss reduction | Fixed original token/unit weights versus documented alternative | Implicit reweighting |
| Gradient zero atoms | Original support; a controlled shifted-support case | Discontinuity/near-zero mass sensitivity |

Change one factor at a time first. Preserve the original experiment as the baseline. For a shifted-support diagnostic, recheck the intended moment equalities after shifting. A same-sign distribution with gradients bounded away from zero can remove a sign-limit effect at step one, while finite epsilon and later optimizer history can still matter.

**Acceptance:** enumerated actual-update predictions and measured optimizer updates agree to a dtype-appropriate, predeclared tolerance. If they do not, stop interpretation and fix the implementation or the claimed gradient law.

### 2.3 An optimizer-matched frozen-Jacobian comparator

Do not compare Adam on the full network only with a plain affine-SGD toy learner and call the difference representation learning. Starting from a common full snapshot \((\theta_0,m_0,v_0,\ldots)\), compare:

| Model response | Optimizer |
|---|---|
| Actual network \(f_\theta\) | Actual Adam, with full state |
| Frozen-Jacobian model \(f_{\theta_0}+J_{\theta_0}(\theta-\theta_0)\) | The same Adam implementation and initial state |
| Actual network | Declared SGD control |
| Frozen-Jacobian model | The same declared SGD control |
| Exact scalar affine-update construction | Its mathematical reference update |

Each branch maintains and updates its own valid optimizer moments after the common initialization. Do not force it to borrow the full network's later \(m_t,v_t\). Preserve losses, supervised targets, normalization, data order and task evaluation where the models permit the same definition.

The frozen-Jacobian model is not automatically an affine learner: a nonlinear loss and Adam normalization can still make its learning map nonlinear. Log both the Jacobian change and the function linearization residual. An Adam gap that persists under a frozen Jacobian shows that evolving features are not necessary for that gap. A difference between actual and frozen models is evidence about linearization/feature evolution, not by itself a complete causal decomposition.

### 2.4 Mandatory history regression test: sign alone is insufficient

Use a two-step scalar control with zero initial moments, \(\beta_1=\beta_2=1/2\), no weight decay/clipping, bias correction, and gradient histories

\[
(1,-3/4)\quad\text{and}\quad(2,-3/4).
\]

The gradient signs are identical across histories. At step two,

\[
m_2=-1/8\quad\text{versus}\quad+1/8,\qquad
\widehat v_2=17/24\quad\text{versus}\quad41/24.
\]

Thus the second parameter updates have opposite signs. In the zero-epsilon control, all denominators here remain nonzero and the first parameter updates are exactly equal. With the actual positive epsilon, compute both first updates exactly; they need not be identical. The second-step opposite-direction conclusion remains.

This regression test refutes a general multi-step sufficiency claim based only on first-step gradient signs. It does not refute the restricted affine moment theorem.

## 3. Priority B: repair and test state coverage

The reported Theorem 19 failure must be reproduced against its exact source version, then classified. Uniform concentration over a frozen finite collection of audit cells does not extend automatically to arbitrary later neural states. Nearest-checkpoint assignment by parameter distance supplies no missing coverage theorem.

Use the following operational notation, consistent with the formal patch. For a fixed candidate \(j\), at an audited anchor \(x\), measure fixed features \(\phi_x\), and let

\[
D_x=\max_\ell |(P_x-Q_{j,x})\phi_{x,\ell}|.
\]

For a later actual state \(z\) and candidate \(j\), explicitly bound or diagnose the signed-discrepancy transport using those same fixed audited features:

\[
a_{tj}(z)=
\max_\ell
\left|(P_z-Q_{j,z})\phi_{x,\ell}
      -(P_x-Q_{j,x})\phi_{x,\ell}\right|.
\]

If the practical metric instead uses state-dependent features \(\phi_z\), add

\[
\alpha_{zx}=\sup_y\|\phi_z(y)-\phi_x(y)\|_\infty.
\]

Then its current-state feature discrepancy is bounded by \(D_x+a_{tj}(z)+2\alpha_{zx}\). If the actual current-state continuation response has an approximation

\[
g_z=c_z+b_z^\top\phi_z+r_z,\qquad
\|b_z\|_1\le B_z,\quad \|r_z\|_\infty\le\tau_z,
\]

its local response error is at most

\[
B_z\big(D_x+a_{tj}(z)+2\alpha_{zx}\big)+2\tau_z.
\]

Use simultaneous confidence bounds for the observed anchor discrepancy, and include coverage/exit and other confidence terms required by the composed theorem. Define a common message domain or explicit transport map before comparing features across states. If such a domain does not exist, do not compute the displayed norm by silently aligning unrelated coordinates.

For a **certificate**, the transport, feature-drift and response-residual bounds must be independently valid on the claimed state/message range. A few observed small values, small parameter distance, or low fitted residual do not establish these bounds. An empirical nearest-checkpoint experiment remains a diagnostic.

Required coverage tests:

1. Reproduce the original missing-state counterexample.
2. Verify the corrected guarantee when every actual finite state is covered.
3. Deliberately omit a reachable state and show that setting its transport/residual/exit term to zero is invalid.
4. Enumerate a finite example with nonzero transport and verify the corrected bound.
5. Hold parameters fixed but vary legitimate optimizer/history state; verify that parameter-only coverage can miss a changed response.
6. In the controlled neural learner, compare anchored predictions against fresh measurements at actually reached states. Report residual distributions and failures; do not promote them to a uniform theorem.

Keep the finite noisy-feature stability check: a response approximation can have small residual yet amplify measurement errors through large coefficients. The relevant quantity includes both approximation error and coefficient-weighted feature uncertainty.

## 4. Priority C: distinguish initial demonstration coverage from new information

Plot \(K\), the count of initial complete demonstrations per task family, separately from the number and type of additional queries. More replay does not increase either independent initial coverage or new observations.

Implement an exactly checkable hidden-bit model:

- A task family has \(m\) observable subcontexts \(j\), each with an unknown independent fair bit \(b_j\).
- A complete teacher demonstration at subcontext \(j\) reveals \(b_j\).
- Initial demonstration contexts are IID with probabilities \(q_j\); deployment context probabilities are \(w_j\), with \(\sum_jq_j=\sum_jw_j=1\).
- The student can store one decision per observable subcontext. Its Bayes-optimal decision is correct for observed bits and guesses for unobserved bits.

The exact expected regret after \(K\) initial demonstrations is

\[
R_K=\frac12\sum_{j=1}^m w_j(1-q_j)^K.
\]

An allowed additional query reveals one chosen previously unobserved bit. With uniform \(q_j=w_j=1/m\), conditional on \(C_K\) distinct initially observed contexts, \(Q\) distinct new queries leave regret

\[
R_{K,Q}\mid C_K=\frac{(m-C_K-Q)_+}{2m}.
\]

Compute the occupancy distribution of \(C_K\) exactly or by an independently checked recursion. Compare passive additional demonstrations, active unseen-context queries, and repeated replay of the same information. Count each query and its feedback type. For nonuniform deployment weights, test choosing high-weight unobserved contexts.

This finite model tests identification and coverage under an explicit access contract. Its curves are not a theorem for arbitrary language-model training or a guarantee that every environment can reveal a bit with one query. Keep a separate actual-learner curve to distinguish information availability from the learner's ability to use it.

## 5. Priority D: validate optimizer-aware proxies at G5

Candidate features should track the actual update and future task response, not just input moments or first-step signs. Start with a small frozen feature family:

- The actual proposed parameter update \(\Delta\theta\), projected using a declared pilot-only or fixed-random map.
- The current and next optimizer state, including \(m,v\), step counter, and the normalization/preconditioning factors.
- The learner parameters or task-response features needed to interpret an update's direction.
- Short-tail and selected full-tail real task responses, with their true costs.
- Interactions between actual updates and independently fitted task-response directions, when measurable.

Importance weights, sign features, gradient projections and learned task-response weights are **proxies**. None is sufficient merely because it is interpretable. Do not set approximation or coverage error to zero after observing a high correlation.

Freeze candidates on pilot data. Test on held-out **states, task response functions, and optimizer recipes**. Include fresh versus warm moments, changed epsilon and a held-out learning rate. Keep the split axes separate so a failure can be attributed to its scope.

Primary metric evaluation is whether the proxy chooses better real training feedback/providers on held-out cases. Also report signed effect prediction, magnitude error, calibration, wrong-sign decisions and abstention. Compare against input-moment error, mean gradient, sign-only features, actual first-step update, task identity, training-step index, random scores and full-tail gold measurements on the affordable subset.

A proxy that fits first-step effects but fails later tails does not validate multi-step information sufficiency. A negative sign-only result may support adding optimizer state; it does not by itself prove the expanded features sufficient.

## 6. Gold auditing: cost-aware evidence without changing the theorem

The v0.14 Hoeffding calculation of 1,016 independent paired training replicas **per candidate**, for \(M=4,\delta=.05,e\le.1\), is a sufficient worst-case sample size for that bound. It is **not a lower bound** on the samples needed by every valid procedure or by the particular experiment.

Before considering that expense:

1. Run a paired-variance pilot with correctly coupled reference/candidate marginals and independent training replicas.
2. Estimate variability attributable to blocks, suffix training and final deployment separately.
3. If reference replicas are shared across candidates, retain the resulting covariance and count the reference work once. Sharing does not create independent reference evidence.
4. Audit only predeclared sparse training rounds or choose the next round from past information with valid conditional accounting.
5. Consider an applicable variance-adaptive fixed-sample or sequential confidence procedure, with its assumptions and multiplicity stated. Do not reuse a fixed-\(n\) radius after outcome-dependent stopping.
6. Forecast full-tail updates and deployment episodes using measured throughput.

For cheaper **aggregate method evidence**, randomize independent full training runs to a reference arm and a complete candidate method arm, then evaluate both on the same held-out task set. This directly estimates average terminal improvement of the whole training procedure without estimating a local certificate at every state. It does not supply local sufficiency, a statewise gate, or a theorem about one realized training run.

Keep Theorem 20's distinctions: its high-probability result concerns predictable cumulative drift; its terminal-value conclusions are unconditional expectations with the stated failure-budget penalties. A practical proxy gate or an underpowered gold pilot must not be reported as a high-probability terminal-performance certificate.

## 7. Minimal execution sequence and stop criteria

The following are **interfaces to implement or map to the supplied repository**. They are not claims that the external C23 package has already been received or that these commands have run.

~~~bash
python -m eic.cli ingest-c23 --manifest external/c23_manifest.json
python -m eic.cli audit-adam-map --config configs/v015_controlled.yaml
python -m eic.cli test-optimizer-history --config configs/v015_controlled.yaml
python -m eic.cli compare-frozen-jacobian --config configs/v015_controlled.yaml
python -m eic.cli test-state-coverage-patch --config configs/v015_controlled.yaml
python -m eic.cli test-hidden-bit-coverage --config configs/v015_controlled.yaml
python -m eic.cli validate-response-proxies --config configs/v015_controlled.yaml
python -m eic.cli forecast-gold-audit --config configs/v015_controlled.yaml
~~~

Run these in order, preserving raw records. The controlled suite should remain small enough for exact support enumeration and multiple independent checks. No broad Gemma sweep is required to diagnose the reported Adam result.

Only after the controlled map, history, coverage and held-out proxy tests pass should the coding agent run a **micro-LoRA pilot on the actual specified checkpoint**, google/gemma-4-12B-it, if a usable runtime and checkpoint access are provided. Use the original black-box teacher contract. Do not substitute a smaller checkpoint or another model and report it as this pilot. Limit the first pilot to one task family, one or a few full-state checkpoints, a small number of feasible feedback providers, and a separately held-out task set. Measure restoration fidelity, throughput and proxy signal before scaling.

| Gate | Pass condition | Failure action |
|---|---|---|
| External reproduction | Code, manifest and raw estimands reproduce | Report precisely what is missing or differs |
| Actual Adam map | Enumerated finite-epsilon map matches executed updates | Fix implementation/semantics before interpretation |
| History control | Same-sign histories expose different state-dependent updates | Diagnose incorrect state or recurrence |
| Frozen-Jacobian comparison | Optimizer-matched branches are valid and differences interpretable | Repair comparator; do not infer feature drift |
| Coverage patch | Finite covered case and explicit failure/transport cases match the corrected statement | Repair statement or implementation |
| Proxy G5 | Held-out provider choices improve over simple controls with reported uncertainty | Revise/drop that proxy, retaining independent theory branches |
| Micro-LoRA feasibility | Exact requested checkpoint, usable runtime, full-state restore and measurable signal | Report resource/access or measurement blocker |
| Scale-up | A useful reference and method effect justify the projected cost | Keep controlled scope or revise the method |

Coupling-specific negative results affect the coupling branch. Reference-fidelity, query-selection, feedback-content, coverage and task/learner-adaptation results must retain their own evidence and scope.

## 8. Required compact return package

Return one report plus raw artifacts sufficient to recompute it:

1. C23 reproduction status and the exact coverage failure.
2. Per-support Adam map table, with epsilon placement, correction, clipping, zero atoms and loss weights.
3. The same-sign history regression test and optimizer-matched frozen-Jacobian comparison.
4. Corrected state-coverage/feature-transport checks, distinguishing certified finite terms from empirical neural diagnostics.
5. \(K\)-coverage and additional-query curves for the exact hidden-bit model.
6. Held-out proxy predictions and actual training responses, including negative cases.
7. Paired variance and cost forecast, with reference reuse and audit-round accounting.
8. Whether a real micro-LoRA pilot ran; if so, checkpoint IDs, updates, raw deployment results and restore logs; otherwise exact blockers.
9. A claim ledger separating reproduced facts, exact finite results, user-reported observations, empirical neural evidence and unresolved theory.

The intended outcome is a sharper unified information theory and an implementable distillation method whose useful scope is measured. Neither the current Adam gap nor one repaired coverage theorem establishes universal neural sufficiency or an award-level novelty claim.

