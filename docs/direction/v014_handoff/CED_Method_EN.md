# Consequence-Calibrated Experience Distillation

**Version:** 0.14, 10 October 2026 UTC.  
**Status:** Proposed method family derived from the environment-information framework. No neural efficacy, global novelty, or award-level claim is established.

## 1. Research objective

Use a small number \(K\) of initial complete teacher demonstrations per task family, plus explicitly accounted permitted feedback, to train a student that succeeds independently in the real task environment. Choose the feedback interface by its effect on the actual student's subsequent learning and deployment.

The theoretical contribution is intended to be the unified characterization of required environment information. CED is its algorithmic research program. It is not restricted to world models, packet coupling, a particular moment order, or a universal scalar quality score.

The motivating distinction is operational:

- Acquisition changes what can be known about relevant decisions.
- Compilation changes how already acquired evidence enters the learner.
- Selection/calibration tests the effects of the resulting updates.

These operations can reduce different terms of the theory's representation–information–algorithm decomposition. A transformation of a fixed transcript cannot be credited with acquiring new environmental facts.

## 2. Fixed project contract

| Item | Contract |
|---|---|
| Initial support | \(K\) complete trusted teacher demonstrations per target task family |
| Main \(K\) | 32; sensitivity \(4,8,16,32\), conditional on enough independent episodes |
| Teacher | \`gpt-6-luna\`, black-box sampled outputs |
| Student | \`google/gemma-4-12B-it\`; exact revision, tokenizer/template, trainable modules pinned |
| Model/component provenance | Preserve the project's exclusion of Chinese checkpoints and model-derived components |
| Teacher internals | No required logits, probabilities, gradients, hidden states, or weights |
| Environment | Real/reference executable task environment; WM optional |
| Deployment | Student alone with frozen observation, tools, memory, decoding, and retry rules |
| Main scientific aim | Useful environment information and generalizable distillation, with full cost accounting |
| Current execution status | Training runtime/access prerequisites not established; see runtime audit |

An account's entitlement and a successful model load must be demonstrated by the executing coding agent. A public model identifier is not that demonstration. Model substitution requires a visible project change, not an automatic fallback.

## 3. Units, state, and the response being estimated

A complete valid feedback unit is an episode or explicitly permitted training fragment with all conditioning context, teacher/sample outputs, real observations, terminal status, supervision mask, provenance, and parent identifiers. A candidate may supply a packet of such units or a declared sequence of micro-updates.

The complete learning snapshot contains model weights, trainable adapters, optimizer state, learning-rate scheduler, gradient scaler, RNG streams, sampler cursor, gradient accumulation state, buffers, and any ongoing environment/agent state.

For recipe \(f\), state \(z\), and candidate provider \(S_j\), define
\[
G_f(z,j)=
\mathbb E\!\left[
J_{E_f}\!\left(\operatorname{Deploy}
\left(\operatorname{Tail}_{f}^{0}(U_f(z,Y_j))\right)\right)
\right],
\]
where \(Y_j\sim S_j(\cdot\mid z)\). The tail is a declared reference **learning** continuation, not merely an action rollout of a frozen policy. Its remaining update count and behavior are fixed before the comparison.

The signed target is \(A_f(z,j)=G_f(z,j)-G_f(z,0)\). A content-level response \(g_f(z,y)\) fixes the message instead of integrating the provider. They are linked by \(G_f(z,j)=\mathbb E_{Y_j}g_f(z,Y_j)\).

Restore before every arm; apply real updates; regenerate any adaptive teacher/student/environment feedback under that arm. Shared exogenous seeds reduce comparison noise only when each arm's marginal remains correct. Reusing the same fixed tail dataset changes the estimand if the reference protocol is interactive.

## 4. Three primary branches and one diagnostic branch

### 4.1 Acquisition branch: expose missing consequences

Start with the initial demonstrations. Identify candidate training questions from student failures or disagreement about action consequences. Execute only allowed training queries: ordinary rollout, teacher response at a student prefix, a restorable branch, or an environment-provided check.

Finite controlled tasks use an explicit environment ambiguity set and the robust decision criterion from the theory. For neural tasks, failure localization, model disagreement, and uncertainty are proposals. Their success must be tested against random, frequency-based, and cost-sensitive query allocation at equal access/cost.

**Primary ablation:** fixed original transcript with arbitrarily many replay/rewrite operations versus genuinely new feedback at controlled query counts. Keep \(K\) unchanged and disclose extra information.

**Failure means:** the acquisition heuristic does not find useful evidence. It does not contradict the passive lower bound, nor justify relabeling a generated explanation as a real observation.

### 4.2 Compilation branch: preserve useful information in learnable form

Given the same acquired evidence bank, build a finite set of executable providers. A starting menu is:

1. Whole-episode teacher SFT/behavior cloning.
2. Teacher-labeled student-prefix corrections.
3. Authentic failure/recovery units with explicit action masks.
4. Cost-sensitive or student-continuation supervision using observable outcomes.
5. A response-calibrated mixture/representation of the above, learned only on the development partition.

Every provider logs what information it drops, retains, or transforms. Retain whole dependencies when the test class needs joint information. A “compressed” provider is not assumed sufficient because its token count or supervised loss is small.

For finite tabular tasks, preserve the derived joint moments or solve the exact noisy-feature discrepancy problem. For neural tasks, learn response features and independently measure whether they predict and select real learning gains. No theorem says monomials of token embeddings are the relevant moments.

**Primary ablation:** identical evidence with different compilation, controlled exposure, tokens, loss weights, and actual optimizer steps. Separately vary exposure when the research question requires it.

### 4.3 Coverage and transfer branch: make a reusable interface

Construct a response panel over development task families and learner recipes. Derive or learn a feature bank from pilot forks, then freeze it.

For a finite panel, select providers by either:

- a small worst-case response discrepancy to a useful reference; or
- a large conservative signed advantage across the predeclared cells.

A robust minimum over task/recipe cells and a predeclared weighted mean answer different objectives; record which one is used. A panel certificate covers that panel. Generalization to a new family requires its response approximation/coverage conditions or an empirical held-out evaluation.

Transfer the **method** by freezing its predictor, compilation rules, calibration recipe, and hyperparameters before acquiring the new family's permitted support. Separately test the frozen **policy** without adaptation.

### 4.4 Optional coupling branch: isolate joint information

Keep the complete-unit bank, all unit/token weights, per-slot marginals, total exposures, and optimizer-step count fixed. Change only valid grouping or ordering.

A practical first experiment takes four whole episodes and two actual updates with two episodes per update. Compare the three pair partitions and their two update orders, using balanced schedules. Evaluate each entire two-update schedule; a sum of independently measured pair scores is only a proposal heuristic.

Preserve per-example loss normalization. Isolate attention across unrelated episodes. Different token-length batches must not silently change example weights, gradient accumulation, truncation, or the number of updates.

Compare against random grouping, independent pairing, antithetic gradient pairing, gradient-diverse batches, and an exposure-matched ordering method. Check the strict additive-SGD null before interpreting a nonzero effect.

The coupling proposition identifies nonadditive response as the only source of fixed-marginal sensitivity. It is not a novelty claim about antithetic sampling, optimal transport, or minibatch ordering. Failure here stops this branch only.

## 5. Learning a usable consequence metric

The method should earn the word “consequence” through held-out measurements.

### 5.1 Gold measurement panel

At each frozen snapshot, draw multiple independent packets per provider. For each draw:

1. Restore the complete learner and environment state.
2. Execute the real candidate update or macro-update.
3. Run the specified reference-learning tail.
4. Evaluate the resulting student independently on the designated validation tasks.
5. Run a valid paired reference arm.

An update/tail replica, rather than each deployment episode of that replica, is the sampling unit for between-training variability. Deployment episodes estimate a trained student's value and have their own nested variance. Save both levels.

Use multiple tail lengths as different estimands. A zero-tail or short-tail ranking can be useful, but it is not automatically a ranking under the full training recipe. Verify how rankings change.

### 5.2 Candidate proxies

Use an ascending baseline ladder:

| Proxy | What it observes | Main limitation |
|---|---|---|
| Teacher success or confidence | Teacher/source behavior | Student mismatch and omitted feedback |
| Student supervised loss | Immediate fitting difficulty | Not deployment consequence |
| Gradient norm/similarity | Local update geometry | May omit optimizer, noise, and long-tail effects |
| GRACE/LESS-style features | Structured student-gradient information | Must validate beyond their training distribution |
| Frozen-student continuation score | Action-level consequences | Does not perform a parameter update |
| Short actual-update tail | A closer but truncated consequence | Horizon/recipe bias |
| Learned response predictor | Features plus actual fork labels | Calibration, leakage, and transfer risk |
| Full-tail gold forks | The declared target | High computational and environment cost |

Start with a transparent ridge or small bounded-output predictor using frozen student-side features and recipe/state descriptors. Compare a nonlinear predictor only if it adds held-out value. Fitting a powerful critic is not itself evidence of novel information.

For a finite response matrix, singular values and low-rank fits are exploratory diagnostics. Frobenius reconstruction error is not a uniform worst-cell certificate. Measure maximum residual with simultaneous uncertainty on a held-out finite panel where possible; mark continuous/out-of-family residuals as unverified.

### 5.3 Required metric outcomes

Report on unseen fork replicas and unseen task/recipe groups:

- Absolute and signed prediction error with uncertainty.
- Pairwise ranking accuracy and sign error near the selection boundary.
- Empirical coverage and width of declared intervals.
- Actual selection regret against the best measured candidate in the same finite menu.
- Harmful-selection frequency and reference-fallback rate.
- Incremental utility over cheaper proxies at matched total cost.
- Relationship between local predicted gains and complete training-run outcomes.

Spearman correlation alone is inadequate: it can look strong while the selected top candidate is harmful, while all useful distinctions occur near zero, or while evaluation costs dominate the gain.

## 6. Selection rules and their guarantee modes

### 6.1 Frozen finite-panel fidelity mode

Freeze cells, features, and candidates from pilot data. Use fresh calibration messages/forks. Select a provider minimizing empirical discrepancy and record its simultaneous uncertainty.

For finite known responses, the exact noisy-feature objective is
\[
\inf_b\left[
\operatorname{osc}(g-\Phi b)+\sum_j\epsilon_j|b_j|
\right].
\]
This makes approximation error and estimation stability part of one criterion. It is a finite LP, not a free neural oracle. The selected provider must subsequently use the declared fresh training law.

When a useful reference gain is independently established, combine it with the discrepancy guarantee. Otherwise report fidelity only.

### 6.2 Gold actual-state adaptive mode

At an actual state, freeze a finite candidate menu and collect independent paired **full-tail** contrasts. With per-candidate sample count \(n_t\), use
\[
e_t=\sqrt{2\log(2M_t/\delta_t)/n_t}.
\]
The reference has lower bound exactly zero. Choose the largest positive lower bound, or the reference if none is positive. Apply a fresh selected update; do not promote the luckiest audit-trained replica.

Under the proof's exact restoration, independence, full-tail, and confidence-allocation conditions, the unconditional expected performance change is at least \(-\sum_t\delta_t\), and at least the expected sum of selected lower bounds minus \(2\sum_t\delta_t\). The simultaneous high-probability statement is about conditional drift, not one realized final score.

This is a costly gold option, not the default large-model training loop. With four candidates and a 0.05 failure budget at one state, a Hoeffding radius of 0.1 needs 1016 paired update/tail replicas per candidate. The pilot's two packet draws per cell are for diagnosis and cost estimation, not such a certificate.

### 6.3 Practical proxy mode

Use an independently calibrated predictor or a short-tail response to propose candidates. Freeze each candidate menu before its validating audit. A valid uniform proxy-bias allowance can be subtracted from the lower bound; absent that allowance, label selection **empirical**.

A cheap proposal can save search even when its scores are not certified. Compare against direct actual-update search using the same total compute/environment budget. Refresh a predictor only with declared training/validation feedback; never consume sealed deployment outcomes.

## 7. Pseudocode

~~~text
Inputs:
  task-family split, initial support D_K, exact student checkpoint
  declared learner recipe f and useful-reference protocol S_0
  permitted training query access, resource envelope, candidate menu builder

1  Freeze policy visibility, deployment rules, family split, and all ledgers.
2  Verify one genuine update, full-state restoration, and task execution.
3  Run pilot-only acquisition/compilation candidates.
4  At pilot snapshots, execute paired actual-update/reference-tail experiments.
5  Fit response features/predictors; compare against cheaper baselines.
6  Freeze finite candidates, features, metric, and calibration protocol.
7  Independently calibrate fidelity or signed advantage on declared cells.
8  For each final training run from the same permitted initialization:
9      Gather only authorized training feedback; record new information.
10     Build valid complete units with provenance.
11     At selected intervention rounds:
12         If gold on-demand audit is enabled and feasible:
13             freeze current candidates; audit full tails;
14             choose positive lower bound or reference; apply fresh update.
15         Else:
16             use the frozen empirical provider/selector; mark its mode.
17     Save all actual updates, checkpoints, and accounting.
18 Evaluate the final student alone on the sealed deployment partition.
19 Freeze the method and repeat on held-out families and learner recipes.
20 Report paired outcomes, uncertainty, failures, and total cost.
~~~

The selector need not run at every gradient step. Macro-updates and sparse audited interventions make the estimand and cost explicit. Every schedule change alters the recipe and must be recorded.

## 8. Baselines and access parity

The main output-only teacher track includes strong teacher SFT/BC, on-policy teacher-label correction, cost-sensitive/student-continuation imitation, gradient-based selection, trajectory/gradient matching where applicable, direct actual-update selection, and the optional coupling controls.

HaDES/Behaviour Distillation is a close conceptual predecessor for optimizing datasets by post-training environmental performance. MetaOPD is a close predecessor for weighting through actual optimizer-aware updates. GRACE and LESS are strong gradient-based comparators. FutureBridge-OPD is a close rollout-consequence comparator. Merely combining “actual update” and “task return” is not a sufficient novelty claim.

Faithful methods requiring teacher token probabilities/scoring belong in a separate same-access track when that access is established. An output-only adaptation must be labeled as an adaptation. Do not call a baseline reproduced if its defining access or objective was removed.

Budget match the complete method, including acquisition, selector fitting, failed candidates, auditing, reference-tail training, calibration, and evaluation. Also report informative-access matching separately: equal compute does not imply equal teacher/environment evidence.

## 9. Stage gates

| Stage | Required deliverable | Continue criterion | Negative result |
|---|---|---|---|
| Runtime | Actual model load, backward step, checkpoint/restore, environment step | Verified execution and measured resource forecast | Produce blocker report; do not fake a neural run |
| Finite theory | Reproduced identities, witnesses, dimensions, duality, rates | Correct construction and declared assumptions | Repair implementation or theorem before claims |
| Neural metric pilot | Real paired update/tail panel and cost ledger | Reproducible signal and useful held-out selection | Revise metric or stop its branch |
| Compilation/acquisition | Controlled interventions on authentic feedback | Gains beyond strong same-access controls | Reject the proposed intervention, preserve negative evidence |
| Coupling, optional | Matched marginals and passed additive null | Effect beyond generic batching explanations | Stop coupling branch only |
| Full distillation | Independent trained seeds and student-only task evaluation | Meaningful interval-supported improvement or informative boundary | Report lack of efficacy |
| Transfer | Frozen method on new families/recipes/domain | Reproducible gains with calibrated uncertainty | Restrict scope and analyze missing response coverage |
| Contribution | Final nearest-work and theory/empirics reconciliation | Distinct theorem/method claim supported by evidence | No first-ever or award-level assertion |

## 10. What is genuinely open

The proposed synthesis is a learner-relative interface theory with an acquisition/verification bridge. The individual use of loss weighting, actual-update validation, moments, pairing, or confidence bounds has prior art. A substantial method contribution needs a specific mechanism that follows from the theory and produces robust, transferable benefits unexplained by those baselines.

Possible successful outcomes include an efficient response approximation with proven scope, a useful acquisition-and-compilation rule with tight guarantees on a structured adaptive class, or a controlled neural result that validates the theory's distinctive interaction/recipe predictions. The project should not force a coupling method if another branch better realizes the central theory.

Read \`Independent_Novelty_Audit_EN.md\` before writing contribution language. Read \`Coding_Agent_Execution_Plan_EN.md\` for the full run protocol. Nothing in this method specification is evidence that a neural experiment has already succeeded.
