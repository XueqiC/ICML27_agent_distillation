# Coding-agent execution specification: environment information for few-shot agent distillation

Version: 0.14. Prepared on 2026-10-10 UTC.
Status: execution specification and independent implementability review. Real neural-agent training has not been performed in this workspace.

## 1. Mission and scientific contract

Implement the environment-information framework, train the specified student in executable agent environments, and determine which measurable information discrepancies predict the effect of training on real deployment utility. Deliver both positive and negative results. The goal is to establish a defensible empirical scope for the theory and to assess the proposed distillation method, not to force a particular paper conclusion.

The environment is an information source. A world model is one optional source implementation. The experiment must work with real interaction, black-box teacher responses, complete replay blocks, and structured feedback before any world model is introduced. Do not reintroduce a spectral interpretation or a capability-atom pipeline.

The main theory distinguishes:

1. **Information access:** whether permitted experiments can reveal distinctions that matter for a student decision.
2. **Information transmission:** which relationships the training interface preserves.
3. **Information utilization:** how the actual learner update transforms those relationships into deployment performance.

These are separate questions. An accurate environment predictor, low imitation loss, a strong teacher, and a student that improves are not interchangeable observations.

The unifying object is **the environment information required relative to a declared learner and task family**. The experimental chain is: identify decision-relevant distinctions under allowed queries; derive or estimate continuation-response tests; find feedback representations/providers that approximately preserve those tests; compose local errors through actual training; and verify the resulting decisions and deployment utility. Query selection, feedback content, state coverage, and adaptation across task/learner families are all method branches of this framework. Packet coupling is one diagnostic branch. It is neither the sole novelty claim nor a project-wide success condition.

The term **few-shot** means that each target task family initially supplies \(K\) complete trusted teacher demonstrations. It does not mean \(K\) task families, \(K\) tokens, or \(K\) total pieces of information. Additional teacher requests, environment branches, labels, retrieved knowledge, and training rollouts must be recorded separately. Pretraining is part of the initial information of both models. Do not describe synthetic restatements of an existing block as new independent environment evidence.

The named method and its final objective are defined in the accompanying v0.14 theory/method document. This specification governs implementation and evidence. If a theorem condition conflicts with a practical implementation, report the discrepancy explicitly and classify the result as an empirical extension.

## 2. What has and has not been verified

A read-only audit of the current workspace found:

| Item | Observed status |
|---|---|
| Python | 3.12.14 |
| numpy, scipy, PyYAML | Importable |
| PyTorch, Transformers, datasets, PEFT, TRL, Accelerate | Not installed |
| Gymnasium, ALFWorld | Not installed |
| nvidia-smi, torchrun, accelerate, sbatch, srun | Not present on PATH |
| Agent-training repository, checkpoint manifest, benchmark manifest | Not found in the task workspace |
| Existing material | Theory documents and finite-construction verification code |
| AGENTS.md | No matching file found under /workspace |
| Remote cluster, CUDA device, model weights, paid API access | Not established |

Absence of nvidia-smi alone is not a mathematical proof that no accelerator exists. In conjunction with the missing training stack, it is sufficient to say that a usable GPU training runtime has not been established. Historical mentions of clusters or GPUs do not establish current authorization, credentials, access, or availability.

The original requested identifiers have public official pages:

- Teacher: **gpt-6-luna**, sampled black-box outputs. Official page: https://developers.openai.com/api/docs/models/gpt-6-luna
- Student: **google/gemma-4-12B-it**. Official publisher page: https://huggingface.co/google/gemma-4-12B-it
- Google model documentation: https://ai.google.dev/gemma/docs/core/model_card_4

These checks verify identifiers, not an account's entitlement or a successful local load. Before execution, pin the student repository revision, configuration, tokenizer/processor, chat template, and every trainable module. Record the teacher's requested and returned model identifiers and response metadata. Do not silently substitute another checkpoint or teacher. No Chinese checkpoints or model-derived components are permitted by the project constraint.

There are no neural scores, runtime-throughput measurements, confidence intervals from training, or successful agent runs to report at this stage.

## 3. Read before writing training code

Read the full current v0.14 theory, proof audit, novelty audit, and method specification. Then read these v0.13 documents as the explicit mathematical baseline:

- Environment_Information_Theory_v013_ZH.md
- Stage1_Information_Validation_v013_ZH.md
- verify_environment_information_v013.py
- results_v013.json

Extract a machine-readable theorem ledger with these fields:

| Field | Meaning |
|---|---|
| claim_id | Stable identifier, independent of section numbering |
| exact_claim | The statement actually tested |
| assumptions | All mathematical and access assumptions |
| exact_or_empirical | Exact finite theorem, conditional bound, or empirical extension |
| observable | What is measured and with which sampling unit |
| failure_interpretation | Implementation bug, assumption violation, unsupported proxy, or claim falsification |
| evidence_artifacts | Raw records, run IDs, and recomputation script |

In particular, joint-moment sufficiency and the \(n^{1-r}\) expansion apply to their declared independent-block affine/polynomial learners. They are not a theorem about arbitrary language-model fine-tuning. The general continuation-value identity is exact with a complete learner state; estimating its terms in a large model is a separate statistical and computational problem.

## 4. Deliverable repository and command contract

Create a reproducible repository with the following responsibilities. Names below are **interfaces to implement**, not claims that the modules already exist.

| Component | Required responsibility |
|---|---|
| eic/runtime.py | Read-only runtime inventory and prerequisite gate |
| eic/manifests.py | Task-family splits, hashes, source genealogy, and immutable locks |
| eic/teachers.py | Black-box sampled teacher requests and response cache |
| eic/environments/ | Reset/step/evaluate/snapshot wrappers with access accounting |
| eic/learner.py | Actual parameter updates, complete checkpointing, exact recipe metadata |
| eic/providers.py | Finite candidate interfaces that return complete experience blocks |
| eic/branches.py | Paired training forks and reference continuation semantics |
| eic/metrics.py | Frozen gradient/update/consequence feature extraction |
| eic/calibration.py | Candidate selection, feature freezing, confidence bookkeeping |
| eic/evaluation.py | Independent real deployment, task-level records |
| eic/statistics.py | Paired and clustered inference, power, equivalence, multiplicity |
| eic/cli.py | Explicit stages and nonzero failure codes |
| tests/ | Restore, masking, split, accounting, and small mathematical checks |
| configs/ | Frozen configurations with concrete resolved dependencies |
| reports/ | Claims table, results, failures, costs, and provenance |

After implementation, these commands must be available:

~~~bash
python -m eic.cli audit-runtime --config experiment_config_v014.yaml
python -m eic.cli verify-theory --config experiment_config_v014.yaml
python -m eic.cli freeze-manifest --config experiment_config_v014.yaml
python -m eic.cli smoke-train --config experiment_config_v014.yaml
python -m eic.cli collect-support --config experiment_config_v014.yaml
python -m eic.cli pilot --config experiment_config_v014.yaml
python -m eic.cli freeze-protocol --config experiment_config_v014.yaml
python -m eic.cli measure-consequences --config experiment_config_v014.yaml
python -m eic.cli train --config experiment_config_v014.yaml --run-manifest runs/main.jsonl
python -m eic.cli evaluate --config experiment_config_v014.yaml --sealed-manifest manifests/deploy.jsonl
python -m eic.cli analyze --config experiment_config_v014.yaml
python -m eic.cli audit-claims --config experiment_config_v014.yaml
~~~

The first commands may fail with a structured blocker report. A failure must not trigger a model substitution, mock training, invented score, or a switch to a toy environment presented as the requested benchmark. The command name alone is not proof of implementation: supply its source, tests, and recorded exit status.

Do not install unpinned newest packages in the final experiment. Resolve working versions in a pilot, save a lock file or image digest, and run subsequent experiments from that resolution.

## 5. Runtime and model gates

Proceed in this order:

1. Establish the actual assigned training host, storage quota, accelerator type and memory, driver/runtime compatibility, scheduler if any, and permitted execution route.
2. Verify the exact model is accessible, pin its revision, and load its actual configuration before choosing a model class or adapter target names. A multimodal checkpoint must not be assumed to expose the same training API as an older text-only model.
3. Build a small text-only tool conversation with the official template. Inspect token IDs and response boundaries. Fix the model's thinking/output format and tool-call parser across all arms.
4. Freeze student-visible history, truncation, memory, admissible tools, retry count, context size, decoding, and stop conditions. These belong to the deployment policy class.
5. Perform one forward pass, one backward pass, one optimizer step, checkpoint save, process restart, restore, and one matched continuation step.
6. Measure actual peak device memory, tokens/second, optimizer-step time, checkpoint size, restore time, and evaluation throughput.
7. Forecast every branch experiment and full training wave from these measured quantities before launching the wave.

Default pilot recipe to investigate, not a theorem-prescribed optimum: text-only adapter fine-tuning, bf16 when supported, AdamW, learning-rate grid \(\{10^{-5},3\cdot10^{-5},10^{-4}\}\), and rank \(\{8,16\}\). Resolve adapter target modules from the actual checkpoint. Choose the main recipe using pilot data only and apply it consistently. Preserve a held-out recipe with a different learning rate and either optimizer or adapter rank for later transfer tests. Quantization is a separate factor and must be frozen if used; do not silently change it to fit memory.

A minimal smoke test must show nonzero trainable parameter updates, finite losses, correctly masked supervised tokens, and a saved checkpoint that changes the action distribution on an allowed toy training batch. This is evidence of a working update path, not evidence of agent improvement.

## 6. Environment selection and restoration semantics

The first controlled mechanism environment may be constructed locally, but at least two external agent environments are required for the final empirical generality claim.

Planned external domains:

| Domain | Purpose | Admission condition |
|---|---|---|
| ALFWorld | Language-guided multi-step household interaction | Pinned official implementation/data and verified deterministic restart/replay or genuine snapshot support |
| AppWorld | Stateful tool/API interaction and recovery | Pinned official implementation/data, automatic evaluator, and validated state restoration |

Official sources: https://github.com/alfworld/alfworld and https://github.com/StonyBrookNLP/appworld . The AppWorld README documents database checkpoint/restore operations. Database restoration does not restore a neural learner or every process state automatically. Verify each required component.

For ALFWorld, use only restart-and-action-prefix replay if it reproduces the entire relevant observation, evaluator, inventory, task-progress, and random-generator state. Match the full state signature at the branch point. Do not assume reset means arbitrary branch restoration.

Training-only resets must be available to every same-access baseline. Deployment uses one frozen reset policy for all methods, normally only episode initialization. A training branch should not give the proposed method extra deployment retries. Report training and deployment reset counts separately.

Every environment wrapper must implement or explicitly reject:

~~~python
reset(task_id: str, seed: int) -> Observation
step(action: Action) -> StepResult
evaluate() -> TaskScore
snapshot() -> EnvironmentStateReference
restore(state: EnvironmentStateReference) -> None
state_signature() -> str
allowed_observation() -> Observation
~~~

State signatures used for audit must not be exposed to a policy if they reveal hidden variables or task solutions. Store ground-truth rewards in evaluator records, not in student input. World model rollouts, if added later, retain source="model" and never count as observations of the real branch.

## 7. Freeze the data partitions and access rules

Use disjoint roles, with grouping performed before any episode or token split:

| Partition | Use |
|---|---|
| support | The \(K\) complete initial demonstrations per task family |
| pilot | Hyperparameters, candidate design, features, and runtime/power estimates |
| reference_fit | Real feedback allowed for reference learning and candidate construction |
| calibration | Frozen candidate and metric selection |
| metric_test | New training forks for validating proxy predictions |
| deployment | Final independent agent performance |
| heldout_recipe | New learner recipes, with no metric refit |
| heldout_family | Entire task templates/families, with no metric refit |
| heldout_domain | Second environment, reported separately |

The same task template, generated world seed, database base instance, paraphrase family, and all descendants of a trajectory must remain in a single appropriate group. A support trajectory and a paraphrase of it are not independent groups. Task IDs alone may fail to detect shared generators.

**Two different transfer questions must be separated.**

- Interface/metric transfer: freeze the metric and selection rule on source families, acquire only the declared \(K\) support demonstrations plus allowed feedback on a new target family, and specialize a new student. This is the primary cross-task test.
- Policy zero-shot transfer: deploy a student on a new family with no specialization. This is optional and is a stronger, different claim.

Declare whether \(K\) support counts are per semantic family or per benchmark task class, and publish the mapping. Do not silently treat a broad environment with many unrelated workflows as one family. Main initial value: \(K=32\); planned sensitivity values \(K\in\{4,8,16,32\}\). If the environment lacks enough independent demonstrations for a chosen value, mark that cell unavailable.

Allocate partitions by concrete identifiers, not percentages alone. If there are too few independent families, enumerate available families and narrow the cross-family claim. No aggregate “cross-task” result may hide that its metric was refit using target-family test outcomes.

## 8. Exact records and manifests

Store immutable JSONL records, with references to larger blobs. Hash canonical records after excluding volatile timestamps. Never store API secrets.

### 8.1 Experience-block record

~~~json
{
  "schema_version": "0.14",
  "block_id": "sha256:...",
  "parent_block_ids": [],
  "source_kind": "real_branch",
  "source_interface_id": "provider_recovery_v1",
  "candidate_set_hash": "sha256:...",
  "split": "reference_fit",
  "environment": "alfworld",
  "environment_revision": "resolved_commit",
  "data_revision": "resolved_revision",
  "task_family_id": "declared_semantic_family",
  "task_template_id": "template_id",
  "task_instance_id": "instance_id",
  "environment_seed": 104729,
  "teacher_request_id": "cached_request_reference",
  "teacher_model_requested": "gpt-6-luna",
  "teacher_model_returned": "provider_returned_identifier",
  "student_checkpoint_id": "checkpoint_reference",
  "student_visible_history_ref": "blob_reference",
  "teacher_visible_fields_hash": "sha256:...",
  "branch_prefix_ref": "blob_reference",
  "steps_ref": "blob_reference",
  "is_complete_episode_or_declared_block": true,
  "block_definition_id": "full_episode_v1",
  "independent_evidence_group_id": "world_draw_001",
  "teacher_label_group_id": "teacher_request_group",
  "real_environment_steps": 18,
  "training_resets": 2,
  "teacher_input_tokens": 1024,
  "teacher_output_tokens": 256,
  "teacher_calls": 1,
  "synthetic_descendant_count": 0,
  "observed_terminal_reward": 1.0,
  "reward_observed_or_predicted": "observed"
}
~~~

The numbers above illustrate the schema; they are not experiment measurements. Prompt tokens, generated reasoning if returned, output tokens, unsuccessful calls, and retries require separate accounting where the API exposes them. Only sampled text/tool outputs available under the black-box contract may be used as teacher targets. Do not assume teacher logits, hidden states, internal reasoning, or gradients.

### 8.2 Training snapshot record

Required: model revision and weight/adaptor checksum; optimizer states including moments; scheduler state; mixed-precision scaler; Python, NumPy, framework CPU and all-device RNG states; data-loader cursor and sampler RNG; replay-buffer contents/order; gradient-accumulation phase and residual gradients; environment state and RNG; agent transcript/memory; decoding state if an action is partially generated; software lock hash; current train/eval mode; running-normalization state; recipe ID; optimizer-step and token counters.

Prefer snapshots at clean optimizer boundaries. If an experiment needs mid-accumulation branching, serialize partial gradients. A parameter-only checkpoint is not a valid learner-state restore for this theory.

### 8.3 Fork record

Required: fork_id; parent checkpoint; candidate provider; sampled block; reference provider; continuation mode; suffix length measured in optimizer updates; complete continuation configuration; coupled random-seed map; reference/candidate output checkpoints; evaluation task IDs; returns; runtime; teacher/environment usage; all restore-test outcomes.

A candidate and reference fork share the parent snapshot. Pairing should couple exogenous randomness where valid, not force the two learners to receive a sequence that is impossible under their declared adaptive data source.

### 8.4 Deployment record

Required: method, run_id, training_seed, support_seed, family/template/instance, environment_seed, decoding_seed, full transcript reference, normalized reward, success, tool errors, invalid actions, timeout/termination reason, reset count, evaluator revision, final checkpoint hash, inference tokens, wall time.

Distinguish infrastructure failures from policy failures with a frozen rule. Never discard poor-return episodes as timeouts after inspecting method identity.

### 8.5 Frozen protocol record

Required: hashes of all partitions, candidate set, metrics, transforms, fitting procedures, hyperparameters, primary estimand, contrasts, family weighting, equivalence margins, multiplicity correction, seed IDs, stopping rule, and allowed analysis changes.

## 9. First executable stage: finite theory and real update path

Reproduce all exact v0.13 checks before relying on the new neural metrics. Add tests for any corrected assumptions in the independent v0.14 proof audit.

Required finite experiments:

- Perfect-teacher passive indistinguishability and the active branch construction.
- Equal student-policy/teacher values with unequal distillation values.
- Matched marginal/mean information with different joint consequences.
- Positive sufficiency examples as well as negative low-order examples.
- Moment-order and independent-block exposure curves, including matched lower-order feedback on parity constructions.
- Scalar versus multivariate task/learner response families of the same maximum degree, with exact and approximate response-space complexity measured separately.
- Nonlinear-update counterexamples.
- Local continuation identity with signed differences.
- Finite calibration with a correctly specified feature set, omitted-feature failures, and coverage failures.

Then train an actual small neural policy on the same finite environment to show where affine conclusions hold approximately or fail. Call this a controlled neural experiment. It cannot replace the specified Gemma student or the external benchmarks.

Avoid a test suite that merely restates implementation. The important tests compare independent formulas, exactly enumerable distributions, interrupted/restored continuations, and invariants violated by realistic bugs.

## 10. Reference learners, candidate providers, and baseline fairness

The reference learner \(A(E)\) must be an actual executable recipe trained using the declared real-information interface. It is not an oracle optimum and need not be better than every synthetic source. Preserve signed learning-value differences.

First establish whether real-information training improves the initial student on pilot tasks. If it does not, the method cannot infer practical improvement solely by matching it. Fix a demonstrable training, formatting, or access defect on pilot data; do not tune against the sealed evaluation set.

Candidate providers may include:

- Complete teacher demonstrations.
- Student-visited contexts with sampled teacher labels.
- Real branch outcomes plus teacher-labelled recovery continuations.
- Replay of intact blocks.
- A finite, frozen family of block-reweighting or batching schemes.
- A later optional generator, with clearly marked provenance.

A training message/block must include the relation between feedback and the action/history that produced it. Rebinding a consequence to the wrong action is an explicit corruption control, never a valid training target.

Core comparisons:

| Arm | Information and objective | Main purpose |
|---|---|---|
| Base | No post-training | Initial capability |
| Teacher SFT | \(K\) complete teacher trajectories | Passive distillation |
| SFT replay/exposure | Same independent blocks, extra training exposure | Separate repeated optimization from new evidence |
| On-policy sampled-output distillation | Teacher samples at student-visited visible histories | Strong same-access interactive baseline |
| DAgger-style aggregation | Iterative student rollouts, teacher labels, aggregated supervised training | Isolate the proposed selection mechanism from standard interaction |
| Cost-sensitive/student-continuation distillation | Same branch feedback; action weights/targets from measured deployment consequences | Isolate known consequence-aware imitation gains |
| Gradient/update matching | Same source bank; match projected or full tractable student updates | Test whether proposed metrics add beyond immediate update matching |
| Trajectory matching | Same source bank; match several student parameter/function updates | Strong learning-dynamics comparator |
| Proposed method | Same access, proposed information acquisition/selection rule | Test the new method |

DAgger-style and on-policy sampled-output arms can coincide under a particular implementation. If they are identical, consolidate them and state this, rather than count a duplicate as an independent strong baseline.

A black-box sampled-output adaptation is not a faithful reproduction of a method that requires teacher logits or differentiable teacher access. Label adaptations explicitly. For any chosen published comparator, inspect the original official code/paper, record necessary access, and document changes before execution.

Use two comparison panels:

1. **Same feedback panel:** all applicable methods receive the exact same acquired block bank. Differences test information transmission/utilization.
2. **End-to-end panel:** each method chooses permitted queries with identical access rights. Differences test acquisition plus transmission; report real information acquisition separately.

Cost matching is a diagnostic, not the central claim. Report training tokens, independent environment draws, teacher tokens/calls, branching rollouts, optimizer steps, GPU hours, inference costs, and deployment tool interactions. Do not conflate them in one “budget” number.

## 11. Measured learning consequences and neural proxies

For a complete learner state \(z\), candidate block law \(S\), and declared reference continuation of length \(L\), define

\[
C_{z,L}(S)
=\mathbb E[J_E(\pi_{\operatorname{Tail}^{E}_{L}(U(z,B_S))})],
\]

and compare it with the same expression for the real reference block source. The update and tail use the actual learner state, not an abstract gradient substitution.

### 11.1 Two continuation protocols

**Adaptive reference tail:** following the forked update, each branch performs the declared reference collection/training procedure conditional on its own current learner state. Couple only exogenous seeds that preserve both marginals. This is the direct operational counterpart of the general continuation test.

**Frozen replay tail:** both branches continue on a precommitted real block sequence. This is cheaper and meaningful for a frozen replay learner, but it is a different continuation operator from an adaptive online learner. Report it as such. Matching frozen-replay effects does not certify an adaptive tail.

All suffix interactions, teacher requests, training updates, checkpoint restores, and deployment evaluations count as method cost. The “one queried block” is not the full cost of its consequence estimate.

### 11.2 Proxy ladder

Evaluate the following separately:

| Metric | Estimation unit | Principal failure mode |
|---|---|---|
| Supervised loss, action agreement, teacher perplexity | Complete block | May be unrelated to deployment value |
| Mean projected gradient | Independent blocks at a fixed snapshot | Misses covariance, optimizer memory, curvature |
| Mean actual parameter/function update | Actual update from identical snapshot | Misses later nonlinear learning |
| Projected gradient/update covariance | Multiple independent blocks | Estimation noise and omitted directions |
| Joint block statistics | Complete blocks | Selected statistics may omit needed dependencies |
| Short-tail consequence discrepancy | Restored fork plus real deployment | Short tail can misrank long-tail effects |
| Multi-snapshot/multi-tail consequence discrepancy | Frozen panel of forks | Panel may miss actual training states |
| Full remaining-training consequence | Full restored reference continuation | Expensive; still restricted to sampled snapshots/tasks |

Select projections from independent pilot data or from a frozen random seed. Report their dimension and estimation uncertainty. A projection is a computational approximation, not a discovered sufficient statistic by definition. Do not call an empirical small residual a global \(\tau=0\) certificate.

Fit metric-to-learning-value models using pilot/calibration folds, then test them on separate training forks and complete independently trained students. Predicting a model's training loss from its own gradients is not the target. The target is the **signed real deployment-value difference after the declared remaining training**.

### 11.3 Concrete initial fork schedule

Use a measured-cost feasibility pilot with:

- 3 parent checkpoints: early, middle, late along a real reference training run.
- 4 candidate providers, including an intact reference control.
- 2 independent candidate-block draws per checkpoint/provider.
- 2 short continuation lengths: \(L\in\{1,8\}\) optimizer updates.
- 8 common evaluation task instances per finished fork.

This yields 48 candidate fork executions and, without shared-reference reuse, 48 reference executions. There are 96 short-tail learner executions and 768 deployment episodes. There are \(3\cdot4\cdot2\cdot(1+8)\cdot2=432\) tail optimizer updates plus 96 initial block updates if every reference fork is separately executed. Reuse of a reference control changes covariance and must be recorded; it does not create independent replications.

These counts are an engineering pilot, not a sufficient sample size for a statistical claim. Increase block draws, evaluation episodes, checkpoints, or independent parent runs according to measured uncertainty. Add \(L\in\{0,32,\text{full remaining horizon}\}\) only on a preselected audit subset after feasibility is established.

Choose that full-tail audit subset without viewing its full-tail outcomes. Do not pick only short-tail cases on which the proposed metric already looks correct.


### 11.4 CED feedback compilation, selection, and what is certified

The working method is **Consequence-Calibrated Experience Distillation (CED)**. This is a descriptive provisional name, not a claim that consequence-aware data selection is new. Its general unit is an executable feedback interface: which allowed experiment is queried, which observed information is retained, how it is encoded into valid student targets/experience, and how it is supplied across the learner's visited states. A packet-grouping rule is only one special case.

Implement four separable, initially small branches: (a) choose among allowed environment queries that can resolve student-relevant ambiguity; (b) select or compile feedback content using continuation-response tests; (c) expand calibration to training states actually visited, with explicitly measured coverage limits; and (d) reuse the interface/metric across declared task and learner families. Every compiled target must retain observed-versus-predicted provenance and valid action/history semantics. A target with high estimated utility is not evidence that the trained student reaches that utility.

For fidelity, choose a frozen executable interface whose observed response-test discrepancies from a declared real reference are small. For improvement, use signed gains only when the reference and actual deployment estimands are available. Report both objectives separately; small fidelity error does not imply a beneficial reference, while a provider that outperforms the reference should not be rejected solely for a large absolute discrepancy. Freeze which objective each experiment uses.

For a frozen candidate provider \(S_j\) and panel cell \(c\) containing a complete learner snapshot, task distribution, learner recipe and reference-tail specification, measure the signed advantage

\[
A_{j,c}=C_c(S_j)-C_c(S_{\mathrm{ref}}).
\]

For the signed-improvement branch, choose a candidate by the largest conservative lower confidence bound on \(\min_{c\in\mathcal C} A_{j,c}\), subject to the declared tolerances and feasibility constraints. Report every panel cell as well as the minimum. If the method uses an average rather than a minimum, change the declared objective before calibration; an average gain does not certify all cells. The fidelity branch instead minimizes declared response-test discrepancies with its own residual/coverage bookkeeping; these are different objectives, not interchangeable names for the same score.

Every observation is an actual update and continuation followed by real deployment. Cheap predictors can propose candidates or allocate pilot effort. They must not be substituted for real outcomes in a claimed empirical lower confidence bound unless their additional error is separately established.

Freeze the finite candidate family and panel before confirmatory calibration. Allocate failure probability across all candidate/cell comparisons, including separately estimated references. Shared-reference reuse is allowed but its dependence must be retained in analysis; it does not multiply the number of independent reference samples.

A finite-panel comparison certifies only that panel and sampling process. It provides no automatic cross-task or cross-optimizer guarantee. Held-out families and recipes require an independent transfer evaluation.

For a certificate about a downstream learner trained with a selected provider, selection records and fresh downstream training draws must be distinguished. If the provider replays the calibration bank, condition on that fixed bank and explicitly model reuse; do not invoke fresh-draw theory. Adaptive creation of new providers requires a fresh calibration set or an applicable uniform/adaptive analysis.

Recent nearby work includes post-update reference-loss meta-weighting, student-rollout consequence scoring, and training-data optimization for post-training environment return. Therefore, novelty must be evaluated against the complete learner/task-relative information account and its distinctive predictions, with each method branch compared to its closest prior mechanism. Calibrated actual-update response tests and controlled packet coupling are useful experiments, but a gain from one data-selection heuristic alone does not establish the theoretical contribution.

**Adaptive audit boundary, now proof-reviewed.** Section 11, Theorem 20 of Environment_Information_Theory_EN.md gives a finite-condition result for on-demand auditing at the actual current complete learner state. At each audit, freeze the candidate kernels, replica count and conditional failure allocation before sampling; generate independent paired **full reference-tail** experiments; leave the actual learner/environment unchanged; select using the lower confidence bounds with reference fallback; and then apply a **fresh** selected update rather than deploying an audit-trained replica. The reference recipe must ignore auxiliary audit records. The cumulative allocations satisfy \(\sum_t\delta_t\le\delta\).

The reviewed conclusions are deliberately distinct:

- With probability at least \(1-\delta\), the **predictable cumulative drift** is at least the sum of selected lower bounds.
- Unconditionally over audits and actual training, expected terminal gain relative to the reference is at least \(-\delta\), and also at least \(\mathbb E\sum_t B_t-2\delta\).
- Neither statement is a high-probability guarantee on a realized terminal return or a terminal expectation conditioned on all audits passing.

The theorem status is **reviewed_finite_conditions_only**. Neural certification remains disabled until exact restoration, correct full-tail marginals, independent replicas, fresh actual updates, and conditional confidence bookkeeping all pass operational validation. The practical short-tail/predictor branch remains empirical unless its uniform bias allowance is separately justified. A finite snapshot panel is not evidence of coverage of future neural states.

The distribution-free radius is \(e_t=\sqrt{2\log(2M_t/\delta_t)/n_t}\), where \(n_t\) counts independent paired training replicas **per candidate**. With \(M=4\), a single-audit allocation \(\delta=.05\), and desired \(e\le.1\), \(n=1016\) is required for each candidate: 4,064 paired experiments or 8,128 full-tail branch executions without reuse. Splitting the confidence budget across multiple rounds increases this per-round requirement. More deployment episodes of the same trained replica do not create independent update/tail draws. The small pilot above tests feasibility and predictivity; it cannot support a nontrivial gate at this worst-case radius.

Implement the adaptive branch first in finite gold systems, where exact response values and independent replicas can be checked. In neural work it is optional and expensive. Never reuse a fixed-sample Hoeffding radius after outcome-dependent within-audit stopping; use a justified sequential method or a frozen sample count.

## 12. Test whether the metric is useful rather than merely correlated

A valid empirical metric should improve decisions on held-out data.

Required analyses:

1. **Ranking:** Spearman/Kendall association between predicted signed learning change and observed signed change; uncertainty clustered by parent training run and task family.
2. **Magnitude:** Held-out MAE and \(R^2\), with a baseline predicting the training-family mean. A good ranking alone does not establish calibrated bounds.
3. **Sign:** Whether predicted harmful/helpful changes have the declared error rate; report abstentions and the full confusion matrix.
4. **Selection regret:** Select a provider without its final evaluation outcomes, then compare its actual value with that of the best evaluated provider. Include equal allocation and random selection controls.
5. **Incremental value:** Compare a model using only sample count, source identity, immediate loss and mean update against one additionally using the proposed consequence/joint-information features.
6. **Transfer:** Repeat without refitting on held-out task families and held-out learner recipes.
7. **Sensitivity:** Detect deliberately corrupted dependency structure and an intentionally harmful provider.
8. **Reliability:** Independent repeated estimates at the same cell, paired restore checks, and error bars that include block, suffix and deployment randomness.

Negative controls:

- A constant metric.
- Random metric values with the same marginal scale.
- Permuted consequence labels within permissible grouping.
- An arbitrary checkpoint norm or training-step counter.
- A metric trained with task identity alone.
- A feature with sign reversed.
- A teacher-loss metric and immediate action-agreement metric.
- A no-op update paired with itself.
- Identical source laws presented under different provider names.

Permutation must preserve grouped dependence. Permuting individual tokens across tasks is not a valid null test of a task-level association. An index or provider-name feature can expose spurious source leakage even when aggregate correlation looks impressive.

The proposed metric passes the pilot gate only if it improves held-out provider selection over the predeclared simple alternatives. A broad confidence interval is inconclusive; it is not evidence that the metric works or that two providers are equivalent.

## 13. Causal mechanism interventions

Use interventions that independently manipulate information, while explicitly measuring collateral changes.

### 13.1 Independent evidence versus exposure

Compare (a) more independent real blocks, (b) repeated passes over the same blocks, and (c) synthetic descendants of those blocks. Match training exposure in one panel and plot independent evidence count in another. The theorem's \(n\) is the count of independent blocks drawn under its distributional assumptions, not epochs or minibatches.

Do not impose the affine \(n^{1-r}\) exponent on a neural training curve. Treat it as a controlled-task prediction and an exploratory neural hypothesis with a frozen fitting range.

### 13.2 Matched marginal information, changed joint information

Construct whole-block versus regrouped-block sources from the same frozen record bank. For each intervention, verify which marginals, token counts, label frequencies, lengths and mean updates remain matched, using prespecified equivalence tolerances. Preserve valid action-feedback semantics unless the intervention is explicitly labelled semantic corruption.

Two different controls are needed:

- **Batch organization control:** intact legal trajectories remain intact, but correlations among their placement into updates change.
- **Semantic dependency corruption control:** deliberately mismatched feedback/history pairs test sensitivity to lost information but also change target validity.

Do not interpret the latter as a pure covariance manipulation. In neural training, reordering may also change curriculum, gradients, optimizer state and context packing; measure these differences.

### 13.3 New decision information

Create training queries at student-visited recovery states and compare against passive demonstration replay. Keep deployment observations unchanged. Compare against same-query on-policy/cost-sensitive baselines. Improvement caused solely by access to extra hidden state belongs to a changed policy class, not an information-transfer success.

### 13.4 Short-tail failure and optimizer dependence

Hold the parent weights fixed while varying valid serialized optimizer state, or compare held-out optimizers. Assess whether an immediate-update metric misranks long-tail effects and whether the consequence metric corrects it. Never combine optimizer states that could not arise in the declared learner unless clearly marked as an adversarial stress test.


### 13.5 Optional coupling-branch gate: fixed-marginal coupling

Before expanding the coupling-specific experiment, ask whether **the joint organization of identical valid trajectory units changes useful learning beyond known reweighting and gradient-diversity methods**. This is one falsifiable branch of the general framework. Its failure stops or narrows the coupling branch only; query-selection, feedback-content, coverage and learner/task-adaptation branches remain eligible for their own tests.

Do not splice histories, swap action-feedback labels, invent teacher reasoning, or change individual trajectory content. The unit is an intact executable teacher or recovery trajectory with a fixed tokenization, supervision mask and per-token weight vector.

A concrete balanced design uses \(M\) immutable units \(e_1,\ldots,e_M\). For a two-slot packet provider specified by a permutation \(\sigma\), its complete audit cycle is

\[
(e_1,e_{\sigma(1)}),\ldots,(e_M,e_{\sigma(M)}).
\]

Randomize the order of the first-slot indices using a precommitted seed. Each unit occurs exactly once in each slot and twice overall. Compare multiple frozen permutations, including random grouping, measured-consequence grouping, gradient-diversity grouping, and variance-reducing/antithetic grouping. Use derangements if self-pairs would introduce an avoidable difference, and impose the same restriction on every provider.

Across a full cycle, both slot marginals and unit counts are exactly identical. If providers instead draw packets independently with replacement, only distributional marginals are equal; measure realized count differences and do not describe them as exact count matching. If the experiment starts from a snapshot and processes a full balanced cycle, its message for theory purposes may be the entire cycle. Its inside-cycle dependence must not be passed off as independent packet draws.

Required invariant ledger:

| Quantity | Fixed in the strict coupling experiment |
|---|---|
| Complete valid trajectory units | Same immutable multiset |
| Teacher outputs and feedback | Exactly the same content |
| Slot marginals | Exactly once per unit per slot in each complete cycle |
| Supervision masks | Same token-level mask per unit |
| Per-token weights | Same stored vector per unit; no implicit provider-specific normalization |
| Total supervised tokens | Same over the full cycle |
| Number of optimizer updates | Same |
| Initial parameters and optimizer state | Identical full snapshot |
| Learning rate schedule | Same at corresponding update index |
| Deployment harness and task draws | Same permissions and paired exogenous seeds |
| Checkpoint/evaluation selection | Fixed before final results |
| Variable | Packet coupling/grouping and a separately declared order factor |

Declare whether the coupling objective also optimizes packet order. If it does, include a factorial comparison of grouping fixed/order random, grouping random/order optimized, and both optimized. Otherwise, do not call an order effect evidence of a grouping-specific mechanism.

The exact update semantics must be published:

| Update case | Loss/accumulation semantics | Predicted interpretation |
|---|---|---|
| Frozen additive full-batch SGD null | Compute all unit gradients at one identical \(\theta\), sum with fixed weights, apply one identical update | Grouping cannot change the result except numerical error |
| Sequential mini-batch SGD | One update per packet; later gradients evaluated at changed \(\theta\); fixed global unit weights | Grouping/order may change the result through update nonlinearity and temporal interaction |
| AdamW packet updates | One optimizer update per packet, with restored moment state and fixed schedule | Grouping can change gradient moments and future updates |
| Gradient accumulation into a fixed super-batch | No optimizer step between microbatches, fixed loss normalization and deterministic model state | Repartitioning the same super-batch is a null when gradients are additive and evaluated at the same state |
| Variable-length token-mean batches | Divide by each packet's actual token count | Changes implicit unit weights; exclude from strict coupling or correct normalization |
| Context packing | Units may attend across boundaries unless explicitly masked | Cross-unit attention changes inputs; forbid in strict coupling |

A meaningful null uses deterministic dropout settings or a frozen token-indexed randomness mapping. Dropout RNG sequence changes caused by reordering are a separate mechanism, not floating-point noise.

Primary strict-coupling objective: use per-unit masked losses with a fixed precomputed normalization and take the same fixed-weight sum within each packet. If per-token supervision is desired, store global token weights and do not renormalize them by provider-specific packet length. Log effective weights after gradient accumulation.

The coupling-specific continuation gate has three requirements:

1. Invariance audit passes and the additive-SGD null is numerically null.
2. A real nonlinear/sequential optimizer exhibits a reproducible coupling effect at fixed information marginals.
3. CED chooses useful couplings on held-out snapshots/families better than random grouping, gradient diversity, antithetic/variance-minimizing grouping and same-data meta-reweighting.

A benefit only from changing token weights, fresh feedback, optimization steps or invalid trace construction fails this branch gate. A genuine coupling effect that CED cannot predict fails the coupling-selection part. A true null is informative: report it and remove unsupported coupling claims. It does not refute learner-relative information requirements or stop the other method branches. Do not search many test sets merely to obtain a positive coupling result.

### 13.6 Direct contemporary controls and access tiers

The independent novelty audit identifies particularly close mechanisms in MetaOPD, FutureBridge-OPD, and HaDES/Behaviour Distillation. Read the exact papers and official implementations listed in that audit before choosing a faithful baseline. Include their accessible objectives as follows:

- Post-update reference-loss meta-weighting on the same complete-unit bank.
- Frozen-student rollout consequence scoring.
- Same-data optimization against post-training task return, where computationally feasible.
- LESS/GRACE-style student gradient relevance or diversity scores.
- Gradient matching and multi-update trajectory matching.
- LOLS-style student-continuation or cost-sensitive imitation.

A teacher-logprob-dependent method cannot be faithfully reproduced with sampled teacher outputs alone. Keep the main black-box track output-only. Label an output-only surrogate as an adaptation, and explain which objective it estimates and which original terms are unavailable. A privileged-scoring track is separate and must not introduce an unapproved substitute teacher or silently change access rights.

The main question is whether the additional CED measurements improve real query, feedback-content, provider or adaptation choices enough to justify their cost and support the proposed information account. Coupling choices are one special case. Beating weak passive SFT alone is insufficient evidence for the method.


### 13.7 Joint theory-prediction panel

Evaluate the information account as a chain rather than reducing it to one coupling effect. Use the following panel. Exact finite checks come first; neural versions are clearly marked empirical extensions.

| Prediction | Controlled comparison | Required observation | Boundary of the claim |
|---|---|---|---|
| Identification depends on allowed experiments | Passive perfect-teacher traces versus a permitted informative branch, with unchanged student deployment observations | Same passive law, different needed decisions; branch feedback resolves the controlled ambiguity | A constructed access lower bound does not imply every real task needs branching |
| Equal static values can hide unequal learning value | Two feedback sources equal on the declared student-policy values and teacher value | Actual trained students have different real returns under the stated construction | This is not a counterexample to every stronger Bellman/value-equivalence definition |
| Lower-order feedback can be insufficient | Even/odd parity feedback laws that match all selected proper-subset or lower-order tests | Full interaction response separates the sources despite those matches | Exact in the controlled declared learner/task class; not an unrestricted neural theorem |
| Degree alone does not determine approximation complexity | Scalar versus multivariate response families with the same maximum degree, plus a structured low-complexity multivariate control | Different exact response-space dimensions or different uniform approximation error at the same feature budget | Empirical low-rank plots are not global sufficient-statistic certificates |
| Required information changes with the update rule | Affine, polynomial/nonlinear and actual neural recipes from matched starting information | Feedback/tests sufficient for one declared learner can fail for another | Do not transplant the affine moment theorem into AdamW |
| Fidelity and improvement are different | A good reference and an ineffective reference, each with high-fidelity candidate information | Small reference gap alone need not improve the initial student; improvement must be measured separately | Negative signed reference gaps are permitted |
| Local response errors compose through training | Exactly enumerable learner process, then restored neural snapshots along real training | Exact telescoping identity in the finite case; quantified short-tail/coverage residuals in neural use | Finite-state identity does not make unobserved neural tails or states free |
| Approximation and measurement stability jointly control information error | Exact finite response approximations with different coefficient magnitudes and feature-error tolerances | Reproduce the primal/dual operational discrepancy, including amplification by unstable coefficients | Small approximation residual alone does not certify accurate measured responses |
| Reusable information is relative to a family | Freeze tests/interface on source tasks and recipes; evaluate new declared families/recipes | Report preserved and failed transfer axes without refitting on their outcomes | Cross-task utility remains a separate empirical test outside proved scope |

For the same-degree complexity experiment, use an enumerable message grid and a finite declared response class. Include the constant function when computing exact linear span dimension. For unrestricted degree-\(d\) polynomial tests in one scalar variable, the nonconstant monomial count is \(d\); for \(k\) coordinates it is \(\binom{k+d}{d}-1\), subject to domain independence and the actual included tests. A structured multivariate class can occupy a much smaller span, so ambient monomial count is an upper bound, not always the needed dimension.

Measure exact rank with rational/high-precision checks where tractable. For approximate compression, report maximum absolute response reconstruction error over the entire declared finite grid and held-out responses. An SVD/Frobenius error may be a useful diagnostic, but cannot be renamed a uniform error bound. Freeze the norm, feature budget, domain and threshold before comparison.

The finite gold suite can additionally reproduce Section 10, Theorem 18's noisy-feature dual: worst-case response discrepancy equals the infimum of twice the uniform approximation residual plus the feature-tolerance-weighted coefficient norm. Solve the finite primal and dual independently and compare. A simple stability control uses a binary response \(g\), a scaled feature \(\phi=\alpha g\), and feature error tolerance \(\epsilon\): exact representation has zero residual but coefficient \(1/\alpha\), with worst-case discrepancy \(\min\{1,\epsilon/\alpha\}\). This checks why an accurate but ill-scaled response representation may be difficult to estimate. No implementation of this optimization over all neural responses is required or assumed.

Use small parity dimensions and message grids that can be fully enumerated; expand only when a concrete unresolved prediction requires it. Neural experiments need not reproduce every finite construction with the 12B model. Their purpose is to test whether a practical response metric predicts useful real training choices, and whether those choices transfer beyond the tasks and recipes that fitted the metric.

## 14. Main neural training matrix

Avoid running every possible ablation before the method or metric has passed its pilot.

**Wave 1: implementability and mechanism.** First complete the small joint theory-prediction panel in controlled systems. Then use one real domain, \(K=32\), and 3 independent pilot training seeds. Run base evaluation, teacher SFT, real-feedback reference training, and one feasible CED query/content/provider branch. Execute a small fork/proxy panel. Add coupling or adaptive gold-audit substudies only if resources and their prerequisites permit. Their failure does not block the other branches. Pilot results are not final statistical evidence.

**Wave 2: primary within-domain comparison.** Freeze the recipe. Train 5 independent seeds for each of 6 predeclared main arms: teacher SFT, on-policy/DAgger sampled-output baseline, cost-sensitive/student-continuation baseline, strongest gradient-or-trajectory baseline chosen using pilot only, equal-access uniform-block control, and the proposed method. Evaluate the base without training. This is 30 training runs per environment. Five is a starting allocation, not a universal guarantee of statistical power.

**Wave 3: external-domain confirmation.** Repeat the frozen selection/metric procedure in the second domain with declared target support and feedback, at least the same 6 main arms if resources permit. Fifty or hundreds of evaluation episodes from one model do not replace independent training seeds. If resources force fewer arms, preregister the reduced comparison and narrow the claim.

**Wave 4: targeted scaling and transfer.** Run the proposed method and the strongest same-access baseline over \(K\in\{4,8,16,32\}\), a frozen exposure axis, and held-out learner recipes. No reuse of sealed deployment outcomes for method design. Additional pilot adaptation in the second domain must be reported as adaptation and tested on a fresh sealed split.

Record all planned and executed cells, including incomplete and failed runs. If power analysis requires more than the initial 5 seeds, set a fixed new total from pilot variance before accessing final outcomes, or use a prespecified sequential-inference method. Do not add seeds only when a p-value is close to a threshold.

## 15. Statistical estimands, uncertainty, and decision rules

### 15.1 Primary utility contrast

Let \(Y_{m,s,f,i}\) be real deployment reward for method \(m\), independent training seed \(s\), task family \(f\), and test instance \(i\). Define a predeclared family-balanced utility, rather than allowing a large easy family to dominate:

\[
\widehat J_m = \frac{1}{S}\sum_s\frac{1}{F}\sum_f
       \frac{1}{n_f}\sum_i Y_{m,s,f,i}.
\]

Use paired support sets, initialization seeds and evaluation tasks across methods where possible. Analyze paired method differences. Preserve all seed-by-family outcomes.

If inference targets new training seeds and new task instances but the observed families are fixed, resample training seeds and instances within fixed families with a dependence-aware crossed procedure. If inference targets new families, resample families as well and acknowledge the limited number of families. Specify the estimand before choosing a bootstrap. With few seeds/families, report all observations and sensitivity to the inference method.

McNemar tests on episodes from one fixed student answer a conditional deployment question. They do not establish robustness to retraining. Token-level standard errors are invalid for the main utility contrast.

### 15.2 Metrics and multiple comparisons

The primary method contrast is proposed versus the strongest predeclared same-access baseline. Declare a small set of secondary contrasts and use Holm correction or a predeclared familywise procedure. Report effect sizes and intervals, not only p-values.

For metrics, report nested held-out prediction: feature/regularization selection occurs without the outer test fold. Leave-one-family-out fitting is appropriate only when the outer family was never used to choose features, horizons or labels. If it was, use a final fresh family split.

### 15.3 Planned practical margins

Set practical margins before final testing. Starting planning values for normalized success are:

- A noninferiority tolerance of 0.02 for preserving a real-reference student's average success.
- A desired improvement of 0.03 over the strongest same-access baseline.
- A metric selection-regret target chosen from pilot uncertainty and task value, not assumed universal.

These are project design choices, not consequences of the theory. Freeze or revise them on pilot data with a written rationale. A claim of noninferiority requires a one-sided interval above the negative margin. “No statistically significant difference” is not evidence of preservation.

Do not automatically claim a formal neural certificate from empirical success. The finite theorem's feature residual, covered state set, and exit probability require actual evidence. If they are unavailable, call the result an empirical risk estimate or diagnostic.

## 16. Train-state restore audit

Before any consequence claim, run the following audit at early and late checkpoints:

1. Continue a reference run for a fixed block sequence and evaluation set.
2. Restart from its saved full snapshot in a fresh process and repeat.
3. Compare parameters, optimizer state, random draws, losses, actions and deployment results.
4. Repeat the no-op-versus-no-op paired fork.
5. Change only one candidate training block and verify that the shared starting state is otherwise identical.

For deterministic settings, require exact state hashes where supported. For known nondeterministic kernels, record a declared numeric tolerance and quantify restart noise across repeats. The treatment signal must be compared with that noise. Do not choose a tolerance after inspecting a desired result.

Never run a fork in the main training process and attempt to “undo” it by restoring only model weights. Isolate forks or restore every complete state component.

## 17. Stop/continue gates

| Gate | Continue only when | Otherwise |
|---|---|---|
| G0: resources | Exact model, real runtime, environment data and execution route are established | Report concrete blocker; do not substitute |
| G1: math | Independent exact checks and corrected theorem implementation agree | Fix discrepancy before neural interpretation |
| G2: learner | Actual parameter update and full restore pass | Repair training path |
| G3: harness | Correct observation permissions, parser, replay/snapshot and evaluator pass | Repair or narrow access claim |
| G4: useful reference | Real-information training has a credible beneficial pilot effect | Diagnose recipe, target format, teacher quality or information access |
| G5: practical metric branch | Held-out response prediction improves its declared query/content/provider decision over simple controls | Drop/modify that proxy on pilot; other valid branches or an affordable gold-measurement branch may continue |
| G6: main method | Same-access effect is measured with prespecified uncertainty | Report negative/inconclusive result without changing test set |
| G7: transfer | Frozen metric/selection survives new families and recipes at stated margin | Narrow reusable-information claim |
| G8: paper claim | Every headline maps to a theorem with assumptions or a completed experiment | Remove unsupported claim |
| C: optional coupling branch | Invariants, additive-SGD null and useful held-out coupling selection pass | Stop or narrow this branch only; continue independently justified query/content/coverage/adaptation work |
| A: adaptive gold-audit branch | Theorem 20's reviewed finite conditions are operationally satisfied: exact restore, full-tail marginals, independent replicas, fresh actual updates, and conditional failure allocation | Keep neural certification disabled; report empirical gold measurements and explicit failed assumptions |

A failure at G4 does not prove the environment-information theory false; it may mean that this reference recipe has no gain to preserve. It blocks a gain-by-reference-preservation claim for that recipe, while an independently justified information-acquisition or feedback-compilation branch can still be tested. A failure at G5 directly weakens that practical surrogate. A method success with metric failure may be useful engineering, but does not substantiate that proposed metric mechanism. Branch failures must be reported separately from failures of shared runtime, mathematical or access prerequisites.

## 18. What the coding agent must return

Return an evidence package containing:

1. Runtime and model verification, with concrete blockers if any.
2. Source repository commit and dependency lock/image digest.
3. Frozen task/data/access and experiment manifests.
4. Baseline fidelity and adaptation notes.
5. Complete learner-state restore audit.
6. Actual trained checkpoint IDs, trainable parameter counts, token/step counts, and training logs.
7. Episode-level deployment records and evaluator outputs.
8. Metric prediction/selection results on held-out forks, families and recipes.
9. Theory-to-experiment claim ledger, including assumptions that failed.
10. Finite proof checks, actual statistical estimates, confidence procedures and effect sizes.
11. Full cost ledger including all branches and reference tails.
12. Negative results, aborted runs, and supported-scope statement.
13. A reproducibility README that regenerates tables from raw records without API requests or retraining.
14. A concise recommendation: continue, revise the metric/method, narrow the claim, or stop.

Do not deliver only training loss curves, an example successful trajectory, a simulator-only score, or a prose assertion of novelty. An independent audit and empirical tests can strengthen confidence in mathematical relevance; empirical training cannot prove an unrestricted theorem or establish an ICML award outcome.

## 19. Independent implementability assessment

The theoretical abstraction is implementable in finite systems because it exposes actual learning-state transitions and bounded terminal utility. Its general neural version has four real difficulties:

- **State fidelity:** optimizer memory, stochastic data order, agent history and environment state all affect the continuation value.
- **Continuation cost:** a faithful consequence query can require many training updates and real deployment episodes.
- **Support and adaptivity:** a finite snapshot bank does not cover all states visited by an adaptive neural learner.
- **Transfer:** a metric fitted to one recipe can learn incidental correlations rather than reusable information requirements.

The proposed experimental design makes those difficulties observable rather than assuming them away. Its decisive evidence would be a frozen, affordable metric that predicts and improves provider choice on held-out families and recipes, and a few-shot distillation method that outperforms strong same-access baselines on real deployment. That evidence has not yet been collected.

## 20. Primary implementation sources checked

- Google, Gemma 4 12B model repository: https://huggingface.co/google/gemma-4-12B-it
- Google, Gemma 4 model documentation: https://ai.google.dev/gemma/docs/core/model_card_4
- OpenAI, GPT-6 Luna model page: https://developers.openai.com/api/docs/models/gpt-6-luna
- Official ALFWorld repository: https://github.com/alfworld/alfworld
- Official AppWorld README, including state checkpointing: https://github.com/StonyBrookNLP/appworld/blob/main/README.md

These sources establish available public interfaces and identifiers. All concrete software versions, model revisions, data licenses/access and environment semantics must still be resolved and recorded in the actual execution runtime.

