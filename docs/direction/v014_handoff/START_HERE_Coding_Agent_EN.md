# Execution handoff to the coding agent

You are implementing and testing a research program, not manufacturing supporting results. Read this file, the complete English framework, both independent audits, the method specification, and the execution plan before writing the training system.

## Mission

Implement **Consequence-Calibrated Experience Distillation (CED)**, perform real student parameter training in executable agent environments, validate which neural measurements predict actual learning consequences, and test transfer across task families and learner recipes.

The primary contribution under investigation is a unified theory of the information an environment must transmit to a declared learner for a declared task family. The chain is:

1. Student execution limits determine attainable capability.
2. Allowed experiments determine which decision-relevant facts can be identified.
3. A message affects capability through the actual update and remaining learning.
4. Continuation-response functions therefore define sufficient information.
5. Their span/approximation width and update composition determine complexity.
6. Independently measured feature errors and responses determine what can be certified.
7. A derived method must improve the final real student and transfer under declared conditions.

Do not force the project into a world model or a single packet-coupling mechanism. Query acquisition, compilation/content, coverage, and learner/task adaptation are independent candidate method branches. Coupling is an optional controlled diagnostic.

## Read in this order

1. \`Environment_Information_Theory_EN.md\`
2. \`Independent_Proof_Audit_EN.md\`
3. \`Independent_Novelty_Audit_EN.md\`
4. \`CED_Method_EN.md\`
5. \`Coding_Agent_Execution_Plan_EN.md\`
6. \`experiment_config_v014.yaml\`
7. \`runtime_audit_v014.json\`
8. Finite verification scripts/results in this package

The English theory is standalone. The included v0.13 material records the earlier exact finite baseline; v0.14 resolves and extends it. Later explicit qualifications in v0.14 take precedence.

## Nonnegotiable scientific distinctions

- \(K\) counts initial complete trusted demonstrations per task family. Additional feedback remains permitted but separately counted.
- Teacher: \`gpt-6-luna\`, black-box sampled outputs. Student: \`google/gemma-4-12B-it\`. Establish actual access and pin versions. Preserve the project’s checkpoint/component provenance restriction.
- The final student acts alone. No test-time teacher or selection oracle is part of the claimed distilled policy.
- A simulated/toy arithmetic check is not neural training. A neural forward pass is not a parameter-training result. A lower training loss is not agent improvement.
- Affine-moment theorems do not automatically hold for Gemma/AdamW. Mark exact theorems, conditional certificates, and empirical neural extensions separately.
- Preserve complete learner state and environment state in forks. Restore optimizer and RNG state, not just weights.
- Apply real updates and full declared reference-learning tails for gold responses. A frozen student's action continuation measures something else.
- Regenerate adaptive tail feedback for each changed learner; couple exogenous randomness only while preserving correct marginals.
- Count independent training replicas separately from deployment episodes of one replica.
- Keep candidate discovery, calibration, metric testing, and sealed deployment disjoint by task/template/genealogy.
- Do not treat a finite checkpoint panel as coverage of a continuous future neural trajectory.
- Do not treat a same-degree scalar approximation example as universal compressibility, or parity hardness as universal neural difficulty.
- Do not report a teacher-logprob baseline as faithfully reproduced in an output-only teacher track.
- Do not tune to a desired positive result or infer ICML quality from theorem count.

## First executable work

The previous workspace audit found no established GPU training runtime, no neural training stack, no agent-training repository, and no verified model/API access. Do not assume these have appeared.

1. Inventory the **actually assigned** execution host, repo, accelerators, model access, environment manifests, and resource envelope. Do not probe unrelated accounts or scan for secrets.
2. If prerequisites remain unavailable, write a structured blocker report and complete the finite/local tasks that are executable. Do not silently substitute a tiny model or mock environment and call it the requested experiment.
3. On an available training host, load the exact student, execute and record one real backward/optimizer step, save/restart/restore, and reproduce a continuation step.
4. Pin ALFWorld and AppWorld or document a justified explicit benchmark change. Verify actual state restoration and evaluation semantics.
5. Profile update, checkpoint, restore, teacher, environment, and deployment costs.
6. Forecast the pilot from measured throughput before launching it.
7. Implement the command/module contract in the execution plan; those modules are specified interfaces, not already existing code.

No invented GPU allocation, elapsed training time, model load, benchmark run, or score is acceptable.

## Minimum empirical program

**Controlled theory phase.** Reproduce passive-teacher ambiguity, active repair, static-value/learning separation, joint-moment hierarchy, parity versus scalar approximation, nonlinear update amplification, independent-block versus replay behavior, feature LP duality, and fidelity-versus-gain reversal.

**Neural metric phase.** At multiple real learner snapshots, compare a finite provider menu with actual updates, multiple independent packet/tail replicas, and nested deployment evaluation. Measure held-out sign/ranking error, calibration, selection regret, harmful selection, and total cost. Compare against supervised loss, gradient metrics, GRACE/LESS-style scores, action-continuation costs, and direct update search.

**Real distillation phase.** Train independent student seeds from identical permitted initialization and support. Compare strong SFT/BC, on-policy correction, cost-sensitive/student-continuation supervision, accessible dataset-distillation controls, and the candidate CED branches. Evaluate the final student alone.

**Transfer phase.** Freeze the method before new task families, changed recipes, and a second domain. Distinguish method transfer with allowed adaptation from frozen-policy transfer.

The execution plan specifies manifests, masks, hierarchical inference, workload counts, baseline access, and stopping criteria.

## Certificate modes

Theorem 19 is a frozen finite-panel audit followed by fresh training under the declared law. Theorem 20 is an adaptive actual-state gold audit with full tails, fresh actual updates, and predictable confidence allocation.

For the adaptive theorem, the simultaneous high-probability statement controls predictable drift. Valid terminal guarantees are **unconditional expectations** with statistical penalties. Do not condition a terminal guarantee on the future-dependent “all audits passed” event.

With four nonreference candidates, failure budget 0.05, and target Hoeffding radius 0.1 at one state, the gold audit requires 1016 paired training replicas per candidate. The small pilot is not a distribution-free nondegradation certificate. Default neural proxy mode remains empirical until every certificate premise is established.

## Required final outputs

- Source repository, pinned dependencies, exact commands, and immutable run manifests.
- Checkpoints with proof of actual parameter updates and complete restoration tests.
- Complete teacher/environment/compute accounting, including unsuccessful candidates.
- Raw per-training-replica and per-deployment-episode records.
- Finite-theorem reproduction and explicit assumption checks.
- Held-out metric validation and selection-regret analysis.
- Real student-only success/failure results with paired, task/seed-aware uncertainty.
- Separate cross-family, cross-recipe, and cross-domain results.
- Baseline access-fidelity table and negative/failed experiments.
- A claim ledger mapping every paper statement to proof, measurement, or unresolved hypothesis.

Report what the evidence permits, including a null result. A failed coupling branch is not a reason to abandon the unified theory; a failed practical proxy is a reason to revise its scope, not to label it certified.
