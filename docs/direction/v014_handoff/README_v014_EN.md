# Environment information theory and few-shot agent distillation — v0.14

This package provides a standalone English theory, independent model-agent proof and novelty audits, a derived method specification, and an implementation plan that requires actual neural-agent training and cross-task evaluation.

The theory asks what information an environment must transmit **relative to the student, learning procedure, and task family**. World models and packet coupling remain possible implementations or experiments. Neither defines the theory's scope.

## Main intellectual structure

Decision-relevant identification → continuation responses of actual learning → exact and approximate information sufficiency → composition-dependent complexity → finite measurement and transfer conditions → acquisition and distillation methods.

The same continuation-response object links these stages. Classical mathematical tools are explicitly attributed. The package does not claim that every lemma is original, that the full historical novelty question is settled, or that a major conference award level has been achieved.

## Files to read

| File | Role |
|---|---|
| Environment_Information_Theory_EN.md | Complete English framework, 21 numbered propositions/lemmas/theorems, proofs, limitations, and contribution ledger |
| Independent_Proof_Audit_EN.md | Independent model-agent review of 40 substantive statements/extensions plus final integration review |
| Independent_Novelty_Audit_EN.md | Component and whole-framework novelty assessment; 31-entry primary reference register and unresolved comparisons |
| CED_Method_EN.md | Consequence-Calibrated Experience Distillation: acquisition, compilation, coverage/transfer, optional coupling, and calibrated selection |
| Coding_Agent_Execution_Plan_EN.md | Detailed real-training implementation, state restoration, data splits, baseline access, metrics, statistical analysis, and stage gates |
| START_HERE_Coding_Agent_EN.md | Direct execution brief for the coding agent |
| experiment_config_v014.yaml | Research configuration; unresolved runtime fields are intentional blockers, not fabricated resources |
| runtime_audit_v014.json | Observed current execution prerequisites |
| claim_ledger_v014.json | Machine-readable statement/scope/evidence starting ledger |
| verify_environment_information_v014.py | Reproducible new finite checks |
| results_v014.json | Actual outputs from those checks |
| theory_checks_v014.png / .svg | Standalone scientific figure from the finite checks |
| requirements_validation_v014.txt | Pinned packages for finite validation, not a neural training environment |
| package_manifest_v014.json | Content hashes for the package and included baseline material |

The ZIP also includes the earlier Chinese v0.13 framework, first-stage protocol, verification script, and results under baseline_v013/. These supply the recorded earlier finite examples. The English v0.14 theory is independently readable and its explicit corrections take precedence.

## What this version adds

- A coherent operational characterization rather than a method-specific framing.
- Exact and noisy-feature approximation dualities, exposing both residual error and coefficient stability.
- A constant-gap multivariate parity witness, alongside scalar approximate compressibility.
- Exact finite-grid statistic counts and a broad-task approximate incompressibility barrier.
- An explicit distinction between a fixed finite-panel audit and adaptive auditing at actual learner states.
- Correct unconditional adaptive performance guarantees with statistical penalties.
- A method and experimental plan that can reject individual branches without prematurely abandoning the general theory.

The existing affine moment hierarchy, averaging bound, cumulant decay law, nonlinear update composition, and passive/active acquisition example are incorporated with their assumptions and proofs.

## What independent review changed

The review was performed by separate model agents with different assignments. It is not external human peer review or machine-checked formal verification.

Material corrections include the confidence range for the two-world testing lower bound; a common initial learning-state law; distinguishing a pointwise message response from a provider-averaged response; independent re-auditing after data-dependent provider construction; complete-state and fresh-update requirements; and the full-polytope assumption for the coupling characterization.

For adaptive selection, simultaneous confidence controls the true conditional drifts along visited states. It must not be rewritten as a conditional terminal guarantee given a future-dependent audit-success event. The final theory states the valid unconditional expected-value inequalities.

The novelty review treats integration as a possible source of originality. It also identifies nearby complete frameworks and methods; the statement “its components have precedents” is neither a proof of no originality nor evidence that the integration is new.

## Completed mathematical checks

All assertions in the new finite-check script passed in the recorded runtime.

| Check | Recorded outcome |
|---|---|
| Noisy-feature duality | Explicit example plus 36 independently formulated random LP primal/dual comparisons; maximum residual about \(8.9\times10^{-16}\) |
| Constant-gap parity | Orders 1–8: all lower-order joint moments agree and final learning values differ by 1 |
| Response dimension | Six finite tensor-grid rank checks match the derived statistic count |
| Scalar approximation | Chebyshev identities and truncation bounds checked for degrees 4, 8, 16, and 32 |
| Independent-block decay | 20 exact rational convolution cases match the predicted formula |
| Coupling and additive null | Additive response has zero width; quadratic response has nonzero width; frozen additive SGD regrouping is invariant |
| Adaptive auditing | Finite analytic expectations satisfy the qualified bounds; a separate example shows finite optimistic selection can have a strictly negative expected gain |
| Robust ambiguity | Independent primal and dual minimax programs agree |

These are finite mathematical checks. They are not real agent-training results or experimental proof of neural generalization.

To rerun after extracting the archive:

~~~bash
python verify_environment_information_v014.py --output results_recomputed.json --plot
~~~

Use requirements_validation_v014.txt in a suitable isolated Python environment if needed. The script needs NumPy and SciPy; plots additionally need Matplotlib. The included baseline script can be run separately. The neural command contract in the execution plan is an interface to implement, not a claim that an agent-training repository already exists.

## Real training status

No neural-agent training, paid teacher requests, model download, or external benchmark execution was performed in this workspace. A usable GPU runtime, neural training packages, an agent-training repository, model weights/access, and executable environment manifests were not established.

The coding-agent plan preserves the requested teacher and student, requires actual parameter updates and checkpoint evidence, and separates training replicas from evaluation episodes. It includes held-out task-family, recipe, and domain validation. Its first gate is to establish the assigned training runtime and verify the model and environment paths.

The small pilot is designed to measure feasibility, costs, and predictive signal. It is not large enough to supply the worst-case distribution-free full-tail certificate. Exact gold auditing is an optional costly control; practical neural proxies remain empirical unless their missing assumptions are independently established.

## Research claim still to earn

The next substantive contribution must show that this operational characterization leads to a nontrivial, measurable information requirement or method that remains useful beyond its exact finite examples. Strong possibilities are a structured neural/adaptive response approximation theorem, a tight acquisition-and-verification result for an important learner class, or a reproducible distillation effect with the theory's distinctive predicted dependencies.

The existing package supports pursuing that objective rigorously. It does not certify priority, cheap universal neural sufficiency, completed real-agent efficacy, or best-paper quality.
