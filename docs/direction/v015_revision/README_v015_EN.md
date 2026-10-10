# v0.15: Coverage erratum and optimizer/few-shot extension

Prepared 10 October 2026 UTC.

## Read in this order

1. `Environment_Information_Theory_EN.md`: the complete framework, now version **0.14.1**, with the missing Theorem 19 coverage premise corrected and a quantitative transport extension. This file is at the root of the new ZIP. Its separately saved main-document identity is retained.
2. `Optimizer_State_and_Few_Shot_Theory_v015_EN.md`: the new addendum, with complete proofs for seven propositions covering transport, Adam's immediate-parameter quotient, approximate sign sufficiency, optimizer memory, the role of K, optimal finite-model extra queries, and adaptive KL accounting.
3. `Proof_Review_v015_EN.md` and `Novelty_Update_v015_EN.md`: separate model-agent reviews. They are not external peer review or formal verification.
4. `Coding_Agent_Followup_v015_EN.md`: the detailed experimental handoff. The external C23 source code/raw data are required for a replication. Command names in that handoff are interfaces to implement, not claims that an existing CLI was executed.
5. `verify_v015_extensions.py` and `results_v015.json`: new finite regression checks executed in this session.

The original **v0.14 ZIP is unchanged** and is included under `historical/` in the new bundle. Its manifest describes the original v0.14 bytes, not the corrected main document. The external validation report supplied by the user concerned original main SHA-256 `9271ae051b6a31fc15907a0469354e46a8d043493f9cf353daca2809c58ac1e6`. Do not attribute the old audit's conclusions to a new unreviewed version, or run its manifest against a folder whose main document has been replaced by this erratum.

## Reproduce the new finite checks

Tested with Python 3.12.14 and NumPy 2.3.5. From this directory:

```bash
python -m pip install -r requirements_v015.txt
python verify_v015_extensions.py --out results_v015_rerun.json
```

The seed is fixed at 15019. Numerical results may differ slightly with floating-point libraries. The verifier raises on a failed condition; it does not silently relax failed assertions. Exact finite enumeration uses rational arithmetic. The original coverage inequality is deliberately required to fail on its counterexample; the regression then verifies the corrected alternatives.

## Executed results

All **13 new groups passed**. These are primary-agent-authored regression checks, separate from the user's independent C1–C23 suite. They do not independently reproduce that suite or the controlled neural runs.

| Group | Test | Result |
|---|---|---|
| T01 | Original coverage counterexample and repairs | Actual gap 1; old bound 0.029604; transport/exit repairs valid |
| T02 | Local transport and optional feature drift | 1,200 random finite instances satisfy the inequalities |
| T03 | Cold Adam identity and positive-epsilon inverse | 480 cases; update error at most 2.23e-16; inverse illustrates poor conditioning |
| T04 | Exact finite quotient dimensions | 25 gradient atoms give 9 sign patterns; dimensions 8 versus 24 |
| T05 | Same marginal signs, different joint signs | Mean updates match; bounded joint utility gap 1 |
| T06 | Matched orders 0, 1, 2 with different normalized updates | SGD mean parameter gap 0; ideal sign gap 1/10; finite-epsilon gap 250/2703 |
| T07 | Margin-sensitive sign approximation | 1,500 finite-law instances; no violation |
| T08 | Same sign histories, different optimizer memory | Second parameter increments +0.019803 and -0.012752 for learning rate 0.1 |
| T09 | Explicit hidden-world posterior and query decision trees | 20 exact K/budget cases agree with the derived optimum |
| T10 | Uniform missing mass and positive-part expectation | 35 exact cases; explicit counterexample to moving expectation inside the positive part |
| T11 | Adaptive query/stop transcripts in two worlds | 10 directional cases obey binary-KL and acquisition bounds |
| T12 | Stopped learning composition | 400 finite Markov instances, all bounds below one and valid |
| T13 | Existing Hoeffding sample allocation | 1,016 is sufficient for the stated radius; no necessity claim |

These tests support arithmetic and finite implementation consistency, not universal mathematical truth or neural generalization. In T12, bounds below one do not imply tight bounds. In T04, the rank test checks the specified finite witness; the proposition's proof supplies the general dimension statement.

During development, T09 initially failed at K=0 because an empty product began as an integer and subsequent division introduced floating point into an exact-rational posterior calculation. The accumulator was changed to start with `Fraction(1)`. No assertion tolerance was loosened. The final file and result reflect that correction.

## Evidence and remaining work

- The user's report of 22 PASS / 1 FAIL, C23 neural gaps, and original-package reproduction remains **user-supplied evidence**. The supplied plot was visually inspected; the external scripts and raw measurements were not available here.
- The theorem patch, optimizer identities/counterexample, K formulas, proofs, and the finite results above are the new work in this package.
- No Gemma, LoRA, or external benchmark training was executed as part of these new checks.
- Cold Adam's sign behavior and the missing-mass argument have direct precedents. Their inclusion does not by itself establish a novel central contribution.
- Priority work is the optimizer-matched neural comparator, actual-state proxy validation, measured audit cost, and a real intended-checkpoint pilot with held-out task consequences. A useful neural approximation/transport theorem remains open.

The bundle is a corrected and extended research framework, not a claim of an award-level completed theory.
