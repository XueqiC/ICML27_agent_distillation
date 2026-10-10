# Independent v0.15 checks

All mathematics is derived from the three documents named in SPEC.md. The package verifier and its result file are forbidden by a runtime read guard. Only the expressly permitted v0.14 independent C23 helpers are imported, unchanged. A runtime filesystem guard restricts writes to this directory; network access is disabled. No repository tests are collected.

Run:

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --output theory/v015_checks/results_v015_independent.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 theory/.venv/bin/python -m pytest -c /dev/null --rootdir=theory/v015_checks --confcutdir=theory/v015_checks -p no:cacheprovider --basetemp=theory/v015_checks/quick/pytest_tmp theory/v015_checks/tests/test_v015_quick.py -q
```

`--quick` reduces random counts and the E2 matrix; it retains every group and every failure assertion. `--only V3,E2` selects groups. Quick artifacts go in `quick/`; full artifacts and figures remain intact. Every FAIL makes the command and the quick test fail. Expected failures of the old theorem and the linear recursion shortcut are affirmative regressions, not skipped assertions.

Master seed: 150150. Every random family uses `SeedSequence([150150, group_number])`, with V1–V7 numbered 1–7 and E0–E7 numbered 100–107. Runtime: 55.14 s. One CPU process; BLAS threads capped at one.

| Group | Status | Instances | Key result |
|---|---|---:|---|
| V1 | PASS | 2572 | old bound 0.02960414 < gap 1; best A4 ratio 1 |
| V2 | PASS | 300 | 300 quotient alphabets; inverse derivative reaches 10000000000000000000000 |
| V3 | PASS | 1660 | largest A9 gap/bound 0.994079 |
| V4 | PASS | 39 | opposite-step witnesses on 80% of beta grid |
| V5 | PASS | 348 | 100 transcript families; strict Jensen 1/2 > 0 |
| V6 | PASS | 297 | best A15 tightness ratio 1 |
| V7 | PASS | 717 | linear witness: 65/32 > 1 |
| E0 | PASS | 72 | R=1: gap 0.076907458; R=4: gap 0.119610952 |
| E1 | PASS | 824 | 110 ablation cells; max relative map error 2.1e-15 |
| E3 | PASS | 18 | 2 neural epsilon-grid witnesses |
| E2 | INFO | 56145 | 39/45 frozen-Adam gaps persist |
| E4 | INFO | 40 | 36/40 anchor prediction failures at 0.001 |
| E5 | PASS | 1070 | 944 exact K/query cells; replay gain zero |
| E6 | INFO | 504 | 432 held-out rows; 38 proxy/split failures to beat best simple control |
| E7 | INFO | 9600 | Hoeffding n=1016; pilot empirical-Bernstein forecast n=559 |

## Methods

- **V1 — Theorem 19 repair; A1–A5, A17 and IID affine (40).** Fraction-valued Markov kernels, backward values and independent forward occupancies.
- **V2 — A2 cold Adam/AdamW map, inverse and minimal quotient dimensions.** Independent recurrence, rational arithmetic, exact ranks/nullspaces and HiGHS separation.
- **V3 — A3 margin-sensitive approximation (A9) and sign TV bound.** Exact rational clipped sums of piecewise-linear coordinate functions.
- **V4 — A4 magnitude-bearing optimizer state and quadratic realization.** Fraction moments, symbolic square roots, positive-epsilon grid and beta-grid search.
- **V5 — A5–A6 hidden-bit coverage, hard query budget and adaptivity.** All bit worlds, ordered reveal histories, Bayes actions and label-dependent decision trees.
- **V6 — A7 two-world adaptive stopping and demonstration information credit.** Exact rational transcript probabilities; direct log KL versus separately accumulated chain.
- **V7 — Sharp geometric recursion; failure of unconditional linear-in-R simplification.** Exact rational recurrence with slack; restart/hill-climb over eta,L,C,R.
- **E0 — C23 reproduction and raw artifact bundle.** Imported original C23 functions; enumerated histories and independently logged recurrence.
- **E1 — Actual finite-epsilon Adam map and one-factor ablations.** Per-support and history enumeration against imported neural_step; separate sign limit.
- **E3 — Same-sign history regression in the executed scalar and neural optimizers.** Shared E1/E2 optimizer; rational scalar oracle; neural soft-label search.
- **E2 — Optimizer-matched frozen-Jacobian comparison.** Five branches from one snapshot; identical within-source histories across branches.
- **E4 — Fresh-state coverage diagnostics and anchored response predictions.** Actual reference trajectories; exact finite-support/full-tail enumeration at observed states.
- **E5 — Hidden-bit K/query curves and achieved tabular learner risk.** Exact integer occupancy inversion, rational risks, actual Adam/SGD logit updates.
- **E6 — G5 held-out provider selection using frozen pilot proxies.** Exact one-boundary/full-reference-tail enumeration; independent state/task/recipe holdouts.
- **E7 — Gold-audit paired variance and cost forecast.** Correct inverse-CDF coupling; exact nested variance and independent paired pilot.

## Reading the evidence

`PASS` means the implemented mathematical or numerical consistency assertions passed. `INFO` marks empirical diagnostics or forecasts; it is not a gate pass or a sufficiency certificate. A scientific negative result remains in the raw results even when numerical consistency checks pass. All assertion failures retain exact inputs under `counterexamples`; rational numbers are strings. LP tolerance is 1e-8, Adam-map tolerance is a predeclared relative 1e-12, neural arithmetic is float64.

The C23 bundle includes exact laws, matched moments, raw gradients/updates/moments, the exact historical Theorem 19 wording and source SHA, 23 independent check definitions and their historical independent results. The supplied 19-file package manifest is metadata: its package verifier/results are deliberately not reread or rerun. Reproduction here refers to the available independent C23 implementation, not an unspecified external model run.

E2 uses all finite histories when the imported helper permits (at most 4096), otherwise 2000 independent histories per source in the full run. The same histories feed all five branches. E4 measures finite message panels at genuinely reached states; its prediction failures do not falsify the repaired bound and do not provide uniform coverage.

E5 uses integer subset Möbius inversion and exact rational risks, then actual tabular Adam/SGD logits. K counts independent complete trajectories; additional queries and replay are separate. Distinct subcontexts are not distinct tasks.

E6 writes its split manifest before any measurements and freezes all proxy coefficients and pilot abstention thresholds before held-out evaluation. Gold is exact finite-history integration for one candidate boundary plus seven reference updates. State, task, epsilon, learning-rate and fresh/warm transfers are reported separately; no held-out result changes a proxy. Negative selection results are explicit. Empirical calibration and cell bootstraps are not uniform certificates.

E7 retains shared-reference covariance, separately enumerates block/suffix/deployment variance, and reports the 1016-replica Hoeffding allocation. Its Maurer–Pontil allocation uses measured pilot variance as a forecast; a future independent fixed-n audit must compute its own radius. Supplied hpg/Gemma timings are scenario inputs only, not measured here.

## Scope and claim ledger

- Reproduced: available C23 controlled network and old coverage counterexample.
- Exact finite evidence: rational mathematics, hidden-bit enumeration, optimizer-history moments. Numerical checks are not formal proofs.
- Empirical: neural comparator gaps, anchor diagnostics, held-out proxy results.
- Forecast: Gemma costs from supplied hpg segment timings.
- No Gemma or micro-LoRA pilot ran: this task authorizes the offline CPU controlled suite only, with no network, GPU, checkpoint access or model substitution. No teacher is called.
- Unresolved: useful uniform neural coverage/response certificates and transfer to actual agent training.

## Artifacts

`results_v015_independent.json` contains provenance, source/code SHA-256s, every group, counterexamples, and artifact hashes. `full_run.log` is the complete progress log. `c23_bundle.json` contains reproduction records. `raw/` holds per-history arrays, proxy rows and paired-pilot data; `figs/E0` through `figs/E7` have PNG and SVG versions. `proxy_split_freeze.json` and `proxy_models_frozen.json` preserve the split and fitted proxy contract. No commits or repository test runs are performed.
