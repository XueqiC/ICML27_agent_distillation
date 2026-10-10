# Independent v0.15 checks

All mathematics is derived from the three documents named in SPEC.md and `docs/direction/v015_extension/Warm_Adam_Response_EN.md` (read in full). The package verifier and its result file are forbidden by a runtime read guard. Only the expressly permitted v0.14 independent C23 helpers are imported, unchanged. A runtime filesystem guard restricts writes to this directory; network access is disabled. No repository tests are collected.

Run:

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --output theory/v015_checks/results_v015_independent.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 theory/.venv/bin/python -m pytest -c /dev/null --rootdir=theory/v015_checks --confcutdir=theory/v015_checks -p no:cacheprovider --basetemp=theory/v015_checks/quick/pytest_tmp theory/v015_checks/tests/test_v015_quick.py -q
```

`--quick` reduces random counts and the E2 matrix; it retains every group and every failure assertion, including every E8 task/rate/horizon/W cell. `--only V3,E2` or `--only E8,E9` selects groups. Quick artifacts go in `quick/`; full artifacts and figures remain intact. Every FAIL makes the command and the quick test fail. Expected failures of the old theorem and the linear recursion shortcut are affirmative regressions, not skipped assertions.

Master seed: 150150. Every random family uses `SeedSequence([150150, group_number])`, with V1–V7 numbered 1–7 and E0–E9 numbered 100–109. Runtime: 121.11 s. One CPU process; BLAS threads capped at one.

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
| E8 | PASS | 1443382 | W0 best ratio 1; W1 PASS on 2496 law pairs; 210 gap cells |
| E9 | INFO | 918 | 918 held-out rows; 0/8 cheap proxy/W pairs improve all three split-mean metrics |

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
- **E8 — Optimizer warmth W0--W3 and C23 moment warm-up.** Float64 adversarial W0; finite-suffix W1 with sampled constants; exact W2; actual trajectories and E2 warm-up matrix.
- **E9 — G5 proxy transfer under lr=0 moment warm-up.** Unchanged E6 provider/features/gold/fits/metrics; frozen E6 pilot and paired cold/warmed held-out cells.

## Reading the evidence

`PASS` means the implemented mathematical or numerical consistency assertions passed. `INFO` marks empirical diagnostics or forecasts; it is not a gate pass or a sufficiency certificate. A scientific negative result remains in the raw results even when numerical consistency checks pass. All assertion failures retain exact inputs under `counterexamples`; rational numbers are strings. LP tolerance is 1e-8, Adam-map tolerance is a predeclared relative 1e-12, neural arithmetic is float64.

The C23 bundle includes exact laws, matched moments, raw gradients/updates/moments, the exact historical Theorem 19 wording and source SHA, 23 independent check definitions and their historical independent results. The supplied 19-file package manifest is metadata: its package verifier/results are deliberately not reread or rerun. Reproduction here refers to the available independent C23 implementation, not an unspecified external model run.

E2 uses all finite histories when the imported helper permits (at most 4096), otherwise 2000 independent histories per source in the full run. The same histories feed all five branches. E4 measures finite message panels at genuinely reached states; its prediction failures do not falsify the repaired bound and do not provide uniform coverage.

E5 uses integer subset Möbius inversion and exact rational risks, then actual tabular Adam/SGD logits. K counts independent complete trajectories; additional queries and replay are separate. Distinct subcontexts are not distinct tasks.

E6 writes its split manifest before any measurements and freezes all proxy coefficients and pilot abstention thresholds before held-out evaluation. Gold is exact finite-history integration for one candidate boundary plus seven reference updates. State, task, epsilon, learning-rate and fresh/warm transfers are reported separately; no held-out result changes a proxy. Negative selection results are explicit. Empirical calibration and cell bootstraps are not uniform certificates.

E7 retains shared-reference covariance, separately enumerates block/suffix/deployment variance, and reports the 1016-replica Hoeffding allocation. Its Maurer–Pontil allocation uses measured pilot variance as a forecast; a future independent fixed-n audit must compute its own radius. Supplied hpg/Gemma timings are scenario inputs only, not measured here.

E8 uses seed `[150150,108]`. W0 samples positive second moments log-uniformly over 1e-12..1, random first moments/gradients, t=1..2000 and five beta pairs including (0.9,0.999); random restarts and hill-climbing maximize the bound ratio. The coordinatewise difference is rationalized before float64 evaluation to avoid cancellation between rounded updates; witnesses also contain direct subtraction. Only relative slack 1e-12 is permitted.

W1 trains 1–4 output-head coordinates of the original C23 network, keeping the remaining parameters fixed. It uses 2–6 messages, one actual Adam boundary and R=0,2,6 reference updates, enumerating all 2^R suffix histories. Reference labels are jointly all zero or all one with equal probability; deployment is C23 moment/parity return. An independent analytic sensitivity recurrence computes b=grad F(0). Central differences on declared actual/surrogate theta and relative-v segments estimate L_i and L_v; radial/convex-hull sampling and Hessian differences estimate H_G. Constants are fixed before law-pair checks and never padded after failures. Each pointwise and equal/unequal-mean two-law bound is checked, with adversarial searches on both law families. W1 PASS means empirical consistency on every measured instance, not a uniform certificate; signed exact-enumeration gaps and b^T dE G are recorded separately.

W2 retains the exact rational sign witness and finite-epsilon (1e-8) value. The same gradient laws are swept over a log kappa grid by varying v at fixed t and m=0. Both raw and preconditioner-normalized gap slopes are reported: changing v also changes the step scale. W3 records coordinate/replica kappa quantiles along actual moment/parity Adam trajectories at eta=.01,.1, with W=0,1,2,5,10,20,50 actual lr=0 warm-up steps. Every warm-up step must preserve parameter bytes. Fits use median=c/(W+s), s=1..79 completed parameter updates; free log-log slopes and R2 retain nonstationarity and poor fits. Infinity at v=0 is retained. This tests the cold-window heuristic, not a telescoping/coverage certificate.

E8 gap sweeps reuse E2 run_network and weighted_result for actual Adam, frozen-Jacobian Adam and tangent affine-SGD, across moment N=2,3,4, parity N=2,3, eta=.01,.1, R=1,4,40 and all seven W. The same finite or sampled histories feed all W/branches. Reference warm-up uses one declared IID P-law prefix per task; reported gaps condition on that prefix. W=0 is the original fresh state. Each Adam branch keeps its own subsequent moments.

E9 uses seed `[150150,109]` and the E6 provider_menu, proxy_panel, proxy_features, fit_proxies and proxy_metrics unchanged. It reuses only the original frozen E6 pilot for fitting, verifies coefficient identity, and writes proxy_split_freeze_E9.json and proxy_models_frozen_E9.json before any new held-out panel. All seven original E6 splits remain; five paired cold/W=5/W=20 panels separate state, task, epsilon, learning-rate and joint recipe shifts. The state-only comparison uses the original warm2-only fit, excluding fresh and both moment-warmed descendants from fitting. Other comparisons use combined E6 pilot models and their original held-out tasks/recipes. Random controls are paired across arms. No warm-up gold fits a model. Gold remains one candidate update plus seven IID reference updates, exactly enumerated, with the original task returns and all ten proxies plus full-tail oracle. Zero predictions, wrong-sign cutoff and abstention follow E6 exactly. Aggregate improvement is a descriptive equal-split mean; per-split neutral/adverse outcomes remain in the JSON.

## Scope and claim ledger

- Reproduced: available C23 controlled network and old coverage counterexample.
- Exact finite evidence: rational mathematics, hidden-bit enumeration, optimizer-history moments. Numerical checks are not formal proofs.
- Empirical: neural comparator gaps, anchor diagnostics, held-out proxy results.
- Forecast: Gemma costs from supplied hpg segment timings.
- No Gemma or micro-LoRA pilot ran: this task authorizes the offline CPU controlled suite only, with no network, GPU, checkpoint access or model substitution. No teacher is called.
- Unresolved: useful uniform neural coverage/response certificates and transfer to actual agent training.

## Artifacts

`results_v015_independent.json` contains provenance, source/code SHA-256s, every group, counterexamples, and artifact hashes. `full_run.log` is the complete progress log. `c23_bundle.json` contains reproduction records. `raw/` holds per-history arrays, proxy rows, pre-extension results and paired-pilot data; `figs/E0` through `figs/E9` have PNG and SVG versions (E8 also has warmth and W2 panels). `proxy_split_freeze.json` and `proxy_models_frozen.json` preserve the original E6 contract; the `_E9.json` files preserve the new frozen contract. `legacy_data_comparison.json` compares old group data before the results file is overwritten. No commits or repository test runs are performed.

Legacy comparison: 15/15 original group data objects byte-identical under canonical JSON serialization. Differences: [].

E7 originally stores wall-clock timing inside its data. Its five timing/CPU-projection fields retain the historical baseline to preserve that data object; current measured timings and every before/after value are separately recorded in `rerun_measurements.E7`. Scientific values and assertions are always recomputed and compared without normalization.

## Warmth results

W0: 6944 random/adversarial states, best ratio 1. W1: 24 continuation panels and 2496 law pairs; largest gap/bound 0.988341. W2 small-kappa log-log slopes: raw gap 1.499457, preconditioner-normalized gap 0.999457.

| W | Moment N=3, eta=.01, P: fitted c | Free log-log slope | R2 for c/(W+s) |
|---:|---:|---:|---:|
| 0 | 0.78095 | -1.1194 | 0.9283 |
| 1 | 0.683 | -1.1215 | 0.85238 |
| 2 | 0.61445 | -1.1289 | 0.81027 |
| 5 | 0.39037 | -1.0738 | 0.98069 |
| 10 | 0.32485 | -0.95835 | 0.96644 |
| 20 | 0.31429 | -0.96097 | 0.99668 |
| 50 | 0.30109 | -0.93794 | 0.98529 |

All 140 trajectory-family fits are retained, including 2 unfittable series. Fits exclude zero/infinite medians but preserve those values in raw records; successful fits are descriptive and need not follow 1/s.

## Cheap proxy results after warm-up

Equal-weight mean changes over the five paired held-out split panels (warm minus cold). Negative regret/wrong-sign changes and positive Spearman changes indicate improvement. These are descriptive changes; all E6 metrics, controls, raw predictions and per-split comparisons remain in the results.

| Proxy | W | Regret change | Wrong-sign change | Spearman change | All three improve |
|---|---:|---:|---:|---:|---|
| input_moment_error | 5 | +0 | +0.0375 | -0.0301639 | no |
| input_moment_error | 20 | +0.000210562 | +0.0625 | -0.207305 | no |
| mean_gradient | 5 | +0 | +0.0125 | +0.0421689 | no |
| mean_gradient | 20 | +0.000210562 | +0.05 | -0.150331 | no |
| first_delta | 5 | +0 | +0.075 | -0.0819569 | no |
| first_delta | 20 | +0.000210562 | +0.0375 | -0.199973 | no |
| delta_plus_m_v | 5 | +0 | +0.05 | -0.0723617 | no |
| delta_plus_m_v | 20 | +0.000210562 | +0.075 | -0.190845 | no |

0/8 cheap proxy/W combinations improve all three aggregate metrics in this panel. These finite held-out observations do not establish general G5 improvement; negative results are preserved.
