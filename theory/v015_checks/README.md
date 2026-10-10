# Independent v0.15 checks

All mathematics is derived from the three documents named in SPEC.md, `docs/direction/v015_extension/Warm_Adam_Response_EN.md`, and `docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md` (read in full). The package verifier and its result file are forbidden by a runtime read guard. Only the expressly permitted v0.14 independent C23 helpers are imported, unchanged. A runtime filesystem guard restricts writes to this directory; network access is disabled. No repository tests are collected.

Run:

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --output theory/v015_checks/results_v015_independent.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 theory/.venv/bin/python -m pytest -c /dev/null --rootdir=theory/v015_checks --confcutdir=theory/v015_checks -p no:cacheprovider --basetemp=theory/v015_checks/quick/pytest_tmp theory/v015_checks/tests/test_v015_quick.py -q
```

`--quick` reduces random counts and the E2 matrix; it retains every group and every failure assertion, including every E8 and E10 task/rate/horizon cell. `--only V8,V9,E10,E11,E12` selects the new extension. Quick artifacts go in `quick/`; full artifacts and figures remain intact. Every group FAIL makes the command and the quick test fail. E10 is an INFO diagnostic with explicit cell PASS/FAIL/UNAVAILABLE statuses: cell failures are scientific negative results, never group PASS. Expected failures of the old theorem and the linear recursion shortcut are affirmative regressions.

Master seed: 150150. Every random family uses `SeedSequence([150150, group_number])`, with V1–V9 numbered 1–9 and E0–E12 numbered 100–112. Runtime: 350.73 s. One CPU process; BLAS threads capped at one.

| Group | Status | Instances | Seconds | Key result |
|---|---|---:|---:|---|
| V1 | PASS | 2572 | 6.08 | old bound 0.02960414 < gap 1; best A4 ratio 1 |
| V2 | PASS | 300 | 1.84 | 300 quotient alphabets; inverse derivative reaches 10000000000000000000000 |
| V3 | PASS | 1660 | 1.23 | largest A9 gap/bound 0.994079 |
| V4 | PASS | 39 | 0.12 | opposite-step witnesses on 80% of beta grid |
| V5 | PASS | 348 | 7.14 | 100 transcript families; strict Jensen 1/2 > 0 |
| V6 | PASS | 297 | 4.26 | best A15 tightness ratio 1 |
| V7 | PASS | 717 | 0.37 | linear witness: 65/32 > 1 |
| V8 | PASS | 3712 | 2.94 | 140 exact polynomials; logistic max ratio 0.182664; 160 rank/LP panels |
| V9 | PASS | 2880000 | 4.79 | 20000 trials/cell; max quantile/bound 0.615277 |
| E0 | PASS | 72 | 0.39 | R=1: gap 0.076907458; R=4: gap 0.119610952 |
| E1 | PASS | 824 | 1.89 | 110 ablation cells; max relative map error 2.1e-15 |
| E3 | PASS | 18 | 0.16 | 2 neural epsilon-grid witnesses |
| E2 | INFO | 56145 | 17.71 | 39/45 frozen-Adam gaps persist |
| E4 | INFO | 40 | 0.36 | 36/40 anchor prediction failures at 0.001 |
| E5 | PASS | 1070 | 23.43 | 944 exact K/query cells; replay gain zero |
| E6 | INFO | 504 | 1.48 | 432 held-out rows; 38 proxy/split failures to beat best simple control |
| E7 | INFO | 9600 | 0.29 | Hoeffding n=1016; pilot empirical-Bernstein forecast n=559 |
| E8 | PASS | 1443382 | 35.85 | W0 best ratio 1; W1 PASS on 2496 law pairs; 210 gap cells |
| E9 | INFO | 918 | 2.92 | 918 held-out rows; 0/8 cheap proxy/W pairs improve all three split-mean metrics |
| E10 | INFO | 990 | 194.43 | 990 cells; 26 empirical bound failures; 10 unavailable derivatives; C23 0.076907458 |
| E11 | PASS | 15 | 1.67 | 5 catalog witnesses; always-active slope 1.99997; warm v term 2.50426e-06 |
| E12 | INFO | 432 | 20.11 | 432 frozen rows; 396 smooth; 56 above margin; margin wrong-sign 0.0 |

## Methods

- **V1 — Theorem 19 repair; A1–A5, A17 and IID affine (40).** Fraction-valued Markov kernels, backward values and independent forward occupancies.
- **V2 — A2 cold Adam/AdamW map, inverse and minimal quotient dimensions.** Independent recurrence, rational arithmetic, exact ranks/nullspaces and HiGHS separation.
- **V3 — A3 margin-sensitive approximation (A9) and sign TV bound.** Exact rational clipped sums of piecewise-linear coordinate functions.
- **V4 — A4 magnitude-bearing optimizer state and quadratic realization.** Fraction moments, symbolic square roots, positive-epsilon grid and beta-grid search.
- **V5 — A5–A6 hidden-bit coverage, hard query budget and adaptivity.** All bit worlds, ordered reveal histories, Bayes actions and label-dependent decision trees.
- **V6 — A7 two-world adaptive stopping and demonstration information credit.** Exact rational transcript probabilities; direct log KL versus separately accumulated chain.
- **V7 — Sharp geometric recursion; failure of unconditional linear-in-R simplification.** Exact rational recurrence with slack; restart/hill-climb over eta,L,C,R.
- **V8 — C1 finite Taylor identities and C3 minimal first-order statistics.** Fraction polynomial arithmetic; certified logistic Hessians; exact ranks and LP witnesses.
- **V9 — C5 dimension-free one- and two-sample concentration.** IID trials; exact implicit high-dimensional sampling; 4-SE binomial margin.
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
- **E10 — C2 complete-state linear response across eleven optimizer configurations.** Exact R=1,4 histories; independent R=40 runs; full-state Richardson contractions and unpadded segment curvature.
- **E11 — C4 clipping, warm Adam, Lion and LoRA witnesses.** Executed maps, actual C23 continuations, linear-logit controls, bilinear sigmoid LoRA.
- **E12 — C6 margin on the unchanged E6 held-out panels.** Read frozen rows/models; exact seven-update reference continuation; empirical segment curvature.

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
- No pretrained Gemma/LoRA pilot ran: E11 is a tiny constructed bilinear sigmoid model. This task authorizes the offline CPU controlled suite only, with no network, GPU, checkpoint access or model substitution. No teacher is called.
- Unresolved: useful uniform neural coverage/response certificates and transfer to actual agent training.

## Artifacts

`results_v015_independent.json` contains provenance, source/code SHA-256s, every group, counterexamples, and artifact hashes. `full_run.log` is the complete progress log. `c23_bundle.json` contains reproduction records. `raw/` holds per-history arrays, proxy rows, pre-extension results and paired-pilot data; `figs/E0` through `figs/E9` have PNG and SVG versions (E8 also has warmth and W2 panels). `proxy_split_freeze.json` and `proxy_models_frozen.json` preserve the original E6 contract; the `_E9.json` files preserve the new frozen contract. `legacy_data_comparison.json` compares old group data before the results file is overwritten. No commits or repository test runs are performed.

Legacy comparison: 17/17 original group data objects byte-identical under canonical JSON serialization. Differences: [].

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


## Optimizer-transformed response methods

V8 uses exact Fraction line-polynomial coefficients. For the infinity norm, the derivative upper bound is j! times the coefficient of t^j in sum_alpha |c_alpha| product_i (b_i+t)^alpha_i, where b_i is the largest absolute coordinate among x and the increment endpoints. This rational coefficient-sum bound contains the whole convex hull; it is not its exact supremum. The logistic quadratic uses the certified analytic bound L²/(6 sqrt(3)) + ||A||op/4, with L the largest ||Az+b|| at a hull vertex. Full runs include 2200 independent instances plus random-restart hill climbing. C3 uses exact quotient ranks, nullspace matches and feature-matching LP witnesses.

V9 uses 20000 IID trials per law/dimension/n cell, reused across the two delta values. All laws have norm at most one. Biased scalar sign vectors use eta=1/sqrt(p); clipped Gaussian is full-dimensional N(0,I/p) with norm clipped at one; spikes are fair signed uniform coordinate vectors. Exact Beta-angle rotational recursion and explicit sparse collisions avoid materializing p=100000 vectors. Coverage assertions allow four binomial standard errors using delta. Fixed-coordinate sign updates still have b=eta sqrt(p).

E10 keeps the C23 one-hot → four tanh → sigmoid network, mean CE and float64. The full state contains theta, m and v. The original SGD and Adam maps and E2 run_network/weighted_result are reused unchanged; the other maps live only in otlr.py. Warm Adam executes a shared IID P prefix at lr=0 for W=5,20. First-order contractions perturb all state blocks together, using central differences at h=.002 and h/2 and Richardson extrapolation; the error diagnostic is |fine-coarse|/3. R=1,4 enumerate every Q history and P suffix. R=40 uses 2000 independent complete runs per law for the gap and 2000 independent random-time Q boundaries with IID P tails for the C2 sum, with SE. Exact R=40 continuation enumeration is not implemented (up to 4^39 suffixes). This limitation is explicit; no sampled result is called exact.

The E10/E12 Mhat2 search normalizes realized increment directions by the Euclidean norm of (delta theta, eta delta m, eta delta v). It samples five segment positions and hill-climbs their directional second derivatives with Richardson differences. It never raises a constant to accommodate a measured gap. E10 samples up to four Q states at every short-horizon time; R=40 samples four observed times per cell (two in quick) and uses their largest constant across the cell. E12 uses every message ray at the frozen boundary. These are empirical directional Taylor bounds, not full Hessian operator-norm certificates. Sign/clipping kinks and cold zero-variance coordinates can violate S1; nonfinite central probes are UNAVAILABLE. Every ratio above one remains a cell FAIL. R=40 also reports residual SE and whether an exceedance remains after four SE; that diagnostic does not inflate the bound or change the point-estimate status. Slopes use eta<=.01 with a 1e-12 numerical floor; zero terms have no fitted slope. The analytic nonzero LoRA witness retains its smaller signal and states that exception.

E11 executes the clipping witness as a direct gradient-law linear-logit experiment. The always-active witness matches mean unit direction but differs in direction second moments; changing magnitudes alone with an identical direction law would give an exactly zero gap. The warm-Adam witness uses symmetric C23 labels and a declared admissible warm state m=0,v=.02,t=20 so theta/m contrasts cancel; generic equal-mean laws need not cancel the theta warmth error. Its W1 comparison bounds the explicit linear v term. Lion records both dominated and flipping controls and distinguishes stored m differences from a potentially zero continuation derivative. LoRA uses d=k=4, ranks 1 and 2, legal soft-label mean CE, B=0 and random A. A is checked bitwise after step one. Cold Adam equality of sketch signs is an epsilon→0 statement; finite-epsilon h_epsilon is checked separately.

E12 reads the original full E6 split, models and held-out rows even in quick mode. It never fits a proxy or modifies E6/E9 artifacts, and checks their hashes before/after. Every smooth row uses the exact seven-update reference suffix and a full complete-state coordinate gradient, cross-checked against directional Richardson. Threshold utility is discontinuous: those rows retain formal numerical scores and proxy comparisons but are ineligible for C6. Zero predictions and the |gold|>1e-10 convention follow E6. The no-wrong-sign assertion above the empirical margin is hard, but is not a uniform certificate.

New artifacts: `otlr.py`, `raw/E10_cells.json`, `raw/E12_margin_rows.json`, and PNG/SVG pairs `figs/V9`, `figs/E10`, `figs/E11`, `figs/E12`. All cell values, differentiation diagnostics, curvature searches, negative results and omissions are retained in the main JSON. `raw/results_before_otlr.json` preserves the pre-extension baseline. Results, README and legacy comparison are published using a same-directory temporary file, fsync and os.replace; concurrent readers never see truncation.

## E10 outcomes and limitations

The group is INFO, with the following explicit per-optimizer cell status. Short-horizon gaps and continuations are exactly enumerated; R=40 quantities are sampled. The slope columns show moment N=3, R=4 at eta=.001,.003,.01. Other task/horizon slopes remain in JSON. Undefined first-order slopes reflect vanishing terms, not successful eta scaling.

| Optimizer | Cell status | Max ratio, R<=4 | Max ratio, all R | R=40 exceedances after 4 SE | Remainder slope | First-order slope | Sign agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| SGD | FAIL | 0.99862 | 14.972 | 0 | 2.9628 | n/a | 0 |
| heavy_ball | FAIL | 0.99839 | 1.6085 | 0 | 2.9541 | n/a | 0 |
| Nesterov | FAIL | 0.99797 | 1.0076 | 0 | 2.9069 | n/a | 0 |
| Adam_cold | PASS | 0.99815 | 0.99815 | 0 | 0.9894 | 1.0034 | 0.37778 |
| Adam_warm5 | PASS | 0.33388 | 0.33388 | 0 | 1.1082 | 1.0393 | 0.62921 |
| Adam_warm20 | PASS | 0.84324 | 0.84324 | 0 | 0.9957 | 0.9957 | 0.58427 |
| AdamW | PASS | 0.98167 | 0.98167 | 0 | 0.98918 | 1.0032 | 0.44444 |
| clipped_SGD | FAIL | 1.2453 | 13.451 | 0 | 2.9909 | 0.99785 | 0.54545 |
| clipped_Adam | UNAVAILABLE | 0.99819 | 0.99819 | 0 | 0.9101 | 1.0082 | 0.5125 |
| signSGD | FAIL | 0.99815 | 2284.8 | 1 | 2.9778 | 1.0008 | 0.57778 |
| Lion | FAIL | 0.99815 | 902.19 | 1 | 2.7129 | 1.0008 | 0.57778 |

Cold Adam C23 (moment N=3, eta=.1, R=1): measured 0.0769074585, transformed prediction 0.0756827915. C4(d) displayed Prediction has a plus eta multiplying grad(V) dot delta h; the actual increment is -eta h, so C1/C2 give the opposite sign. The increment-based prediction here uses the correct minus sign.

Clipping frequencies use the declared equal mixture (P+Q)/2; uniform-alphabet fractions are also recorded. Norms within 1e-12 are treated as ties. A scalar threshold cannot split a tie. The 30–70% target is infeasible for: moment N=3 (chosen frequency 0.25), parity N=3 (chosen frequency 0.25).

Near-linear remainder slopes for stateful Adam do not support the derivation’s general O(eta^2) scaling claim. Dependence on moments through an O(eta) future update alone does not imply that the Hessian stays O(1) in an eta-weighted state norm. No constants were padded to fix this result.

## E12 frozen-panel outcomes

| Split | Smooth rows / all rows | Above margin | Oracle wrong sign, smooth | Oracle wrong sign, margin |
|---|---:|---:|---:|---:|
| overall | 396 / 432 | 56 | 0.28693 | 0 |
| held_out_states | 72 / 72 | 16 | 0.23438 | 0 |
| held_out_task | 0 / 36 | 0 | n/a | n/a |
| held_out_epsilon | 72 / 72 | 16 | 0.28125 | 0 |
| held_out_lr | 72 / 72 | 0 | 0.3125 | n/a |
| held_out_epsilon_lr | 72 / 72 | 0 | 0.3125 | n/a |
| fresh_to_warm | 72 / 72 | 16 | 0.23438 | 0 |
| warm_to_fresh | 36 / 36 | 8 | 0.40625 | 0 |

Proxy comparison on the same smooth rows. Enrichment divides the fraction of a proxy’s failures outside the margin by the fraction of nonzero-gold rows outside it; a large raw fraction alone is not evidence of concentration.

| Proxy | Wrong sign, smooth | Wrong sign, margin | Outside-margin failure enrichment |
|---|---:|---:|---:|
| input_moment_error | 0.34375 | 0.35714 | 0.99263 |
| mean_gradient | 0.35227 | 0.35714 | 0.99738 |
| sign_only | 0.35795 | 0.35714 | 1.0004 |
| first_delta | 0.37784 | 0.35714 | 1.0104 |
| delta_plus_m_v | 0.43466 | 0.35714 | 1.0337 |
| short_tail_1 | 0.17045 | 0 | 1.1892 |
| short_tail_4 | 0.025568 | 0 | 1.1892 |
| random | 0.53125 | 0.53571 | 0.99841 |
| task_identity | 0.4233 | 0.5 | 0.96572 |
| training_step | 0.21023 | 0.21429 | 0.99635 |
| full_tail_gold | 0 | 0 | n/a |
