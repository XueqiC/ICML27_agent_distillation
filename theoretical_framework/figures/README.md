# Theoretical framework figures

Regenerate from the repository root (CPU only, no network):

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/figures/make_figures.py
```

For a subset, append `--only N12,N14`. Each basename produces a vector PDF with embedded Type 42 fonts and a 300 dpi PNG. Captions are embedded in PDF Subject and PNG Description metadata. All figures use constrained layout and explicit colors. The style block uses 9 pt text; a few dense annotations and category labels use 8 pt.

JSON evidence is loaded from the named blobs in Git HEAD; uncommitted result files and `theory/v015_checks/raw/` are never read. The N14 working-tree imports must match HEAD byte for byte. N18–N20 require only stored-data reproduction checks; their specs do not request importing or running otlr.py. Temporary Matplotlib caches are confined to this directory and removed on exit.

Data-source abbreviations:

- **I**: `theory/independent_checks/results_independent_v014.json`
- **V**: `theory/v015_checks/results_v015_independent.json`
- **P**: `theory/v014_repro/results_v014_rerun.json`
- **T**: `docs/direction/v015_revision/results_v015.json`

| Basename | Node | Data sources / key paths | What it shows |
| --- | --- | --- | --- |
| `N01_decomposition` | N01 | exact Fraction instance in N01; P:robust_ambiguity_minimax | Regret decomposition, garbling, and minimax duality. |
| `N02_queries` | N02 | I:claims[id=C2].details.configurations; specified passive formula | Query counts, bound ratios, and passive impossibility. |
| `N03_floor` | N03 | V:groups[id=E5].data.{curves,actual_learners}; T:checks.T11_adaptive_query_KL_enumeration.rows | Hidden-bit information floors, learners, and additive KL budgets. |
| `N04_static_vs_learning` | N04 | specified analytic formulas; I:controlled_neural.details.configurations | Static equivalence versus different trained students. |
| `N05_response` | N05 | exact Fraction instances in N05; I:claims[id=C4].details.cancellation_absolute_drift | Telescoping, response equivalence, and cancellation. |
| `N06_duality` | N06 | exact Fraction instance in N06; I:claims[id=C19].details.families; I:claims[id=C5] | Worst matched-pair discrepancy equals twice approximation error. |
| `N07_moment_order` | N07 | I:claims[id=C8].details.scalar; I:claims[id=C9].details.{grid_ranks,binary_barriers}; P:constant_gap_parity | Necessary moment orders, exact ranks, and the binary barrier. |
| `N08_rates` | N08 | I:claims[id=C18].details.curves; I:claims[id=C11].details.cumulant_cases; I:claims[id=C10] | Independent-evidence rates, cumulant coefficients, and a bound uniform in rounds. |
| `N09_approximation` | N09 | specified exact Chebyshev coefficients; P:scalar_approximation; I:claims[id=C12]; I:claims[id=C19].details.families | Approximation complexity versus exact polynomial degree. |
| `N10_regime` | N10 | I:controlled_neural.details.configurations; I:claims[id=C13]; V:groups[id=V7].data.linear_failure | Local SGD regime, remainder bounds, and geometric error accumulation. |
| `N11_cold_adam` | N11 | V:groups[id=E2].data.comparisons; V:groups[id=E1].data.ablations; V:groups[id=V4].data | Cold Adam, ablations, and magnitude-bearing optimizer memory. |
| `N12_warmth` | N12 | V:groups[id=E8].data.{gaps,W3.per_step,W3.fits,W1.panels} | Warm-up gaps, inverse-count warmth, and W1 residuals. |
| `N13_stability` | N13 | I:claims[id=C22].details.configurations; P:noisy_feature_duality.explicit_example; I:claims[id=C14] | Tolerant duality and the cost of small feature scale. |
| `N14_transport` | N14 | theory/v015_checks/{mathematics,common}.py (read-only, with common’s committed C23 dependency); V:groups[id=V1].data; V:groups[id=E4].data | Reproduced transport certificates, reached-state diagnostics, and uncovered states. |
| `N15_adaptive_audit` | N15 | I:claims[id=C16].details.configurations; V:groups[id=E7].data.{Hoeffding,empirical_Bernstein,variance_decomposition} | Adaptive audit gains, replica cost, and variance sources. |
| `N16_coupling` | N16 | I:controlled_neural.details.{regrouping,additive_null_parameter_residual}; P:coupling_and_additive_null; I:claims[id=C17] | Sequential regrouping effects and additive coupling width. |
| `N17_selection` | N17 | V:groups[id=E6].data.reports; V:groups[id=E9].data.{summary,cheap_proxy_results} | Held-out proxy selection, threshold failures, and warm-up transfer. |
| `N18_linear_response` | N18 | V:groups[id=E10].data.{cells,optimizer_summary,C23_prediction} | Optimizer-increment linear response, remainder orders, and empirical bound checks. |
| `N19_learner_statistics` | N19 | V:groups[id=E11].data | Clipping, warm Adam, Lion, and LoRA first-order statistics. |
| `N20_first_order_audit` | N20 | V:groups[id=V9].data; V:groups[id=E12].data.{bins,summary} | Dimension-free concentration and actual-state first-order margin audits. |

## Data/spec differences

- N03 A: the spec requests K=0…32, but the committed selected curves contain K=0…16. Only stored values are drawn; the panel is labelled with its supported range, with no extrapolation.
- N12 A: moment N=3, η=0.1, R=1, W=50: spec endpoint 0.0004; data 0.000202258761. Displayed 0.0769 → 0.0002.
- N12 A: moment N=3, η=0.1, R=4, W=50: spec endpoint 0.0016; data 0.003328877647. Displayed 0.1196 → 0.0033.
- N12 B: moment N=3, source P has slope -1.119403208 and R² 0.928300307. The separate medians over 9 finite-slope task/source series at η=0.01, W=0 are -1.057698636 and 0.9985193309. Both are now labelled.
- N12 C clarification: pointwise max residual/r = 0.9984871109 (within 0.151%); law-pair max gap/bound = 0.9883409452 (within 1.166%). The spec’s “within 0.2%” applies to the pointwise residual, not the law-pair ratio. Both are labelled separately.
- N11 wording: the plotted SGD gaps are small, not identically zero (max |actual SGD| = 0.003390528294); shifted-support residuals are R=1: -2.435468494e-12, R=4: -1.148025361e-05. Signed values are retained; caption says “small”/“nearly”.
- N18 A: the requested slope-2 guide is anchored to signSGD R=1, whose stored fitted slope is 2.999398618 (approximately cubic). The requested reference guide and actual fitted slope are both retained.
- N18 C: 5 exact cells have nonnumeric ratios and are omitted, never zero-filled. E10 explicitly says smoothness S1 is not certified and curvature estimates are empirical. The caption reports observed ratios, not a verified uniform smoothness certificate.
- N19 C: measured gap 8.753159888e-06 exceeds W1 bound 7.126654275e-06; the source caveat says this bound covers only the linear v contribution (2.50425886e-06), not the full nonlinear gap.
- N20 C: requested bin-row-weighted outside oracle rate = 0.3451124567; combining with the inside rate gives 0.296308675 versus stored eligible rate 0.2869318182 (absolute difference 0.009376856803; tolerance 0.02: PASS). Bin n counts include zero-gold rows excluded from wrong-sign denominators, so row weighting is approximate. Stored proxy rates retain their original denominators.
- N20 caption: zero oracle error from ratio 0.25 is an empirical bin observation; the C6 separator remains at ratio 1. The outside-margin short-tail-4 error is nonzero; the caption does not claim perfect reliability.

## Rendering choices and scope

- N09 B omits exact zero errors from the log curves. A triangle at the threshold marks every first crossing, including the zero-error endpoint for N=4.
- N18 uses a fixed 11-optimizer color order, skips missing numeric values, and verifies all exact-cell maxima before drawing. N19 shows zero Lion increments as text at the log floor. N20 verifies its reconstructed oracle mixture within the specified 0.02 tolerance.
- N11 retains tiny signed SGD and shifted-support gaps. N12 retains the noisy warmth trajectory and signed SGD comparators without smoothing.
- N14 C uses adjacent strip and bar axes inside the requested panel. Its transport certificate is capped as specified and labelled with its full value. N14 A plots the 2,000 reproduced random instances plus the sharp witness; its 2,572-instance annotation is explicitly the committed full V1 summary.
- N16 B uses a labelled schematic position for the additive point because the source stores its width but no min/max or absolute response.
- N17 preserves static/measured/control colors across panels; hatching distinguishes held-out splits. The two C dot plots share their row axis. Only the specified cheap_proxy_results cohort supplies the all-three-metrics count.
- All prescribed panels and data series are included over their committed support (N03’s missing K values are documented above). No data are downsampled. Coincident curves use distinct markers or line styles. N03 direction 1 and N07 optional package rank markers are omitted, as allowed by their specs.
- Panel titles preserve spec wording, with line breaks for fit. N01–N05 use their panel descriptions where no literal title is specified.
- Exact analytic instances use fractions until plotting; N14 reproduces the seeded evaluator and refuses to draw on any mismatch. The W1 source describes an empirical consistency check, not a uniform certificate.

## Determinism

Fixed `svg.hashsalt`, fixed jitter seed, explicit plotting order, fixed metadata, no PDF creation/modification date, and PNG timestamp stripping. To verify, run `CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/figures/verify_figures.py`. It regenerates all 20 figures twice, compares every PDF/PNG plus README and the text audit byte for byte, and writes sizes and SHA-256 hashes to [verification.json](verification.json). That report records the two-run result for all 42 generated files and the exact interpreter/library environment.

## Rendered text audit

`text_audit.json` records the bounding box of every visible text artist at 300 dpi, all intersecting pairs, out-of-canvas text, and any unrendered formula characters. Off-view ticks and invisible descendants are excluded. Bounding boxes are conservative, especially for rotated and multiline text. No overlapping pair is silently exempted.

| Figure | Text artists | Bounding-box overlaps | Outside canvas |
| --- | ---: | ---: | ---: |
| N01 | 58 | 0 | 0 |
| N02 | 50 | 0 | 0 |
| N03 | 64 | 0 | 0 |
| N04 | 60 | 0 | 0 |
| N05 | 47 | 0 | 0 |
| N06 | 46 | 0 | 0 |
| N07 | 54 | 0 | 0 |
| N08 | 59 | 0 | 0 |
| N09 | 68 | 0 | 0 |
| N10 | 65 | 0 | 0 |
| N11 | 65 | 0 | 0 |
| N12 | 58 | 0 | 0 |
| N13 | 50 | 0 | 0 |
| N14 | 79 | 0 | 0 |
| N15 | 56 | 0 | 0 |
| N16 | 36 | 0 | 0 |
| N17 | 103 | 0 | 0 |
| N18 | 74 | 0 | 0 |
| N19 | 95 | 0 | 0 |
| N20 | 65 | 0 | 0 |

## Output sizes

| File | Bytes |
| --- | ---: |
| `N01_decomposition.pdf` | 18,056 |
| `N01_decomposition.png` | 309,056 |
| `N02_queries.pdf` | 21,252 |
| `N02_queries.png` | 357,104 |
| `N03_floor.pdf` | 72,195 |
| `N03_floor.png` | 368,535 |
| `N04_static_vs_learning.pdf` | 20,301 |
| `N04_static_vs_learning.png` | 335,076 |
| `N05_response.pdf` | 24,682 |
| `N05_response.png` | 220,582 |
| `N06_duality.pdf` | 31,108 |
| `N06_duality.png` | 306,800 |
| `N07_moment_order.pdf` | 28,062 |
| `N07_moment_order.png` | 296,626 |
| `N08_rates.pdf` | 30,246 |
| `N08_rates.png` | 405,391 |
| `N09_approximation.pdf` | 29,107 |
| `N09_approximation.png` | 388,906 |
| `N10_regime.pdf` | 35,576 |
| `N10_regime.png` | 336,405 |
| `N11_cold_adam.pdf` | 25,725 |
| `N11_cold_adam.png` | 306,704 |
| `N12_warmth.pdf` | 33,542 |
| `N12_warmth.png` | 487,298 |
| `N13_stability.pdf` | 26,160 |
| `N13_stability.png` | 290,551 |
| `N14_transport.pdf` | 56,844 |
| `N14_transport.png` | 580,804 |
| `N15_adaptive_audit.pdf` | 22,177 |
| `N15_adaptive_audit.png` | 341,054 |
| `N16_coupling.pdf` | 20,422 |
| `N16_coupling.png` | 288,751 |
| `N17_selection.pdf` | 37,293 |
| `N17_selection.png` | 480,522 |
| `N18_linear_response.pdf` | 28,016 |
| `N18_linear_response.png` | 545,424 |
| `N19_learner_statistics.pdf` | 32,777 |
| `N19_learner_statistics.png` | 694,419 |
| `N20_first_order_audit.pdf` | 26,599 |
| `N20_first_order_audit.png` | 348,465 |
