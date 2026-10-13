# Theoretical framework figures — v0.16

Regenerate and verify from the repository root (CPU only; no network):

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python -B theoretical_framework/figures/make_figures.py
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python -B theoretical_framework/figures/verify_figures.py
```

The figures are drawn with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)) from arrays prepared with NumPy ([Harris et al., 2020](#ref-harris2020array)). For a subset use `--only fig5_3,fig7_1`. There are 25 figures: 20 renamed figures (including the revised warmth figure) and the five explicitly specified new figures. Each basename produces a vector PDF with Type 42 fonts and a 300 dpi PNG. PDF Subject and PNG Description contain the caption. The matching specification is `specs/<short id>.md`.

## Font and layout policy

Every figure is 10 inches wide, with at most two panels per row. Full-width panels use additional rows. Base text and axis labels are 14 pt, titles 15 pt, ticks 12 pt, legends and annotations at least 12 pt. At a 900 px display width the smallest text is 15 px (12 × 900 / (10 × 72)). The verifier checks every text artist, including hidden and empty artists, against both the 12 pt and 15 px minima. All figures use constrained layout, explicit colors, sparse numeric ticks, and legends outside crowded axes.

## Provenance

Summary JSON is read only from named Git HEAD blobs. The sole raw-data exception is `theory/v016_checks/results_v016_raw.json.gz`, read from the working tree only after its byte count, SHA-256 and schema match the committed `raw_artifact` entry. It supplies the stored exact warmth grid and the certificate components/variance ratios absent from the summary. The warmth grid can also be computed exactly from its closed form; the generator cross-checks the stored formula. No v0.16 checks are rerun. `theory/v015_checks/raw/` is never read.

The transport figure retains the original seeded finite-evaluator reproduction: 2,000 exact tiny processes, with imported source files required to match HEAD byte for byte before import. No training or experiment suite is invoked. Caches stay in `.mpl-cache/` inside this directory; Python bytecode is disabled. No files are deleted. Superseded figures, specs and verification reports were moved into `_trash/theoretical_framework_figures_v015_20261011/`.

Data-source abbreviations:

- **I**: `theory/independent_checks/results_independent_v014.json`
- **V**: `theory/v015_checks/results_v015_independent.json`
- **P**: `theory/v014_repro/results_v014_rerun.json`
- **T**: `docs/direction/v015_revision/results_v015.json`
- **U**: `theory/v016_checks/results_v016.json`

| Basename | Section | Data sources / key paths | What it shows |
| --- | --- | --- | --- |
| `fig2_1_decomposition` | 2 | `exact Fraction instance in matching spec; P:robust_ambiguity_minimax` | Regret decomposition, garbling, and minimax duality. |
| `fig2_2_queries` | 2 | `I:claims[id=C2].details.configurations; specified passive formula` | Query counts, bound ratios, and passive impossibility. |
| `fig2_3_floor` | 2 | `V:groups[id=E5].data.{curves,actual_learners}; T:checks.T11_adaptive_query_KL_enumeration.rows` | Hidden-bit information floors, learners, and additive KL budgets. |
| `fig3_1_static_vs_learning` | 3 | `specified analytic formulas; I:controlled_neural.details.configurations` | Static equivalence versus different trained students. |
| `fig3_2_response` | 3 | `exact Fraction instances in matching spec; I:claims[id=C4].details.cancellation_absolute_drift` | Telescoping, response equivalence, and cancellation. |
| `fig4_1_duality` | 4 | `exact Fraction instance in matching spec; I:claims[id=C19].details.families; I:claims[id=C5]` | Worst matched-pair discrepancy equals twice approximation error. |
| `fig4_2_moment_order` | 4 | `I:claims[id=C8].details.scalar; I:claims[id=C9].details.{grid_ranks,binary_barriers}; P:constant_gap_parity` | Necessary moment orders, exact ranks, and the binary barrier. |
| `fig4_3_rates` | 4 | `I:claims[id=C18].details.curves; I:claims[id=C11].details.cumulant_cases; I:claims[id=C10]` | Independent-evidence rates, cumulant coefficients, and a bound uniform in rounds. |
| `fig4_4_approximation` | 4 | `specified exact Chebyshev coefficients; P:scalar_approximation; I:claims[id=C12]; I:claims[id=C19].details.families` | Approximation complexity versus exact polynomial degree. |
| `fig4_5_regime` | 4 | `I:controlled_neural.details.configurations; I:claims[id=C13]; V:groups[id=V7].data.linear_failure` | Local SGD regime, remainder bounds, and geometric error accumulation. |
| `fig5_1_linear_response` | 5 | `V:groups[id=E10].data.{cells,optimizer_summary,C23_prediction}` | First-order representation as a tool; remainder and empirical bound checks. |
| `fig5_2_cold_adam` | 5 | `V:groups[id=E2].data.comparisons; V:groups[id=E1].data.ablations; V:groups[id=V4].data` | Cold Adam, ablations, and magnitude-bearing optimizer memory. |
| `fig5_3_warmth` | 5 | `V:groups[id=E8].data.{gaps,W3.per_step,W1.panels}; U:checks[claim_id=C1.exact].data; authenticated raw /checks/C1.exact/{0,1,3,4}` | Warm-up gaps, exact warmth bounds and nonzero plateaus, affine residuals. |
| `fig5_4_learner_statistics` | 5 | `V:groups[id=E11].data` | Exact mean Adam increment; gradient moments only under frozen denominators; clipping, Lion, LoRA. |
| `fig6_1_information_bound` | 6 | `U:checks[claim_id=C2.risk,C2.design,C2.aliasing].data` | Plug-in risk, design duality, and aliasing floors. |
| `fig6_2_learner_weights` | 6 | `U:checks[claim_id=C3.sensitivity,C3.allocation,C3.separation].data` | Learner sensitivities, allocation error and ratios, and scale separation. |
| `fig7_1_amplitude_law` | 7 | `U:checks[claim_id=C4.amplitude].{data,failures}` | Finite response, mean-field and quadratic laws; all 33 crossing settings including one FAIL. |
| `fig7_2_rankers` | 7 | `U:checks[claim_id=C5,C5b].data[0].{overall,v_share_bins,v_share_unavailable}` | Misranking, denominators, and ties for six rankers. |
| `fig8_1_stability` | 8 | `I:claims[id=C22].details.configurations; P:noisy_feature_duality.explicit_example; I:claims[id=C14]` | Tolerant duality and the cost of small feature scale. |
| `fig8_2_transport` | 8 | `theory/v015_checks/{mathematics,common}.py (read-only, with common’s committed C23 dependency); V:groups[id=V1].data; V:groups[id=E4].data` | Reproduced transport certificates, reached-state diagnostics, and uncovered states. |
| `fig8_3_adaptive_audit` | 8 | `I:claims[id=C16].details.configurations; V:groups[id=E7].data.{Hoeffding,empirical_Bernstein,variance_decomposition}` | Adaptive audit gains, replica cost, and variance sources. |
| `fig8_4_first_order_audit` | 8 | `V:groups[id=V9].data; V:groups[id=E12].data.{bins,summary}` | Concentration at fixed b (which can grow with dimension); E12 uses the oracle continuation gradient. |
| `fig8_5_certificate` | 8 | `U:checks[claim_id=C6,C6b].data; authenticated raw /checks/C6/1/raw_replicates/pairs and /checks/C6b/{1,3,5,7}/raw_replicates/pairs` | Zero first-order coverage, strict vs diagnostic finite certificates, variance, and full measurement cost. |
| `fig9_1_coupling` | 9 | `I:controlled_neural.details.{regrouping,additive_null_parameter_residual}; P:coupling_and_additive_null; I:claims[id=C17]` | Sequential regrouping effects and additive coupling width. |
| `fig9_2_selection` | 9 | `V:groups[id=E6].data.reports; V:groups[id=E9].data.{summary,cheap_proxy_results}` | Held-out proxy selection, threshold failures, and warm-up transfer. |

## Scope and data qualifications

- The hidden-bit curves store K=0…16, not the original requested 0…32; no extrapolation is drawn.
- Warm-up endpoints use actual stored values (moment order 3: 0.0769 → 0.0002 for R=1 and 0.1196 → 0.0033 for R=4). The noisy measured median is an early-phase diagnostic, not the stationary mean-field expectation. The affine bound is an empirical residual check.
- The first-order representation is a tool. Five nonnumeric exact-cell ratios are omitted, never zero-filled. Curvature estimates are empirical and smoothness is not certified. The signSGD reference guide has slope 2 while its selected fitted slope is approximately 3.
- The exact Adam statistic is the mean Adam increment. First and second gradient moments suffice only under the frozen-denominator approximation. The warm-Adam affine bound covers the linear second-moment contribution, not the full nonlinear measured gap.
- The concentration constant b can grow with dimension. E12 uses the oracle continuation gradient; its outside-margin oracle mixture is reconstructed by bin-row weighting and checked within 0.02 because gold-zero exclusions change denominators.
- The tested static proxies did not reliably select. The LESS-style cosine uses loss-gradient direction for SGD and actual updates for Adam; the update-oriented diagnostic flips SGD only. Ties remain separate from wrong signs; unavailable share rows stay in the overall denominator.
- The amplitude source remains FAIL: beta2=0.99, rho approximately 0.245, T=50 misses the 10% crossing criterion (10.925%). All 33 settings are accounted for: 26 bracketed roots and seven unbracketed in both finite searches. Unbracketed is not proof of no root.
- First-order certificates certify 0/1,584 eligible pairs. Finite replacement uses B=2 for strict range coverage. Observed-range coverage is diagnostic only. Short-surrogate calibration transfer can fail; wrong certifications are shown. Tail costs include calibration/curvature and use the prescribed M=4 full-audit reference on an M=8 panel (1,016 vs 1,154 replicas). Different accuracy criteria prevent a superiority claim based on costs alone.
- The additive coupling point has a schematic position because only its width is stored. Tiny signed SGD residuals and unsmoothed warmth measurements are preserved.

## Verification and outputs

Fixed metadata, deterministic plotting order, fixed jitter and timestamp-free PDF/PNG outputs are checked across two full regenerations. [verification.json](verification.json) records source hashes, versions, runtime, each output size/hash, and per-figure font minima. [text_audit.json](text_audit.json) records every text font plus visible-text boxes, intersections and clipping. No overlapping text pair is silently exempted.

| Figure | PNG bytes | PDF bytes | Min font pt | Min at 900 px | Text overlaps | Outside canvas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fig2_1_decomposition` | 491,882 | 17,825 | 12 | 15 | 0 | 0 |
| `fig2_2_queries` | 559,509 | 21,032 | 12 | 15 | 0 | 0 |
| `fig2_3_floor` | 617,856 | 72,700 | 12 | 15 | 0 | 0 |
| `fig3_1_static_vs_learning` | 516,076 | 20,354 | 12 | 15 | 0 | 0 |
| `fig3_2_response` | 341,973 | 24,438 | 12 | 15 | 0 | 0 |
| `fig4_1_duality` | 460,135 | 31,153 | 12 | 15 | 0 | 0 |
| `fig4_2_moment_order` | 498,470 | 28,054 | 12 | 15 | 0 | 0 |
| `fig4_3_rates` | 665,780 | 30,296 | 12 | 15 | 0 | 0 |
| `fig4_4_approximation` | 667,243 | 29,617 | 12 | 15 | 0 | 0 |
| `fig4_5_regime` | 563,467 | 35,508 | 12 | 15 | 0 | 0 |
| `fig5_1_linear_response` | 758,728 | 27,722 | 12 | 15 | 0 | 0 |
| `fig5_2_cold_adam` | 482,503 | 25,850 | 12 | 15 | 0 | 0 |
| `fig5_3_warmth` | 836,441 | 169,438 | 12 | 15 | 0 | 0 |
| `fig5_4_learner_statistics` | 1,101,160 | 32,821 | 12 | 15 | 0 | 0 |
| `fig6_1_information_bound` | 503,634 | 36,158 | 12 | 15 | 0 | 0 |
| `fig6_2_learner_weights` | 1,062,541 | 37,162 | 12 | 15 | 0 | 0 |
| `fig7_1_amplitude_law` | 1,103,463 | 31,658 | 12 | 15 | 0 | 0 |
| `fig7_2_rankers` | 1,209,753 | 20,021 | 12 | 15 | 0 | 0 |
| `fig8_1_stability` | 482,900 | 26,481 | 12 | 15 | 0 | 0 |
| `fig8_2_transport` | 845,824 | 56,968 | 12 | 15 | 0 | 0 |
| `fig8_3_adaptive_audit` | 539,680 | 22,012 | 12 | 15 | 0 | 0 |
| `fig8_4_first_order_audit` | 538,676 | 26,636 | 12 | 15 | 0 | 0 |
| `fig8_5_certificate` | 873,219 | 71,296 | 12 | 15 | 0 | 0 |
| `fig9_1_coupling` | 432,791 | 20,389 | 12 | 15 | 0 | 0 |
| `fig9_2_selection` | 695,236 | 37,380 | 12 | 15 | 0 | 0 |
