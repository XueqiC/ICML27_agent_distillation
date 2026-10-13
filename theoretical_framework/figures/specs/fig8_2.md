# Figure spec — fig8_2 · Frozen audit with transport

**Output:** `theoretical_framework/figures/fig8_2_transport.pdf` (vector) and `fig8_2_transport.png` (300 dpi). Layout: full-width rows, at most two subpanels per row, 10 inches wide. English labels. Drawn with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)) by `figures/make_figures.py`.

**Message:** the transport-aware certificate (39a) is never below the exact gap. Transport at reached neural states is nonzero, and measuring it makes the local bound valid. An uncovered state must be charged.

## Panel A — V1: exact gap vs certificate (2,000 random instances)

- **Data (recomputed; per-instance values are not stored in the committed JSON).** Import `theory/v015_checks/mathematics.py` and `common.py`, with the working directory set to `theory/v015_checks`, `CUDA_VISIBLE_DEVICES=''`, and `theory/.venv/bin/python`.
  - Construct `g = common.Group('V1', 'figure', 'figure')`. This seeds `g.rng` with `SeedSequence([150150, 1])`, exactly as `check_V1`.
  - Then call, 2,000 times: `score, record = mathematics.evaluate_markov(g, mathematics.markov_instance(g.rng))`.
  - Collect `float(record['gap'])` and `float(record['bound'])`. The values may be `Fraction` or str; parse with `fractions.Fraction`.
  - **Reproduction check:** `max(score)` must equal `groups[id=V1].data.worst_random.ratio` (0.7357142857…) in `theory/v015_checks/results_v015_independent.json`. If it does not, stop and report; do not plot.
- **Plot:** a scatter of x = bound (39a) and y = exact gap, with small markers (alpha 0.4). Draw the line y = x. Add the sharp witness as a highlighted marker at (1, 1), labelled "sharp transport witness (D=0, a=1)", from `data.sharp_transport_witness` (gap 1, bound 1).
- **Annotation:** "2,572 instances, 0 violations; max ratio 1".
- **Title:** "V1: (39a) ≥ exact gap".

## Panel B — E4: local bound vs gold contrast at 40 reached states

- **Data:** `theory/v015_checks/results_v015_independent.json`, `groups[id=E4].data.observations`. There are 40 rows with fields `gold_contrast`, `diagnostic_bound`, `anchor_prediction`, `a` and `alpha`.
- **Plot:** x = state index, sorted by `diagnostic_bound`. y on a log scale, with a floor of 1e-4. Plot three series:
  - `abs(gold_contrast)` (filled circles);
  - `diagnostic_bound` = B(D+a+2α)+2τ (line with markers);
  - `abs(anchor_prediction - gold_contrast)`, the zero-transport anchored error (open triangles).
- **Reference line:** dashed horizontal at 1e-3, labelled "prediction tolerance".
- **Annotation:** "|gold| ≤ bound at 40/40 states; zero-transport prediction misses 36/40".
- **Title:** "E4: transport must be measured".

## Panel C — Transport at reached states and the uncovered-state witness

- **Left half (strip plot):** E4 `observations[*].a` and `observations[*].alpha` as two jittered strips, with medians marked (0.0195 and 0.139). Draw a horizontal reference line at the median audited D_x (0.188; the anchors' `D` are 0.200, 0.169 and 0.188).
- **Right half (bars):** from `groups[id=V1].data.regression`. Show the exact gap (1.0); (39a) charging a=1 (`corrected_transport_bound`, 1.0296, drawn capped at 1 with a cap marker); (39a) charging exit ρ=1 (`corrected_exit_bound`, 1.0); and "transport term deleted" (`old_bound`, 0.0296), hatched and labelled "invalid without transport".
- **Title:** "Uncovered state: charge a or ρ".

**Caption.** (A) Exact gap versus the transport-aware certificate on 2,000 random finite training processes (V1); no point lies above y = x. The annotation summarizes all 2,572 committed V1 instances. (B) In a controlled network (E4), the measured-transport local bound contains the gold full-tail contrast at all 40 reached states, while zero-transport anchor predictions miss 36. (C) Transport and feature drift at reached states, and the uncovered-state witness. Exact finite computations and a controlled small network; not a real-model result.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../../references.md).

<a id="ref-hunter2007matplotlib"></a>

**John D. Hunter (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering* 9, no. 3, 90–95. DOI [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).
