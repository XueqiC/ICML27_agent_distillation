# Figure spec — fig5_4 · What each learner makes the feedback carry

**Output:** `theoretical_framework/figures/fig5_4_learner_statistics.pdf` (vector) and `fig5_4_learner_statistics.png` (300 dpi). Layout: full-width rows, at most two subpanels per row, 10 inches wide. English labels. Drawn by `figures/make_figures.py` with Matplotlib ([Hunter, 2007](#ref-hunter2007matplotlib)).

**Message:** The exact Adam statistic is the mean Adam increment; first and second gradient moments suffice only under the frozen-denominator approximation. The optimizer decides what the feedback must carry. Clipping ([Pascanu et al., 2013](#ref-pascanu2013difficulty)) makes the clipped mean (or, when always active, only the direction) the statistic. Warm Adam adds the second moment through $`v`$. Momentum-dominated Lion ([Chen et al., 2023](#ref-chen2023symbolic)) makes the parameter step blind to the message. LoRA ([Hu et al., 2022](#ref-hu2022lora)) at initialization sees only a rank-$`r`$ sketch.

All data come from `theory/v015_checks/results_v015_independent.json`, `groups[id=E11].data`.

## Panel A — "Clipped SGD: the clipped mean decides"

- **Data:** `clipped_mean.rows[*]` with fields `eta`, `gap`, `first_order` and `remainder`.
- **Plot:** log–log. `gap` as filled circles, `first_order` as a solid line, `remainder` as open squares, against `eta`. Add a dotted slope-2 guide through the remainder points.
- **Annotation:** "raw means equal; clipped means differ by −1/2" and "remainder slope " + `clipped_mean.remainder_slope` (3 decimals).

## Panel B — "Always-active clipping: only the direction matters"

- **Data:** `always_active.rows[*]` (`eta`, `gap`, `first_order`).
- **Plot:** log–log `gap` vs `eta` with markers, plus a slope-2 guide. Print "first-order term = 0 at every η" as text, because `first_order` is identically 0 and cannot be drawn on a log axis.
- **Annotation:** "gap slope " + `always_active.slope` (5 decimals).

## Panel C — "Warm Adam: E[G²] enters through v"

- **Data:** `warm_Adam.first_order_blocks` (three numbers: θ, m, v), `warm_Adam.predicted_v_term`, `warm_Adam.measured_gap`, `warm_Adam.W1_Lv_kappa_bound`.
- **Plot:** vertical bars on a linear y-axis in units of $`10^{-6}`$: "θ block", "m block", "v block", "measured gap". Draw the predicted v term as a hollow diamond on the v bar, and the warm-Adam affine bound as a dashed horizontal line labelled "warm-Adam affine bound".
- **Annotation:** "equal gradient means (δ₁ = 0)", computed as `max |warm_Adam.delta1|`.

## Panel D — "Lion: momentum dominance blinds the parameter step"

- **Data:** `Lion[0]` (dominated) and `Lion[1]` (control). In each, `increment_difference` is the concatenation of the θ, m and v blocks, each of length `len/3`.
- **Plot:** grouped bars per state ("momentum-dominated m=0.2", "control m=0.005") of `max |·|` over the θ block and over the m block. Use a log y-axis with floor 1e-4, and print "0 (exact)" above the dominated θ bar.
- **Annotation:** the flip fraction for each state, computed as the mean of `flip_probability`.

## Panel E — "LoRA at B=0: only the sketch ΓA₀ᵀ matters"

- **Data:** `LoRA[*]` (rank 1 and rank 2); `rows[*]` with `eta`, `gap`, `range_gap`, `range_first_order`.
- **Plot:** log–log against `eta`:
  - the null-space pair `|gap|` as solid lines, one per rank;
  - the range pair `range_gap` as dashed lines, with `range_first_order` as markers on them.
- Add slope guides 1 (range) and 2 (null).
- **Annotation:** "null-space slope " + `null_gap_slope` per rank, and "A unchanged: " + count of `A_bitwise_audit[*].unchanged` / total per rank.

## Panel F — summary table (text only)

A compact two-column text table: learner → first-order statistic.

| Learner | Statistic |
|---|---|
| SGD, momentum | E G |
| Adam (exact) | mean Adam increment |
| Adam (frozen denominator only) | E G and E G² |
| cold Adam | sign law; E G, E G² in m, v |
| clipped SGD | clipped mean (direction if always active) |
| Lion | flip probability |
| LoRA, step 1 | sketch ΓA₀ᵀ (or its sign law) |

Use text ink and no colors.

**Caption.** Learner-specific first-order statistics. The exact Adam statistic is the mean Adam increment; gradient first and second moments suffice only under the frozen-denominator approximation. (A) Equal raw means, different clipped means: first-order gap under clipping. (B) Always-active clipping: equal mean directions give an O(eta²) gap. (C) Warm Adam: with equal gradient means, the first-order term is carried entirely by the v block; the warm-Adam affine bound covers this linear contribution, not the entire measured gap. (D) Momentum-dominated Lion: the parameter increment does not depend on the message. (E) LoRA at initialization: null-space differences are second order, range differences first order. Controlled small networks; not a real-model result.
