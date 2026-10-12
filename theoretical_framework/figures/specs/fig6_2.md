# Figure spec — learner weights (output `fig6_2_learner_weights`)

**Message.** Learner sensitivity changes how visitation matters. Water-filling improves the tested allocation errors. Scale invariance lets clipped SGD avoid the SGD aliasing floor.

**Layout.** Full-width rows, at most two subpanels per row, 10 inches wide. PDF vector, PNG 300 dpi. Base/labels 14 pt; titles 15 pt; ticks, legends and annotations at least 12 pt (15 px at 900 px width).

**Data.** Git HEAD `theory/v016_checks/results_v016.json`; no raw companion or rerun needed.

**Panel A — measured sensitivity.** `checks[claim_id=C3.sensitivity].data`, selecting `(regime,learner)` = `(frequent,SGD)`, `(frequent,Adam)`, `(isolated,Adam)`, `(short_bias_corrected,Adam)`, `(frequent,always_clipped_SGD)`. x=`d`, y=`mean`, pointwise 95% intervals `ci_low,ci_high`. Log-log axes. Legend lists `slope` and predicted slopes 1,1/2,1,1,1. `R`, `burn_in`, beta and correction settings differ across regimes; absolute heights are not matched-budget comparisons. The transition regime is not an asserted asymptotic series.

**Panel B — realized allocation errors.** `checks[claim_id=C3.allocation].data[0:16].arms.<arm>.mae.mean`, arms `water,uniform,proportional_d,SGD_weights`, vs profile index, log y. The irreducible final row is not an allocation profile.

**Panel C — predicted vs realized MAE ratios.** Same profiles: `ratios.<arm>.mae.{predicted,realized,ci_low,ci_high}`, arms `uniform,proportional_d,SGD_weights`, scatter with stored familywise intervals and y=x. Predictions use square roots of variance ratios, never variance ratios directly as MAE ratios.

**Panel D — scale-invariance separation.** `checks[claim_id=C3.separation].data[*].{K,clipped_mae.mean,SGD_worst_risk,aliasing_floor,clipped_RMS_upper}`. Log-log curves. Clipped error decreases; SGD remains above the 0.45 aliasing floor.

**Caption.** Measured sensitivities carry pointwise 95% intervals; legends give fitted and predicted slopes (1/2 for frequent Adam, 1 for the other shown regimes). Horizons and burn-in differ by regime as specified; absolute heights are not matched-budget comparisons. Water-filling uses the smallest realized MAE in these 16 profiles. Ratio predictions use square roots of variance ratios, with stored familywise intervals. Always-clipped SGD error shrinks with K while SGD retains the aliasing floor; this separation assumes state-owned coordinates and the stated scale model.
