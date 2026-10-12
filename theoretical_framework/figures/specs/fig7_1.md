# Figure spec — amplitude law (output `fig7_1_amplitude_law`)

**Message.** The mean-field formula tracks the finite response over the stored amplitude grid; the quadratic law is local. A failed boundary crossing check remains visible.

**Layout.** Full-width rows, at most two subpanels per row, 10 inches wide. PDF vector, PNG 300 dpi. Base/labels 14 pt; titles 15 pt; ticks, legends and annotations at least 12 pt (15 px at 900 px width).

**Data.** Git HEAD `theory/v016_checks/results_v016.json`; all required maps are in the summary. No checks rerun and no raw companion needed.

**Response panels.** `checks[claim_id=C4.amplitude].data[0,7,21,27]`: x=`lambdas`; measured finite response=`rollout.mean` with `rollout.ci_low,rollout.ci_high` (95%, pointwise); exact mean-field=`mean_field`; quadratic law=`predicted`. The last row is the failed boundary setting. Titles use `beta2,rho,T`. Plot every amplitude, including negative values and ineligible quadratic points. Do not substitute the different first-order quantity `val.mean` for the finite rollout.

**Crossing panel.** All 33 `checks[claim_id=C4.amplitude].data` rows: x=`mean_field_zero_crossing`, y=`measured_rollout_zero_crossing`. Plot the 26 numeric pairs and explicitly count seven null/null pairs as “unbracketed in both searches”. The latter remain accounted for, never zero-filled or described as globally root-free. Draw y=x and the ±10% criterion band `x/1.1 <= y <= x/0.9`, matching the stored convention `abs(predicted/measured-1) <= 0.1`. Highlight `checks[claim_id=C4.amplitude].failures[0].case=27`: beta2=.99, rho≈.245, T=50, crossing error .1092509814, predicted 68.6180 vs measured 77.0341. Label FAIL. Do not relax tolerance or hide the row.

**Context.** `C4.costate` is the independent adjoint check and `C4.ranking` retains separate first-order and finite-response ranking results. They are read as context, not confused with the amplitude response. The optional costate histogram is omitted.

**Caption.** Four representative finite-response sweeps compare measured Delta F(lambda) with pointwise 95% intervals, the exact stationary mean-field formula, and the local quadratic law. The quadratic law is only local; the exact formula is mean-field, not an exact stochastic rollout identity. All 33 settings are accounted for: 26 bracketed roots and 7 unbracketed in both bounded searches. The single FAIL (beta2=0.99, rho approximately 0.245, T=50) has 10.925% crossing error against a 10% threshold. The shaded criterion is abs(predicted/measured - 1) <= 0.10. Unbracketed searches do not prove global nonexistence.
