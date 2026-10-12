# Figure spec — rankers (output `fig7_2_rankers`)

**Message.** Finite paired replacement improves point ranking on this panel; oracle first-order scores still fail. Ranking accuracy does not establish certification.

**Layout.** Full-width rows, at most two subpanels per row, 10 inches wide. PDF vector, PNG 300 dpi. Base/labels 14 pt; titles 15 pt; ticks, legends and annotations at least 12 pt (15 px at 900 px width).

**Data.** Git HEAD `theory/v016_checks/results_v016.json`; summary only, no rerun.

**Panels and exact paths.** `checks[claim_id=C5b].data[0].overall` and `checks[claim_id=C5b].data[0].v_share_bins[*].rankers`, with bin endpoints `lo,hi`. Six keys: `less_cosine`, `less_utility_orientation`, `parameter_only`, `full_state`, `measured_aligned`, `paired_replacement`. Bars use `wrong_rate`; each label prints `wrong/available` and `ties`. Verify the first five overall rows equal `checks[claim_id=C5].data[0].overall` exactly. Overall is 1,687 available nonzero-gold pairs; the stored denominator includes split multiplicities. `v_share_unavailable` rows remain in the overall panel and are counted in the caption.

**Naming caveat.** The stored “LESS-style cosine” is loss-gradient-oriented for SGD but actual-update-oriented for Adam. Its label/footnote states this mixed convention. The update-oriented cosine (`less_utility_orientation`) reverses SGD only. Do not invent an all-optimizer loss-gradient cosine absent from the data. Parameter-only and full-state scores use oracle continuation gradients. CRN first order uses the aligned surrogate. Finite replacement uses the aligned surrogate at r=256. These proxies are not the official LESS implementation.

**Caption.** LESS-style cosine uses positive loss-gradient direction for SGD and actual parameter updates for Adam. The update-oriented cosine reverses SGD orientation, leaving Adam unchanged; these are proxies, not official LESS reproduction. Parameter-only and full-state first-order scores use oracle continuation gradients; CRN first order uses the aligned surrogate. Finite paired replacement uses the aligned surrogate at r=256. Bars report wrong signs, denominators and ties separately, preserving the stored convention. Second-moment-channel shares are oracle diagnostics used only for reporting; 165 rows have unavailable shares and remain in the overall panel. Ranking accuracy is not certification.
