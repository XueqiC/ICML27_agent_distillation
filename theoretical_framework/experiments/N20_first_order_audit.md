# N20 · Measuring the first-order statistic at the actual states — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N20_first_order_audit.md) · [中文](N20_first_order_audit_ZH.md)

**Verdict.** N20 holds.

**V9 — dimension-free estimation (Theorem C5).** 144 cells × 20,000 trials, 2,880,000 instances, all PASS.

- Every empirical failure rate is far below its nominal $\delta$: at most 0.00095 at $\delta=0.05$ and 0.00765 at $\delta=0.2$.
- The realized deviation quantile never exceeds 0.615 of the bound.
- That ratio does not grow from $p=1$ to $p=100{,}000$ parameters: it is 0.615 at $p=1$ and 0.574 at every larger $p$.

**E12 — selection by first-order margin (Theorem C6), on the frozen E6 held-out panels.**

- On the 56 rows where the first-order margin exceeds the remainder bound, the oracle first-order score $\langle\nabla V(x),\mu^{\rm cand}-\mu^{\rm ref}\rangle$ has **zero wrong signs**, as C6 requires.
- On the same 56 rows, every static proxy of E6 is wrong 35.7% of the time; random selection is wrong 53.6%. Inside the margin, what those proxies lack is the continuation gradient at the actual state, condition (i) of Corollary C6′.
- Outside the margin, even the oracle is wrong on 28.7% of eligible rows. Measured four-update tails are wrong on 3.0% of those rows. Higher-order tail effects decide there, which is where node [N17](N17_ced_method.md) prescribes measured tails.

![Dimension-free estimation and the first-order margin](../figures/N20_first_order_audit.png)

[PDF](../figures/N20_first_order_audit.pdf)

## V9 · Dimension-free concentration of the mean increment

**Why.** C5 claims that the empirical mean increment concentrates at rate $b(1+\sqrt{2\log(1/\delta)})/\sqrt n$ in any dimension. LoRA adapters have tens of millions of parameters. A dimension-dependent failure would show as a failure rate above $\delta$, or as a realized quantile that rises toward the bound as $p$ grows.

**Setup.**

- Dimensions $p\in\lbrace1,10,1000,100000\rbrace$, sample sizes $n\in\lbrace8,32,128\rbrace$, confidence $\delta\in\lbrace0.05,0.2\rbrace$, and one- and two-sample statistics.
- Three bounded laws with $b=1$ (`data.laws`):
  - *sign:* a biased scalar sign times a fixed unit sign vector, the normalized sign-type increment $\eta=1/\sqrt p$, with different means under $P$ and $Q$;
  - *clipped Gaussian:* $N(0,I/p)$ clipped at norm $1$;
  - *sparse spikes:* a uniformly chosen coordinate with a fair sign, norm $1$.
- High-dimensional resultants are sampled exactly, by isotropy and a Beta-angle recursion, without forming $p$-vectors (`data.implicit_sampling`).
- 20,000 independent trials per cell; seed `SeedSequence([150150, 9])`.

**Procedure.** In each cell:

1. record the fraction of trials whose deviation exceeds the bound, and assert it is at most $\delta$ plus four binomial standard errors;
2. record the empirical $(1-\delta)$ quantile of the deviation divided by the bound.

**Result** (`groups[id=V9].data.rows`):

| $p$ | cells | worst failure rate at $\delta=0.05$ | worst failure rate at $\delta=0.2$ | lowest coverage | largest quantile/bound |
|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 0.00095 | 0.00765 | 0.99235 | 0.615 |
| 10 | 36 | 0.00030 | 0.00315 | 0.99685 | 0.574 |
| 1,000 | 36 | 0.00050 | 0.00355 | 0.99645 | 0.574 |
| 100,000 | 36 | 0.00040 | 0.00335 | 0.99665 | 0.574 |

By law, the largest quantile/bound is 0.574 (sign), 0.408 (clipped Gaussian) and 0.615 (sparse spikes); the overall maximum is `data.max_quantile_over_bound` = 0.615. For the one-sample clipped Gaussian at $\delta=0.05$, the ratio falls with $p$, from 0.406 to 0.291, at every $n$.

**What this shows.** The bound is valid and conservative by a factor of at least 1.6 in every cell, and nothing in it deteriorates with dimension. The bound is dimension-free for a fixed increment bound $b$. For unnormalized sign-type increments $b=\eta\sqrt p$, and the first-order differences have the same scale, so the relative accuracy is still set by $1/\sqrt n$ (`data.caveat`).

## E12 · The first-order margin on the frozen E6 panels

**Why.** C6 predicts that a first-order score ranks two feedback laws correctly whenever its margin exceeds the remainder bound. Corollary C6′ predicts why the cheap proxies of node [N17](N17_ced_method.md) (check E6) failed:

- inside the margin they lack the continuation gradient, condition (i);
- outside the margin no first-order score is guaranteed, condition (ii).

E12 tests both predictions on the exact rows on which E6 evaluated those proxies, without refitting anything.

**Setup.**

- The E6 held-out panels and frozen proxy models, read unchanged; their SHA-256 hashes match before and after (`data.frozen_sha256_before`, `_after`).
- Each row is one candidate update at a learner state, followed by the declared seven reference updates, with the exact full-tail gold contrast.
- The oracle score is $\langle\nabla V^0(x),\mu^{\rm cand}-\mu^{\rm ref}\rangle$, with $\nabla V^0$ the exact continuation gradient of that row's reference tail.
- The remainder bound uses sampled, unpadded segment curvature.
- The 36 rows of the threshold task have a discontinuous return. They violate S$_1$ and are ineligible for C6.
- Seed `SeedSequence([150150, 112])`.

**Procedure.**

1. Compute the oracle score and the remainder bound for each row.
2. Split rows by the ratio $\lvert$score$\rvert$ / bound.
3. Report wrong-sign rates of the oracle, of every frozen E6 proxy, of measured short tails and of random selection, overall, per split and per ratio bin.
4. Assert zero oracle wrong signs where the ratio exceeds 1.

Following E6, a zero prediction does not count as a wrong sign, and only rows with nonzero gold enter the rates.

**Result — overall** (`groups[id=E12].data.summary[split=overall]`):

| Quantity | Value |
|---|---:|
| rows / eligible (smooth) rows | 432 / 396 |
| rows with margin $>$ remainder bound | 56 (14.1% of eligible) |
| oracle wrong-sign rate inside the margin | **0** |
| oracle wrong-sign rate, all eligible rows | 0.287 |
| static proxies, inside the margin (input moment, mean gradient, sign, $\Delta\theta$, $\Delta\theta$ + $m,v$) | 0.357 each |
| measured short tail of 1 / 4 updates, inside the margin | 0 / 0 |
| short tail of 1 / 4 updates, outside the margin | 0.203 / 0.030 |
| random selection, inside the margin | 0.536 |
| full-tail gold | 0 |

**Result — by ratio $\lvert$oracle score$\rvert$ / remainder bound** (`data.bins`):

| ratio | rows | oracle | mean gradient | $\Delta\theta$ + $m,v$ | short tail 1 | short tail 4 | random |
|---|---:|---:|---:|---:|---:|---:|---:|
| $[0,0.25)$ | 316 | 0.371 | 0.371 | 0.485 | 0.221 | 0.033 | 0.533 |
| $[0.25,0.5)$ | 12 | 0 | 0.250 | 0.083 | 0 | 0 | 0.333 |
| $[0.5,1)$ | 12 | 0 | 0 | 0 | 0 | 0 | 0.667 |
| $[1,2)$ | 8 | 0 | 0 | 0 | 0 | 0 | 0.625 |
| $[2,\infty)$ | 48 | 0 | 0.417 | 0.417 | 0 | 0 | 0.521 |

**Result — by split** (`data.summary`):

| split | eligible rows | margin rows | oracle wrong signs inside / overall |
|---|---:|---:|---|
| held-out states | 72 | 16 | 0 / 0.234 |
| held-out $\epsilon$ | 72 | 16 | 0 / 0.281 |
| held-out learning rate | 72 | 0 | — / 0.313 |
| held-out $\epsilon$ and learning rate | 72 | 0 | — / 0.313 |
| fresh → warm | 72 | 16 | 0 / 0.234 |
| warm → fresh | 36 | 8 | 0 / 0.406 |
| held-out threshold task | 0 (36 ineligible) | — | — |

**What this shows.**

- **C6 holds.** Wherever the first-order margin exceeds the remainder bound, the oracle first-order score never ranks wrongly, in each of the four splits that contain margin rows. It is also never wrong once the ratio passes 0.25, so the empirical remainder bound is conservative.
- **Inside the margin, static proxies fail for lack of $\nabla V$.** On the 48 rows with ratio $\ge2$, where first order decides, the static proxies are wrong 41.7% of the time and the oracle 0%. They use the same update statistics without the state- and task-specific continuation gradient, so they violate condition (i) of C6′.
- **Outside the margin, the tail decides.** On the 316 rows with ratio $<0.25$, even the oracle is wrong 37.1% of the time. Measured four-update tails are wrong 3.3% of the time. This regime calls for the measured short tails of N17, and for full tails on threshold tasks, where S$_1$ fails.
- **The margin is rare in this panel.** Only 14% of eligible rows meet it, and none in the held-out learning-rate splits, whose step sizes are ten times those of the pilot recipes (Adam $\eta=0.1$, SGD $\eta=1$; `theory/v015_checks/proxy_split_freeze.json`, `recipe_shift_axes`). This matches the remainder orders of node [N18](N18_optimizer_transformed_response.md).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V9,E12 --output theory/v015_checks/quick/n20.json
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: `groups[id=V9].data` (`rows`, `laws`, `max_quantile_over_bound`, `caveat`) and `groups[id=E12].data` (`summary`, `bins`, `cell_audits`, `limitations`, frozen hashes).
- Frozen inputs used unchanged: `theory/v015_checks/proxy_split_freeze.json` and `proxy_models_frozen.json`.
- Code: `theory/v015_checks/otlr.py` (`check_V9`, `check_E12`).
- Suite figures: `theory/v015_checks/figs/V9.png`, `figs/E12.png` (and `.svg`).
- Figure: `../figures/N20_first_order_audit.pdf` and `.png` (spec `../figures/specs/N20.md`).
