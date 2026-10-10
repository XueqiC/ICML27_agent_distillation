# N12 · Optimizer warmth switches the first-order feedback statistic — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N12_optimizer_warmth.md) · [中文](N12_optimizer_warmth_ZH.md)

**Verdict.** N12 holds. Check group E8 passes on 1,443,382 instances with 10,825 assertions, and it confirms each part of the node.

- **Lemma W0** holds on 6,944 adversarially searched optimizer states. Its best lhs/rhs ratio is $1.0$, so the bound is tight.
- **Proposition W1** holds on all 2,496 law pairs, with constants fixed before any pair was evaluated. The largest gap-to-bound ratio is 0.988.
- **Proposition W2.** The cold separation equals $\lvert w\rvert\eta/2$ exactly. At warm states the gap, normalized by the preconditioned step scale, scales like $\kappa^{1.00}$, as the normalization term of W1 predicts.
- **Corollary W3.** Warmth decays like $1/(W+s)$ under near-stationary gradients: at $\eta=0.01$ and $W=0$ the median log-log slope over the 9 fitted task/source series is $-1.06$, with median $R^2=0.999$. Warm-up evaluations and real evaluations are interchangeable for warmth.
- **Cold window in the controlled network.** A lr = 0 moment warm-up of $W=10$ removes the cold-window gap at short horizons. For moment $N=3$, $\eta=0.1$, the gap falls from 0.0769 to 0.0004 at $R=1$ and from 0.1196 to 0.0016 at $R=4$.
- **Long horizons.** At $R=40$ warm-up no longer drives the gap to zero: gaps up to 0.049 remain with $W\ge10$. Where the residual is largest, the SGD comparator is nonzero as well (up to 0.011). This matches the warmth-independent curvature term of W1.

![Warmth controls the cold-window gap; warmth decays like 1/(W+s); W1 residual vs bound](../figures/N12_warmth.png)
[PDF](../figures/N12_warmth.pdf)

All numbers come from `theory/v015_checks/results_v015_independent.json → groups[id=E8]` (seed `[150150,108]`). This is a controlled neural setting, not Gemma.

## E8-W0 — coordinatewise surrogate error

**Why.** W0 is the only place where the true Adam denominator is replaced by the frozen one. If the bound $\eta u_i\min\lbrace \kappa_i/2,1\rbrace $ failed anywhere, W1 would fail with it.

**Setup.**
- Random states: $v$ log-uniform on $[10^{-12},1]$; random $m$, gradients and $t\in\lbrace 1,\dots,2000\rbrace $; five $(\beta_1,\beta_2)$ pairs including $(0.9,0.999)$.
- Search: 24 random restarts × 80 hill-climbing steps that maximize lhs/rhs.
- Arithmetic: the difference $\theta^+-\tilde\theta^+$ is computed in rationalized exact arithmetic, to avoid cancellation between rounded updates; direct float subtraction is also kept.
- Tolerance: relative slack $10^{-12}$ only.

**Result** (`data.W0`).

| Quantity | Value |
|---|---|
| States checked (`n_instances`) | 6,944 |
| Violations | 0 |
| Best lhs/rhs (`best_ratio`) | $1.0000000000000002$; witness in the saturated branch $\kappa\gg1$ |

**What this shows.** W0 is valid and cannot be improved by a constant factor. As $\kappa\to0$ the bound's first branch is also asymptotically tight, since $(1-(1+\kappa)^{-1/2})/(\kappa/2)\to1$.

## E8-W1 — warm Adam is first-order affine in the consumed gradient

**Why.** W1 is the central claim of the node. At a warm state, two message laws with equal mean consumed gradient differ in response by at most $\mathbb E_{P_0}r+\mathbb E_{P_1}r$. Laws with unequal means differ by at most $\lvert b^\top\Delta\mathbb EG\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r$.

**Setup.**
- States: actual warm states of the controlled C23 network ($t$ between 21 and 200).
- Trainable coordinates: 1–4 output-head coordinates; the rest are fixed.
- Messages: 2–6 per panel.
- Boundary and suffix: one actual Adam boundary, followed by an IID reference suffix of $R\in\lbrace 0,2,6\rbrace $ updates on all-zero or all-one labels, with probability $1/2$ each. All $2^R$ suffix histories are enumerated exactly.
- Grid: $\eta\in\lbrace 0.01,0.1\rbrace $; deployment is the C23 moment ($N=3$) or parity ($N=2$) return.
- Constants were estimated by predeclared central differences *before* any law pair was evaluated, and never enlarged afterwards:
  - $L_i$ along actual-to-surrogate segments;
  - $L_v$ along second-moment segments with $\bar k=1$;
  - $H_G$ as the sum of absolute Hessian entries over radial and convex-hull samples.
- $b$: computed by analytic forward sensitivities and checked against independent finite differences.
- Law pairs: random and adversarial pairs with equal and with unequal means.

**Result** (`data.W1`).

| Quantity | Value |
|---|---|
| Panels × law pairs | 24 × 104 = 2,496 |
| Pairs violating the two-law bound | 0 |
| Largest gap/bound over all pairs (`best_ratio`) | 0.988 |
| Largest pointwise $\lvert\psi-c-b^\top G\rvert/r$ (`panels[*]`) | 0.998 |
| Analytic $b$ vs finite differences | agreement $\le10^{-8}$ in every panel |
| Warmth of the tested states | $\max\kappa\le0.081$ in every panel |
| Curvature constant vs step size | e.g. moment, $p=1$, $R=0$: $H_G=2.1\times10^{-6}$ at $\eta=0.01$ and $1.3\times10^{-4}$ at $\eta=0.1$ |

**What this shows.** The bound of W1 holds and is nearly attained (0.988) on actual network states, across suffix lengths and step sizes. The curvature constant grows roughly like $\eta^2$, as Remark 2 of the proof anticipates.

This is empirical consistency with estimated constants, not a certificate. The proof supplies the validity, and the check confirms that the stated residual terms are the operative ones.

## E8-W2 — cold separation and the warm scaling

**Why.** W2 says that a cold state separates equal-mean laws at first order in $\eta$. W1 says that a warm state shrinks the same separation in proportion to the warmth.

**Setup.**
- Cold: the exact rational laws of W2(a), with $\eta\in\lbrace 0.01,0.1\rbrace $ and $w\in\lbrace -0.7,1\rbrace $, under the ideal sign map and under $\epsilon=10^{-8}$.
- Warm: the same unscaled laws at $t=1000$ and $m=0$, sweeping $v$ so that $\kappa_{\max}$ ranges over 33 log-spaced values in $[10^{-6},10^2]$.

**Result** (`data.W2`).

| Quantity | Value |
|---|---|
| Cold gap, $\eta=0.01$, $w=-0.7$ | ideal $7/2000=0.0035$; $\epsilon=10^{-8}$: $0.00349999986$ |
| Cold gap, $\eta=0.1$, $w=1$ | ideal $1/20$; $\epsilon=10^{-8}$: $0.0499999980$ |
| Warm: log-log slope of raw gap vs $\kappa$ (`loglog_slope`) | 1.4995 |
| Warm: slope after normalizing by the preconditioned step scale (`normalized_loglog_slope`) | 0.9995 |

**What this shows.** The cold gap is exactly $\lvert w\rvert\eta/2$, which is first order in $\eta$. At a warm state, changing $v$ changes both $\kappa$ and the step scale $u\propto\kappa^{1/2}$. The raw gap therefore scales like $\kappa^{3/2}$, and the normalized gap like $\kappa^{1}$. That is the $\eta\,u\,\kappa$ form of the W1 normalization term.

## E8-W3 — how warmth evolves; warm-up is interchangeable with real steps

**Why.** Corollary W3 rests on the mean-field rate $\bar\kappa(s)\in[1/(\beta_2s),1/(\beta_2^ss)]$. The step to realized warmth, $\kappa\approx c/(W+s)$, is the one approximation in the node, so it must be tested on actual trajectories.

**Setup.**
- Trajectories: actual Adam trajectories of the controlled network (moment and parity tasks; $\eta\in\lbrace 0.01,0.1\rbrace $; 128 replicas).
- Warm-up: $W\in\lbrace 0,1,2,5,10,20,50\rbrace $ lr = 0 steps on the reference law. Every warm-up step preserves the parameter bytes (asserted).
- Recorded: quantiles of coordinate warmth at every step.
- Fit: median $=c/(W+s)$ for $s=1,\dots,79$, with a free log-log slope.

**Result** (`data.W3`).

| Setting | Median slope | Median $R^2$ |
|---|---|---|
| $\eta=0.01$, $W=0$ | $-1.06$ | 0.999 |
| $\eta=0.01$, $W\in\lbrace 5,10,20,50\rbrace $ | $-1.07$ to $-1.14$ | 0.95–0.99 |
| $\eta=0.1$, $W=0$ | $-1.18$ | 0.94 |
| $\eta=0.1$, $W\in\lbrace 10,20,50\rbrace $ | $-1.55$ to $-1.93$ | 0.55–0.68 |

Interchangeability: moment $N=3$, $\eta=0.01$, source $P$, median $\kappa$ at an equal prior evaluation count $W+s$ (`data.W3.per_step`):

| $W+s$ | $W=0$ | $W=5$ | $W=20$ | $W=50$ |
|---|---|---|---|---|
| 51 | 0.00661 | 0.00642 | 0.00619 | 0.00619 |
| 79 | 0.00405 | 0.00395 | 0.00413 | 0.00379 |

**What this shows.**
- Under near-stationary gradients ($\eta=0.01$), warmth decays like $1/(W+s)$, as W3(a) predicts.
- At $\eta=0.1$ the consumed gradients shrink as training converges, and warmth decays faster than $1/(W+s)$. The hypothesis $\kappa\le c/(W+s)$ of W3(b) is then met with room to spare.
- Warmth depends on the total number of absorbed gradient evaluations, not on whether they came from a warm-up or a real update.

## E8 gap sweep — the cold window in a real network

**Why.** W3(b) predicts how the response gap between equal-moment sources behaves in a full training run.
- The warmth-controlled part concentrates in the first updates and is removed by warm-up.
- The curvature part does not depend on warmth and accumulates with the horizon.

**Setup.**
- Code: the E2 code (actual Adam, frozen-Jacobian Adam, tangent affine SGD) is reused unchanged.
- Grid: 5 tasks (moment $N=2,3,4$; parity $N=2,3$) × $\eta\in\lbrace 0.01,0.1\rbrace $ × $R\in\lbrace 1,4,40\rbrace $ × $W\in\lbrace 0,1,2,5,10,20,50\rbrace $ = 210 cells.
- Histories: identical across $W$ and branches. They are enumerated exactly when there are at most 4096, otherwise 2000 independent runs per source.
- Warm-up: one declared IID prefix from the reference law.

**Result** (`data.gaps`).

| Horizon | Gap at $W=0$ (max over cells) | Gap at $W\ge10$ (max over cells) | $\lvert$gap$(W{=}10)\rvert/\lvert$gap$(W{=}0)\rvert$ | SGD comparator (max) |
|---|---|---|---|---|
| $R=1$ | 0.0769 | 0.0006 | median 0.009, max 0.034 (5 cells with gap > $10^{-3}$) | 0.0034 over $R\le4$ |
| $R=4$ | 0.1196 | 0.0064 | median 0.025, max 0.226 (7 cells) | (same) |
| $R=40$ | 0.0436 | 0.0487 | not driven to zero | 0.0110 |

Headline cells:

| Cell | $W=0$ | $W=1$ | $W=2$ | $W=5$ | $W=10$ | $W=20$ | $W=50$ | tangent SGD |
|---|---|---|---|---|---|---|---|---|
| moment $N=3$, $\eta=0.1$, $R=1$ | 0.0769 | 0.0460 | 0.0311 | 0.0032 | 0.0004 | 0.0004 | 0.0002 | 0.0000 |
| moment $N=3$, $\eta=0.1$, $R=4$ | 0.1196 | 0.1016 | 0.0881 | 0.0206 | 0.0016 | 0.0037 | 0.0033 | 0.0000 |
| parity $N=2$, $\eta=0.1$, $R=40$ | 0.0436 | 0.0477 | 0.0441 | 0.0488 | 0.0487 | 0.0391 | 0.0391 | 0.0110 |

With $W\in\lbrace 1,2\rbrace $, $\kappa\approx1/(W+1)$ is still of order one. W1 then gives no control, and the gap can move either way; for example, moment $N=2$, $\eta=0.1$, $R=1$ goes from 0.0175 to $-0.0194$ at $W=1$. For $W\ge10$ the short-horizon gaps are uniformly small.

**What this shows.**
- At short horizons, the gap between equal-moment sources is a cold-window effect, and a lr = 0 moment warm-up of $W=10$ removes a median 99% of it at $R=1$ and 97.5% at $R=4$, as W1–W3 predict.
- At long horizons, warm-up no longer drives the gap to zero. The largest residuals, parity $N=2$ and moment $N=2$ at $\eta=0.1$, stay at 0.039–0.049 and 0.005–0.036 for every $W\ge10$. The SGD comparator is nonzero in those cells (0.011 and 0.0067). The residual there is the curvature/trajectory term of W1 accumulated over 40 updates, which SGD shares; it is not a cold-start effect.
- For few-shot runs, warmth decides which statistic a feedback source must match in the first updates. Over long horizons, nonlinearity of the trajectory is what remains.

(Whether warm-up changes the predictive value of cheap proxies is a method question. It is answered in node [N17](N17_ced_method.md).)

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E8 --output theory/v015_checks/quick/e8.json
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json → groups[id=E8]`: W0, W1 panels, W2 sweep, W3 `per_step` and `fits`, and the 210 `gaps` cells.
- `theory/v015_checks/figs/E8.png`, `E8_warmth.png`, `E8_W2.png`: the original panels.
- `theory/v015_checks/raw/E8_trajectories.npz`: per-replica trajectories (gitignored; regenerated by the command above; SHA-256 recorded in `data.W3.raw_sha256`).
- Code: `theory/v015_checks/warmth.py` (`check_E8`).
