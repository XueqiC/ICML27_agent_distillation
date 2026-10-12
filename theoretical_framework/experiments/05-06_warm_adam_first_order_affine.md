# Proposition 5.6 · Warm Adam is first-order affine in the consumed gradient

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-06_warm_adam_first_order_affine.md) · [Proof](../proofs/05-06_warm_adam_first_order_affine.md) · [Experiments](../experiments/05-06_warm_adam_first_order_affine.md) · [中文](05-06_warm_adam_first_order_affine_ZH.md)

**What is tested here.** [Check 5.9](../experiments/05-06_warm_adam_first_order_affine.md#c5-9) tests the bound on 2,496 law pairs with constants fixed in advance. [Check 5.13](../experiments/05-08_evolution_warmth.md#c5-13), whose full description is on the page of Proposition 5.8, follows the equal-moment gap over a training run.

**Verdict.**

- **Proposition 5.6 (warm Adam is first-order affine).** The bound holds on all 2,496 law pairs, with constants fixed in advance. The largest gap/bound is 0.988.

**Setting used on this page.**

The objects below are restated so that this page can be read on its own; the general notation is that of [Definitions 1.3–1.4](../README.md#s1-3).

- **State.** The complete learner state is $`z=(\theta,o)\in\mathcal Z\subseteq\mathbb R^p\times\mathbb R^q`$. Here $`\theta`$ are the trainable parameters, for example LoRA adapters, and $`o`$ is the optimizer state, for example Adam's moments $`m,v`$ and the step counter. When the counter matters it is written explicitly.
- **Feedback and gradients.** At round $`t`$ a feedback law $`P_t(\cdot\mid z)`$ emits a message $`Y`$. The learner computes a consumed gradient $`g=G_t(z,Y,\xi)\in\mathbb R^p`$ after loss normalization and accumulation, with $`\xi`$ the declared update randomness. Clipping, if any, is part of the optimizer.
- **Optimizer.** The optimizer is a measurable increment map $`z^+=z+\Delta_t(z,g)`$. The compared feedback laws share the optimizer and the gradient map and differ only in the message law. A change of optimizer is a different $`\Delta`$, compared in the same way.
- **Letters reused locally.** In this section $`s`$ counts gradient evaluations absorbed by Adam's second moment (Proposition 5.8). It is unrelated to the agent states $`s`$ of Section 6. The clipping threshold is $`c`$; the weight-decay coefficient is $`\lambda_{\rm wd}`$.

The learning kernel of a law $`P`$ is

```math
(K_t^Pf)(z)=\mathbb E_{Y\sim P_t(\cdot\mid z),\,\xi}\,f\big(z+\Delta_t(z,G_t(z,Y,\xi))\big).
```

For a law $`P`$ at $`(t,z)`$, the message-level mean increment is $`D_t(z,y)=\mathbb E_\xi\Delta_t(z,G_t(z,y,\xi))`$, and

```math
\mu_t^P(z)=\mathbb E_{Y\sim P}D_t(z,Y),\qquad m_{k,t}^P(z)=\mathbb E_{Y\sim P,\,\xi}\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert^k,\qquad \mathbb E_P\Delta^{\otimes j}=\mathbb E_{Y\sim P,\,\xi}\Delta_t(z,G_t(z,Y,\xi))^{\otimes j}.
```

- **Continuation.** $`V^P_{t+1}:\mathcal Z\to[0,1]`$ is the continuation value of protocol $`P`$: the later updates run under $`P`$, then the student is deployed and scored by $`F_E`$ ([Definition 1.4](../README.md#s1-4)). $`L(P)=\mathbb E\,F_E(Z_R^P)`$, and $`d_t^Q`$ is the law of $`Z_t`$ under protocol $`Q`$.
- **Norm.** A norm $`\lVert\cdot\rVert`$ on $`\mathbb R^{p+q}`$ is fixed, with dual norm $`\lVert\cdot\rVert_*`$.

**Adam and AdamW.** Adam is bias-corrected, with $`\epsilon`$ outside the square root, optional decoupled weight decay (AdamW) and no AMSGrad. An update with counter $`t\ge1`$ acts on $`z=(\theta,m,v,t-1)`$ and a consumed gradient $`G\in\mathbb R^p`$. Coordinatewise,

```math
m^+=\beta_1m+(1-\beta_1)G,\qquad v^+=\beta_2v+(1-\beta_2)G^{\odot2},\qquad n=\frac{m^+}{1-\beta_1^{\,t}},\qquad d=\sqrt{\frac{v^+}{1-\beta_2^{\,t}}},
```

```math
\theta^+=(1-\eta\lambda_{\rm wd})\theta-\eta\,\frac{n}{d+\epsilon}.
```

- The hyperparameters satisfy $`\beta_1\in[0,1)`$, $`\beta_2\in(0,1)`$, $`\eta>0`$, $`\lambda_{\rm wd}\ge0`$, $`\epsilon\ge0`$. Arithmetic is exact real arithmetic.
- The next complete state is $`z^+=(\theta^+,m^+,v^+,t)`$.
- A *cold* state has $`m=v=0`$ and $`t=1`$. A *warm* state has $`v_i>0`$ for every coordinate $`i`$.

*Warm-state notation ([Definition 5.4](../statements/05-04_warmth_frozen_denominator_surrogate.md#s5-4)).*

Consider an Adam update with counter $`t\ge1`$ at $`z=(\theta,m,v,t-1)`$, and assume $`v_i>0`$, or $`\epsilon>0`$ whenever some $`v_i=0`$.

- *Retained denominator, warmth, relative step.* For a consumed gradient $`G`$,

```math
a_i=\sqrt{\frac{\beta_2v_i}{1-\beta_2^{\,t}}},\qquad\kappa_i(G)=\frac{(1-\beta_2)G_i^2}{\beta_2v_i}\ \ (v_i>0),\qquad\kappa_i(G)=\infty\ \ (v_i=0),\qquad u_i(G)=\frac{\lvert n_i(G)\rvert}{a_i+\epsilon}.
```

- When $`v_i>0`$, $`d_i=a_i\sqrt{1+\kappa_i}`$ and $`v_i^+=\beta_2v_i\,(1+\kappa_i)`$. The warmth $`\kappa_i`$ is the relative perturbation that the new gradient makes to the retained second moment.
- *Frozen-denominator surrogate.* With the fixed diagonal preconditioner $`\Phi=\mathrm{diag}(1/(a_i+\epsilon))`$,

```math
\tilde\theta^+(G)=(1-\eta\lambda_{\rm wd})\theta-\eta\,\Phi\,n(G),\qquad\tilde\Delta(z,G)=\big(\tilde\theta^+(G)-\theta,\;m^+(G)-m,\;v^+(G)-v\big),\qquad\tilde z^+(G)=\big(\tilde\theta^+(G),\,m^+(G),\,\beta_2v,\,t\big).
```

- $`\tilde\theta^+`$ is one momentum-SGD step with the fixed preconditioner $`\Phi`$. $`\tilde\Delta`$ is the *frozen-denominator increment*: its $`(\theta,m)`$ blocks are affine in $`G`$ and its $`v`$ block is affine in $`G^{\odot2}`$. $`\tilde z^+`$ additionally freezes the second moment at $`\beta_2v`$; it is the comparison state of Proposition 5.6. For $`g\in\mathbb R^p`$, $`\tilde z^+(g)`$ is the same map evaluated at consumed gradient $`g`$.

**Status labels.** `PASS` means the implemented consistency assertions passed. `INFO` marks an empirical diagnostic: each cell gets its own PASS, FAIL or UNAVAILABLE status, and the group as a whole makes no pass/fail claim. Unless a check says otherwise, every network is the controlled small network of `v014:C23`: one-hot context, 4 tanh units, a sigmoid output, mean cross-entropy, float64. None of these networks is a real model.

**Statement under test ([Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6)).** Fix a round $`t\ge1`$, a state $`z=(\theta,m,v,t-1)`$ and a finite message set $`Y`$ with consumed gradients $`G(y)`$. Let $`V:\mathcal Z\to[0,1]`$ be the continuation value after the update (the declared reference tail followed by deployment, with tail randomness integrated), $`\psi(y)=V(z^+(y))`$ the one-boundary response, and $`F(g)=V(\tilde z^+(g))`$. Assume, with all states generated from $`z`$ by messages in $`Y`$ and the counter fixed at $`t`$:

- **(W)** $`v_i>0`$ for every coordinate (warm state);
- **(L)** $`\lvert V(\vartheta,m,v)-V(\vartheta',m,v)\rvert\le\sum_iL_i\lvert\vartheta_i-\vartheta_i'\rvert`$ on the pairs $`\big(\theta^+(y),\tilde\theta^+(y)\big)`$ at $`(m,v)=(m^+(y),v^+(y))`$;
- **$`(\mathrm S_v)`$** there are $`L_v\ge0`$ and a cap $`\bar k\in(0,\infty]`$ such that, at $`(\vartheta,m)=(\tilde\theta^+(y),m^+(y))`$, $`\lvert V(\vartheta,m,\beta_2v\odot(1+k))-V(\vartheta,m,\beta_2v)\rvert\le L_v\max_i\min\lbrace k_i,\bar k\rbrace`$ for $`k=\kappa(y)`$ (since $`V\in[0,1]`$ this always holds with $`L_v\bar k\ge1`$; the content lies in small $`L_v`$);
- **(H)** $`F`$ is twice continuously differentiable on the convex hull $`\mathcal C`$ of $`\lbrace 0\rbrace \cup\lbrace G(y):y\in Y\rbrace`$, and $`\lvert s^\top\nabla^2F(g)\,s\rvert\le H_G\lVert s\rVert_\infty^2`$ for all $`g\in\mathcal C`$ and $`s\in\mathbb R^p`$.

Let $`\psi_0=F(0)`$ and $`b=\nabla F(0)`$; both are independent of $`y`$. Then for every $`y\in Y`$

```math
\big\lvert\psi(y)-\psi_0-b^\top G(y)\big\rvert\le\eta\sum_iL_iu_i(y)\min\lbrace \tfrac{\kappa_i(y)}2,1\rbrace +L_v\max_i\min\lbrace \kappa_i(y),\bar k\rbrace +\tfrac12H_G\lVert G(y)\rVert_\infty^2=:r(y),
```

and for any two message laws $`P_0,P_1`$ on $`Y`$

```math
\big\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\big\rvert\le\min\Big\lbrace 1,\;\big\lvert b^\top(\mathbb E_{P_0}G-\mathbb E_{P_1}G)\big\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r\Big\rbrace .
```

- In particular, laws with equal mean consumed gradient differ in response by at most $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$.
- If $`V`$ is differentiable in $`(\theta,m)`$, then the first-order statistic is the mean preconditioned consumed gradient, with weight

```math
b=(1-\beta_1)\Big[-\frac{\eta}{1-\beta_1^{\,t}}\,\Phi\,\nabla_\theta V+\nabla_mV\Big]\Big|_{\tilde z^+(0)}.
```

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.9](../experiments/05-06_warm_adam_first_order_affine.md#c5-9) | [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6), [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11) | `v015:E8` (warm-affine part) | PASS |
| [Check 5.13](../experiments/05-08_evolution_warmth.md#c5-13) | [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6)–[5.8](../statements/05-08_evolution_warmth.md#s5-8) over a training run | `v015:E8` (`data.gaps`) | PASS |

<a id="c5-9"></a>

## Check 5.9 · Warm Adam is first-order affine in the consumed gradient (Proposition 5.6, Remark 5.11)

**Verdict.** On actual warm states of the controlled network, the bound holds for all 2,496 law pairs, with constants fixed before any pair was evaluated. The largest gap/bound is 0.988, and the largest pointwise residual/bound is 0.998.

**Why.** [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6) bounds response differences at a warm state:

- Two message laws with equal mean consumed gradient differ in response by at most $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$.
- Laws with unequal means differ by at most $`\lvert b^\top\Delta\mathbb EG\rvert+\mathbb E_{P_0}r+\mathbb E_{P_1}r`$.

**Setup** (`v015:E8`, warm-affine part).

- States: actual warm states of the network, with $`t`$ between 21 and 200.
- Trainable coordinates: 1–4 output-head coordinates; the rest are fixed.
- Messages: 2–6 per panel.
- Boundary and suffix:
  - one actual Adam boundary;
  - then an IID reference suffix of $`R\in\lbrace 0,2,6\rbrace`$ updates on all-zero or all-one labels, each with probability $`1/2`$;
  - all $`2^R`$ suffix histories are enumerated exactly.
- $`\eta\in\lbrace 0.01,0.1\rbrace`$. Deployment is the moment ($`N=3`$) or parity ($`N=2`$) return.
- Constants were estimated by predeclared central differences *before* any law pair was evaluated, and never enlarged afterwards:
  - $`L_i`$ along actual-to-surrogate segments;
  - $`L_v`$ along second-moment segments, with $`\bar k=1`$;
  - $`H_G`$ as the sum of absolute Hessian entries over radial and convex-hull samples.
- $`b`$ comes from analytic forward sensitivities and is checked against independent finite differences.
- Law pairs: random and adversarial pairs, with equal and with unequal means.

**Result** (`v015:E8`, warm-affine part).

| Quantity | Value |
|---|---|
| panels × law pairs | 24 × 104 = 2,496 |
| pairs violating the two-law bound | 0 |
| largest gap/bound over all pairs | 0.988 |
| largest pointwise $`\lvert\psi-c-b^\top G\rvert/r`$ | 0.998 |
| analytic $`b`$ against finite differences | agreement $`\le10^{-8}`$ in every panel |
| warmth of the tested states (maximum over coordinates and messages) | $`\max\kappa\le0.081`$ in every panel |
| curvature constant against step size | for example, moment, $`p=1`$, $`R=0`$: $`H_G=2.1\times10^{-6}`$ at $`\eta=0.01`$ and $`1.3\times10^{-4}`$ at $`\eta=0.1`$ |

Panel C of [Figure 5.3](../experiments/05-08_evolution_warmth.md#c5-11) plots the pointwise residual against its bound.

**Why it supports Proposition 5.6.**

- The bound holds and is nearly attained on actual network states, across suffix lengths and step sizes.
- The curvature constant grows roughly like $`\eta^2`$, as the curvature term of the bound anticipates.
- The warmth condition is checked as a maximum over coordinates and messages, not as a quantile. This is the form [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11) requires.
- Limit of the evidence: the constants are estimated, so this is empirical consistency, not a certificate. The proof supplies validity. The check confirms that the stated residual terms are the operative ones.

### [Check 5.13](../experiments/05-08_evolution_warmth.md#c5-13) · The cold window over a training run (Propositions 5.6–5.8)

**What it tests here.** This check is described in full on the page of Proposition 5.8. For Proposition 5.6 it tests: [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6)–[5.8](../statements/05-08_evolution_warmth.md#s5-8) over a training run.

**Verdict.**

- At short horizons, the gap between equal-moment sources is a cold-window effect. A lr = 0 moment warm-up of $`W=10`$ removes a median 99% of it at $`R=1`$ and 97.5% at $`R=4`$. For moment $`N=3`$, $`\eta=0.1`$ the gap falls from 0.0769 to 0.0004 at $`R=1`$, and from 0.1196 to 0.0016 at $`R=4`$.
- At $`R=40`$, gaps up to 0.049 remain for $`W\ge10`$. Where the residual is largest, the SGD comparator is nonzero as well (up to 0.011).

**Setup in brief.**

- The branches of [Check 5.7](../experiments/05-03_cold_start_adam_sign_quotient.md#c5-7) (actual Adam, frozen-Jacobian Adam, tangent affine SGD) are reused unchanged.
- Grid: 5 tasks (moment $`N=2,3,4`$; parity $`N=2,3`$) × $`\eta\in\lbrace 0.01,0.1\rbrace`$ × $`R\in\lbrace 1,4,40\rbrace`$ × $`W\in\lbrace 0,1,2,5,10,20,50\rbrace`$ = 210 cells.
- Histories are identical across $`W`$ and across branches. They are enumerated exactly when there are at most 4,096 of them; otherwise each source gets 2,000 independent runs.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`:
  - `v015:V8`, and `v015:E10` (per-cell records in `data.cells`; summaries in `data.optimizer_summary`; the cold-Adam cell in `data.C23_prediction`; the design in `data.design`);
  - `v015:V2`, `v015:V3`, `v015:V4`, `v015:E0`, `v015:E1`, `v015:E2`, `v015:E3`;
  - `v015:E8`: the frozen-denominator, warm-affine, cold-separation and warmth-trajectory parts (the last with `per_step` and `fits`), and the 210 `data.gaps` cells;
  - `v015:E11` (`data.clipped_mean`, `data.always_active`, `data.warm_Adam`, `data.Lion`, `data.LoRA`).
- `theory/v015_checks/c23_bundle.json`: exact laws, network, optimizer code and per-history raw steps.
- `theory/v015_checks/raw/E8_trajectories.npz`: per-replica warmth trajectories. It is gitignored and regenerated by the first command; its SHA-256 is recorded as `raw_sha256` in the warmth-trajectory part of `v015:E8`.
- `theory/v016_checks/results_v016.json`: claims `v016:C1.exact`, `v016:C1.moments` and `v016:C1.accumulation`. The full arrays are in `theory/v016_checks/results_v016_raw.json.gz`, which is gitignored; its SHA-256 is in `raw_artifact`.
- `theory/independent_checks/results_independent_v014.json`: `controlled_neural`.
- `docs/direction/v015_revision/results_v015.json`: package checks `pkg:T03`–`pkg:T08`.
- Code:
  - `theory/v015_checks/otlr.py` (`check_V8`, `check_E10`, `check_E11`);
  - `theory/v015_checks/warmth.py` (`check_E8`);
  - `theory/v016_checks/check_v016.py` (`v016:C1`);
  - specifications in `theory/v015_checks/README.md` and `theory/v016_checks/SPEC.md`.
- Figures: `../figures/fig5_1_linear_response`, `fig5_2_cold_adam`, `fig5_3_warmth` and `fig5_4_learner_statistics`, each as `.pdf` and `.png`. The specs are in `../figures/specs/fig5_1.md` to `fig5_4.md`.
