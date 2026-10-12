# Proposition 5.7 · Cold steps separate equal-mean feedback at first order

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-07_cold_steps_separate_equal_mean.md) · [Proof](../proofs/05-07_cold_steps_separate_equal_mean.md) · [Experiments](../experiments/05-07_cold_steps_separate_equal_mean.md) · [中文](05-07_cold_steps_separate_equal_mean_ZH.md)

**What is tested here.** [Check 5.10](../experiments/05-07_cold_steps_separate_equal_mean.md#c5-10) verifies the cold gap $`\lvert w\rvert\eta/2`$ exactly and measures how the normalized gap scales with the warmth at warm states.

**Verdict.**

- **Proposition 5.7 (cold steps separate equal-mean feedback).** The cold gap equals $`\lvert w\rvert\eta/2`$ exactly. At warm states, the normalized gap scales like $`\kappa^{0.9995}`$.

**Setting used on this page.**

The objects below are restated so that this page can be read on its own; the general notation is that of [Definitions 1.3–1.4](../README.md#s1-3).

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

**Sign map.** For $`\epsilon>0`$, $`h_\epsilon(g)_i=g_i/(\lvert g_i\rvert+\epsilon)`$. For $`\epsilon=0`$, the *ideal sign update* is $`h_0(g)_i=\mathrm{sign}(g_i)`$, with $`\mathrm{sign}(0)=0`$. This convention is part of the definition; a literal floating-point Adam with $`\epsilon=0`$ would compute $`0/0`$.

**Utilities.** An *immediate-parameter utility* is a bounded measurable function of $`\theta_1`$. A *complete-state utility* is a bounded measurable function of the next complete state $`(\theta_1,m_1,v_1,1)`$. Consumed-gradient laws $`P,Q`$ are probability laws on $`\mathbb R^p`$. In the finite statements they are supported on a finite alphabet $`\mathcal A=\lbrace g^{(1)},\dots,g^{(m)}\rbrace`$. Put $`\bar\theta=(1-\eta\lambda_{\rm wd})\theta_0`$.

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

**Statement under test ([Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7)).** Take a cold state ($`m=v=0`$, $`t=1`$), the ideal sign map $`h_0`$ (Proposition 5.3), a scalar parameter and the immediate utility $`V(\theta)=\tfrac12+w\theta`$, restricted to a neighbourhood where $`V\in[0,1]`$.

- **(a) Equal means.** Let $`P_0(G=1)=\tfrac14`$, $`P_0(G=-\tfrac13)=\tfrac34`$ and $`P_1(G=0)=1`$. Both laws have mean $`0`$, and $`\lvert\mathbb E_{P_0}\psi-\mathbb E_{P_1}\psi\rvert=\lvert w\rvert\eta/2`$.
- **(b) Equal moments to any fixed order.** For $`N\ge1`$, let $`P_N(Z=i/N)=\binom Ni/2^{N-1}`$ for even $`i`$ and $`Q_N(Z=i/N)=\binom Ni/2^{N-1}`$ for odd $`i`$, $`0\le i\le N`$ (the laws of [Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order.md#s4-4)), and $`G=q-Z`$ with $`q\in(0,1/N)`$. The two gradient laws agree on all moments of order below $`N`$, and $`\lvert\mathbb E_{P_N}\psi-\mathbb E_{Q_N}\psi\rvert=\lvert w\rvert\eta\,2^{2-N}`$.
- In both cases the separation is first order in $`\eta`$. At a warm state with the same utility, Proposition 5.6 bounds the gap of equal-mean laws by $`\eta\lvert w\rvert\big(\mathbb E_{P_0}[u\min\lbrace \kappa/2,1\rbrace ]+\mathbb E_{P_1}[u\min\lbrace \kappa/2,1\rbrace ]\big)`$.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.10](../experiments/05-07_cold_steps_separate_equal_mean.md#c5-10) | [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7) | `v015:E8` (cold-separation part) | PASS |
| [Check 5.13](../experiments/05-08_evolution_warmth.md#c5-13) | [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6)–[5.8](../statements/05-08_evolution_warmth.md#s5-8) over a training run | `v015:E8` (`data.gaps`) | PASS |

<a id="c5-10"></a>

## Check 5.10 · Cold separation and warm scaling (Proposition 5.7)

**Verdict.**

- The cold gap between equal-mean laws equals $`\lvert w\rvert\eta/2`$ exactly. For example, it is $`7/2000`$ at $`\eta=0.01`$, $`w=-0.7`$, and $`1/20`$ at $`\eta=0.1`$, $`w=1`$.
- At warm states, the gap normalized by the preconditioned step scale grows like $`\kappa^{0.9995}`$.

**Why.**

- [Proposition 5.7](../statements/05-07_cold_steps_separate_equal_mean.md#s5-7) says that a cold state separates equal-mean laws at first order in $`\eta`$.
- [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6) says that at a warm state the same separation is only of the size of the warmth term $`\eta\,u\,\kappa`$.

**Setup** (`v015:E8`, cold-separation part).

- Cold: the exact rational laws of the proposition, with $`\eta\in\lbrace 0.01,0.1\rbrace`$ and $`w\in\lbrace -0.7,1\rbrace`$, under both the ideal sign map and $`\epsilon=10^{-8}`$.
- Warm: the same unscaled laws at $`t=1000`$ and $`m=0`$. $`v`$ is swept so that $`\kappa_{\max}`$ takes 33 log-spaced values in $`[10^{-6},10^2]`$.

**Result** (`v015:E8`, cold-separation part).

| Quantity | Value |
|---|---|
| cold gap, $`\eta=0.01`$, $`w=-0.7`$ | ideal $`7/2000=0.0035`$; at $`\epsilon=10^{-8}`$: $`0.00349999986`$ |
| cold gap, $`\eta=0.1`$, $`w=1`$ | ideal $`1/20`$; at $`\epsilon=10^{-8}`$: $`0.0499999980`$ |
| warm: log–log slope of the raw gap against $`\kappa`$ | 1.4995 |
| warm: slope after normalizing by the preconditioned step scale | 0.9995 |

**Why it supports Proposition 5.7.**

- The cold gap is exactly $`\lvert w\rvert\eta/2`$, which is first order in $`\eta`$.
- At a warm state, changing $`v`$ changes both $`\kappa`$ and the step scale $`u\propto\kappa^{1/2}`$. The raw gap therefore scales like $`\kappa^{3/2}`$, and the normalized gap like $`\kappa^{1}`$. This is the $`\eta\,u\,\kappa`$ form of the warmth term.

### [Check 5.13](../experiments/05-08_evolution_warmth.md#c5-13) · The cold window over a training run (Propositions 5.6–5.8)

**What it tests here.** This check is described in full on the page of Proposition 5.8. For Proposition 5.7 it tests: [Propositions 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6)–[5.8](../statements/05-08_evolution_warmth.md#s5-8) over a training run.

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
