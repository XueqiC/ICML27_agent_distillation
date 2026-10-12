# Remark 5.11 · Checking the warm-state conditions on a real model

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-11_checking_warm_state_conditions.md) · [Experiments](../experiments/05-11_checking_warm_state_conditions.md) · [中文](05-11_checking_warm_state_conditions_ZH.md)

**What is tested here.** One check, described in full on the page of Proposition 5.6, records warmth quantiles alongside the warm-affine bound; it illustrates why quantiles alone do not verify the condition.

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

**Statement under test ([Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11)).**

- The warm-state error of Proposition 5.6 is the sensitivity-weighted sum $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ plus the term $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$. A statement about quantiles of $`\kappa`$, such as "most coordinates have small warmth", does not verify either term: a few coordinates with large $`L_iu_i\kappa_i`$ can dominate the sum, and the second term depends on the maximum.
- What should be reported on a real model is therefore the weighted sum $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ and its largest terms, together with $`\max_i\kappa_i`$. The sensitivities $`L_i`$ are properties of the continuation value and are not observable from the update alone; any substitute $`\hat L_i`$, for example the magnitude of a held-out validation gradient, is a proxy and must be reported as one.
- A measured non-additivity of the update map (how far $`\Delta`$ is from affine in the message) is a property of the optimizer map, not of the task response. Whether the response is affine must be measured on the response itself, for example by the paired-replacement scores of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6).
- By Proposition 5.8(a), the relevant reference value for the warmth of a typical coordinate late in training is the plateau $`(1-\beta_2)/\beta_2`$, not zero.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.9](../experiments/05-06_warm_adam_first_order_affine.md#c5-9) | [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6), [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11) | `v015:E8` (warm-affine part) | PASS |

### [Check 5.9](../experiments/05-06_warm_adam_first_order_affine.md#c5-9) · Warm Adam is first-order affine in the consumed gradient (Proposition 5.6, Remark 5.11)

**What it tests here.** This check is described in full on the page of Proposition 5.6. For Remark 5.11 it tests: [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6), [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11).

**Verdict.** On actual warm states of the controlled network, the bound holds for all 2,496 law pairs, with constants fixed before any pair was evaluated. The largest gap/bound is 0.988, and the largest pointwise residual/bound is 0.998.

**Setup in brief.**

- States: actual warm states of the network, with $`t`$ between 21 and 200.
- Trainable coordinates: 1–4 output-head coordinates; the rest are fixed.
- Messages: 2–6 per panel.

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
