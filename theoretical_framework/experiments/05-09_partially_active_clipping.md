# Proposition 5.9 · Partially active clipping

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-09_partially_active_clipping.md) · [Proof](../proofs/05-09_partially_active_clipping.md) · [Experiments](../experiments/05-09_partially_active_clipping.md) · [中文](05-09_partially_active_clipping_ZH.md)

**What is tested here.** [Check 5.14](../experiments/05-09_partially_active_clipping.md#c5-14) shows that two laws with equal raw means but different clipped means have a first-order gap, so the statistic is the clipped mean. [Check 5.15](../experiments/05-09_partially_active_clipping.md#c5-15) tests the always-active case $`p_{\rm act}=1`$, where equal mean directions give a first-order term of exactly $`0`$. The partial-activation bound itself has no separate numerical check.

**Verdict.**

- **Proposition 5.9 and Theorem 5.10(e) (clipping).**
  - Equal raw means with different clipped means give a first-order gap. Its remainder has slope 2.0015.
  - With clipping always active ($`p_{\rm act}=1`$), equal mean directions give a first-order term of exactly 0.
  - The partial-activation bound of Proposition 5.9 is a two-line inequality and has no separate numerical check.

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

**Status labels.** `PASS` means the implemented consistency assertions passed. `INFO` marks an empirical diagnostic: each cell gets its own PASS, FAIL or UNAVAILABLE status, and the group as a whole makes no pass/fail claim. Unless a check says otherwise, every network is the controlled small network of `v014:C23`: one-hot context, 4 tanh units, a sigmoid output, mean cross-entropy, float64. None of these networks is a real model.

**Statement under test ([Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9)).** Let $`\mathrm{clip}_c(g)=g\min\lbrace1,c/\lVert g\rVert\rbrace`$ with $`c>0`$, set $`G/\lVert G\rVert=0`$ when $`G=0`$, and let $`p_{\rm act}=\Pr(\lVert G\rVert>c)`$ be the activation rate under a law of $`G`$. Then

```math
\mathbb E\,\mathrm{clip}_c(G)=\mathbb E\big[G\,1\lbrace\lVert G\rVert\le c\rbrace\big]+c\,\mathbb E\Big[\frac{G}{\lVert G\rVert}1\lbrace\lVert G\rVert>c\rbrace\Big],\qquad \Big\lVert\mathbb E\,\mathrm{clip}_c(G)-c\,\mathbb E\frac{G}{\lVert G\rVert}\Big\rVert\le c\,(1-p_{\rm act}).
```

- For two laws, the clipped means differ from the scaled mean directions by at most $`c\big[(1-p^P_{\rm act})+(1-p^Q_{\rm act})\big]`$ in total. For instance, at an activation rate of $`0.9`$ under both laws, replacing the clipped mean by the mean direction errs by at most $`0.2c`$ in the difference.
- The always-active case $`p_{\rm act}=1`$ is the only one in which the mean direction is exactly the clipped mean.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.14](../experiments/05-09_partially_active_clipping.md#c5-14) | [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9), [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(e) | `v015:E11` (clipped mean) | PASS |
| [Check 5.15](../experiments/05-09_partially_active_clipping.md#c5-15) | [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9) at $`p_{\rm act}=1`$, [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(e) | `v015:E11` (always active) | PASS |

<a id="c5-14"></a>

## Check 5.14 · Clipping: the statistic is the clipped mean (Proposition 5.9, Theorem 5.10(e))

**Verdict.** Two gradient laws have equal raw means but clipped means that differ by $`-\tfrac12`$. They separate at first order, and the predicted gap $`\tfrac\eta2\sigma'(0.4)`$ matches the measured gap with a remainder of log–log slope 2.0015.

**Why.**

- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(e) says that under clipping the feedback must match $`\mathbb E\,\mathrm{clip}_c(G)`$, not $`\mathbb E\,G`$.
- If clipping changed only magnitudes, two laws with equal raw means would agree at first order. A first-order gap at equal raw means therefore confirms the clipped mean as the statistic.
- The two laws are differently active. $`P`$ is always clipped and $`Q`$ never is. So the case split of [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9) is exercised on both of its branches.

**Setup** (`v015:E11`, clipped mean).

- A sigmoid with a linear logit at $`\theta=0.4`$.
- $`P(G=4)=\tfrac14`$ and $`P(G=-\tfrac43)=\tfrac34`$; $`Q=\delta_0`$.
- Clip at $`c=1`$; clipped SGD, i.e. SGD with global-norm clipping ([Pascanu et al., 2013](#ref-pascanu2013difficulty)).
- $`\eta\in\lbrace 10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\rbrace`$.
- The gradients $`4`$ and $`-4/3`$ cannot occur in the fixed network at its initialization, so a linear-logit model realizes them.

**Procedure.**

- Assert that the raw means agree and that the clipped means differ by $`-\tfrac12`$.
- Compute the exact gap $`(P-Q)\sigma(\theta+\Delta\theta)`$ and the predicted first-order gap $`\eta\,\sigma'(0.4)/2`$.
- Assert the analytic second-order bound $`\lvert\text{gap}-\text{first order}\rvert\le\eta^2/(12\sqrt3)`$.

**Result** (`v015:E11`, `data.clipped_mean`).

| $`\eta`$ | measured gap | first-order prediction | remainder |
|---:|---:|---:|---:|
| 0.001 | $`1.20107\times10^{-4}`$ | $`1.20130\times10^{-4}`$ | $`2.37\times10^{-8}`$ |
| 0.01 | $`1.19892\times10^{-3}`$ | $`1.20130\times10^{-3}`$ | $`2.38\times10^{-6}`$ |
| 0.1 | $`1.17675\times10^{-2}`$ | $`1.20130\times10^{-2}`$ | $`2.46\times10^{-4}`$ |

The remainder's log–log slope is 2.0015.

![Figure 5.4: the first-order statistic by learner](../figures/fig5_4_learner_statistics.png)

*Figure 5.4.* Five witnesses of the learner-specific first-order statistic ([Checks 5.14](../experiments/05-09_partially_active_clipping.md#c5-14)–[5.18](../experiments/05-10_first_order_statistic_by_learner.md#c5-18)):
- (A) Equal raw means but different clipped means give a first-order gap under clipping.
- (B) With clipping always active, equal mean directions give an $`O(\eta^2)`$ gap.
- (C) Warm Adam with equal gradient means: the first-order term is carried entirely by the $`v`$ block, as predicted.
- (D) Momentum-dominated Lion: the parameter increment does not depend on the message.
- (E) LoRA at initialization: null-space differences are second order, range differences first order.

[PDF](../figures/fig5_4_learner_statistics.pdf)

**Why it supports Theorem 5.10(e).** The raw mean gradient predicts a zero gap, while the clipped mean predicts the measured gap to first order, with an $`O(\eta^2)`$ remainder.

<a id="c5-15"></a>

## Check 5.15 · Always-active clipping: only the direction matters (Proposition 5.9 with full activation, Theorem 5.10(e))

**Verdict.** When every gradient is clipped and the two laws have the same mean unit direction, the first-order term is exactly 0 and the gap is second order (log–log slope 1.99997).

**Why.** When clipping is always active ($`p_{\rm act}=1`$), the error bound $`c(1-p_{\rm act})`$ of [Proposition 5.9](../statements/05-09_partially_active_clipping.md#s5-9) is 0. The clipped mean is then exactly $`c\,\mathbb E[G/\lVert G\rVert]`$. Two laws with the same mean direction must agree at first order, whatever their magnitudes.

**Setup** (`v015:E11`, always active).

- Two-dimensional gradients with $`\lVert G\rVert\ge2>c=1`$.
- $`P`$ is uniform on $`\lbrace(\pm2,0)\rbrace`$ and $`Q`$ is uniform on $`\lbrace(0,\pm4)\rbrace`$. Both mean unit directions are 0, and their second direction moments differ.
- The same linear-logit model and $`\eta`$ grid as [Check 5.14](../experiments/05-09_partially_active_clipping.md#c5-14).

**Procedure.** Assert that the first-order term is exactly zero, and fit the slope of the exact gap.

**Result** (`v015:E11`, `data.always_active`).

| $`\eta`$ | first-order term | measured gap |
|---:|---:|---:|
| 0.001 | 0 | $`7.11\times10^{-8}`$ |
| 0.01 | 0 | $`7.11\times10^{-6}`$ |
| 0.1 | 0 | $`7.06\times10^{-4}`$ |

The gap's log–log slope is 1.99997.

**Why it supports Proposition 5.9 and Theorem 5.10(e).**

- With clipping always active, magnitudes drop out at first order. Only the second-order direction moments separate these laws.
- If the two laws also had the same law of directions, their increment laws would coincide exactly.
- Limit of the evidence: for $`0<p_{\rm act}<1`$, Proposition 5.9 bounds how far "only the direction matters" is from the truth, by $`c(1-p_{\rm act})`$. That intermediate case has no dedicated numerical check.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

The checks use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)). The linear programs of `v015:V2` and `v015:V8` are solved by `scipy.optimize.linprog` with the HiGHS solver ([Huangfu & Hall, 2018](#ref-huangfu2018parallelizing)).

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
