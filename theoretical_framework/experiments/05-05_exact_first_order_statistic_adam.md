# Proposition 5.5 · The exact first-order statistic of Adam and the frozen-denominator error

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-05_exact_first_order_statistic_adam.md) · [Proof](../proofs/05-05_exact_first_order_statistic_adam.md) · [Experiments](../experiments/05-05_exact_first_order_statistic_adam.md) · [中文](05-05_exact_first_order_statistic_adam_ZH.md)

**What is tested here.** [Check 5.8](../experiments/05-05_exact_first_order_statistic_adam.md#c5-8) searches adversarially for violations of the frozen-denominator bound (b). [Check 5.16](../experiments/05-10_first_order_statistic_by_learner.md#c5-16), whose full description is on the page of Theorem 5.10, shows that at equal gradient means the first-order term is carried by the $`v`$ block, as (a) and (c) imply.

**Verdict.**

- **Proposition 5.5(b) (frozen-denominator error).** No violation on 6,944 adversarially searched states. The best ratio is 1.0, so the bound is tight.

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

**Adam and AdamW.** Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)) is bias-corrected, with $`\epsilon`$ outside the square root, optional decoupled weight decay (AdamW; [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)) and no AMSGrad. An update with counter $`t\ge1`$ acts on $`z=(\theta,m,v,t-1)`$ and a consumed gradient $`G\in\mathbb R^p`$. Coordinatewise,

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

**Statement under test ([Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)).** In the setting of Definition 5.4:

- **(a) Exact statistic.** The Adam increment is $`\Delta=(\Delta\theta,\Delta m,\Delta v,1)`$ with $`\Delta\theta=-\eta\lambda_{\rm wd}\theta-\eta\,n(G)/(d(G)+\epsilon)`$, $`\Delta m=(1-\beta_1)(G-m)`$ and $`\Delta v=(1-\beta_2)(G^{\odot2}-v)`$. The first-order statistic of Proposition 5.1 is the exact mean $`\mu=\mathbb E\,\Delta(z,G)`$. Its parameter block is a nonlinear functional of the law of $`G`$: if $`\beta_1<1`$ and $`v_i>0`$ or $`\epsilon>0`$ for some coordinate $`i`$, there are two finitely supported laws with equal $`\mathbb E\,G`$ and equal $`\mathbb E\,G^{\odot2}`$ whose parameter blocks $`\mu_\theta`$ differ.
- **(b) Frozen-denominator error.** For every coordinate and every $`G`$,

```math
\lvert\theta_i^+(G)-\tilde\theta_i^+(G)\rvert\le\eta\,u_i(G)\,\min\lbrace \kappa_i(G)/2,\;1\rbrace .
```

- **(c) Two approximations, kept apart.** The frozen-denominator mean increment $`\tilde\mu=\mathbb E\,\tilde\Delta(z,G)`$ is determined exactly by $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$, and $`\lvert\mu_{\theta,i}-\tilde\mu_{\theta,i}\rvert\le\eta\,\mathbb E[u_i\min\lbrace\kappa_i/2,1\rbrace]`$ while the $`m`$ and $`v`$ blocks agree exactly. The statement "warm Adam needs only $`(\mathbb E\,G,\ \mathbb E\,G^{\odot2})`$" therefore rests on two separate approximations:
  - the first-order approximation of Proposition 5.1, whose remainder is controlled by the second moments of the increment, with the block structure described in Remark 5.2;
  - the frozen-denominator approximation, whose error is controlled by (b) and is small in the warmth.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.8](../experiments/05-05_exact_first_order_statistic_adam.md#c5-8) | [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)(b) | `v015:E8` (frozen-denominator part) | PASS |
| [Check 5.16](../experiments/05-10_first_order_statistic_by_learner.md#c5-16) | [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(c), [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)(a) | `v015:E11` (warm Adam) | PASS |

<a id="c5-8"></a>

## Check 5.8 · Frozen-denominator error (Proposition 5.5(b))

**Verdict.** On 6,944 adversarially searched optimizer states there is no violation of $`\lvert\theta_i^+-\tilde\theta_i^+\rvert\le\eta\,u_i\min\lbrace\kappa_i/2,1\rbrace`$. The best ratio is 1.0000000000000002, which is equality up to floating-point rounding, so the bound cannot be improved by a constant factor.

**Why.** Part (b) is the only place where the true Adam denominator is replaced by the frozen one. If the bound failed anywhere, [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6) would fail with it.

**Setup** (`v015:E8`, frozen-denominator part; seed `[150150,108]`).

- Random states:
  - $`v`$ log-uniform on $`[10^{-12},1]`$;
  - random $`m`$, gradients and $`t\in\lbrace 1,\dots,2000\rbrace`$;
  - five $`(\beta_1,\beta_2)`$ pairs, including $`(0.9,0.999)`$.
- Search: 24 random restarts × 80 hill-climbing steps that maximize lhs/rhs.
- The difference $`\theta^+-\tilde\theta^+`$ is computed in rationalized exact arithmetic, to avoid cancellation between rounded updates. The direct float subtraction is also kept.
- Tolerance: relative slack $`10^{-12}`$ only.

**Result** (`v015:E8`, frozen-denominator part).

| Quantity | Value |
|---|---|
| states checked | 6,944 |
| violations | 0 |
| best lhs/rhs | $`1.0000000000000002`$; the witness is in the saturated branch $`\kappa\gg1`$ |

**Why it supports Proposition 5.5(b).**

- The bound is valid, and its saturated branch is attained.
- As $`\kappa\to0`$, the first branch is asymptotically tight as well, since $`(1-(1+\kappa)^{-1/2})/(\kappa/2)\to1`$.

### [Check 5.16](../experiments/05-10_first_order_statistic_by_learner.md#c5-16) · Warm Adam: the second moment enters at first order through the optimizer state (Theorem 5.10(c), Proposition 5.5(a))

**What it tests here.** This check is described in full on the page of Theorem 5.10. For Proposition 5.5 it tests: [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)(c), [Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)(a).

**Verdict.**

- With equal gradient means, the parameter block and the $`m`$ block contribute exactly 0. The $`v`$ block contributes $`(1-\beta_2)\langle\nabla_vV,\delta_2\rangle=2.504\times10^{-6}`$, as predicted to $`10^{-12}`$.
- So laws with equal means are not first-order equivalent for warm Adam.

**Setup in brief.**

- The network with $`k=1`$ context and $`p=13`$ parameters.
- A warm state with $`m=0`$, $`v=0.02`$ in every coordinate and step counter 20.
  - It is an admissible warm state, not a draw from an IID warm-up.
  - The symmetry of the label laws is what makes the parameter term vanish.
- Adam with $`\beta=(0.9,0.999)`$ and $`\eta=0.03`$.

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
