# Remark 5.11 · Checking the warm-state conditions on a real model

*Experiments*

[← Framework](../README.md) · [Statement](../statements/05-11_checking_warm_state_conditions.md) · [Experiments](../experiments/05-11_checking_warm_state_conditions.md) · [中文](05-11_checking_warm_state_conditions_ZH.md)

**What is tested here.** One check on the controlled network and two real-model checks.

- [Check 5.9](../experiments/05-06_warm_adam_first_order_affine.md#c5-9), described in full on the page of Proposition 5.6, records warmth quantiles alongside the warm-affine bound; it illustrates why quantiles alone do not verify the condition.
- [Check 5.19](#c5-19), real model: the pre-registered warm criterion **fails at all six measured states**, because its denominator cancels. Against the signed first-order term (post hoc, labelled so) the sensitivity-weighted frozen-denominator error is about 10% of the first-order term from the p03 checkpoint on, and larger than it early in training.
- [Check 5.20](#c5-20), real model: the pre-registered criterion for response non-additivity is met at p10 (median 0.855) and narrowly missed at update 8 (0.188 against 0.2), with only four replicas.

**Setting used on this page.**

The objects below are restated so that this page can be read on its own; the general notation is that of [Definitions 1.3–1.4](../README.md#s1-3).

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

**Real-model setting (for [Check 5.19](#c5-19) and [Check 5.20](#c5-20)).**

- *Recipe.* ALFWorld ([Shridhar et al., 2021](#ref-shridhar2021alfworld)) Seed-SFT on 20 trusted demonstrations (seed 0); student `google/gemma-4-12B-it` with LoRA rank 16, $`\alpha=32`$, dropout 0 (65,568,768 trainable coordinates); AdamW, constant learning rate $`10^{-5}`$, $`\epsilon=10^{-8}`$, global gradient clipping at 1.0; at least 512 supervised tokens per update; 469 updates in the full run.
- *States.* p03, p10 and p40 are the checkpoints after 31,221, 104,070 (update 115) and 416,280 supervised tokens.
- *Surrogate.* $`\tilde F`$ is the mean teacher-forced log-likelihood on 40 held-out teacher-success trajectories that never enter training. The pre-registration specified 80; 40 were used to cut the scoring cost, a recorded deviation.
- *Sensitivity proxy.* $`\hat L_i=\lvert\partial\tilde F/\partial\theta_i\rvert`$, a proxy for the sensitivity $`L_i`$ and not a Lipschitz bound.
- Both checks were pre-registered with their criteria before any job ran. They are single-seed exploration runs.

**Statement under test ([Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11)).**

- The warm-state error of Proposition 5.6 is the sensitivity-weighted sum $`\eta\sum_iL_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ plus the term $`L_v\max_i\min\lbrace\kappa_i,\bar k\rbrace`$. A statement about quantiles of $`\kappa`$, such as "most coordinates have small warmth", does not verify either term: a few coordinates with large $`L_iu_i\kappa_i`$ can dominate the sum, and the second term depends on the maximum.
- What should be reported on a real model is therefore the weighted sum $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$ and its largest terms, together with $`\max_i\kappa_i`$. The sensitivities $`L_i`$ are properties of the continuation value and are not observable from the update alone; any substitute $`\hat L_i`$, for example the magnitude of a held-out validation gradient, is a proxy and must be reported as one.
- A measured non-additivity of the update map (how far $`\Delta`$ is from affine in the message) is a property of the optimizer map, not of the task response. Whether the response is affine must be measured on the response itself, for example by the paired-replacement scores of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6).
- By Proposition 5.8(a), the relevant reference value for the warmth of a typical coordinate late in training is the plateau $`(1-\beta_2)/\beta_2`$, not zero.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 5.9](../experiments/05-06_warm_adam_first_order_affine.md#c5-9) | [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6), [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11) | `v015:E8` (warm-affine part) | PASS |
| [Check 5.19](#c5-19) | the sensitivity-weighted error on a real model | `v017:R1.r0a` | pre-registered criterion FAIL; post-hoc INFO |
| [Check 5.20](#c5-20) | response non-additivity on a real model | `v017:R2.r0b` | INFO (criterion met at p10, missed at update 8) |

### [Check 5.9](../experiments/05-06_warm_adam_first_order_affine.md#c5-9) · Warm Adam is first-order affine in the consumed gradient (Proposition 5.6, Remark 5.11)

**What it tests here.** This check is described in full on the page of Proposition 5.6. For Remark 5.11 it tests: [Proposition 5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6), [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11).

**Verdict.** On actual warm states of the controlled network, the bound holds for all 2,496 law pairs, with constants fixed before any pair was evaluated. The largest gap/bound is 0.988, and the largest pointwise residual/bound is 0.998.

**Setup in brief.**

- States: actual warm states of the network, with $`t`$ between 21 and 200.
- Trainable coordinates: 1–4 output-head coordinates; the rest are fixed.
- Messages: 2–6 per panel.

<a id="c5-19"></a>

## Check 5.19 · The sensitivity-weighted frozen-denominator error on a real model (Remark 5.11)

**Verdict.**

- **Pre-registered criterion: FAIL at all six states.** The criterion was $`E_{\rm fd}\le0.1\times\lvert\sum_i\hat L_i\Delta\theta_i\rvert`$. Its denominator multiplies the nonnegative proxies $`\hat L_i`$ by signed steps $`\Delta\theta_i`$, so it cancels: at p10 it is $`7.2\times10^{-6}`$, about 200 times smaller than the signed first-order term. The criterion was mis-specified, and its failure is reported as such.
- **Post hoc (labelled so).** Against the signed first-order term $`\lvert\sum_i(\partial\tilde F/\partial\theta_i)\Delta\theta_i\rvert`$ the ratio is about 0.1 at p03, p10 and p40 and above 1 in the first updates. The frozen-denominator description therefore carries an error of roughly 10% of the first-order term at warm checkpoints, which is not negligible.

**Why.** [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11) says that the warm-state error is a sensitivity-weighted sum and that warmth quantiles do not check it. This is the first measurement of that sum on a real model.

**Setup.**

- Six states: after updates 1, 4 and 8, and the checkpoints p03, p10 and p40. The pre-registration listed nine; the probe's cold and warm-up states had not been saved, a recorded deviation.
- At each state, one consumed batch: per coordinate, the warmth $`\kappa_i`$, the relative step $`u_i`$ and the realized step $`\Delta\theta_i`$, in float64; the gradient of $`\tilde F`$ gives $`\hat L_i`$.
- $`E_{\rm fd}=\eta\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$, the pre-registered ratio $`E_{\rm fd}/\lvert\sum_i\hat L_i\Delta\theta_i\rvert`$, and the post-hoc ratio $`E_{\rm fd}/\lvert\sum_i(\partial\tilde F/\partial\theta_i)\Delta\theta_i\rvert`$.

**Result.**

| State | $`E_{\rm fd}`$ | pre-registered ratio | criterion ($`\le0.1`$) | post-hoc ratio |
|---|---|---|---|---|
| update 1 | 1.19 | 35,472 | FAIL | 29.7 |
| update 4 | 0.0340 | 2,548 | FAIL | 2.13 |
| update 8 | 0.104 | 97.4 | FAIL | 0.984 |
| p03 | $`3.38\times10^{-4}`$ | 19.0 | FAIL | 0.0997 |
| p10 | $`1.48\times10^{-4}`$ | 20.5 | FAIL | 0.0972 |
| p40 | $`1.71\times10^{-4}`$ | 4.09 | FAIL | 0.0837 |

The stored warmth quantiles of the same runs (field `old_quantiles` of each record) put only 0.014% of the coordinates above $`\kappa=1`$ at p10, and the median warmth at $`3.1\times10^{-4}`$. Read by quantiles, the state looks fully warm; the weighted sum says the frozen-denominator error is still a tenth of the first-order effect.

**Why it supports Remark 5.11.** The remark's central point is confirmed on a 12B model: quantiles of $`\kappa`$ suggested a negligible error, while the sensitivity-weighted sum, dominated by few coordinates, is about 10% of the first-order term. The check also documents a second lesson of the remark that the pre-registration missed: a first-order comparison term must keep the signs of the sensitivities, or it cancels.

<a id="c5-20"></a>

## Check 5.20 · Response non-additivity on a real model (Remark 5.11)

**Verdict.** INFO. With the per-token surrogate of the pre-registration, the median response non-additivity is 0.855 at p10, above the pre-registered threshold 0.2 and inside the predicted interval $`0.631\pm0.25`$; at update 8 it is 0.188, just below the threshold and below the predicted interval. With four replicas the estimates are noisy.

**Why.** [Remark 5.11](../statements/05-11_checking_warm_state_conditions.md#s5-11) separates non-additivity of the update map from non-additivity of the task response, and says the second must be measured on the response itself. This check does that with paired replacements at the real states.

**Setup.**

- States: update 8 and p10. Four disjoint pairs of batches $`(A,B)`$, the same as in the pilot probe.
- For each pair, branches $`\lbrace0,A,B,A\cup B\rbrace`$: one replaced update followed by a common-random-number tail of $`T=10`$ reference updates, $`r=4`$ replicas ([Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6)).
- $`\delta\tilde F(X)=\tilde F(X)-\tilde F(0)`$ and $`\nu_{\rm resp}=\lvert\delta\tilde F(A\cup B)-w_A\delta\tilde F(A)-w_B\delta\tilde F(B)\rvert/\lvert\delta\tilde F(A\cup B)\rvert`$, with token weights $`w`$; the reported number is the median over the four pairs of the replica-mean value.
- Pre-registered: criterion "median above 0.2 means non-additive at the response level"; prediction "median within 0.25 of the update map's value", which is 0.631 at p10 for clipped AdamW.

**Result.**

| State | median $`\nu_{\rm resp}`$, per-token (pre-registered surrogate) | median $`\nu_{\rm resp}`$, geometric token probability | criterion $`>0.2`$ |
|---|---|---|---|
| update 8 | 0.188 | 0.191 | not met (borderline) |
| p10 | 0.855 | 0.886 | met |

At p10 the mean effect of the union batch is smaller than the per-replica scatter of the paired scores, so $`\nu_{\rm resp}`$ is a ratio of small, noisy numbers; four replicas are too few to say more than the sign of the conclusion.

**Why it supports Remark 5.11.** The measurement is possible, and it is on the response rather than on the update map, as the remark requires. It does not establish a precise value: the threshold verdicts at both states are within the noise that four replicas allow.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V8,E10,V2,V3,V4,E0,E1,E2,E3,E8,E11 --output theory/v015_checks/quick/s5.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v016_checks/check_v016.py --only C1 --output /tmp/s5_c1.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C23 --output /tmp/s5_c23.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package checks T01–T13
```

The two real-model checks recompute the reported numbers from the stored run outputs:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
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
- `theory/v017_checks/results_v017.json`: entries `v017:R1.r0a` and `v017:R2.r0b`; functions `r_r0a` and `r_r0b` in `theory/v017_checks/check_v017.py`. The stored run outputs are in `theory/v017_checks/records/r0a/` (one JSON file per state, with per-block breakdowns and the stored warmth quantiles) and `theory/v017_checks/records/r0b/` (merged paired scores and per-pair, per-replica non-additivity).
- Code:
  - `theory/v015_checks/otlr.py` (`check_V8`, `check_E10`, `check_E11`);
  - `theory/v015_checks/warmth.py` (`check_E8`);
  - `theory/v016_checks/check_v016.py` (`v016:C1`);
  - specifications in `theory/v015_checks/README.md` and `theory/v016_checks/SPEC.md`.
- Figures: `../figures/fig5_1_linear_response`, `fig5_2_cold_adam`, `fig5_3_warmth` and `fig5_4_learner_statistics`, each as `.pdf` and `.png`. The specs are in `../figures/specs/fig5_1.md` to `fig5_4.md`.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-huangfu2018parallelizing"></a>

**Qi Huangfu, J. A. Julian Hall (2018).** Parallelizing the dual revised simplex method. *Mathematical Programming Computation* 10, no. 1, 119–142. DOI [10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
