# Theorem 7.1 · State-channel decomposition

*Experiments*

[← Framework](../README.md) · [Statement](../statements/07-01_state_channel_decomposition.md) · [Proof](../proofs/07-01_state_channel_decomposition.md) · [Experiments](07-01_state_channel_decomposition.md) · [中文](07-01_state_channel_decomposition_ZH.md)

**What is tested here.** Part (a) of the theorem: that the backward costate recursion gives the exact gradient of the continuation value with respect to the complete Adam/AdamW state $`(\theta,m,v)`$. The channel split (b) and the closed form (d) follow algebraically from (a). **Verdict: PASS.** The adjoint gradient matches central finite differences in all 1,024 directional comparisons, worst relative error $`2.8\times10^{-7}`$ (tolerance $`10^{-6}`$), across 256 configurations that include bias correction, decoupled weight decay, $`\epsilon>0`$ and active global-norm clipping. An independent one-step implementation of the recursion, written separately from that suite, agrees with central differences to $`7.5\times10^{-11}`$ ([Check 7.7](#c7-7)). Part (f), the gradient under state-dependent sampling, matches central differences to $`5.2\times10^{-11}`$, while the costate average alone is off by $`3.2\times10^{-3}`$ ([Check 7.10](#c7-10)).

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 7.1](#c7-1) | (a), full tails | `v016:C4.costate` | PASS |
| [Check 7.7](#c7-7) | (a), one step, independent implementation | `v017:C15.adjoint` | PASS |
| [Check 7.10](#c7-10) | (f), state-dependent sampling | `v017:C23.sampling` | PASS |

**Setting used on this page.**

- $`z=(\theta,m,v,s)`$ is the complete Adam state; AdamW ([Kingma & Ba, 2015](#ref-kingma2015adam); [Loshchilov & Hutter, 2019](#ref-loshchilov2019decoupled)) with $`\beta_1,\beta_2,\eta,\epsilon`$, decoupled weight decay $`\lambda_{\rm wd}`$ and bias-correction factors $`q_{1,s},q_{2,s}`$ maps a consumed gradient $`g`$ to the increment $`\Delta(z,g)`$ ([Proposition 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)).
- After a message at round $`t`$, the reference tail runs $`R-1-t`$ updates with consumed gradients $`g_k(\theta)`$ (possibly global-norm clipped), and $`V_{t+1}(z)`$ is the final score $`F(\theta_R)`$ reached from $`z`$.
- The costate $`\chi_{t+1}=(\chi^\theta_{t+1},\chi^m_{t+1},\chi^v_{t+1})`$ is the output of the backward recursion of the theorem; "adjoint gradient" below means $`\chi_{t+1}`$.

**Statement under test ([Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1)).** If $`F`$ is $`C^1`$, every tail gradient $`g_k`$ is $`C^1`$ in $`\theta`$ near the tail trajectory (with clipping away from its threshold), and $`v'_k>0`$ along the tail, then $`V_{t+1}`$ is differentiable at $`z_t`$ in $`(\theta,m,v)`$ and $`\nabla V_{t+1}(z_t)=\chi_{t+1}`$, where $`\chi`$ solves, from $`\chi_R=(\nabla F,0,0)`$,

```math
\begin{aligned}
\chi^m_k&=\beta_1\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big),\qquad \chi^v_k=\beta_2\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big),\\
\chi^\theta_k&=(1-\eta\lambda_{\rm wd})\chi^\theta_{k+1}+J_k^\top\Big[(1-\beta_1)\big(\chi^m_{k+1}-\eta D_1^{(k)}\chi^\theta_{k+1}\big)+2(1-\beta_2)\,\mathrm{diag}(g_k)\big(\chi^v_{k+1}+\eta D_2^{(k)}\chi^\theta_{k+1}\big)\Big],
\end{aligned}
```

with $`D_1^{(k)}=\mathrm{diag}\big(q_{1,s_k}/(\sqrt{q_{2,s_k}v'_k}+\epsilon)\big)`$, $`D_2^{(k)}=\mathrm{diag}\big(q_{1,s_k}q_{2,s_k}m'_k/(2\sqrt{q_{2,s_k}v'_k}(\sqrt{q_{2,s_k}v'_k}+\epsilon)^2)\big)`$ and $`J_k=\partial g_k/\partial\theta`$ (the clipping Jacobian when clipping is active). The first-order value of a message then splits into a parameter, a momentum and a second-moment channel.

<a id="c7-1"></a>

## Check 7.1 · Costate recursion against finite differences (Theorem 7.1)

**Verdict.** PASS. Worst directional relative error $`2.82\times10^{-7}`$ (tolerance $`10^{-6}`$) over 1,024 comparisons.

**Why.** [Theorem 7.1(a)](../statements/07-01_state_channel_decomposition.md#s7-1) claims that the reverse recursion gives the exact gradient of the continuation value in $`(\theta,m,v)`$. That gradient includes the bias-correction factors, decoupled weight decay, $`\epsilon`$ and the Jacobian of global-norm clipping. A wrong block anywhere in the recursion shows up as a mismatch along some direction.

**Setup.**

- Exact AdamW in float64 on quadratic and logistic losses, $`p\in\lbrace5,20\rbrace`$ parameters, terminal score $`\mathrm{sigmoid}(a^\top\theta)`$, $`\eta=0.007`$.
- Full factorial over: bias correction off/on; $`\epsilon\in\lbrace10^{-8},10^{-3}\rbrace`$; decoupled weight decay $`0`$ or $`0.03`$; $`\beta_1\in\lbrace0,0.9\rbrace`$; $`\beta_2\in\lbrace0.99,0.999\rbrace`$; global-norm clipping off or at $`0.15`$. That is 256 configurations.
- Tails of 20, 60 or 200 updates, starting from states with positive $`v`$.
- For each configuration, central differences of $`V_{t+1}`$ along random $`\theta`$-only, $`m`$-only, $`v`$-only and joint directions, with a step ladder $`h=2\times10^{-4},\,h/2`$ and Richardson extrapolation.
- A finite-difference step is halved whenever a perturbed trajectory changes any clipping decision. The decision uses forward clipping patterns only, never agreement with the costate.
- Tolerance: relative error $`\le10^{-6}`$, with denominator $`\max(\lvert\text{adjoint}\rvert,\lvert\text{FD}\rvert,10^{-10})`$. No automatic differentiation is used.

**Result.**

- 1,024/1,024 directional comparisons within tolerance; worst relative error $`2.82\times10^{-7}`$.
- All 1,024 accepted finite-difference steps stay on the same clipping branch as the unperturbed trajectory. Two attempted steps crossed a clipping threshold and were halved.
- The comparisons include the exactly zero momentum derivatives at $`\beta_1=0`$.

**Why it supports the statement.**

- The test covers every term of the recursion, including the extensions beyond the plain Adam step: bias correction, decoupled decay, $`\epsilon`$, and the non-symmetric clipping Jacobian.
- It is a test of the identity in (a). The channel split (b) and the closed form (d) are algebraic consequences.

**Limits.** The losses are smooth and low-dimensional; derivatives at a clipping threshold are excluded by design, as in the theorem.

<a id="c7-7"></a>

## Check 7.7 · One step of the costate recursion, independent implementation (Theorem 7.1)

**Verdict.** PASS. The one-step costate map, with a nonzero Hessian, bias correction, $`\epsilon>0`$ and decoupled weight decay, agrees with central finite differences to $`7.5\times10^{-11}`$ (tolerance $`10^{-8}`$).

**Why.** [Check 7.1](#c7-1) tests whole tails in one code base. This check re-derives one step of the recursion from scratch, in a separate suite, so that an error shared by the recursion and its test harness in that code base would show up here. One step is the building block: the recursion of [Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1) composes it backwards.

**Setup.**

- One coordinate of AdamW at step counter $`s=5`$ with $`\beta_1=0.9`$, $`\beta_2=0.999`$, learning rate $`0.07`$, $`\epsilon=0.03`$ and decoupled weight decay $`0.1`$; state $`(\theta,m,v)=(0.7,0.2,0.8)`$.
- The consumed gradient is $`g(\theta)=\theta-0.3`$, so the Hessian is $`1`$, not $`0`$.
- A fixed next-state costate $`(\chi^\theta,\chi^m,\chi^v)=(0.4,-0.2,0.6)`$; the objective is its inner product with the next state $`(\theta',m',v')`$. Its exact gradient with respect to $`(\theta,m,v)`$ is one step of the recursion, with $`D_1`$, $`D_2`$ and $`J=1`$ as in the theorem.
- Central differences with step $`10^{-6}`$ in each of the three coordinates; tolerance $`10^{-8}`$ on the Euclidean error.

**Result.** Euclidean error between the recursion and the finite-difference gradient: $`7.5\times10^{-11}`$.

**Why it supports the statement.** Every block of one step appears in the test: $`\partial m'/\partial m=\beta_1`$, $`\partial v'/\partial v=\beta_2`$, $`\partial\theta'/\partial m'=-\eta D_1`$, $`\partial\theta'/\partial v'=\eta D_2`$, the bias-correction factors inside $`D_1,D_2`$, the weight-decay factor $`1-\eta\lambda_{\rm wd}`$, and the Jacobian term through $`g(\theta)`$. A wrong block would give an error of the size of that block, far above the tolerance.

**Limits.** One coordinate and one step; the composition over long tails is the subject of [Check 7.1](#c7-1).

<a id="c7-10"></a>

## Check 7.10 · The sampling term under state-dependent sampling (Theorem 7.1(f))

**Verdict.** PASS. With a tail branch drawn with a probability that depends on the state, the gradient of the continuation value equals the costate average plus the sampling term to $`5.2\times10^{-11}`$. The costate average alone misses the momentum component by $`3.2\times10^{-3}`$, about 13% of that component.

**Why.** Part (e) needs tail randomness whose law does not depend on the state. When the feedback of a later update is chosen by looking at the student, part (f) adds the term $`\sum_\omega V_{t+1,\omega}\nabla p_\omega`$; this check shows that the term is needed even with two branches, and that it enters only the blocks on which the branch probability depends.

**Setup.**

- One coordinate of Adam with bias correction, $`\beta_1=0.9`$, $`\beta_2=0.999`$, learning rate $`0.05`$ and $`\epsilon=10^{-3}`$; the tail starts at step counter $`7`$ from $`(\theta,m,v)=(0.3,0.1,0.05)`$.
- The first tail update consumes the gradient $`\theta-a_\omega`$ of one of two data points, $`a_0=-0.5`$ or $`a_1=1.2`$; branch $`1`$ is drawn with probability $`\sigma(2\theta+3m)`$, which depends on the parameter and on the momentum but not on $`v`$. The second tail update consumes $`\theta-0.4`$. The score is $`F(\theta)=\sigma(2\theta)`$.
- Branch values, their gradients and the gradient of the branch probability are computed by central differences with step $`10^{-6}`$; the full value $`V=\sum_\omega p_\omega V_\omega`$ is differentiated the same way.

**Result.**

| Quantity | $`\theta`$ | $`m`$ | $`v`$ |
|---|---|---|---|
| gradient of the continuation value | $`0.455851`$ | $`-0.024319`$ | $`0.013323`$ |
| costate average $`\sum_\omega p_\omega\chi_\omega`$ | $`0.453708`$ | $`-0.027533`$ | $`0.013323`$ |
| sampling term $`\sum_\omega V_\omega\nabla p_\omega`$ | $`0.002143`$ | $`0.003214`$ | $`0`$ |

- Error of the formula of (f): $`5.2\times10^{-11}`$ (tolerance $`10^{-7}`$). Error of the costate average alone: $`3.2\times10^{-3}`$.
- Replacing $`V_\omega`$ by $`V_\omega-b`$ with $`b`$ the mean branch value, and the score-function form $`\mathbb E_\omega[V_\omega\nabla\log p_\omega]`$, change the sampling term by at most $`7\times10^{-17}`$.

**Why it supports the statement.** The sampling term is of the size of the effect, not of rounding: dropping it changes the momentum component of the continuation gradient by 13%. Its $`v`$-component is exactly zero because the only sampling step looks at the starting state through $`\theta`$ and $`m`$, so the branch probability does not depend on $`v`$, as (f) states. A sampling step at a later update would make it depend on $`v`$ through the tail dynamics.

**Limits.** Two branches and two tail updates, with derivatives by finite differences; the check verifies the product-rule formula, not an estimator of the sampling term from rollouts.

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C4
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- The costate group (`v016:C4.costate`) takes about 16 s on rai; the full suite without `--only` about 104 s.
- Seeds: `numpy.SeedSequence([160160, group, subcase, ...])`, PCG64, float64, single BLAS thread.

The independent one-step check of [Check 7.7](#c7-7) and the sampling check of [Check 7.10](#c7-10):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v016_checks/results_v016.json` — claim id `v016:C4.costate`: assertions, per-configuration errors.
- `theory/v016_checks/results_v016_raw.json.gz` — raw arrays referenced by JSON pointers. Not committed; SHA-256 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`.
- `theory/v016_checks/SPEC.md` (section on `v016:C4`), `check_v016.py` and its tests.
- `theory/v017_checks/results_v017.json` — entries `v017:C15.adjoint` and `v017:C23.sampling` (functions `c_adjoint` and `c_sampling` in `theory/v017_checks/check_v017.py`).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-loshchilov2019decoupled"></a>

**Ilya Loshchilov, Frank Hutter (2019).** Decoupled weight decay regularization. *7th International Conference on Learning Representations (ICLR)*. arXiv [1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
