# Theorem 7.1 · State-channel decomposition

*Experiments*

[← Framework](../README.md) · [Statement](../statements/07-01_state_channel_decomposition.md) · [Proof](../proofs/07-01_state_channel_decomposition.md) · [Experiments](07-01_state_channel_decomposition.md) · [中文](07-01_state_channel_decomposition_ZH.md)

**What is tested here.** Part (a) of the theorem: that the backward costate recursion gives the exact gradient of the continuation value with respect to the complete Adam/AdamW state $`(\theta,m,v)`$. The channel split (b) and the closed form (d) follow algebraically from (a). **Verdict: PASS.** The adjoint gradient matches central finite differences in all 1,024 directional comparisons, worst relative error $`2.8\times10^{-7}`$ (tolerance $`10^{-6}`$), across 256 configurations that include bias correction, decoupled weight decay, $`\epsilon>0`$ and active global-norm clipping.

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

## Reproduce

From the repository root (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py --only C4
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python -m pytest -q -p no:cacheprovider theory/v016_checks
```

- The costate group (`v016:C4.costate`) takes about 16 s on rai; the full suite without `--only` about 104 s.
- Seeds: `numpy.SeedSequence([160160, group, subcase, ...])`, PCG64, float64, single BLAS thread.

The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v016_checks/results_v016.json` — claim id `v016:C4.costate`: assertions, per-configuration errors.
- `theory/v016_checks/results_v016_raw.json.gz` — raw arrays referenced by JSON pointers. Not committed; SHA-256 `1777069872ea80c09dfd5260d610871b68e41a10721eba2826dcd4d647874f69`.
- `theory/v016_checks/SPEC.md` (section on `v016:C4`), `check_v016.py` and its tests.

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
