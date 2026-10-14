# Proposition 8.14 · Anchor reuse and a paired drift certificate

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-14_anchor_reuse_drift_certificate.md) · [Proof](../proofs/08-14_anchor_reuse_drift_certificate.md) · [Experiments](08-14_anchor_reuse_drift_certificate.md) · [中文](08-14_anchor_reuse_drift_certificate_ZH.md)

**What is tested here.**

- **(a), (b), (d) on synthetic instances.** The reuse residual at a drifted checkpoint is 3.375, below the bound 3.477; the squared operator drift is 0.094, below the paired second moment 0.330; the debiased paired product is unbiased to $`1.1\times10^{-16}`$ (**PASS**).
- **(c) at the measured real-model noise.** The real-model noise reference (one checkpoint, 300 dev tasks, two identical servers) gives 241 and 242 successes with 15 task flips. At that noise the strict drift radius is 0.718 with 10 pairs, 0.507 with 40, 0.404 with 100 and 0.227 with 1,000, and reaching 0.03 needs about $`4.9\times10^6`$ pairs. The residual it is meant to bound is trivially at most 0.5. The certificate is **valid but vacuous** at this scale.

**Setting used on this page.**

- Reference measure $`\rho`$, $`\mathcal H=L^2_0(\rho)`$; $`p`$ tests with weights $`\omega`$; response operators $`\mathcal R_0`$ at an anchor $`z_0`$ and $`\mathcal R_z`$ at a new checkpoint $`z`$, with response vectors $`g_0(y),g_z(y)\in[0,1]^p`$.
- $`\hat{\mathcal R}_0`$ is the anchor estimate, with error $`\epsilon_0`$, singular values $`\hat\sigma_k(z_0)`$ and projection $`\hat\Pi_r`$ onto its first $`r`$ right singular vectors.
- Paired experiment: random messages $`y_i\sim\rho`$, bounded outputs $`X^{(i)}_z,X^{(i)}_0`$ with the correct conditional means, $`D^{(i)}=X^{(i)}_z-X^{(i)}_0`$, $`W_i=\lVert D^{(i)}\rVert_\omega^2`$; $`M`$ frozen checkpoint pairs, $`n`$ messages each.

**Statement under test ([Proposition 8.14](../statements/08-14_anchor_reuse_drift_certificate.md#s8-14)).** If $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\Delta_z`$,

```math
\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}\le\hat\sigma_{r+1}(z_0)+\epsilon_0+\Delta_z ;
```

the drift satisfies $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2\le\mathbb E_\rho\lVert g_z-g_0\rVert_\omega^2\le\mathbb EW_i`$; with probability at least $`1-\delta`$, for all $`M`$ pairs, $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}\le\hat\Delta_z=\sqrt{\min\lbrace1,\bar W_z+\sqrt{\log(M/\delta)/(2n)}\rbrace}`$; and the product of two conditionally independent pair differences is an unbiased estimate of $`\lVert g_z(y)-g_0(y)\rVert_\omega^2`$.

| Check | Tests | Code ID | Result |
|---|---|---|---|
| [Check 8.20](#c8-20) | (a), (b), (d) | `v017:C19.anchor` | PASS |
| [Check 8.21](#c8-21) | (c), (d) at the measured noise | `v017:C20.scale`, `v017:R5.noise` | PASS; valid but vacuous |

<a id="c8-20"></a>

## Check 8.20 · Reuse residual, paired drift bound and the debiased product (Proposition 8.14(a), (b), (d))

**Verdict.**

- The residual of the anchor's rank-3 interface at the drifted checkpoint, 3.375, is below $`\hat\sigma_4(z_0)+\epsilon_0+\Delta_z=3.477`$.
- The squared operator drift, 0.094, is below the paired second moment, 0.330.
- The debiased product $`\langle D^{(1)},D^{(2)}\rangle_\omega`$ has conditional mean $`\lVert g_z-g_0\rVert_\omega^2`$ up to $`1.1\times10^{-16}`$.

All three: **PASS**.

**Why.** (a) is the reuse inequality with the full operator drift; (b) replaces an unknown operator difference by a measurable second moment of paired outputs; (d) is the debiasing that removes the process noise from that second moment. Each claim is an exact inequality or identity, so an instance with nonzero drift, nonzero noise and an exact computation tests it directly.

**Setup.**

- *Reuse.* Anchor $`A`$ with independent standard normal $`7\times10`$ entries, known exactly ($`\epsilon_0=0`$); new checkpoint $`A+E`$ with $`E`$ entries $`0.02`$ times standard normal; $`r=3`$; $`\Delta_z=\lVert E\rVert_{\rm op}`$.
- *Drift.* Six tests and eight messages; response values $`g_0,g_z`$ uniform on $`[0,1]`$; $`\rho`$ and $`\omega`$ Dirichlet with unit parameters. Outputs are Bernoulli with a common uniform variable inside a pair, $`X_z=\mathbf 1\lbrace U\le g_z\rbrace`$, $`X_0=\mathbf 1\lbrace U\le g_0\rbrace`$, so the marginals are correct and $`\mathbb E(X_z-X_0)^2=\lvert g_z-g_0\rvert`$.
- *Debiased product.* Two independent pairs per message (independent coupling); the conditional mean of the product is computed by exact enumeration over the three values of each difference, for every test and message.
- Seeded with `SeedSequence([20261014, tag])`; NumPy ([Harris et al., 2020](#ref-harris2020array)).

**Result** (`theory/v017_checks/results_v017.json`, entry `v017:C19.anchor`):

| quantity | value | bound or target |
|---|---:|---:|
| $`\lVert(A+E)(I-\hat\Pi_3)\rVert_{\rm op}`$ | 3.3746 | 3.4765 |
| $`\lVert\mathcal R_z-\mathcal R_0\rVert_{\rm op}^2`$ | 0.0942 | 0.3296 (paired second moment) |
| debiased product, max error of the conditional mean | $`1.1\times10^{-16}`$ | 0 |

**Why it supports the statement.**

- The reuse residual respects the triangle-inequality bound; the anchor's singular vectors were not assumed to be stable.
- The paired second moment bounds the squared drift, as (b) states, and exceeds it substantially here because the second moment contains the coupling noise $`\lvert g_z-g_0\rvert-(g_z-g_0)^2`$.
- The product of two conditionally independent differences has exactly the squared conditional mean as expectation, which is what removes that noise in (d).

<a id="c8-21"></a>

## Check 8.21 · The drift certificate at the measured evaluation noise (Proposition 8.14(c), (d))

**Verdict.**

- With the noise level measured on the real model, the strict radius $`\hat\Delta_z`$ is 0.718, 0.507, 0.404 and 0.227 for 10, 40, 100 and 1,000 pairs ($`M=10`$, $`\delta=0.05`$). A radius of 0.03 needs about $`4.9\times10^6`$ pairs, i.e. about $`10^7`$ continuation branches.
- The quantity it bounds, $`\lVert\mathcal R_z(I-\hat\Pi_r)\rVert_{\rm op}`$, is trivially at most $`\lVert\mathcal R_z\rVert_{\rm HS}\le0.5`$, so below about 40 pairs the certificate is worse than the trivial bound and above it remains far from useful: **valid but vacuous**.
- The estimator floor of $`\hat\Delta_z`$ at this noise is $`\sqrt{\bar W}\approx0.013`$; the debiased product of (d) removes it, but its radius still decays like $`n^{-1/4}`$.

**Why.** Proposition 8.14 makes reuse across checkpoints checkable with random messages instead of a full catalog. Whether that check is informative depends on the noise of a single paired run and on the range-based concentration term. The real-model noise reference provides the first number; the second is fixed by $`n`$, $`M`$ and $`\delta`$.

**Setup.**

- *Real-model noise reference* (`v017:R5.noise`): the final checkpoint of the reference cold-start training run, evaluated on 300 ALFWorld ([Shridhar et al., 2021](#ref-shridhar2021alfworld)) dev tasks on two identically configured inference servers; per-task success records stored under `theory/v017_checks/records/noise/`.
- If the 15 flips were symmetric $`\pm1/300`$ changes, the standard deviation of one score difference would be $`\sqrt{15}/300=0.0129`$ and $`\bar W=(\sqrt{15}/300)^2=1.67\times10^{-4}`$ on one test. This is serving noise only: two continuations with different training randomness would add their own noise, so the real $`\bar W`$ of a drift experiment is at least this large. One server pair at one checkpoint does not establish that the evaluation noise is irreducible or independent and identically distributed.
- Radius: $`\hat\Delta=\sqrt{\min\lbrace1,\bar W+\sqrt{\log(M/\delta)/(2n)}\rbrace}`$ with $`M=10`$, $`\delta=0.05`$ (`v017:C20.scale`).

**Result** (entries `v017:R5.noise` and `v017:C20.scale`):

| noise reference | value |
|---|---:|
| tasks | 300 |
| successes, server a / server b | 241 / 242 |
| task flips (a only / b only) | 15 (7 / 8) |
| SD of one score difference if flips are symmetric | 0.0129 |

| pairs $`n`$ | Hoeffding term | $`\hat\Delta`$ |
|---:|---:|---:|
| 10 | 0.515 | 0.718 |
| 40 | 0.257 | 0.507 |
| 100 | 0.163 | 0.404 |
| 1,000 | 0.051 | 0.227 |
| $`\approx4.9\times10^6`$ | — | 0.030 |

**Why it supports the statement.**

- The certificate of (c) is valid as stated; what the check shows is its size. The Hoeffding term pays the full range of $`W\in[0,1]`$ and does not shrink with the small observed $`\bar W`$, so the radius is dominated by $`n^{-1/4}`$ behaviour long before the noise floor matters.
- The vacuity is a property of this estimator and this concentration inequality. It is not a proof that drift across checkpoints cannot be certified cheaply; it marks what is open ([Remark 8.17](../statements/08-17_open_problems.md#s8-17)): a certificate whose radius scales with the effect.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v017_checks/check_v017.py
```

The suite uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## Artifacts

- `theory/v017_checks/results_v017.json`: entries `v017:C19.anchor`, `v017:C20.scale`, `v017:R5.noise`; code `theory/v017_checks/check_v017.py`, functions `c_anchor`, `c_scale`, `r_noise`.
- Real-model records: `theory/v017_checks/records/noise/a/per_task.tsv`, `theory/v017_checks/records/noise/b/per_task.tsv` (copied from the cluster run directory of the noise reference).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
