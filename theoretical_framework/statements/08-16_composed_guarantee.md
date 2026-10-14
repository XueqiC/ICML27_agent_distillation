# Corollary 8.16 · Composed guarantee from reuse, estimation and compilation

*Statement*

[← Framework](../README.md) · [Statement](08-16_composed_guarantee.md) · [Proof](../proofs/08-16_composed_guarantee.md) · [Experiments](../experiments/08-16_composed_guarantee.md) · [中文](08-16_composed_guarantee_ZH.md)

**In one paragraph.** The pieces of §3, §6 and §8 fit into one guarantee for a whole continued-training run. At every certified state the local response difference between the target feedback and the compiled legal feedback is bounded by four measurable terms: the information–compilation radius, the anchor interface's discarded spectrum, the anchor's estimation error, and the drift from the anchor to the current state. Summed along the run with the stopped joint decomposition, they bound the final gap together with the dynamic, terminal and exit terms, none of which is set to zero. The guarantee is valid. At the measured scale it is vacuous for any realistic density bound $`B`$: its terms are absolute precisions from bounded-range concentration and do not shrink with the effect of one step. With ten replicates the floor of a single step is $`B\times1.296`$, about 2.6 at the two-candidate worst case $`B=2`$, against a trivial bound of 1 on the whole gap.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- *Two protocols.* The reference (target) protocol emits $`P^0_t(\cdot\mid z)`$ and the alternative emits a compiled legal law $`P^S_t(\cdot\mid z)`$; both use the same actual update $`U_t`$ and start from the same initial law. $`L_E(0)`$ and $`L_E(S)`$ are their final real-task scores; $`\mathbb E_0`$, $`\mathbb E_S`$ are expectations along the two chains.
- *Approximate value* (as in [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4)). Fixed functions $`\widehat V_t\in[0,1]`$, $`t\le R`$; reference Bellman residual $`r^0_t=K^0_t\widehat V_{t+1}-\widehat V_t`$; terminal error $`e_R=F_E-\widehat V_R`$; approximate response $`\bar g_{t,z}(y)=\mathbb E_{\xi_t}\widehat V_{t+1}(U_t(z,y,\xi_t))`$; local response difference $`d_t(z)=\int\bar g_{t,z}\,d(P^0_t-P^S_t)(\cdot\mid z)`$.
- *Response operators.* A common reference measure $`\rho`$ on the legal message space, $`\mathcal H=L^2_0(\rho)`$, a finite test set $`\mathcal A`$ with weights $`\omega_a>0`$, and the response operator $`\mathcal R_{t,z}`$ of [Definition 4.8](04-08_response_operator.md#s4-8) at each state. The test set contains, for every $`(t,z)`$, a test $`a_t`$ whose response is $`\bar g_{t,z}`$ (apply the actual update, then read $`\widehat V_{t+1}`$).
- *Anchors.* Anchors $`j=1,\dots,J`$ with estimated operators $`\hat{\mathcal R}_j`$ ([Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12)) and true operators $`\mathcal R_j`$. The first $`r_j`$ singular triples $`(\hat\sigma_{j,k},\hat u_{j,k},\hat v_{j,k})`$ of $`\hat{\mathcal R}_j`$ give message features $`\phi_j=(\hat v_{j,1},\dots,\hat v_{j,r_j})`$, the coefficient map $`C_j\mu=\sum_{k\le r_j}\hat\sigma_{j,k}\mu_k\hat u_{j,k}`$, and the mode means $`\mu^j_P=\mathbb E_P\phi_j`$. $`\hat\Pi_j`$ projects onto $`\hat v_{j,1},\dots,\hat v_{j,r_j}`$.
- *Information–compilation radius* (as in [Proposition 6.16](06-16_information_compilation_radius.md#s6-16)). For a compact confidence set $`\mathcal M\subset\mathbb R^{r_j}`$ for the target's mode means and the compact set $`\mathcal K_j=\lbrace C_j\mu^j_Q:Q\text{ legal}\rbrace`$ of legally reachable outputs,

```math
\mathfrak r(\mathcal M,\mathcal K_j)=\min_{q\in\mathcal K_j}\ \sup_{\mu\in\mathcal M}\ \lVert C_j\mu-q\rVert_\omega .
```

- *Certified states.* On an event $`\mathcal E`$ of the audit data, a state $`z`$ at round $`t`$ is certified for the alternative chain, $`z\in C^S_t`$, when, with the anchor $`j=j(t,z)`$ chosen for it:
  1. $`\lVert\hat{\mathcal R}_j-\mathcal R_j\rVert_{\rm op}\le\epsilon_j`$ and $`\lVert\mathcal R_{t,z}-\mathcal R_j\rVert_{\rm op}\le\Delta_{t,z}`$;
  2. $`\mu^j_{P^0_t(\cdot\mid z)}\in\mathcal M_{t,z}`$, and the compiled law satisfies $`\sup_{\mu\in\mathcal M_{t,z}}\lVert C_j\mu-C_j\mu^j_{P^S_t(\cdot\mid z)}\rVert_\omega\le\mathfrak r_{t,z}+\eta_{\rm opt}`$ with $`\mathfrak r_{t,z}=\mathfrak r(\mathcal M_{t,z},\mathcal K_j)`$;
  3. both laws have densities with respect to $`\rho`$, and $`h_{t,z}=dP^0_t/d\rho-dP^S_t/d\rho`$ satisfies $`\lVert h_{t,z}\rVert_\rho\le B_{t,z}`$.

  $`\tau^S`$ is the first round at which the alternative chain is outside $`C^S_t`$; for the reference chain any certified sets $`C^0_t`$ may be used (for example the whole space), with exit time $`\tau^0`$.

<a id="s8-16"></a>

**Corollary 8.16 (Composed guarantee from reuse, estimation and compilation).** In this setting, on the event $`\mathcal E`$:

**(a) Local bound.** For every certified $`z\in C^S_t`$, with $`j=j(t,z)`$,

```math
\lvert d_t(z)\rvert\le\beta_t(z):=\frac{\mathfrak r_{t,z}+\eta_{\rm opt}+B_{t,z}\big(\hat\sigma_{j,r_j+1}+\epsilon_j+\Delta_{t,z}\big)}{\sqrt{\omega_{a_t}}} .
```

**(b) Run-level bound.**

```math
\lvert L_E(0)-L_E(S)\rvert\le\mathbb E_S\sum_{t<\tau^S}\beta_t(Z_t)+\mathcal E_{\rm dyn}+\mathcal E_{\rm terminal}+\mathcal E_{\rm exit},
```

with $`\mathcal E_{\rm dyn}=\mathbb E_S\sum_{t<\tau^S}\lvert r^0_t(Z_t)\rvert+\mathbb E_0\sum_{t<\tau^0}\lvert r^0_t(Z_t)\rvert`$, $`\mathcal E_{\rm terminal}=\mathbb E_0[\lvert e_R\rvert\mathbf 1\lbrace\tau^0=R\rbrace]+\mathbb E_S[\lvert e_R\rvert\mathbf 1\lbrace\tau^S=R\rbrace]`$ and $`\mathcal E_{\rm exit}=\Pr_0(\tau^0<R)+\Pr_S(\tau^S<R)`$. No term is set to zero.

**(c) Size at the measured scale.** With one test ($`\omega_{a_t}=1`$), $`B_{t,z}=2`$, anchor error $`\epsilon_j=\varepsilon_n`$ from [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12) at $`n=10`$ replicates on a catalog of 20 candidates ($`\varepsilon_n=0.578`$), and the drift radius of [Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) at the measured evaluation noise with 10 pairs ($`\hat\Delta=0.718`$), the per-step floor $`B(\epsilon_j+\Delta_{t,z})`$ is 2.59 before the radius terms and before the sum over steps ([Check 8.23](../experiments/08-16_composed_guarantee.md#c8-23)); these values use $`p=1`$ test, $`m=20`$ candidates and $`\delta=0.05`$, so $`\epsilon_j+\Delta_{t,z}\approx1.296`$. The floor is linear in $`B_{t,z}`$. $`B=2`$ is the worst case for a mixture of two candidates; under the mixture measure of the 20-candidate catalog the worst case is $`\sqrt{40}\approx6.3`$; a compiled law close to its target can have a much smaller $`B`$, and $`B=0.5`$ gives 0.65 per step. Even then the floors of two or more audited steps sum to more than 1, and the floor does not shrink with the effect of a step. Since $`\lvert L_E(0)-L_E(S)\rvert\le1`$ always, the bound is valid but vacuous at this scale unless $`B_{t,z}`$ is small and only one step is audited.

**What it says.**

- Interface reuse, response estimation, information acquisition and legal compilation enter one guarantee for continued training, each through its own measurable term, and the dynamic, terminal and exit terms are kept.
- The guarantee pays the full response drift, so a change of the retained modes' coefficients between anchor and state is paid too, not only drift in the discarded directions.
- With bounded-range concentration, each step's terms are absolute precisions. The sum over steps therefore grows with the number of steps even when each step's effect is tiny, and at the measured scale the bound is far above the trivial bound.

**Intuition.** Theorem-level bookkeeping is exact; what makes it uninformative is that every estimated term is certified at a precision that does not know how small the quantity it bounds is. A certificate of a sum of many small effects built from absolute per-term precisions is a sum of many large radii.

**Role in the framework.**

- It is the composed statement of the framework's maintenance chain: [Proposition 6.16](06-16_information_compilation_radius.md#s6-16) (what to acquire, what can be compiled), [Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate.md#s8-12) (estimation, reuse, cost), and [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4) (accumulation along the run).
- Its practical role at the current scale is as an error budget: it says which term to reduce, not that the gap is small.

**Comments on the statement.**

- **Three problems are distinct.**
  1. Estimating the gain of one fixed protocol pair: directly compare the two protocols, or estimate the signed identity of Proposition 3.4(a) with paired runs.
  2. Certifying every local response simultaneously, which is what (a)–(b) do.
  3. Guaranteeing sufficiency for a whole class of future protocols.

  Their costs differ. With $`R`$ terms and per-query noise $`\sigma`$, estimating the signed sum to mean squared error $`\varepsilon^2`$ needs about $`\sigma^2R^2/\varepsilon^2`$ queries under uniform allocation, while certifying each term to $`\varepsilon/R`$ and adding absolute values needs about $`\sigma^2R^3/\varepsilon^2`$ ([Check 8.23](../experiments/08-16_composed_guarantee.md#c8-23): 400 against 8,000 at $`R=20`$, $`\sigma=\varepsilon=1`$). A random factorial design over the $`R`$ choices, as in datamodels ([Ilyas et al., 2022](#ref-ilyas2022datamodels)), also reaches the $`R^2`$ scale for all terms at once when the responses are close to additive. The $`R^3`$ figure is the cost of this certification scheme, not a lower bound for the problem.
- **What makes it informative.** The realistic route is to let the approximate value carry the long-range part: each term then needs only one-step updates and evaluations of $`\widehat V`$, and the cost moves into the Bellman residuals and into fitting $`\widehat V`$. This is the structure of doubly robust off-policy evaluation, a value model plus a correction, which has a matching lower bound in some cases ([Jiang & Li, 2016](#ref-jiang2016doubly)); a version for continued training must be compared with it, and with predictive-state temporal-difference learning ([Boots & Gordon, 2010](#ref-boots2010predictive)).
- **Where the event must hold.** $`\mathcal E`$ must be valid at every certified state and for every adaptive choice of anchor and compiled law. If the conditions are established only at finitely many audited states, only those states are certified: the chain must actually pass through them (audits at the states it reaches, with validity conditional on the visit), or the exit term absorbs the rest. Elsewhere one needs uniform control.
- **Per-test cost.** The factor $`1/\sqrt{\omega_{a_t}}`$ can be avoided by auditing the row of $`a_t`$ directly.

**Derivation outline.**

- Decompose $`\mathcal R_{t,z}h=\hat{\mathcal R}_j\hat\Pi_jh+\hat{\mathcal R}_j(I-\hat\Pi_j)h+(\mathcal R_{t,z}-\hat{\mathcal R}_j)h`$.
- The first term equals $`C_j(\mu^j_{P^0}-\mu^j_{P^S})`$, bounded by the radius condition because the target's mode means lie in $`\mathcal M_{t,z}`$.
- The other two are bounded by $`B_{t,z}\hat\sigma_{j,r_j+1}`$ and $`B_{t,z}(\epsilon_j+\Delta_{t,z})`$.
- Evaluate at the test $`a_t`$, whose response difference is $`d_t(z)`$, and pay $`1/\sqrt{\omega_{a_t}}`$.
- Insert into Proposition 3.4(d).

**Uses.**

- [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4) — stopped joint decomposition
- [Proposition 6.16](06-16_information_compilation_radius.md#s6-16) — information–compilation radius
- [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12) — anchor estimate
- [Proposition 8.14](08-14_anchor_reuse_drift_certificate.md#s8-14) — drift between anchor and state
- [Proposition 8.15](08-15_two_fidelity_response_estimation.md#s8-15) — cheaper anchor estimation

**Used by.**

- [Remark 8.17](08-17_open_problems.md#s8-17) — what remains to make the composed guarantee informative

## Related work

- The run-level step is the performance-difference bookkeeping of [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4) ([Kakade & Langford, 2002](#ref-kakade2002approximately)). A value model with a correction term is the doubly robust construction of off-policy evaluation ([Jiang & Li, 2016](#ref-jiang2016doubly)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-boots2010predictive"></a>

**Byron Boots, Geoffrey J. Gordon (2010).** Predictive state temporal difference learning. *Advances in Neural Information Processing Systems 23 (NIPS)*, Curran Associates. [proceedings.neurips.cc/paper/2010/hash/3473decccb0509fb264818a7512a8b9b](https://proceedings.neurips.cc/paper_files/paper/2010/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").

<a id="ref-jiang2016doubly"></a>

**Nan Jiang, Lihong Li (2016).** Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48:652–661. [proceedings.mlr.press/v48/jiang16](https://proceedings.mlr.press/v48/jiang16.html).

<a id="ref-kakade2002approximately"></a>

**Sham Kakade, John Langford (2002).** Approximately optimal approximate reinforcement learning. *Proceedings of the 19th International Conference on Machine Learning (ICML)*, 267–274, Morgan Kaufmann. [dl.acm.org/doi/10.5555/645531.656005](https://dl.acm.org/doi/10.5555/645531.656005).
