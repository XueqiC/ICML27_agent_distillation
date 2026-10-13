# Corollary 8.9 · The first-order version, its margin criterion, and its cap

*Statement*

[← Framework](../README.md) · [Statement](../statements/08-09_first_order_version_margin_criterion.md) · [Proof](../proofs/08-09_first_order_version_margin_criterion.md) · [Experiments](../experiments/08-09_first_order_version_margin_criterion.md) · [中文](08-09_first_order_version_margin_criterion_ZH.md)

**In one paragraph.** Gradient-based scores rank feedback laws by the first-order term of their effect. This corollary says exactly when such a ranking is guaranteed (the margin criterion), what a measured first-order certificate must pay on top of the statistical radius, and which pairs it can never certify (the cap). It also separates scores that use the exact task continuation gradient, which only controlled experiments have, from measurable certificates.

**Setting used on this page.**

The objects of [Definitions 1.3–1.4](../README.md#s1-3) are used throughout.

- Rounds are $`t=0,\dots,R-1`$, and $`z`$ is a complete learner state. For a first-order optimizer the actual update is $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$.
- At state $`z`$ in round $`t`$ the reference emits $`Y\sim P^0_{t,z}`$, and candidate $`j\in\lbrace1,\dots,M\rbrace`$ emits $`Y\sim P^j_{t,z}`$. The kernel of candidate $`j`$ is $`K^j_t`$.
- The reference continuation is $`V^0_{t+1}`$, with $`V^0_R=F_E\in[0,1]`$. The response is $`g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi))\in[0,1]`$. Integrating the message gives the kernel identity

```math
(K^j_tV^0_{t+1})(z)=\mathbb E_{Y\sim P^j_{t,z}}\,g_{t,z}(Y),\qquad j\in\lbrace0,1,\dots,M\rbrace .\qquad\text{(K)}
```

- $`L_E(j)=\mathbb E\,F_E(Z^j_R)`$ for the process driven by candidate $`j`$. All processes start from the same initial law of $`Z_0`$.
- The *local advantage* of candidate $`j`$ at $`(t,z)`$ is

```math
A_t(z,j)=(K^j_t-K^0_t)V^0_{t+1}(z)\in[-1,1].
```

- It is the effect on the final score of replacing one reference round by one round of candidate $`j`$, followed by the reference tail. By [Theorem 3.2](../statements/03-02_continuation_response_characterization.md#s3-2), the whole-run difference is a sum of such terms over the states the alternative process reaches.

*The paired-replacement score* ([Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6)).

- Fix an audited state $`z`$ at round $`t`$ (a complete learner state, including the optimizer state), conditionally on the history that produced it. The candidates are $`j=0,1,\dots,M`$ with $`M\ge1`$; $`j=0`$ is the reference. Candidate $`j`$ has message law $`P^j=P^j_t(\cdot\mid z)`$, and a message $`Y`$ produces the increment $`\Delta^j=\Delta_t(z,G_t(z,Y,\xi))`$, update randomness included.
- A *surrogate tail* $`T_\omega`$ maps a state to a final state by running a declared (possibly truncated) reference tail with randomness $`\omega`$. A *surrogate objective* $`\tilde F`$ scores the final state (for example, the teacher-forced log-likelihood of held-out successful trajectories, mapped to $`[0,1]`$). Write

```math
\varphi_\omega(u)=\tilde F\big(T_\omega(z+u)\big),\qquad\tilde L(P^j)=\mathbb E\,\varphi_\omega(\Delta^j),
```

where $`\omega`$ is independent of $`\Delta^j`$. $`\tilde L(P^j)`$ is the surrogate outcome of one round with candidate $`j`$ followed by the surrogate tail.

- *Replicas* $`\ell=1,\dots,r`$, $`r\ge2`$: draw a fresh $`\omega_\ell`$ and, for every candidate, one message and its increment $`\Delta^j_\ell`$. All candidates in replica $`\ell`$ share $`\omega_\ell`$ (*common random numbers*); their messages may also be coupled through common random numbers, as long as each $`\Delta^j_\ell`$ has its correct marginal law. Replicas are independent of each other.
- If a candidate's message law has finite support, the draw may be replaced by the exact weighted average over the support; the randomness then comes only from $`\omega`$.
- The score of the ordered pair $`(j,k)`$ is

```math
Z^{jk}_\ell=\varphi_{\omega_\ell}(\Delta^j_\ell)-\varphi_{\omega_\ell}(\Delta^k_\ell),\qquad\hat S^{jk}=\frac1r\sum_{\ell=1}^rZ^{jk}_\ell .
```

- On the task side, $`L_F(P^j)=\mathbb E\,F_E\big(T^{\rm full}(z+\Delta^j)\big)`$ is the outcome of one round with candidate $`j`$ followed by the full reference tail and the task objective. In the notation of the setting, $`L_F(P^j)=(K^j_tV^0_{t+1})(z)`$, so $`L_F(P^j)-L_F(P^0)=A_t(z,j)`$.

*Its assumptions* ([Assumption 8.7](../statements/08-07_certificate_assumptions.md#s8-7)).

- **(I) Independent replicas.** Conditionally on the history that produced $`z`$, the vectors $`(Z^{jk}_\ell)_{j,k}`$, $`\ell=1,\dots,r`$, are i.i.d.; $`r`$ is fixed before the scores are observed.
- **(B) Bounded scores.** For every pair, $`Z^{jk}_\ell`$ lies almost surely in a known interval of length $`B`$. For example $`\tilde F\in[0,1]`$ gives $`B\le2`$.
- **(A) Surrogate alignment.** There are $`\alpha>0`$ and $`\varepsilon_{\rm sur}\ge0`$ such that, for all pairs,

```math
\Big\lvert\big(L_F(P^j)-L_F(P^k)\big)-\alpha\big(\tilde L(P^j)-\tilde L(P^k)\big)\Big\rvert\le\alpha\,\varepsilon_{\rm sur}.
```

$`\varepsilon_{\rm sur}`$ absorbs both the difference between the surrogate and the task objective and the truncation of the tail.

<a id="s8-9"></a>

**Corollary 8.9 (The first-order version, its margin criterion, and its cap).** Use the setting of [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6) and let $`\mu^j=\mathbb E\Delta^j`$ and $`m^j_2=\mathbb E\lVert\Delta^j\rVert^2`$.

- **(a) Margin criterion.** Let $`V`$ be a continuation that is twice differentiable along the reached segments with $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$. If $`\lvert\langle\nabla V(z),\mu^j-\mu^k\rangle\rvert>\tfrac{M_2}{2}(m^j_2+m^k_2)`$, then $`(K^j_t-K^k_t)V(z)`$ has the sign of $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$. At run level, let $`T_{jk}=\sum_t\mathbb E_{d^k_t}\langle\nabla V^j_{t+1}(Z_t),\mu^j_t-\mu^k_t\rangle`$ and $`B_{jk}=\sum_t\mathbb E_{d^k_t}\tfrac{M_{2,t}}2(m^j_{2,t}+m^k_{2,t})`$ be the first-order term and the remainder bound of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) with $`P_j`$ in the role of $`P`$; here $`d^k_t`$ is the state law at round $`t`$ of the process driven by $`P_k`$. If $`\lvert T_{jk}\rvert>B_{jk}`$, then $`L(P_j)-L(P_k)`$ has the sign of $`T_{jk}`$.
- **(b) Measured first-order certificate.** Let $`\tilde V(u)=\mathbb E_\omega\varphi_\omega(u)`$ be twice differentiable along $`\lbrace su:s\in[0,1]\rbrace`$ for the reached increments $`u`$, with $`\lvert D^2\tilde V(su)[u,u]\rvert\le\tilde M\lVert u\rVert^2`$, and let $`\tilde s^{jk}=\langle\nabla\tilde V(0),\mu^j-\mu^k\rangle`$. For a step $`h\in(0,1]`$ define $`Z^{jk}_\ell(h)=\big(\varphi_{\omega_\ell}(h\Delta^j_\ell)-\varphi_{\omega_\ell}(h\Delta^k_\ell)\big)/h`$ with mean $`\hat S^{jk}(h)`$, and assume (I), (A), and (B) with range length $`B_h`$. Let $`e^{jk}_{\rm stat}(h)`$ be the radius of [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) with $`B_h`$ in place of $`B`$, and

```math
e^{jk}_{\rm fd}(h)=\tfrac12h\tilde M\big(m^j_2+m^k_2\big),\qquad\tilde e^{jk}_2=\tfrac12\tilde M\big(m^j_2+m^k_2\big).
```

Then with probability at least $`1-\delta`$, simultaneously for all ordered pairs,

```math
L_F(P^j)-L_F(P^k)\ \ge\ \alpha\big(\hat S^{jk}(h)-e^{jk}_{\rm stat}(h)-e^{jk}_{\rm fd}(h)-\tilde e^{jk}_2-\varepsilon_{\rm sur}\big).
```

- **(c) Cap.** On the event of (b), a pair is certified by (b) only if $`\tilde s^{jk}>\tilde e^{jk}_2+\varepsilon_{\rm sur}`$. A pair whose first-order term does not exceed the second-order remainder bound is never certified at first order, whatever its true gap. In particular this holds for pairs whose mean increments agree along $`\nabla\tilde V(0)`$, which differ only through second and higher moments of the increment; [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) can certify such pairs. In population terms, (b) certifies the lower bound $`\tilde s^{jk}-\tilde e^{jk}_2\le\tilde L(P^j)-\tilde L(P^k)`$, the quantity that Theorem 8.8 estimates directly.
- **(d) Oracle scores are not certificates.** A score $`\langle\nabla V(z),\mu^j-\mu^k\rangle`$ computed with the exact continuation gradient of the task, as in controlled experiments that differentiate through the whole reference tail, satisfies (a) wherever the margin holds. It is not a measurable certificate: $`\nabla V(z)`$ and $`M_2`$ are not available from feedback samples, and checking the margin needs both. A cheap score ranks correctly by (a) only if (i) its difference equals the first-order term with the optimizer-transformed mean increment of the actual learner ([Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10)) at the actual state, and (ii) the margin holds. Scores built from the raw mean gradient, or with coefficients shared across states and tasks, violate (i); any score used where the margin fails violates (ii).

**What it says.** A first-order score ranks two candidates correctly whenever their first-order difference exceeds the second-order remainder. Measured by finite differences, the first-order certificate pays a finite-difference term and the remainder on top of the statistical radius, and it can never certify a pair whose first-order term is below the remainder, whatever the true gap.

**Intuition.** Candidates that differ mainly through second moments of their increments, as under Adam, are invisible to any first-order certificate.

**Role in the framework.** It marks where gradient-based scores are trustworthy and why Theorem 8.8 measures finite replacement instead; it also separates oracle diagnostics from measurable certificates.

**Comments on the statement.**

- **Why the first-order version is weak.** Two effects add. The a-priori range of $`Z(h)`$ grows like $`1/h`$ (for $`\tilde F\in[0,1]`$, $`B_h\le2/h`$), so the range term of $`e_{\rm stat}(h)`$ grows as $`h`$ shrinks, while $`e_{\rm fd}(h)`$ grows with $`h`$. And the remainder $`\tilde e_2`$ does not shrink with $`r`$ at all.
- **Where the margin holds.** When every coordinate on which the continuation is curved moves by $`O(\eta)`$ (SGD, momentum, sign updates on the parameter block), the first-order term is $`O(\eta)`$ and the remainder $`O(\eta^2)`$ per round, so the margin holds at small step sizes. Under Adam the moment increments $`\Delta m=(1-\beta_1)(G-m)`$ and $`\Delta v=(1-\beta_2)(G\odot G-v)`$ are $`O(1)`$, while the continuation depends on $`(m,v)`$ only through steps of size $`\eta`$. The first-order term and the remainder are then both $`O(\eta)`$, and a small step size does not imply the margin. This is the regime of the second-moment channel in §7 ([Theorem 7.1](../statements/07-01_state_channel_decomposition.md#s7-1)).

**Derivation outline.**

- The margin criterion is the second-order Taylor bound: the remainder is smaller than the first-order term, so it cannot flip the sign.
- For the measured version, Taylor's theorem along $`s\mapsto\tilde V(shu)`$ gives a finite-difference bias of order $`h`$ and, at $`h=1`$, the remainder; empirical Bernstein bounds the sampling error as in Theorem 8.8.
- The cap follows by chaining these inequalities on the confidence event.

**Illustration.** Under Adam the moment increments are of order one while the continuation depends on them only through steps of size $`\eta`$; the first-order term and the remainder are then of the same order, so a small learning rate does not make the margin hold.

**Uses.**

- [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1) — first-order expansion and its remainder
- [Theorem 5.10](../statements/05-10_first_order_statistic_by_learner.md#s5-10) — the optimizer-transformed mean increment
- [Proposition 8.4](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) — Taylor bound at one state
- [Definition 8.6](../statements/08-06_crn_paired_replacement_score.md#s8-6) — the paired-replacement score
- [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) — the finite certificate it is compared with

**Used by.**

- [Remark 7.5](../statements/07-05_where_first_order_rankers_fail.md#s7-5) — where first-order rankers fail
- [Remark 9.3](../statements/09-03_derivation_design_rules.md#s9-3) — when first-order screening is allowed

## Related work

- First-order data-attribution scores, such as influence functions ([Koh & Liang, 2017](#ref-koh2017understanding)), TracIn ([Pruthi et al., 2020](#ref-pruthi2020estimating)), In-Run Data Shapley ([Wang et al., 2025](#ref-wang2025data)) and metagradients ([Engstrom et al., 2025](#ref-engstrom2025optimizing)), use gradients of an objective with respect to training data or hyperparameters. [Corollary 8.9](../statements/08-09_first_order_version_margin_criterion.md#s8-9) states when such first-order rankings are guaranteed and why they cannot certify pairs whose first-order term is below the remainder. Datamodels ([Ilyas et al., 2022](#ref-ilyas2022datamodels)) estimate finite retraining effects by regression over many training runs; Theorem 8.8 measures one-round finite effects at a fixed state with paired tails.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-engstrom2025optimizing"></a>

**Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry (2025).** Optimizing ML training with metagradient descent. arXiv preprint. arXiv [2503.13751](https://arxiv.org/abs/2503.13751).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").

<a id="ref-koh2017understanding"></a>

**Pang Wei Koh, Percy Liang (2017).** Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1885–1894. [proceedings.mlr.press/v70/koh17a](https://proceedings.mlr.press/v70/koh17a.html).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-wang2025data"></a>

**Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia (2025).** Data Shapley in one training run. *13th International Conference on Learning Representations (ICLR)*. arXiv [2406.11011](https://arxiv.org/abs/2406.11011).
