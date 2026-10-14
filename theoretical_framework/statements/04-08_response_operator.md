# Definition 4.8 · Response operator at a checkpoint

*Statement*

[← Framework](../README.md) · [Statement](04-08_response_operator.md) · [中文](04-08_response_operator_ZH.md)

**In one paragraph.** [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2) counts how many feature means must match for a whole response class, in the uniform norm. To measure *how much* each direction of feedback difference matters, and to compress with an explicit metric, collect the responses of many declared tests at one checkpoint into a single linear operator: it maps a difference between two feedback laws to the vector of score differences it causes on every test. This page fixes that operator, the measures that define its geometry, and the elementary facts that make it a Hilbert–Schmidt operator with a singular value decomposition. Its spectrum is studied in [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9).

**Setting used on this page.**

- A checkpoint $`(t,z)`$: round $`t`$ and complete learner state $`z`$, including optimizer state, buffers and history ([Definition 1.3](../README.md#s1-3)). The message space $`\mathcal Y`$ is a standard Borel space; the actual update is $`U_t(z,y,\xi_t)`$.
- A declared set of *tests* $`\mathcal A`$, also a standard Borel space. A test $`a\in\mathcal A`$ is a declared task score $`F_a\in[0,1]`$ together with a declared continuation protocol after round $`t`$. Different tests may use different continuation lengths or different legal training plans; tests with different remaining budgets must be labelled as different tests.
- The *response* of test $`a`$ is $`g_a(y)=\mathbb E[F_a\text{ after the continuation of test }a\mid z,\ \text{this round's message }y]\in[0,1]`$, the one-boundary response of [Definition 1.4](../README.md#s1-4) with the test's own continuation. $`(a,y)\mapsto g_a(y)`$ is jointly measurable.
- A *reference message measure* $`\rho`$ on $`\mathcal Y`$ and *test weights* $`\omega`$ on $`\mathcal A`$, both probability measures, declared before any feedback is selected.

<a id="s4-8"></a>

**Definition 4.8 (Response operator at a checkpoint).**

- The input space is $`H=L_0^2(\rho)=\lbrace h:\mathbb E_\rho h=0,\ \mathbb E_\rho h^2<\infty\rbrace`$ with $`\langle h,h'\rangle_\rho=\mathbb E_\rho[hh']`$. The centred response is $`\tilde g_a=g_a-\mathbb E_\rho g_a`$.
- The *response operator* $`\mathcal R_{t,z}:H\to L^2(\omega)`$ is

```math
(\mathcal R_{t,z}h)(a)=\langle\tilde g_a,h\rangle_\rho . \qquad(4.10)
```

- Its singular value decomposition, which exists by the facts below, is written $`\mathcal R_{t,z}h=\sum_{j\ge1}\sigma_j\langle v_j,h\rangle_\rho\,u_j`$ with $`\sigma_1\ge\sigma_2\ge\dots\ge0`$, orthonormal *input modes* $`v_j\in H`$ and orthonormal *output modes* $`u_j\in L^2(\omega)`$. The subscript $`(t,z)`$ is dropped when the checkpoint is fixed.

**Facts that come with the definition.** Each has a one-line justification.

- *Exact score differences.* If $`P,Q`$ have densities $`p,q\in L^2(\rho)`$, then $`h=p-q\in H`$ and $`(\mathcal Rh)(a)=\mathbb E_Pg_a-\mathbb E_Qg_a`$ for every test $`a`$. Indeed $`\mathbb E_\rho h=0`$, so $`\langle\tilde g_a,h\rangle_\rho=\int g_a\,dP-\int g_a\,dQ`$.
- *Where the linearity comes from.* $`\mathcal R`$ is linear because expectations are linear in the law of the message. The update $`U_t`$, the optimizer state and every later step may be arbitrarily nonlinear; nothing is linearized.


*Hilbert–Schmidt.* $`\mathcal R`$ is the integral operator with kernel $`\tilde g_a(y)`$, so

```math
\lVert\mathcal R\rVert_{\rm HS}^2=\mathbb E_{a\sim\omega}\mathrm{Var}_{Y\sim\rho}\,g_a(Y)\le\tfrac14, \qquad(4.11)
```

because a $`[0,1]`$-valued variable has variance at most $`\tfrac14`$. Hence $`\mathcal R`$ is compact, the singular value decomposition exists ($`L^2`$ spaces over standard Borel spaces are separable), and $`\sum_j\sigma_j^2\le\tfrac14`$. No finite rank is assumed.

- *Rank caps.* One test gives rank at most 1; $`p`$ tests give rank at most $`p`$; $`m`$ messages give rank at most $`m-1`$. A low rank caused by few declared tests carries no structural information.
- *Mixture reference measures.* For finitely many candidate laws $`P^1,\dots,P^J`$, the choice $`\rho=J^{-1}\sum_jP^j`$ gives every candidate and every convex mixture of candidates a density in $`[0,J]`$, and any two such laws a difference with $`\lVert h\rVert_\rho^2\le2J`$, since $`\mathbb E_\rho(p-q)^2\le\mathbb E_\rho(p^2+q^2)\le J(\mathbb E_\rho p+\mathbb E_\rho q)=2J`$. For two candidates mixed equally this is $`\lVert h\rVert_\rho^2\le4`$. A message, candidate or support outside the declared catalog needs a new $`\rho`$ or an explicit coverage cost.
- *What the operator depends on.* $`\rho`$, $`\omega`$, the declared tests, the checkpoint (parameters and optimizer state) and the declared continuation protocols. Changing $`\omega`$ changes the singular values and must be reported. An invertible reparametrization of the parameters that leaves the actual process and tests unchanged leaves $`\mathcal R`$ unchanged: the operator is defined by observable consequences.
- *Several checkpoints.* Different checkpoints generally receive different feedback differences $`h_z`$. Rows of $`\mathcal R_{t,z}`$ from different checkpoints must not be stacked as if one common input $`h`$ acted on all of them; the conditional version is [Proposition 8.13](08-13_shared_features_across_checkpoints.md#s8-13).

**What it says.**

- At a fixed checkpoint, the effect of replacing one feedback law by another on every declared test is one linear map applied to the density difference.
- The map is Hilbert–Schmidt with total squared gain at most $`\tfrac14`$, so its singular values rank the directions of feedback difference by how much score change they cause.

**Intuition.** The operator asks, for each direction in which two feedback laws can differ, how much every declared test would move after the learner consumes this round's message and finishes its declared training.

**Role in the framework.**

- It is the object whose spectrum gives an explicit-metric version of the sufficiency counts of §4 ([Theorem 4.9](04-09_spectral_sufficiency.md#s4-9)), whose dynamic counterparts are the Hankel matrices of [Theorem 4.10](04-10_hankel_rank_predictive_realization.md#s4-10), and whose estimation, reuse and cost are the subject of [Propositions 8.12–8.15](08-12_finite_catalog_spectral_certificate.md#s8-12).
- [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12) factors it through the expected change of the next predictive state.

**Uses.**

- [Definitions 1.3–1.4](../README.md#s1-3): complete learner state and one-boundary response.
- [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2): responses are the right invariant.

**Used by.**

- [Theorem 4.9](04-09_spectral_sufficiency.md#s4-9): spectral sufficiency and linear interface complexity.
- [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12): the factorization through the predictive state.
- [Proposition 8.12](08-12_finite_catalog_spectral_certificate.md#s8-12): finite-catalog estimation of the spectrum.
- [Proposition 8.13](08-13_shared_features_across_checkpoints.md#s8-13): shared message features across checkpoints.

## Related work

- Predicting how outputs change when the training data change is the subject of datamodels ([Ilyas et al., 2022](#ref-ilyas2022datamodels)); the response operator is a state-conditional, one-round, continuation-based relative of that object, and the counterfactual matrix itself is not new.
- The singular value decomposition of a compact operator and the optimality of its truncation go back to Eckart and Young in the matrix case ([Eckart & Young, 1936](#ref-eckart1936approximation)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-eckart1936approximation"></a>

**Carl Eckart, Gale Young (1936).** The approximation of one matrix by another of lower rank. *Psychometrika* 1, no. 3, 211–218. DOI [10.1007/BF02288367](https://doi.org/10.1007/BF02288367).

<a id="ref-ilyas2022datamodels"></a>

**Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry (2022).** Datamodels: Understanding predictions with data and data with predictions. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, PMLR 162:9525–9587. [proceedings.mlr.press/v162/ilyas22a](https://proceedings.mlr.press/v162/ilyas22a.html); preprint arXiv [2202.00622](https://arxiv.org/abs/2202.00622) (titled "Datamodels: Predicting predictions from training data").
