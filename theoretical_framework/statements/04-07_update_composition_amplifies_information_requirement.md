# Theorem 4.7 · Update composition amplifies the information requirement

*Statement*

[← Framework](../README.md) · [Statement](04-07_update_composition_amplifies_information_requirement.md) · [Proof](../proofs/04-07_update_composition_amplifies_information_requirement.md) · [Experiments](../experiments/04-07_update_composition_amplifies_information_requirement.md) · [中文](04-07_update_composition_amplifies_information_requirement_ZH.md)

**In one paragraph.** For a learner that averages experience blocks, [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) found that the information a source must carry is its moments up to the deployment length $`N`$. Real learners are not averages: each update depends nonlinearly on the current parameters and on the new data. This theorem shows what that does. Composed polynomial updates multiply the required moment order, and the bound is attained; a discontinuous (threshold) update breaks moment sufficiency altogether; for one smooth SGD step the first-order statistic is the mean gradient and the second-order statistic its second moment; and per-step errors add up linearly only under a stability condition. The information requirement is a property of the learner composed with the task, not of the data alone.

**Setting used on this page.**

- *Polynomial updates.* The student parameter is $`p\in[0,1]^k`$, the learner starts from a common $`p_0`$ and processes independent blocks $`Z_1,\dots,Z_R`$ with values in a finite set $`\mathcal Y\subset[0,1]^k`$; block $`Z_i`$ has law $`P_i`$ under the reference source and $`Q_i`$ under the alternative. The update is $`p_i=U_i(p_{i-1},Z_i)`$. A deployment with at most $`N`$ decisions has a polynomial value $`J(p)`$ of total degree at most $`N`$ ([Theorem 4.4(a)](04-04_affine_learners_moment_order.md#s4-4)).
- *Smooth local setting.* $`\theta\in\mathbb R^p`$ is a parameter vector and $`V`$ a continuation value: the expected final score when training continues from $`\theta`$ under the reference procedure ([Definition 1.4](../README.md#s1-4)). One SGD step with step size $`\eta`$ and consumed-gradient law $`P`$ maps $`V`$ to $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$; $`\mu_P=\mathbb E_PG`$ is the mean gradient and $`M_P=\mathbb E_PGG^\top`$ its second moment. $`D^3V`$ is the third derivative.
- *Error recursion.* $`e_t`$ is an accumulated error after $`t`$ steps; $`\eta`$, $`L`$ and $`C`$ are a step size, a per-step amplification constant and a per-step error constant.

<a id="s4-7"></a>

**Theorem 4.7 (Update composition amplifies the information requirement).** The setting has three parts.

- **Polynomial updates.** Fix $`R\ge1`$ and a common $`p_0\in[0,1]^k`$. Independent blocks $`Z_1,\dots,Z_R`$ take values in a finite set $`\mathcal Y\subset[0,1]^k`$; block $`Z_i`$ has law $`P_i`$ under the reference source and $`Q_i`$ under the alternative. The learner is $`p_i=U_i(p_{i-1},Z_i)`$, where $`U_i`$ maps $`[0,1]^k\times\mathcal Y`$ into $`[0,1]^k`$ and every coordinate of $`U_i`$ is a polynomial with
  - **(D1)** degree at most $`a_i\ge1`$ in $`p_{i-1}`$ (as a polynomial in $`p_{i-1}`$ whose coefficients are polynomials in $`Z_i`$);
  - **(D2)** degree at most $`b_i\ge1`$ in $`Z_i`$ (as a polynomial in $`Z_i`$ whose coefficients are polynomials in $`p_{i-1}`$).
- The deployment value $`J:[0,1]^k\to[0,1]`$ is a polynomial of total degree at most $`N`$, which holds for every deployment of [Definition 4.3](04-03_binary_context_student_affine_learner.md#s4-3) with at most $`N`$ decisions (Theorem 4.4(a)).
- **Smooth local setting.** $`\theta\in\mathbb R^p`$, $`V:\mathbb R^p\to\mathbb R`$ is a continuation value (Definition 1.4), and one SGD step with consumed-gradient law $`P`$ acts as $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$, with $`\mu_P=\mathbb E_PG`$ and $`M_P=\mathbb E_PGG^\top`$.
  - **(T3)** $`V`$ is $`C^3`$ on every segment $`[\theta,\theta-\eta g]`$ with $`g`$ in the supports of $`P`$ and $`Q`$, and $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$ there.
- **Error recursion.** A nonnegative sequence satisfies $`e_0=0`$ and $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$ for constants $`\eta,L,C\ge0`$.

Then:

- **(a) Degree composition.** Under (D1)–(D2), $`J(p_R)`$, as a polynomial in $`(Z_1,\dots,Z_R)`$, has total degree in block $`Z_i`$ at most

```math
D_i=N\,b_i\prod_{t=i+1}^Ra_t. \qquad(4.8)
```

- Consequently, if $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$ for every $`i`$ and every $`\lvert\alpha\rvert\le D_i`$, then $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$. If all blocks share one source law, matching through $`\max_iD_i`$ suffices.
- **(b) Attainment.** For every $`R\ge1`$ there are updates in the class and a deployment with $`N=1`$ whose response is $`Z_1^{2^{R-1}}`$. The scalar laws of order $`M=2^{R-1}`$ from [Theorem 4.4(c)](04-04_affine_learners_moment_order.md#s4-4) agree on all moments below $`M`$ yet give different learning values, so (4.8) cannot be improved in general.
- **(c) A threshold learner.** For the learner $`p=\mathbf 1\lbrace Z>1/2\rbrace`$ and $`J(p)=p`$ ($`N=1`$), the laws $`P(Z=0)=\tfrac13`$, $`P(Z=\tfrac34)=\tfrac23`$ and $`Q(Z=\tfrac12)=1`$ have $`\mathbb E_PZ=\mathbb E_QZ=\tfrac12`$, while $`\mathbb E_PJ(p)=\tfrac23`$ and $`\mathbb E_QJ(p)=0`$.
- **(d) Local SGD expansion.** Under (T3),

```math
(K^P-K^Q)V(\theta)=-\eta\nabla V(\theta)^\top(\mu_P-\mu_Q)+\frac{\eta^2}{2}\langle\nabla^2V(\theta),M_P-M_Q\rangle+\mathcal E,\qquad\lvert\mathcal E\rvert\le\frac{M_3\eta^3}{6}\big(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3\big). \qquad(4.9)
```

- **(e) Error accumulation.** Under the error recursion,

```math
e_R\le C\eta^2\sum_{s=0}^{R-1}(1+\eta L)^s=\frac{C\eta}{L}\big((1+\eta L)^R-1\big)\quad(L>0),
```

- with equality when the recursion holds with equality. The linear bound $`e_R\le RC\eta^2`$ holds when $`L=0`$; for $`L>0`$ it fails in general: with $`\eta=\tfrac12`$ and $`L=C=1`$ the equality recursion gives $`e_4=\tfrac{65}{32}>1=RC\eta^2`$.

**What it says.**

- (a) When each update is polynomial, the moment order a source must match for an early block is the deployment length times the degrees of all later updates.
- (b) This multiplication really happens: squaring updates turn the first block into $`Z_1^{2^{R-1}}`$, and laws matched below that order still differ.
- (c) With a discontinuous update, even equal means give completely different students; no moment order is sufficient in general.
- (d) For one smooth SGD step, two feedback laws differ to first order through their mean gradients and to second order through their gradient second moments, up to an explicit third-order remainder.
- (e) Per-step errors of size $`C\eta^2`$ accumulate geometrically, and adding them up linearly needs $`\eta LR=O(1)`$ or a contractive error map.

**Intuition.** The same data fed to different learners needs different amounts of information. A later nonlinear update re-processes everything an earlier block did, so the earlier block's higher moments become visible.

**Role in the framework.** It shows that the actual optimizer must be modelled, which is the subject of §5. Part (d) is the SGD case of the general first-order representation [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1); cold and warm Adam are treated in [Proposition 5.3](05-03_cold_start_adam_sign_quotient.md#s5-3) and [Propositions 5.5](05-05_exact_first_order_statistic_adam.md#s5-5)–[5.6](05-06_warm_adam_first_order_affine.md#s5-6).

**Scope.**

- With one convex-affine update ($`a_i=b_i=1`$), (4.8) recovers $`D_i=N`$, which is Theorem 4.4(b). Each nonlinear later update multiplies the requirement on all earlier blocks.
- Degree is coordinate-dependent: an invertible reparameterization can change the polynomial form of an update. The intrinsic object remains the response class and its approximate dimension ([Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2)); degree is a tractable certificate in the chosen probability coordinates.
- (4.9) is local and one-step. For a stateful optimizer the increment of the complete learner state replaces the parameter step (Proposition 5.1).
- (e) is sharp for its recursion; a bound linear in $`R`$ requires $`\eta LR=O(1)`$, since $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$, or a contractive error map.

**Derivation outline.**

- (a) Pull $`J`$ backward through the updates: substituting a degree-$`(a_t,b_t)`$ map into a degree-$`d_t`$ polynomial multiplies the degree in $`p`$ by $`a_t`$ and creates degree $`b_td_t`$ in $`Z_t`$; independence then factors the expectation into per-block moments.
- (b) $`U_1(p,z)=z`$ followed by coordinatewise squaring, with the scalar witnesses of Theorem 4.4(c) at order $`2^{R-1}`$.
- (c) A direct computation with $`P(0)=\tfrac13`$, $`P(\tfrac34)=\tfrac23`$ against $`Q(\tfrac12)=1`$.
- (d) Third-order Taylor expansion of $`V(\theta-\eta g)`$ with integral remainder, then expectation.
- (e) Induction and a geometric sum; the recursion $`e_{t+1}=\tfrac32e_t+\tfrac14`$ shows that the linear simplification can fail.

**Illustration.** Feed the same two block laws to an averaging learner and to a learner that squares its parameter after each block. The first needs moments up to the deployment length; the second needs moments up to the deployment length times $`2^{R-1}`$.

**Uses.**

- [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) — polynomial deployment values and the scalar witnesses.
- [Definition 4.3](04-03_binary_context_student_affine_learner.md#s4-3) — deployments with at most $`N`$ decisions.
- [Definition 1.4](../README.md#s1-4) — continuation values.

**Used by.**

- [Proposition 5.1](05-01_optimizer_transformed_expansion.md#s5-1) — the expansion for any stateful optimizer.
- [Proposition 5.3](05-03_cold_start_adam_sign_quotient.md#s5-3) — cold-start Adam, where the update is not affine.

## Related work

- Part (d) is the one-step moment expansion on which weak approximations of SGD by stochastic modified equations are built ([Li et al., 2019](#ref-li2019stochastic)): the mean and the second moment of the stochastic gradient fix the expected effect of one step up to a remainder of third order in $`\eta`$. Part (e) is the step that turns such one-step errors into a bound over many steps.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-li2019stochastic"></a>

**Qianxiao Li, Cheng Tai, Weinan E (2019).** Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations. *Journal of Machine Learning Research* 20, no. 40, 1–47. [jmlr.org/papers/v20/17-526](https://jmlr.org/papers/v20/17-526.html).
