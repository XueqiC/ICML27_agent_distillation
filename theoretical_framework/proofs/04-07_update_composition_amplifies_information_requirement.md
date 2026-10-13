# Theorem 4.7 · Update composition amplifies the information requirement

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-07_update_composition_amplifies_information_requirement.md) · [Proof](04-07_update_composition_amplifies_information_requirement.md) · [Experiments](../experiments/04-07_update_composition_amplifies_information_requirement.md) · [中文](04-07_update_composition_amplifies_information_requirement_ZH.md)

**What is proved here.** When each training update is a polynomial in the current parameter and the new data, pulling the deployment value backward through the updates multiplies its degree, so an early block's higher moments become relevant; the multiplication is attained by squaring updates. A threshold update defeats moment matching entirely. For one smooth SGD step a third-order Taylor expansion identifies the first- and second-order statistics, and a geometric sum bounds how per-step errors accumulate.

**Setting used on this page.**

- *Polynomial updates.* The student parameter is $`p\in[0,1]^k`$; the learner starts from a common $`p_0`$ and processes independent blocks $`Z_1,\dots,Z_R`$ with values in a finite set $`\mathcal Y\subset[0,1]^k`$; block $`Z_i`$ has law $`P_i`$ under the reference source and $`Q_i`$ under the alternative; the update is $`p_i=U_i(p_{i-1},Z_i)`$.
- *Deployment value.* For a deployment with at most $`N`$ decisions, $`J`$ is a polynomial of total degree at most $`N`$ in $`p`$ ([Theorem 4.4(a)](../statements/04-04_affine_learners_moment_order.md#s4-4)).
- *Scalar witnesses* ([Theorem 4.4(c)](../statements/04-04_affine_learners_moment_order.md#s4-4)): for every order $`M\ge1`$ there are laws $`P_M,Q_M`$ on $`\lbrace0,\tfrac1M,\dots,1\rbrace`$ that agree on all moments below order $`M`$ and differ at order $`M`$ by $`\Delta_M=(-1)^MM!/(2^{M-1}M^M)\ne0`$.
- *Smooth local setting.* $`\theta\in\mathbb R^p`$, and $`V`$ is a continuation value: the expected final score when training continues from $`\theta`$ under the reference procedure ([Definition 1.4](../README.md#s1-4)). One SGD step with step size $`\eta`$ and consumed-gradient law $`P`$ acts as $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$; $`\mu_P=\mathbb E_PG`$, $`M_P=\mathbb E_PGG^\top`$; $`D^3V`$ is the third derivative.
- *Error recursion.* A nonnegative sequence $`e_t`$ with constants $`\eta,L,C\ge0`$.

**Statement ([Theorem 4.7](../statements/04-07_update_composition_amplifies_information_requirement.md#s4-7)).** The setting has three parts.

- **Polynomial updates.** Fix $`R\ge1`$ and a common $`p_0\in[0,1]^k`$. Independent blocks $`Z_1,\dots,Z_R`$ take values in a finite set $`\mathcal Y\subset[0,1]^k`$; block $`Z_i`$ has law $`P_i`$ under the reference source and $`Q_i`$ under the alternative. The learner is $`p_i=U_i(p_{i-1},Z_i)`$, where $`U_i`$ maps $`[0,1]^k\times\mathcal Y`$ into $`[0,1]^k`$ and every coordinate of $`U_i`$ is a polynomial with
  - **(D1)** degree at most $`a_i\ge1`$ in $`p_{i-1}`$ (as a polynomial in $`p_{i-1}`$ whose coefficients are polynomials in $`Z_i`$);
  - **(D2)** degree at most $`b_i\ge1`$ in $`Z_i`$ (as a polynomial in $`Z_i`$ whose coefficients are polynomials in $`p_{i-1}`$).
- The deployment value $`J:[0,1]^k\to[0,1]`$ is a polynomial of total degree at most $`N`$, which holds for every deployment of [Definition 4.3](../statements/04-03_binary_context_student_affine_learner.md#s4-3) with at most $`N`$ decisions (Theorem 4.4(a)).
- **Smooth local setting.** $`\theta\in\mathbb R^p`$, $`V:\mathbb R^p\to\mathbb R`$ is a continuation value (Definition 1.4), and one SGD step with consumed-gradient law $`P`$ acts as $`(K^PV)(\theta)=\mathbb E_{G\sim P}V(\theta-\eta G)`$, with $`\mu_P=\mathbb E_PG`$ and $`M_P=\mathbb E_PGG^\top`$.
  - **(T3)** $`V`$ is $`C^3`$ on every segment $`[\theta,\theta-\eta g]`$ with $`g`$ in the supports of $`P`$ and $`Q`$, and $`\lvert D^3V(\vartheta)[h,h,h]\rvert\le M_3\lVert h\rVert^3`$ there.
- **Error recursion.** A nonnegative sequence satisfies $`e_0=0`$ and $`e_{t+1}\le(1+\eta L)e_t+C\eta^2`$ for constants $`\eta,L,C\ge0`$.

Then:

- **(a) Degree composition.** Under (D1)–(D2), $`J(p_R)`$, as a polynomial in $`(Z_1,\dots,Z_R)`$, has total degree in block $`Z_i`$ at most

```math
D_i=N\,b_i\prod_{t=i+1}^Ra_t. \qquad(4.8)
```

- Consequently, if $`\mathbb E_{P_i}Z^\alpha=\mathbb E_{Q_i}Z^\alpha`$ for every $`i`$ and every $`\lvert\alpha\rvert\le D_i`$, then $`\mathbb E_PJ(p_R)=\mathbb E_QJ(p_R)`$. If all blocks share one source law, matching through $`\max_iD_i`$ suffices.
- **(b) Attainment.** For every $`R\ge1`$ there are updates in the class and a deployment with $`N=1`$ whose response is $`Z_1^{2^{R-1}}`$. The scalar laws of order $`M=2^{R-1}`$ from Theorem 4.4(c) agree on all moments below $`M`$ yet give different learning values, so (4.8) cannot be improved in general.
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

**Proof.** *(a).* Define backward continuation polynomials $`\Phi_R(p)=J(p)`$ and $`\Phi_{t-1}(p;Z_t,\dots,Z_R)=\Phi_t(U_t(p,Z_t);Z_{t+1},\dots,Z_R)`$ for $`t=R,\dots,1`$, so that $`J(p_R)=\Phi_0(p_0;Z_1,\dots,Z_R)`$. We show by backward induction that $`\Phi_t`$ has degree at most $`d_t:=N\prod_{s=t+1}^Ra_s`$ in $`p`$ ($`d_R=N`$) and degree at most $`D_i`$ in $`Z_i`$ for each $`i>t`$.

- *Base case $`t=R`$.* $`\Phi_R=J`$ has degree at most $`N`$ in $`p`$ and contains no blocks.
- *Inductive step.* Write $`\Phi_t=\sum_{\lvert\gamma\rvert\le d_t}c_\gamma(Z_{t+1},\dots,Z_R)\,p^\gamma`$, where each $`c_\gamma`$ has degree at most $`D_i`$ in $`Z_i`$ for $`i>t`$. Substitute $`p=U_t(p',Z_t)`$. The monomial $`p^\gamma`$ becomes a product of $`\lvert\gamma\rvert`$ coordinate polynomials of $`U_t`$, each of degree at most $`a_t`$ in $`p'`$ by (D1) and at most $`b_t`$ in $`Z_t`$ by (D2). Degrees of products add, so the substituted monomial has degree at most $`a_td_t=d_{t-1}`$ in $`p'`$ and at most $`b_td_t=Nb_t\prod_{s>t}a_s=D_t`$ in $`Z_t`$. The coefficients $`c_\gamma`$ involve neither $`p'`$ nor $`Z_t`$, so the degrees in $`Z_i`$, $`i>t`$, are unchanged.

At $`t=0`$, $`p=p_0`$ is a constant, so $`J(p_R)`$ has degree at most $`D_i`$ in each $`Z_i`$. Expand $`J(p_R)=\sum c_{\alpha_1\cdots\alpha_R}\prod_iZ_i^{\alpha_i}`$, a finite sum with $`\lvert\alpha_i\rvert\le D_i`$. Independence gives $`\mathbb E\prod_iZ_i^{\alpha_i}=\prod_i\mathbb E_{P_i}Z_i^{\alpha_i}`$, and under the matching hypothesis every factor is the same under $`P_i`$ and $`Q_i`$. If $`P_i=P`$ and $`Q_i=Q`$ for all $`i`$, matching through $`\max_iD_i`$ implies matching through each $`D_i`$.

*(b).* Let $`U_1(p,z)=z`$ and $`U_i(p,z)=p\odot p`$ for $`i\ge2`$, so $`b_1=1`$ and $`a_i=2`$ for $`i\ge2`$ (the other degrees are bounded by 1). Both maps keep $`[0,1]^k`$ invariant. Deploy once with $`J(p)=p_1`$ ($`N=1`$). By induction $`p_i=Z_1^{2^{i-1}}`$ coordinatewise, so the response is $`Z_{1,1}^{2^{R-1}}`$, while (4.8) gives $`D_1=1\cdot1\cdot2^{R-1}`$. For the laws $`P_M,Q_M`$ with $`M=2^{R-1}`$, moments of order below $`M`$ agree and $`\mathbb E_{P_M}Z^M-\mathbb E_{Q_M}Z^M=\Delta_M\ne0`$. Later blocks are unused, which (D2) permits.

*(c).* $`\mathbb E_PZ=\tfrac13\cdot0+\tfrac23\cdot\tfrac34=\tfrac12=\mathbb E_QZ`$. Under $`P`$, $`\mathbf 1\lbrace Z>1/2\rbrace=1`$ exactly when $`Z=\tfrac34`$, which has probability $`\tfrac23`$. Under $`Q`$, $`Z=\tfrac12`$ is not greater than $`\tfrac12`$, so the value is 0.

*(d).* Fix $`g`$ in either support and put $`\varphi(\tau)=V(\theta-\tau\eta g)`$, $`\tau\in[0,1]`$. By (T3) $`\varphi\in C^3[0,1]`$, and Taylor's theorem with integral remainder gives $`\varphi(1)=\varphi(0)+\varphi'(0)+\tfrac12\varphi''(0)+\int_0^1\tfrac{(1-\tau)^2}{2}\varphi'''(\tau)\,d\tau`$, with

```math
\varphi'(0)=-\eta\nabla V(\theta)^\top g,\qquad\varphi''(0)=\eta^2g^\top\nabla^2V(\theta)g,\qquad\varphi'''(\tau)=-\eta^3D^3V(\theta-\tau\eta g)[g,g,g].
```

By (T3) $`\lvert\varphi'''\rvert\le M_3\eta^3\lVert g\rVert^3`$, and $`\int_0^1(1-\tau)^2/2\,d\tau=1/6`$. Hence $`V(\theta-\eta g)=V(\theta)-\eta\nabla V^\top g+\tfrac{\eta^2}{2}\langle\nabla^2V,gg^\top\rangle+\rho(g)`$ with $`\lvert\rho(g)\rvert\le\tfrac{M_3\eta^3}{6}\lVert g\rVert^3`$. Take $`\mathbb E_P`$ and $`\mathbb E_Q`$ and subtract: $`V(\theta)`$ cancels, the linear and quadratic terms give the two displayed terms, and $`\mathcal E=\mathbb E_P\rho(G)-\mathbb E_Q\rho(G)`$ obeys the stated bound by the triangle inequality.

*(e).* Induction on $`t`$ gives $`e_t\le C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s`$: the case $`t=0`$ is the empty sum, and $`e_{t+1}\le(1+\eta L)C\eta^2\sum_{s=0}^{t-1}(1+\eta L)^s+C\eta^2=C\eta^2\sum_{s=0}^t(1+\eta L)^s`$. Each step uses only the recursion, so equality propagates. The closed form is the geometric sum. For the witness, $`e_{t+1}=\tfrac32e_t+\tfrac14`$ gives $`e_1=\tfrac14`$, $`e_2=\tfrac58`$, $`e_3=\tfrac{19}{16}`$, $`e_4=\tfrac{65}{32}`$. $`\square`$

## Remarks

- **The information order belongs to the learner composed with the deployment.** With one convex-affine update ($`a_i=b_i=1`$), (4.8) recovers $`D_i=N`$, which is [Theorem 4.4(b)](../statements/04-04_affine_learners_moment_order.md#s4-4) (moments through the deployment length are necessary and sufficient). Each nonlinear later update multiplies the requirement on all earlier blocks.
- **Degree is coordinate-dependent.** An invertible reparameterization can change the polynomial form of an update. The intrinsic object remains the response class and its approximate dimension ([Theorem 4.2](../statements/04-02_sufficiency_duality_transfer.md#s4-2), the span and approximation criteria); degree is a tractable certificate in the chosen probability coordinates.
- **Non-polynomial learners.** (c) shows that once the update is discontinuous no moment order is sufficient in general; even one decision imposes more than a first-moment requirement.
- **Scope of (4.9).** The expansion is local and one-step: its first-order statistic is the mean consumed gradient and its second-order statistic the gradient second moment. For a stateful optimizer the increment of the complete learner state replaces the parameter step ([Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1), the first-order representation for any stateful optimizer); cold and warm Adam are treated in [Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3) and [Propositions 5.5](../statements/05-05_exact_first_order_statistic_adam.md#s5-5)–[5.6](../statements/05-06_warm_adam_first_order_affine.md#s5-6).
- **Accumulation needs stability.** (e) is sharp for its recursion. A bound linear in $`R`$ requires $`\eta LR=O(1)`$, since $`\sum_{s<R}(1+\eta L)^s\le Re^{\eta LR}`$, or a contractive error map.

## Related work

- Taylor expansion with integral remainder and the geometric accumulation of per-step errors are classical; they are credited as such and not claimed as new. The same one-step moment expansion, followed by the accumulation of one-step errors, underlies the weak approximation of SGD by stochastic modified equations ([Li et al., 2019](#ref-li2019stochastic)).
- What is added is their use to show that the information a feedback source must carry is set by the learner composed with the deployment, not by the data alone.
