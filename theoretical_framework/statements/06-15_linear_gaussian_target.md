# Proposition 6.15 · Linear-Gaussian target: exact risk, whitened response, one-query gain

*Statement*

[← Framework](../README.md) · [Statement](06-15_linear_gaussian_target.md) · [Proof](../proofs/06-15_linear_gaussian_target.md) · [Experiments](../experiments/06-15_linear_gaussian_target.md) · [中文](06-15_linear_gaussian_target_ZH.md)

**In one paragraph.** [Theorem 6.3](06-03_learner_relative_lower_bound.md#s6-3) and [Lemma 6.4](06-04_design_minimax_equality.md#s6-4) give the information geometry of one learner-relative functional to leading order. When the evidence is linear and Gaussian, the same geometry is exact for a whole vector of learning consequences: the minimax risk of estimating them is a trace, its eigenvalues measure how hard each consequence direction is to identify from the evidence already in hand, and the risk reduction of one more query has a closed form. This gives a computable query rule and separates two quantities that are easy to confuse: how large a consequence is, and how poorly the current evidence identifies it.

**Setting used on this page.**

- **Linear-Gaussian evidence.** The unknown environment parameter is $`\vartheta\in\mathbb R^d`$. All evidence (demonstrations and queries) is one Gaussian linear observation $`y=D\vartheta+\epsilon`$, $`\epsilon\sim\mathcal N(0,\Sigma)`$, with $`D`$ and $`\Sigma\succ0`$ known. The parameter is unrestricted in $`\mathbb R^d`$.
- **Learning-consequence target.** A known linear map $`f(\vartheta)=C\vartheta\in\mathbb R^k`$: for example the first-order effects of the oracle feedback on $`k`$ declared tests, as in [Definition 6.2](06-02_learner_relative_functional_information_norm.md#s6-2) with several task directions. It is not required that $`\vartheta`$ itself be recovered.
- **Loss.** $`\lVert x\rVert_W^2=x^\top Wx`$ with a known weight $`W\succeq0`$ on the $`k`$ outputs.
- **Information and target matrices.** $`I=D^\top\Sigma^{-1}D`$ (the Fisher information of $`y`$) and $`B=C^\top WC`$. $`I^+`$ is the Moore–Penrose pseudo-inverse.
- **A query.** One further observation $`a^\top\vartheta+\varepsilon`$ with unit noise, independent of $`y`$; it adds $`aa^\top`$ to $`I`$.

<a id="s6-15"></a>

**Proposition 6.15 (Linear-Gaussian target: exact risk, whitened response, one-query gain).** In this setting:

**(a) Exact minimax risk.** If $`I\succ0`$, the generalized least-squares estimator $`\hat f=C\hat\vartheta`$, $`\hat\vartheta=I^{-1}D^\top\Sigma^{-1}y`$, has constant risk

```math
\mathbb E\lVert\hat f-f\rVert_W^2=\mathrm{tr}(BI^{-1})=\sum_j\lambda_j,\qquad Bv_j=\lambda_jIv_j ,
```

and this is the minimax risk over $`\vartheta\in\mathbb R^d`$.

**(b) Singular information.** Suppose $`W^{1/2}Cv=0`$ for every $`v\in\ker I`$ (for example $`C\ker I=\lbrace0\rbrace`$). Then $`\hat f=CI^+D^\top\Sigma^{-1}y`$ has constant risk $`\mathrm{tr}(BI^+)`$, which is the minimax risk, and it is unbiased when $`C\ker I=\lbrace0\rbrace`$. If instead $`W^{1/2}Cv\ne0`$ for some $`v\in\ker I`$, the minimax risk over $`\mathbb R^d`$ is $`+\infty`$. On a bounded parameter set that contains two points differing along such a $`v`$, the minimax risk is not infinite but has a positive floor that no estimator removes, because the evidence has the same law at both points.

**(c) Information-whitened response.** For $`I\succ0`$ put $`\mathcal R_I=W^{1/2}CI^{-1/2}`$. Its squared singular values are the $`\lambda_j`$ of (a), so $`\mathrm{tr}(BI^{-1})=\lVert\mathcal R_I\rVert_{\mathrm{HS}}^2`$.

**(d) Exact one-query gain.** For $`I\succ0`$, adding the query $`a`$ lowers the minimax risk by exactly

```math
\Delta(a)=\frac{a^\top I^{-1}BI^{-1}a}{1+a^\top I^{-1}a}.
```

**(e) The scalar case.** For $`k=1`$, $`C=c^\top`$ and $`W=1`$, $`B=cc^\top`$ and the risk is $`c^\top I^{-1}c`$, the criterion of the c-optimal design in [Lemma 6.4](06-04_design_minimax_equality.md#s6-4). Several targets give a weighted A- or L-optimal criterion $`\mathrm{tr}(BI^{-1})`$, not a scalar c-optimal one.

**What it says.**

- In the linear-Gaussian model the risk of estimating a vector of learning consequences is known exactly, attained by generalized least squares, and spread over the eigen-directions of $`I^{-1/2}BI^{-1/2}`$.
- Two spectra must be kept apart. The response operator of [Definition 4.8](04-08_response_operator.md#s4-8) measures how large a consequence a feedback difference causes; the whitened operator $`\mathcal R_I`$ measures how poorly the current evidence identifies each consequence direction. A large consequence that is already well identified is not worth querying.
- One extra query reduces the risk by an explicit amount, so candidate queries can be ranked by $`\Delta(a)`$, or by $`\Delta(a)`$ per unit cost.

**Intuition.**

- Generalized least squares has error covariance $`I^{-1}`$; the target sees that error through $`C`$ and weighs it by $`W`$, which is all the trace says.
- A query is worth what it removes from the error covariance in the directions the target cares about; the Sherman–Morrison formula computes exactly that.

**Role in the framework.**

- It is the fully proved special case of query design in §6: [Theorem 6.3](06-03_learner_relative_lower_bound.md#s6-3) and [Theorem 6.5](06-05_achievability.md#s6-5) give the general learner-relative bound only to leading order, while here every quantity is exact.
- It gives [Proposition 6.16](06-16_information_compilation_radius.md#s6-16) a computable instance of the step "acquire evidence that separates the consequence directions".
- What the special case does not license, when one returns to a real learning system:
  - Choosing queries greedily by $`\Delta(a)/\mathrm{cost}(a)`$ is a one-step rule; it is not claimed to be optimal for a whole budget.
  - In a general regular parametric model, $`I`$ must be replaced by a local information and $`C`$ by the derivative of the **complete** consequence function; local approximation, identifiability and the estimator's properties then need their own proof.
  - If $`f(\vartheta)=\int g_\vartheta(y)P_\vartheta(dy)`$, a change of environment can change the later training, the occupancy and the test outcome as well as the feedback law, so $`\partial f=\int g_\vartheta\,\partial P_\vartheta+\int(\partial g_\vartheta)\,dP_\vartheta`$ where differentiation and integration can be exchanged. Only when the response mechanism is held fixed does the derivative reduce to the response operator applied to $`\partial P_\vartheta`$.

**Derivation outline.**

- (a): the error of $`\hat\vartheta`$ is Gaussian with covariance $`I^{-1}`$, so the risk is $`\mathrm{tr}(WCI^{-1}C^\top)`$. Gaussian priors $`\mathcal N(0,\tau^2I_d)`$ have Bayes risk $`\mathrm{tr}(B(I+\tau^{-2}I_d)^{-1})`$, which increases to the same trace as $`\tau\to\infty`$; no estimator can beat every Bayes risk in the worst case.
- (b): when $`W^{1/2}C`$ vanishes on $`\ker I`$, the bias of $`CI^+D^\top\Sigma^{-1}y`$ is invisible to the loss and the same computation works on the identifiable directions; when it does not, the Bayes risks grow without bound, and on a bounded set Le Cam's two-point argument ([Le Cam, 1973](#ref-lecam1973convergence)) gives the floor.
- (c): $`\mathcal R_I^\top\mathcal R_I=I^{-1/2}BI^{-1/2}`$, whose eigenvalues are the generalized eigenvalues of $`(B,I)`$.
- (d): the Sherman–Morrison formula for $`(I+aa^\top)^{-1}`$, inserted in the trace.

**Uses.**

- [Definition 6.2](06-02_learner_relative_functional_information_norm.md#s6-2): the learner-relative functional, here with several task directions.
- [Lemma 6.4](06-04_design_minimax_equality.md#s6-4): the scalar criterion and the pseudo-inverse convention.

**Used by.**

- [Proposition 6.16](06-16_information_compilation_radius.md#s6-16): the computable acquisition step.
- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2): query selection in the linear-Gaussian case.

**Status.** proved.

## Related work

- Generalized least squares, its minimax optimality under Gaussian noise and the trace criterion are classical; the A-, L- and c-optimality criteria and the role of the information matrix are developed in [Pukelsheim, 1993](#ref-pukelsheim1993optimal), going back to [Elfving, 1952](#ref-elfving1952optimum).
- What is specific here is the target $`C`$: a vector of learning consequences of the declared learner and tests rather than the environment parameter itself, which is why the whitened operator, not the raw parameter covariance, decides which queries are worth making.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-lecam1973convergence"></a>

**Lucien Le Cam (1973).** Convergence of estimates under dimensionality restrictions. *The Annals of Statistics* 1, no. 1, 38–53. DOI [10.1214/aos/1193342380](https://doi.org/10.1214/aos/1193342380).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).
