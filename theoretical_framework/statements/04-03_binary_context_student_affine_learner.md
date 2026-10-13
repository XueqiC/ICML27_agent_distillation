# Definition 4.3 · Binary-context student and convex-affine learner

*Statement*

[← Framework](../README.md) · [Statement](04-03_binary_context_student_affine_learner.md) · [中文](04-03_binary_context_student_affine_learner_ZH.md)

**In one paragraph.** [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2) says that the information two feedback laws must share is the span of the response class. To see what that span is for an actual learner and task, this definition fixes a model in which everything can be computed: a student that makes binary decisions in a few contexts, deployments that are finite decision trees, experience that arrives as independent blocks, and a learner that takes a fixed convex average of the blocks. §4 computes the exact information order, the rates and the effect of nonlinear updates in this model.

**Setting used on this page.**

- An environment's deployment returns lie in $`[0,1]`$, and $`J_E(\pi)`$ is the expected return of a policy ([Definition 1.1](../README.md#s1-1)). Here the policy is parameterized by the student's action probabilities $`p`$.
- "IID" means independent and identically distributed.

<a id="s4-3"></a>

**Definition 4.3 (Binary-context student and convex-affine learner).**

- **Student.** The student has $`k`$ binary-action contexts and a parameter $`p\in[0,1]^k`$. At a decision with context $`j`$ it takes action 1 with conditional probability $`p_j`$, given the entire past.
- **Deployment.** The deployment is a finite tree with three kinds of nodes:
  - *chance nodes*, whose transition probabilities do not depend on $`p`$;
  - *decision nodes*, each labelled by a context $`j`$ that may depend on the history but not on $`p`$ directly;
  - *terminal nodes*, carrying rewards in $`[0,1]`$ that do not depend on $`p`$.
- Every root-to-leaf path contains at most $`N`$ decision nodes, and a context may recur along a path. $`J_E(p)`$ is the expected terminal reward.
- **Experience.** An experience block is $`Z\in[0,1]^k`$, with arbitrarily dependent coordinates. Blocks $`Z_1,\dots,Z_R`$ are IID from $`P`$ (or from $`Q`$).
- **Convex-affine learner.** With $`p_0\in[0,1]^k`$ common and fixed and weights fixed independently of the data,

```math
p_R=w_0p_0+\sum_{i=1}^Rw_iZ_i,\qquad w_i\ge0,\quad\sum_{i=0}^Rw_i=1 .
```

- **Notation.** For $`\alpha\in\mathbb N^k`$, $`p^\alpha=\prod_jp_j^{\alpha_j}`$, $`\lvert\alpha\rvert=\sum_j\alpha_j`$, and $`m_P(\alpha)=\mathbb E_PZ^\alpha`$. For a polynomial $`f=\sum_\alpha c_\alpha p^\alpha`$, $`\lVert f\rVert_{\rm coef}=\sum_\alpha\lvert c_\alpha\rvert`$.
- Then $`p_R\in[0,1]^k`$, and the *learning value* of a source is $`\mathbb E\,J_E(p_R)`$.

**What it says.**

- The student is a table of action probabilities, one per context; a deployment asks it to act at most $`N`$ times.
- A source of experience is a law $`P`$ on blocks; the learner averages $`R`$ independent blocks with weights chosen in advance.
- The learning value of a source is the expected deployment return of the student it produces.

**Intuition.** This is the simplest learner whose outcome depends on more than the mean of the data: the deployment value is nonlinear in $`p`$ because one deployment can consult the same or different contexts several times.

**Role in the framework.**

- It is the computable instance of the general theory: for this model [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) identifies the response span exactly as the joint moments up to order $`N`$.
- The assumptions are restrictive on purpose. On-policy feedback that depends on the current student, data-dependent weights, and reparameterizations such as $`p=\sigma(\theta)`$ all leave this model; §4 shows that leaving it changes the information order ([Theorem 4.7](04-07_update_composition_amplifies_information_requirement.md#s4-7)), and §5 treats real optimizers.

**Uses.**

- [Definition 1.1](../README.md#s1-1) — environments, policies and returns.

**Used by.**

- [Theorem 4.4](04-04_affine_learners_moment_order.md#s4-4) — moment order is necessary and sufficient.
- [Theorem 4.5](04-05_round_independent_bound_averaging_rates.md#s4-5) — round-independent bound and rates.
- [Proposition 4.6](04-06_exact_order_not_approximate_complexity.md#s4-6) — approximate order of $`p^N`$ and parity.
- [Theorem 4.7](04-07_update_composition_amplifies_information_requirement.md#s4-7) — nonlinear and composed updates.

## Related work

- The deployment is a one-player extensive-form game with chance moves ([Kuhn, 1953](#ref-kuhn1953extensive)), the contexts playing the role of information sets, and $`p`$ is a behaviour strategy: one action probability per context, used at every decision with that context. Because a context may recur along a path, $`J_E`$ can have degree above 1 in a single coordinate $`p_j`$.
