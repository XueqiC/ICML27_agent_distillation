# Proposition 4.13 · Finite-horizon Gramians, and what control theory does not transfer

*Statement*

[← Framework](../README.md) · [Statement](04-13_finite_horizon_gramians.md) · [Proof](../proofs/04-13_finite_horizon_gramians.md) · [Experiments](../experiments/04-13_finite_horizon_gramians.md) · [中文](04-13_finite_horizon_gramians_ZH.md)

**In one paragraph.** Viewing training as a controlled system adds one useful idea and invites several wrong ones. The useful idea: a direction of the learner's state matters for the declared future only if feedback can reach it and the future tests can see it, and over a finite horizon, for the linearized map with the remainders $`\epsilon_t,\zeta_t`$ set aside, this is exactly measured by two Gramians, without any stability assumption; for actual training the identity holds only up to those remainders. A direction that is complicated in the environment but unreachable by legal feedback, or reachable but invisible to the tests, contributes nothing; a small change of optimizer state can matter a lot if the tests see it strongly. The wrong ones: the a-priori error bounds of balanced truncation, the stability of linear time-invariant systems, and the assumption that a low-rank fit of one Hankel block is a causal model. This page proves the first and lists the second.

**Setting used on this page.**

*Optional local linearization.* Where the actual update is differentiable along a reference path, write state, input and output perturbations as

```math
\delta z_{t+1}=A_t\,\delta z_t+B_t\,\delta u_t+\epsilon_t,\qquad \delta o_t=C_t\,\delta z_t+\zeta_t ,
```

with the nonlinear remainders $`\epsilon_t,\zeta_t`$ kept, not dropped. $`\delta z`$ is a perturbation of the complete learner state ([Definition 1.3](../README.md#s1-3)), $`\delta u`$ a perturbation of the feedback, and $`\delta o`$ of the declared outputs.

- *Past and future.* Split the horizon at $`\tau`$. With zero initial perturbation and the remainders set aside, $`\delta z_\tau=\mathscr C_\tau\,\delta u_{\rm past}`$ and $`\delta o_{\rm future}=\mathscr O_\tau\,\delta z_\tau`$, where $`\mathscr C_\tau=[\Phi(\tau,1)B_0,\dots,\Phi(\tau,\tau)B_{\tau-1}]`$, $`\mathscr O_\tau`$ stacks $`C_t\Phi(t,\tau)`$ for $`t=\tau,\dots,N`$, and $`\Phi(t,j)=A_{t-1}\cdots A_j`$ with $`\Phi(j,j)=I`$. The *finite-horizon Hankel operator* is $`H_\tau=\mathscr O_\tau\mathscr C_\tau`$.
- Past inputs carry a declared energy norm and future outputs a declared weighted norm; both are absorbed into $`B_t`$ and $`C_t`$ by square-root weighting, so all norms below are Euclidean.
- *Gramians.* $`W_c=\mathscr C_\tau\mathscr C_\tau^\top`$ (reachability) and $`W_o=\mathscr O_\tau^\top\mathscr O_\tau`$ (observability); $`W_c^+`$ is the Moore–Penrose inverse.

<a id="s4-13"></a>

**Proposition 4.13 (Finite-horizon Gramians, and what control theory does not transfer).** In this setting, with no stability or contraction assumption on the $`A_t`$:

**(a) Hankel singular values.** For every $`j`$,

```math
\sigma_j(H_\tau)^2=\lambda_j\big(W_c^{1/2}W_oW_c^{1/2}\big), \qquad(4.25)
```

both sequences in decreasing order and padded with zeros.

**(b) Gain of a state direction.** For a reachable direction $`x\in\mathrm{range}\,W_c`$, the least input energy that produces $`\delta z_\tau=x`$ is $`x^\top W_c^+x`$, and every input that produces it gives the same future output energy $`x^\top W_ox`$. Hence

```math
\sigma_1(H_\tau)^2=\sup_{x\in\mathrm{range}\,W_c,\ x\ne0}\frac{x^\top W_o\,x}{x^\top W_c^+\,x}, \qquad(4.26)
```

and $`H_\tau=0`$ exactly when $`W_oW_c=0`$, i.e. when every reachable direction is unobservable.

**What it says.**

- Over a finite horizon, the input–output gain of the linearized training map (remainders $`\epsilon_t,\zeta_t`$ set aside) splits exactly into how cheaply feedback can push the state in a direction and how strongly the future tests see that direction. Neither alone determines it. For actual training this holds only up to the remainders.
- A direction that is complicated in the environment but unreachable, or reachable but invisible, contributes no gain. A direction of optimizer state, such as Adam's moments, with little immediate effect on the parameters can still carry a large gain if the future tests see it strongly.
- No stability is needed; the $`A_t`$ may have gains above 1.

**Intuition.** Feedback is an input, the deployment scores are outputs, and the learner's complete state sits in between. What matters for distillation is what passes through: reachable by the input and visible at the output.

**What does not transfer.**

- *Balanced-truncation error bounds.* Balancing and truncating by Hankel singular values goes back to [Moore (1981)](#ref-moore1981principal), and its classical a-priori error bound holds for stable linear time-invariant systems and their realizations. It is not a guarantee for nonlinear, nonstationary Adam training and is not used anywhere in the framework.
- *A low-rank Hankel block is not a causal model.* The best low-rank approximation of one Hankel block need not have the shift structure of any finite-dimensional causal model. Closure must be checked separately ([Proposition 4.11](04-11_static_not_dynamic_closure_error.md#s4-11)).
- *The linearization is optional.* Whether the local model above describes real training must be established by measuring $`\epsilon_t`$ and $`\zeta_t`$, or the analysis must be restricted to an explicitly local one. The base identities of the framework, [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2) and [Proposition 3.4](03-04_joint_decomposition_approximate_value.md#s3-4), use the exact kernels and do not depend on it.
- *A failed mechanism is not repaired by a new language.* The real-model test of the stationary amplitude law ([Corollary 7.2](07-02_stationary_amplitude_law.md#s7-2)) did not support its prediction; describing the response spectrally does not fix that failure, and which mechanism caused it needs its own separating experiment.
- *Training as a controlled system is not itself new.* Sequential machine teaching already treats teaching as an optimal control problem ([Lessard, Zhang & Zhu, 2019](#ref-lessard2019optimal)).

**Role in the framework.**

- It gives the finite-horizon meaning of the factorization of [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12): there, $`\mathcal T_z`$ is a one-step reachability map and $`C`$ an observability map; here the same split holds over many steps, with exact Gramians.
- It explains why the complete optimizer state must be modelled ([Theorem 7.1](07-01_state_channel_decomposition.md#s7-1)): directions of $`(m,v)`$ can carry gain that the immediate parameter change does not show.

**Derivation outline.**

- (a) $`H^\top H=\mathscr C^\top W_o\mathscr C`$, and $`XY`$ and $`YX`$ have the same nonzero eigenvalues with the same multiplicities; apply this twice.
- (b) The minimum-norm solution of $`\mathscr Cu=x`$ is $`\mathscr C^+x`$, with squared norm $`x^\top(\mathscr C\mathscr C^\top)^+x`$. The output depends on $`u`$ only through $`x=\mathscr Cu`$. Inputs in $`\ker\mathscr C`$ produce no output, so the supremum of $`\lVert Hu\rVert^2/\lVert u\rVert^2`$ is attained at minimum-norm inputs.

**Uses.**

- [Proposition 4.12](04-12_response_predictive_state_factorization.md#s4-12): the one-step factorization.

**Used by.** No later result depends on this proposition; it is the interpretation behind the complete-state bookkeeping of [Theorem 7.1](07-01_state_channel_decomposition.md#s7-1).

## Related work

- Reachability and observability Gramians, and the Hankel singular values they determine, are the basis of balanced realization ([Moore, 1981](#ref-moore1981principal)); (a) is their finite-horizon, time-varying form.
- Sequential machine teaching formulates the choice of training examples for a known learner as optimal control ([Lessard, Zhang & Zhu, 2019](#ref-lessard2019optimal)); see also iterative machine teaching ([Liu et al., 2017](#ref-liu2017iterative)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-lessard2019optimal"></a>

**Laurent Lessard, Xuezhou Zhang, Xiaojin Zhu (2019).** An optimal control approach to sequential machine teaching. *Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 89:2495–2503. [proceedings.mlr.press/v89/lessard19a](https://proceedings.mlr.press/v89/lessard19a.html).

<a id="ref-liu2017iterative"></a>

**Weiyang Liu, Bo Dai, Ahmad Humayun, Charlene Tay, Chen Yu, Linda B. Smith, James M. Rehg, Le Song (2017).** Iterative machine teaching. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:2149–2158. [proceedings.mlr.press/v70/liu17b](https://proceedings.mlr.press/v70/liu17b.html).

<a id="ref-moore1981principal"></a>

**Bruce C. Moore (1981).** Principal component analysis in linear systems: Controllability, observability, and model reduction. *IEEE Transactions on Automatic Control* 26, no. 1, 17–32. DOI [10.1109/TAC.1981.1102568](https://doi.org/10.1109/TAC.1981.1102568).
