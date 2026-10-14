# Proposition 4.12 · Responses factor through the next predictive state

*Proof*

[← Framework](../README.md) · [Statement](../statements/04-12_response_predictive_state_factorization.md) · [Proof](04-12_response_predictive_state_factorization.md) · [Experiments](../experiments/04-12_response_predictive_state_factorization.md) · [中文](04-12_response_predictive_state_factorization_ZH.md)

**What is proved here.** (a) is conditioning on the message and the mean-zero property of $`h`$. (b) is injectivity of $`C`$. (c) propagates equal predictive states through invariant kernels. (d) is Cauchy–Schwarz on the residual part.

**Setting used on this page.**

- $`H=L_0^2(\rho)`$; tests $`a=1,\dots,p`$ with weights $`\omega_a>0`$; $`(\mathcal R_zh)(a)=\langle\tilde g_a,h\rangle_\rho`$ ([Definition 4.8](../statements/04-08_response_operator.md#s4-8)).
- $`g_a(y)=\mathbb E[V_a(U_t(z,y,\xi_t))\mid z,y]`$; bounded features $`\psi`$ with $`V_a=k_a+c_a^\top\psi`$ (or $`+e_a`$ in (d)); $`C`$ with rows $`c_a^\top`$; $`\Gamma_z(y)=\mathbb E[\psi(U_t(z,y,\xi_t))\mid z,y]`$; $`\mathcal T_zh=\mathbb E_\rho[\Gamma_zh]`$.

**Statement ([Proposition 4.12](../statements/04-12_response_predictive_state_factorization.md#s4-12)).** (a) $`\mathcal R_z=C\mathcal T_z`$. (b) If $`\mathrm{rank}\,C=d`$, $`\ker\mathcal R_z=\ker\mathcal T_z`$, so equal declared responses are equivalent to $`\mathbb E_P\Gamma_z=\mathbb E_Q\Gamma_z`$. (c) With invariance under the declared continuation kernels (or observation operators), equal $`\mathbb E\Gamma_z`$ gives equal later declared expectations. (d) With residuals, $`\mathcal R_z=C\mathcal T_z+\mathcal E_z`$ and $`\lVert\mathcal E_zh\rVert_\omega\le\epsilon_{\rm pred}\lVert h\rVert_\rho`$.

**Proof.**

**(a)** By the tower property and linearity of conditional expectation, $`g_a(y)=\mathbb E[k_a+c_a^\top\psi(U_t(z,y,\xi_t))\mid z,y]=k_a+c_a^\top\Gamma_z(y)`$. For $`h\in H`$, $`\mathbb E_\rho h=0`$, so constants are orthogonal to $`h`$ and

```math
(\mathcal R_zh)(a)=\langle\tilde g_a,h\rangle_\rho=\mathbb E_\rho[g_ah]=k_a\mathbb E_\rho h+c_a^\top\mathbb E_\rho[\Gamma_zh]=c_a^\top\mathcal T_zh .
```

$`\Gamma_z`$ is bounded because $`\psi`$ is, so $`\mathcal T_zh`$ is well defined for $`h\in L^2(\rho)`$.

**(b)** If $`\mathcal T_zh=0`$ then $`\mathcal R_zh=C\mathcal T_zh=0`$. Conversely, if $`\mathcal R_zh=0`$ then $`C(\mathcal T_zh)=0`$, and full column rank means $`Cx=0\Rightarrow x=0`$, so $`\mathcal T_zh=0`$. For laws with densities $`p,q\in L^2(\rho)`$, take $`h=p-q`$: $`(\mathcal R_zh)(a)=\mathbb E_Pg_a-\mathbb E_Qg_a`$ ([Definition 4.8](../statements/04-08_response_operator.md#s4-8)) and $`\mathcal T_zh=\int\Gamma_z\,dP-\int\Gamma_z\,dQ`$.

**(c)** Let $`\mu_P`$ be the law of the state after this round when the message is drawn from $`P`$; then $`\mathbb E_{\mu_P}\psi=\mathbb E_P\Gamma_z`$, and likewise for $`Q`$. With $`\psi'=(\mathbf 1,\psi)`$, invariance gives matrices $`A_b`$ with $`K_b\psi'=A_b\psi'`$ for every declared control $`b`$, and by [Theorem 4.10(a)](../statements/04-10_hankel_rank_predictive_realization.md#s4-10) $`\mathbb E_{\mu K_w}\psi'=A_{b_\ell}\cdots A_{b_1}\mathbb E_\mu\psi'`$ for every word $`w`$. Equal $`\mathbb E_{\mu_P}\psi'`$ and $`\mathbb E_{\mu_Q}\psi'`$ therefore give equal $`\mathbb E\psi'`$ after every declared word, and every declared test value is an affine function of $`\psi`$. For observation-adaptive continuations, use the conditioned update (4.20) along each observation branch: by induction on the branch, the unnormalized vectors $`M_{b_\ell,o_\ell}\cdots M_{b_1,o_1}\mathbb E_\mu\psi'`$ coincide for the two laws, so every branch has the same probability and the same posterior predictive state.

**(d)** As in (a), $`g_a=k_a+c_a^\top\Gamma_z+\bar e_a`$, so $`(\mathcal R_zh)(a)=c_a^\top\mathcal T_zh+\mathbb E_\rho[\bar e_ah]`$, and $`\mathbb E_\rho[\bar e_ah]=\langle\bar e_a-\mathbb E_\rho\bar e_a,h\rangle_\rho`$ because $`h`$ has mean zero. By Cauchy–Schwarz, $`\lvert(\mathcal E_zh)(a)\rvert\le\lVert\bar e_a-\mathbb E_\rho\bar e_a\rVert_\rho\lVert h\rVert_\rho`$; squaring, weighting by $`\omega_a`$ and summing gives (4.24), and $`\mathrm{Var}_\rho\bar e_a\le\mathbb E_\rho\bar e_a^2`$. $`\square`$

## Related work

- The factorization is the one-step form of describing a controlled process by predictions of tests ([Singh, James & Rudary, 2004](#ref-singh2004predictive)); its consequence (c) is the defining property of a sufficient predictive state.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-singh2004predictive"></a>

**Satinder Singh, Michael R. James, Matthew R. Rudary (2004).** Predictive state representations: A new theory for modeling dynamical systems. *Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI)*, 512–519, AUAI Press. arXiv [1207.4167](https://arxiv.org/abs/1207.4167).
