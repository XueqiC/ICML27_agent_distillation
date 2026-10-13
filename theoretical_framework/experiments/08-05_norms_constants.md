# Remark 8.5 · Norms and constants

*Experiments*

[← Framework](../README.md) · [Statement](../statements/08-05_norms_constants.md) · [Experiments](../experiments/08-05_norms_constants.md) · [中文](08-05_norms_constants_ZH.md)

**What is tested here.** The checks summarized below also test this item. Each summary states what the check tests for this item, its setup and its key numbers; the full description is on the page linked in its heading.

**Setting used on this page.**

Use the objects of [Proposition 5.1](../statements/05-01_optimizer_transformed_expansion.md#s5-1): consumed gradient $`G_t(z,Y,\xi)`$, increment $`\Delta_t(z,g)`$, mean increment $`\mu^P_t(z)`$, second increment moment $`m^P_{2,t}(z)=\mathbb E_P\lVert\Delta_t\rVert^2`$ and continuation $`V^P_{t+1}`$. The norm is the Hilbert norm of the state space (Euclidean in finite dimension).

- **Bounded increments (B).** There is $`b<\infty`$ with $`\lVert\Delta_t(z,G_t(z,Y,\xi))\rVert\le b`$ almost surely, for the compared laws at the audited states. Examples: clipped SGD on the parameter block, $`b=\eta c`$; sign-type updates on $`p`$ parameters, $`b=\eta\sqrt p`$; SGD with gradients bounded by $`\bar G`$, $`b=\eta\bar G`$.
- **Curvature (S).** At the audited state, $`V:=V^P_{t+1}`$ is twice differentiable along the segments $`\lbrace z+s\Delta:s\in[0,1]\rbrace`$ reached by the increments, with $`\lvert D^2V(z+s\Delta)[\Delta,\Delta]\rvert\le M_2\lVert\Delta\rVert^2`$; write $`\Lambda=\lVert\nabla V(z)\rVert`$.
- **Audit samples.** At an audited state $`z`$, draw $`n_P`$ messages $`Y\sim P_t(\cdot\mid z)`$ and $`n_Q`$ messages $`Y\sim Q_t(\cdot\mid z)`$, independently and with independent update randomness, and compute the increments $`X=\Delta_t(z,G_t(z,Y,\xi))`$. Their empirical means are $`\bar X_P,\bar X_Q`$ and their empirical second moments $`\hat m^P_2,\hat m^Q_2`$. This takes $`n_P+n_Q`$ consumed-gradient evaluations at $`z`$ and no training tail.

**Statement under test ([Remark 8.5](../statements/08-05_norms_constants.md#s8-5)).**

- The concentration bound of [Proposition 8.4(a)](../statements/08-04_measuring_mean_increment_actual_states.md#s8-4) holds in the Hilbert norm of the state space; its proof uses the inner product (vanishing cross terms). It does not extend to arbitrary norms with the same constant; for example, in the sup norm on $`\mathbb R^p`$ the analogous bound carries a $`\sqrt{\log p}`$ factor.
- The bound contains no explicit dimension, but $`b`$ itself can grow with the dimension. Under cold-start Adam every coordinate moves by about $`\eta`$ ([Proposition 5.3](../statements/05-03_cold_start_adam_sign_quotient.md#s5-3)), so $`\lVert\Delta\theta\rVert`$ can reach $`\eta\sqrt p`$.
- A ranking decision needs only the scalar projection $`\langle\nabla V(z),\Delta\rangle`$, whose fluctuation can be far smaller than $`\lVert\nabla V(z)\rVert\,b`$. [Theorem 8.8](../statements/08-08_crn_paired_replacement_certificate.md#s8-8) goes further: it measures the finite replacement effect directly and needs no bound $`b`$.
- **Cost comparison.** A full-tail audit at one state needs the replica count of [Theorem 8.3(d)](../statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) per candidate; (b) needs $`n_P+n_Q`$ gradient evaluations per candidate. What (b) cannot do alone is supply $`\Lambda`$ and $`M_2`$.

### [Check 8.13](../experiments/08-04_measuring_mean_increment_actual_states.md#c8-13) · Concentration of the mean increment (Proposition 8.4(a), Remark 8.5)

- **What it tests for Remark 8.5.** Whether the concentration bound of Proposition 8.4(a) deteriorates with dimension at a fixed increment bound $`b`$, and how the caveat that $`b`$ itself can grow with dimension shows up.
- **Setup.** Dimensions $`p\in\lbrace1,10,1000,100000\rbrace`$, sample sizes $`n\in\lbrace8,32,128\rbrace`$, $`\delta\in\lbrace0.05,0.2\rbrace`$, three bounded laws with $`b=1`$ (sign-type, clipped Gaussian, sparse spikes), 20,000 trials per cell, sampled exactly without forming $`p`$-vectors.
- **Key numbers.** In all 144 cells the failure rate is far below $`\delta`$; the largest realized quantile over the bound is 0.615 at $`p=1`$ and 0.574 at $`p=100{,}000`$, so nothing grows with $`p`$ at fixed $`b`$. For unnormalized sign-type increments, such as the first step of cold-start Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)), $`b=\eta\sqrt p`$, and the first-order differences have the same scale, so the relative accuracy is still set by $`1/\sqrt n`$.
- **Verdict.** PASS. The bound is free of explicit dimension in the Hilbert norm, as the remark states; the scale of $`b`$ is a separate question.

## Reproduce

Reproduction commands and result files are listed on the pages of the primary checks: [Proposition 8.4](../experiments/08-04_measuring_mean_increment_actual_states.md). The check uses NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
