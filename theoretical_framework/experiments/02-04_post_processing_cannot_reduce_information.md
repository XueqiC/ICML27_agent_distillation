# Proposition 2.4 · Post-processing cannot reduce information regret

*Experiments*

[← Framework](../README.md) · [Statement](../statements/02-04_post_processing_cannot_reduce_information.md) · [Proof](../proofs/02-04_post_processing_cannot_reduce_information.md) · [Experiments](02-04_post_processing_cannot_reduce_information.md) · [中文](02-04_post_processing_cannot_reduce_information_ZH.md)

**What is tested here.** No check is dedicated to this proposition alone; two checks test it. [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity.md#c2-1) garbles the transcript of 500 random problems through random and adversarially chosen channels: no garbling raises the Bayes value, and augmenting by independent data leaves it exactly unchanged (PASS). [Check 2.6](../experiments/02-10_few_shot_information_floor.md#c2-6) measures replay in the hidden-bit model: replay gains exactly 0 at all 944 exact grid points (PASS).

**Setting used on this page.**

- A finite set $`\mathcal E`$ of environments with a prior $`\mu`$, conditional on the common initial information $`I_0`$ (pretrained model, known rules, the $`K`$ trusted demonstrations).
- Finite policy classes $`\Pi\subseteq\Pi_{\rm ref}`$ with values $`J_E(\pi)\in[0,1]`$: $`\Pi`$ is what the student can deploy, $`\Pi_{\rm ref}`$ a reference class (for example, what the teacher can execute).
- The transcript $`D`$: everything the learner observes beyond $`I_0`$, including every experiment it may run; its law given $`E`$ is $`P(D\mid E)`$, and $`\mu(E\mid D)`$ is the posterior.
- A learner: a kernel $`\lambda(\cdot\mid D)`$ that outputs a policy $`\widehat\pi\in\Pi`$; its private randomness is independent of $`E`$ given $`D`$.
- $`J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi)`$ and $`J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi)`$ are the best values in environment $`E`$; $`B(D)=\max_{\pi\in\Pi}\mathbb E[J_E(\pi)\mid D]`$ is the Bayes value, the best expected return a student in $`\Pi`$ can reach given $`D`$.
- The three regret terms $`\mathcal R_{\rm rep}`$, $`\mathcal R_{\rm info}`$, $`\mathcal R_{\rm alg}`$ are defined in [Definition 2.1](../statements/02-01_bayes_value_three_regret_terms.md#s2-1).
- A post-processing is a kernel $`\kappa(\cdot\mid D)`$ that does not depend on $`E`$, so $`E\to D\to D'`$ is a Markov chain.

**Statement under test ([Proposition 2.4](../statements/02-04_post_processing_cannot_reduce_information.md#s2-4)).** Let $`D'\sim\kappa(\cdot\mid D)`$ be a randomized post-processing whose kernel $`\kappa`$ does not depend on $`E`$, so that $`E\to D\to D'`$ is a Markov chain. Then

```math
\mathbb E\max_{\pi}\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D], \qquad(2.5)
```

that is, $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$. If $`Z\perp E\mid(D,I_0)`$, then $`B(D,Z)=B(D)`$, so adding $`Z`$ to the transcript leaves $`\mathcal R_{\rm info}`$ unchanged.

### [Check 2.1](../experiments/02-02_regret_decomposition_decision_ambiguity.md#c2-1) · Regret decomposition, ambiguity criterion, duality and data processing

- *What it tests for this proposition.* Inequality (2.5): an environment-independent post-processing cannot raise the Bayes value; and the augmentation clause $`B(D,Z)=B(D)`$.
- *Setup.* 500 random finite decision problems with a 4-symbol transcript. Each transcript is garbled through a random 3-symbol channel, a post-processing in the sense of Blackwell ([Blackwell, 1953](#ref-blackwell1953equivalent)); on the first 30 instances also through all $`3^4=81`$ deterministic channels, which are the extreme points of the convex objective. An independent $`Z`$ is appended to test augmentation. Exact rational arithmetic.
- *Result.* The largest $`B(D')-B(D)`$ over all garblings, including the adversarial extreme-point ones, is exactly 0, and $`B(D,Z)=B(D)`$ holds exactly on every instance.
- *Verdict.* PASS.

### [Check 2.6](../experiments/02-10_few_shot_information_floor.md#c2-6) · Demonstration-count curves, replay and an actual learner

- *What it tests for this proposition.* Replay is an environment-independent post-processing, so it must leave the information floor unchanged.
- *Setup.* Exact hidden-bit risk curves for $`m\in\lbrace4,8,16\rbrace`$ contexts, $`K=0,\dots,32`$ demonstrations and budgets $`0,1,2,4`$; the *replay* access type shows already seen labels again, at the same budget as extra demonstrations or targeted queries.
- *Result.* The replay gain is exactly 0 in all 944 cells; for $`m=8`$ the replay curve coincides with "$`K`$ demonstrations only", $`\tfrac12(7/8)^K`$.
- *Verdict.* PASS.

## Reproduce

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)) and SciPy ([Virtanen et al., 2020](#ref-virtanen2020scipy)).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C1 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only E5 --output theory/v015_checks/quick/sec2.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: the entry of `v014:C1`; code `theory/independent_checks/check_v014_independent.py` (`check_C1`).
- `theory/v015_checks/results_v015_independent.json`: the group of `v015:E5`; code `theory/v015_checks/experiments.py` (`check_E5`).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-blackwell1953equivalent"></a>

**David Blackwell (1953).** Equivalent comparisons of experiments. *The Annals of Mathematical Statistics* 24, no. 2, 265–272. DOI [10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).

<a id="ref-harris2020array"></a>

**Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, Travis E. Oliphant (2020).** Array programming with NumPy. *Nature* 585, no. 7825, 357–362. DOI [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

<a id="ref-virtanen2020scipy"></a>

**Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, SciPy 1.0 Contributors (2020).** SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods* 17, no. 3, 261–272. DOI [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
