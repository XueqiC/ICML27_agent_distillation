# Proposition 2.11 · Demonstrations and queries share one information budget

*Experiments*

[← Framework](../README.md) · [Statement](../statements/02-11_shared_information_budget.md) · [Proof](../proofs/02-11_shared_information_budget.md) · [Experiments](02-11_shared_information_budget.md) · [中文](02-11_shared_information_budget_ZH.md)

**What is tested here.** [Check 2.5](../experiments/02-11_shared_information_budget.md#c2-5) tests the budget inequality (2.11) on 297 adaptive query-and-stop processes with exact transcript probabilities: it is never violated, and a single informative query attains it with equality (ratio 1.0), so its constant cannot be improved (PASS). The package construction `pkg:T11` of [Check 2.7](../experiments/02-10_few_shot_information_floor.md#c2-7) confirms it with score-based stopping (PASS).

**Setting used on this page.**

- Two worlds $`v\in\lbrace0,1\rbrace`$; $`K`$ IID demonstrations with per-demonstration KL at most $`\kappa_D^{(v)}`$; queries with per-outcome KL at most $`\kappa_X^{(v)}`$ given any history $`h`$; a world-independent query-and-stop rule with $`N\le N_{\max}`$.
- $`\mathrm{kl}(a,b)`$ is the binary KL divergence; $`H_{t-1}`$ is the history before query $`t`$.

**Statement under test ([Proposition 2.11](../statements/02-11_shared_information_budget.md#s2-11)).** Consider two-world adaptive acquisition.

- The world is $`v\in\lbrace 0,1\rbrace`$, and the two worlds require different final decisions.
- Each of $`K`$ IID demonstrations satisfies $`\mathrm{KL}(P_v^{\rm demo}\Vert P_{1-v}^{\rm demo})\le\kappa_D^{(v)}<\infty`$.
- A common history-dependent rule selects queries and stops at $`N\le N_{\max}<\infty`$. The rule's selection, stopping and private randomization kernels do not depend on $`v`$.
- For every allowed query $`x`$ and every preceding history $`h`$ on the relevant support, $`\mathrm{KL}(P_v(\cdot\mid h,x)\Vert P_{1-v}(\cdot\mid h,x))\le\kappa_X^{(v)}<\infty`$.
- The final decision errs with probability at most $`\delta`$ in each world, where $`0<\delta<1/2`$.

Then

```math
K\kappa_D^{(v)}+\kappa_X^{(v)}\,\mathbb E_vN\ge\mathrm{kl}(1-\delta,\delta). \qquad(2.11)
```

If $`0<\kappa_X^{(v)}<\infty`$,

```math
\mathbb E_vN\ge\frac{[\mathrm{kl}(1-\delta,\delta)-K\kappa_D^{(v)}]_+}{\kappa_X^{(v)}}. \qquad(2.12)
```

If $`\kappa_X^{(v)}=0`$ and $`K\kappa_D^{(v)}<\mathrm{kl}(1-\delta,\delta)`$, no query budget achieves the stated errors.

<a id="c2-5"></a>

## Check 2.5 · Demonstration information credits the adaptive query budget (Proposition 2.11)

**Verdict.** PASS: 297 instances, 2,969 assertions, worst residual $`2.7\times10^{-15}`$. The inequality (2.11) is never violated and is attained with equality (ratio 1.0) by a single informative query.

**Why.** Proposition 2.11, like the change-of-measure inequality of best-arm identification that it extends ([Kaufmann et al., 2016](#ref-kaufmann2016complexity)), must hold for *every* common history-dependent query-and-stop rule. A process with $`\mathrm{kl}(1-\delta,\delta)>K\kappa_D+\kappa_X\mathbb E_vN`$ would falsify it; a ratio far below 1 everywhere would mean the bound is loose.

**Setup.**

- Two-world processes with $`K\in\lbrace 1,2\rbrace`$ symmetric informative demonstrations and $`N_{\max}\le4`$, two random binary query channels, and randomized history-dependent selection and stopping rules shared by both worlds.
- Exact rational transcript probabilities. The transcript KL is computed directly and, separately, as the adaptive chain sum; logarithms are in float64 with tolerance $`10^{-8}`$.
- 160 random processes, an adversarial hill-climb (8 restarts × 16 steps), and two constructed cases: a tight case ($`K=0`$, one binary query with channel $`(\tfrac45,\tfrac15)`$, $`\delta=\tfrac15`$) and a zero-information case ($`\kappa_D=\kappa_X=0`$).
- Seed `SeedSequence([150150, 6])`.

**Result.**

| Quantity | Value | JSON key (entry `v015:V6`) |
|---|---|---|
| Instances / assertions / worst residual | 297 / 2,969 / 2.7e-15 | `n_instances`, `n_assertions`, `worst_residual` |
| Random processes satisfying the premise $`0<\delta<\tfrac12`$ | 159 / 160 | `data.random_premise_count` |
| Largest ratio $`\mathrm{kl}(1-\delta,\delta)/(K\kappa_D+\kappa_X\mathbb EN)`$, random processes | 0.9973 | `data.worst_random.score` |
| Largest ratio, adversarial search | 0.9760 | `data.adversarial.best.score` |
| Ratio, constructed tight case | **1.0** (KL $`=\mathrm{kl}(\tfrac45,\tfrac15)=0.8318`$, $`\mathbb EN=1`$) | `best_ratio`, `data.tight` |
| Zero-information case | transcript KL $`=0`$; the two error probabilities sum to 1 | `data.zero_information` |

**Why it supports the statement.**

- The KL chain identity (direct transcript KL equals the accumulated chain) holds exactly under adaptive stopping, which is the core of the proof.
- (2.11) is never violated, and a single informative query attains it with equality, so its constant cannot be improved.
- When demonstrations and queries carry no information, uniform identification is impossible, as the last clause of Proposition 2.11 states.

### [Check 2.7](../experiments/02-10_few_shot_information_floor.md#c2-7) · Independent package constructions

- *What it tests for this proposition.* The two-sided sandwich behind (2.11) and (2.12): binary-KL lower bound ≤ exact transcript KL ≤ $`K\kappa_D+\kappa_X\mathbb EN`$, and $`\mathbb EN`$ at least the bound (2.12).
- *Setup.* Package construction `pkg:T11`: two worlds, demonstration law $`(\tfrac3{10},\tfrac7{10})`$, two query channels, score-based stopping, $`N_{\max}=4`$, $`K\le4`$; exact enumeration.
- *Result.* All 10 directional rows satisfy the sandwich and the bound on $`\mathbb EN`$; the ratio of lower to upper side is 0.37–0.56.
- *Verdict.* PASS.

## Reproduce

The scripts use NumPy ([Harris et al., 2020](#ref-harris2020array)).

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V6 --output theory/v015_checks/quick/sec2.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: the group of `v015:V6`; code `theory/v015_checks/mathematics.py` (`check_V6`).
- `docs/direction/v015_revision/results_v015.json`: `checks.T11_adaptive_query_KL_enumeration.rows`; code `docs/direction/v015_revision/verify_v015_extensions.py`.
