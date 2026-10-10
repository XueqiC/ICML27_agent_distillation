# N03 · Few-shot information floor — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N03_few_shot_information_floor.md) · [中文](N03_few_shot_information_floor_ZH.md)

**Verdict.** N03 holds. All checks pass, exact in rational arithmetic except for the logarithms in KL:

- **V5** (348 instances, 26,214 assertions, worst residual 0). In all 100 brute-force cases ($d\le5$, $K\le4$, $n\le3$), the Bayes-optimal risk equals (A11). In all 100, the best label-adaptive query decision tree equals the greedy largest-weight rule (A13). In 80 of the cases a trajectory reveals several coordinates at once, so the result does not depend on singleton reveals.
- **V6** (297 instances). The two-world bound (A15) holds on every adaptive query/stop process, and is *tight*: the ratio $\operatorname{kl}(1-\delta,\delta)/(K\kappa_D+\kappa_X\mathbb EN)$ reaches exactly $1$.
- **E5** (944 exact $(K,\text{query})$ cells). Replay gain is $0$ in every cell. Targeted queries beat extra passive demonstrations at equal budget. An actual Adam or SGD learner never goes below the information floor.
- **Package checks T09, T10, T11** reproduce the same formulas on independent constructions.

![Information floor in K, query value, and the KL budget](../figures/N03_floor.png)

[PDF](../figures/N03_floor.pdf)

### V5 — Propositions A5–A6: hidden-bit coverage, hard query budget, adaptivity

**Why.**

- (A11) claims the Bayes risk is $\tfrac12\sum_jw_j(1-q_j)^K$ for *any* within-trajectory dependence of reveal events.
- (A13) claims that no label-adaptive query policy beats greedy-by-weight.
- (A14) claims that the positive part must stay inside the expectation.

A brute-force Bayes or decision-tree value differing from the formula would falsify these, as would an instance with $\mathbb E[(M-n)_+]<(\mathbb EM-n)_+$.

**Setup.**

- 100 cases: $d=1,\dots,5$ (20 each), $K=0,\dots,4$, $n=0,\dots,3$.
- Random rational weights $w$ and random reveal-set laws over bitmasks.
- Enumeration covers all $2^d$ bit worlds and every ordered reveal history; the largest case has 2,321 transcripts.
- Bayes actions are taken per posterior. Label-dependent query decision trees are minimized exactly; the largest has 271 query nodes.
- The uniform-occupancy law of $M_K$ is computed by two independent methods, a subset recursion and inclusion–exclusion, for $d=2,\dots,5$ and $K\le4$.
- An adversarial hill-climb (8 restarts × 20 steps) searches for a violation of Jensen's inequality or of greedy optimality.
- Seed `SeedSequence([150150, 5])`; `fractions.Fraction` throughout.

**Result.**

| Quantity | Value | JSON key (in `groups[id=V5]`) |
|---|---|---|
| Instances / assertions / worst residual | 348 / 26,214 / 0 | `n_instances`, `n_assertions`, `worst_residual` |
| Cases with Bayes risk = (A11) | 100 / 100 | `data.cases[*].Bayes` vs `.A11` |
| Cases with adaptive optimum = greedy (A13) | 100 / 100 | `data.cases[*].adaptive` vs `.greedy` |
| Cases with multi-coordinate reveal sets | 80 / 100 | `data.cases[*].reveal` (bitmask keys) |
| Cases where $\mathbb E[(M-n)_+]>(\mathbb EM-n)_+$ strictly | 36 / 100 | `data.cases[*].E_positive` vs `.positive_E` |
| Strict witness | $d=2,K=1,n=1$: $\mathbb E[(M-1)_+]=\tfrac12$ vs $(\mathbb EM-1)_+=0$; risk $\tfrac5{56}$ | `data.strict_Jensen` |
| Two occupancy methods agree; (A14) curve exact | all $d\le5$, $K\le4$, $n\le3$ | `data.uniform` |
| Largest violation found by adversarial search | $0$ | `data.adversarial.best.score` |

**What this shows.** The information floor and the optimal-query value are exactly the closed forms of N03, including correlated reveal events within a trajectory. Value-adaptive querying never helped, as the independence argument predicts. Moving the positive part outside the expectation would be wrong in more than a third of the cases.

### V6 — Proposition A7: demonstration information credits the adaptive query budget

**Why.** (A15) must hold for *every* common history-dependent query/stop rule. A process with $\operatorname{kl}(1-\delta,\delta)>K\kappa_D+\kappa_X\mathbb E_vN$ would falsify it. A ratio far below 1 everywhere would mean the bound is loose.

**Setup.**

- Two-world processes with $K\in\lbrace 1,2\rbrace $ symmetric informative demonstrations and $N_{\max}\le4$.
- Two random binary query channels.
- Randomized history-dependent selection and stopping rules, shared by both worlds.
- Exact rational transcript probabilities. The transcript KL is computed directly and, separately, as the adaptive chain sum; logarithms are in float64 with tolerance $10^{-8}$.
- 160 random processes, plus an adversarial hill-climb (8 restarts × 16 steps), plus two constructed cases:
  - a tight case: $K=0$, one binary query with channel $(\tfrac45,\tfrac15)$, $\delta=\tfrac15$;
  - a zero-information case: $\kappa_D=\kappa_X=0$.
- Seed `SeedSequence([150150, 6])`.

**Result.**

| Quantity | Value | JSON key (in `groups[id=V6]`) |
|---|---|---|
| Instances / assertions / worst residual | 297 / 2,969 / 2.7e-15 | `n_instances`, `n_assertions`, `worst_residual` |
| Random processes satisfying the premise $0<\delta<\tfrac12$ | 159 / 160 | `data.random_premise_count` |
| Largest ratio, random processes | 0.9973 | `data.worst_random.score` |
| Largest ratio, adversarial search | 0.9760 | `data.adversarial.best.score` |
| Ratio, constructed tight case | **1.0** (KL $=\operatorname{kl}(\tfrac45,\tfrac15)=0.8318$, $\mathbb EN=1$) | `best_ratio`, `data.tight` |
| Zero-information case | transcript KL $=0$; the two error probabilities sum to 1 | `data.zero_information` |

**What this shows.** The KL chain identity (direct transcript KL = accumulated chain) holds exactly for adaptive stopping. (A15) is never violated. It is attained with equality by a single informative query, so its constant cannot be improved. When demonstrations and queries carry no information, uniform identification is impossible.

### E5 — Hidden-bit $K$ curves, replay, and an actual learner

**Why.** This separates three things the theory keeps apart:

- independent new evidence (more trajectories or targeted queries);
- replay of seen evidence, which has zero information value by Proposition 1.4 of N01;
- the gap between information availability and a learner's actual use of it.

**Setup.**

- Exact curves of $R_{\rm info}(K,\cdot)$ for $m\in\lbrace 4,8,16\rbrace $ contexts, uniform and nonuniform $q$ and $w$, $K=0,\dots,32$, and query budgets $0,1,2,4$. The computation uses integer subset Möbius inversion with rational risks.
- Three access types at equal budget:
  - *active*: queries target unseen contexts;
  - *passive*: the same budget is spent on extra random demonstrations;
  - *replay*: seen labels are shown again.
- An actual tabular learner (one logit per context) trained by 12 updates of separable summed cross-entropy with Adam or SGD on the revealed bits; unseen logits stay at 0. Deployment samples Bernoulli actions.
- Seed `SeedSequence([150150, 105])`.

**Result.**

| Quantity | Value | JSON key (in `groups[id=E5]`) |
|---|---|---|
| Exact cells / assertions / worst residual | 944 / 7,450 / 5.6e-17 | `data.curves`, `n_assertions`, `worst_residual` |
| Replay gain | $0$ in all 944 cells | `data.curves[*].replay_gain` |
| $m=8$ uniform, $K=4$: no query / 4 passive demos / 4 active queries | $0.2931$ / $0.1718$ / $0.0431$ ($\tfrac{2401}{8192}$, $\tfrac{5764801}{33554432}$, $\tfrac{353}{8192}$) | `data.curves[m=8,uniform,K=4]` |
| $m=8$ uniform, $K=8$: no query / 4 active | $0.1718$ / $0.0013$ | same |
| Actual learner rows | 5,664 (Adam, SGD × active, passive, replay) | `data.actual_learners` |
| $\min(\text{learner risk}-\text{information floor})$ | $0.0$ (never below the floor) | `data.actual_learners[*].risk`, `.information_floor` |
| Learner error on *revealed* bits after 12 updates | Adam 0.239, SGD 0.372 | `data.actual_learners[*].observed_bit_error` |

**What this shows.**

- Every exact curve equals (A11)/(A13).
- Replay leaves the floor unchanged in every cell.
- At equal budget, targeted queries remove far more risk than additional passive demonstrations: at $K=4$, $m=8$, four active queries reach $0.043$ against $0.172$. This is the coverage effect (A12)–(A13).
- The actual learner stays above the floor. Its excess over the floor comes from imperfect fitting of revealed bits, which is the $\mathcal R_{\rm alg}$ term of N01. Information availability and information use are therefore separately visible.

### Package checks T09, T10, T11

**Why.** These are independent implementations from the theory package and give a reproduction on different constructions.

**Setup and result.**

| Check | Construction | Result | JSON key (`docs/direction/v015_revision/results_v015.json`) |
|---|---|---|---|
| T09 | $d=3$, $w=(\tfrac12,\tfrac13,\tfrac16)$, reveal sets $\varnothing,\lbrace 0,1\rbrace ,\lbrace 1,2\rbrace ,\lbrace 0,2\rbrace $ with probabilities $\tfrac1{10},\tfrac2{10},\tfrac3{10},\tfrac4{10}$ ($q=(\tfrac35,\tfrac12,\tfrac7{10})$); full posterior enumeration and decision-tree minimization, $K\le4$, $n\le3$ | 20/20 cases equal the formula. E.g. $K=0$: $\tfrac12,\tfrac14,\tfrac1{12},0$; $K=4$: $\tfrac{2099}{120000},\tfrac1{40000},\tfrac1{120000},0$ | `checks.T09_exact_posterior_and_adaptive_query_optimum.exact_curves` |
| T10 | Uniform $d=4$, $K\le6$, $n\le4$ | 35/35 exact. At $K=3,n=2$: correct risk $\tfrac1{128}$; the formula with the positive part outside the expectation gives 0 | `checks.T10_uniform_coverage_and_positive_part` |
| T11 | Two worlds, demonstration law $(\tfrac3{10},\tfrac7{10})$, two query channels, score-based stopping, $N_{\max}=4$, $K\le4$ | 10/10 directional rows satisfy binary-KL lower $\le$ exact transcript KL $\le K\kappa_D+\kappa_X\mathbb EN$, and $\mathbb EN\ge$ (A16). Lower/upper ratio 0.37–0.56 | `checks.T11_adaptive_query_KL_enumeration.rows` |

**What this shows.** Three further independent constructions agree with (A11), (A13), (A14) and (A15)–(A16).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V5,V6,E5 --output theory/v015_checks/quick/n03.json
theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py --out theory/v015_checks/quick/results_v015_package_rerun.json
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: `groups[id=V5|V6|E5]`.
- `theory/v015_checks/mathematics.py` (`check_V5`, `check_V6`) and `theory/v015_checks/experiments.py` (`check_E5`).
- `docs/direction/v015_revision/results_v015.json`: `checks.T09_*`, `checks.T10_*`, `checks.T11_*`.
