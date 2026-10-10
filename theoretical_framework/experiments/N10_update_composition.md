# N10 · Update composition amplifies the information requirement — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N10_update_composition.md) · [中文](N10_update_composition_ZH.md)

**Verdict.** N10 holds: each of its four statements is confirmed by exact computation, and the controlled network locates where the local SGD expansion applies.

- **Theorem 17** (degree composition). In C13, all 12 random polynomial update chains have actual block degrees exactly equal to the bound $D_i=Nb_i\prod_{t>i}a_t$. Blockwise moment matching through $D_i$ preserves the learning value exactly in every chain.
- **Attainment and the threshold learner.** The squaring witness separates sources that match every lower moment, at degrees 1, 2, 4 and 8 (gaps $-1$, $1/4$, $3/256$, $315/16777216$). The threshold learner separates two equal-mean laws, $2/3$ against $0$.
- **Expansion (35).** The remainder stays below its $\eta^3$ bound in every case; the largest remainder-to-bound ratio is 0.274.
- **Error accumulation.** V7 confirms the geometric bound with exact equality, and it confirms that a linear-in-$R$ simplification fails without a stability condition ($65/32>1$).
- **Regime boundary (C23).** The tangent affine-SGD prediction agrees with the actual network in all 35 SGD settings with $\eta\le0.1$ and at most 4 updates, and with $\eta=0.01$ over 40 updates. Its largest deviation there is $2.5\times10^{-4}$. At $\eta\ge1$ it departs in 35 of 42 settings, by up to 0.53.

![Regime map of the local expansion and exact composition checks](../figures/N10_regime.png)
[PDF](../figures/N10_regime.pdf)

## C13 — degree composition, attainment, threshold learner, expansion (35)

**Why.** Theorem 17 claims that the response degree in block $i$ is at most $Nb_i\prod_{t>i}a_t$, and that this many moments suffice. A wrong degree count would make the moment-matching sufficiency false. A loose count would leave the bound unattained. The expansion (35) is used as the first-order description of SGD in nodes N11–N12, so its remainder bound must hold.

**Setup.**
- 12 random chains with $R\in\lbrace 1,2,3\rbrace $ and $N\in\lbrace 1,2\rbrace $. The degrees are $a_i,b_i\in\lbrace 1,2\rbrace $, and $p_i=\lambda_ip_{i-1}^{a_i}+(1-\lambda_i)Z_i^{b_i}$ with random $\lambda_i\in\lbrace \tfrac14,\tfrac12,\tfrac34\rbrace $.
- Exact symbolic pullback (SymPy) and exact rational block moments. For block $i$ the laws are the scalar laws (26) of order $D_i+1$.
- The squaring witness for $R=1,\dots,4$, and the threshold laws of Proposition 17b.
- For (35): 15 random cubic $V$ on $\mathbb R^3$ with random discrete gradient laws on 6 atoms, and the same with a logistic $V(\theta)=\sigma(w^\top\theta)$. Step sizes are $\eta\in\lbrace 0.01,0.1,0.5,1\rbrace $.

**Procedure.**
- Expand $J(p_R)$, read off each block degree, and compare it with $D_i$.
- Evaluate both learning values with moments matched through $D_i$.
- For (35), compute the exact difference $(K^P-K^Q)V(\theta)$ and the first two expansion terms, and compare the remainder with $\tfrac{M_3\eta^3}6(\mathbb E_P\lVert G\rVert^3+\mathbb E_Q\lVert G\rVert^3)$.
  - For cubics, $M_3$ is the Frobenius norm of the constant third-derivative tensor, which upper-bounds the operator norm.
  - For the logistic, $M_3=\lVert w\rVert^3/8$, because $\lvert\sigma'''\rvert\le1/8$ globally.

**Result.** Source: `theory/independent_checks/results_independent_v014.json → claims[id=C13]`.

| Quantity | Value |
|---|---|
| Status, instances, assertions, worst residual | PASS, 72, 371, 0.0 |
| Declared vs actual block degrees (`details.degrees`, 12 chains) | equal in every chain, e.g. $a=(2,2,2)$, $b=(2,2,1)$, $N=2$: $D=(16,8,2)$, actual $(16,8,2)$ |
| Value equality under blockwise matching | exact in all 12 chains |
| Squaring witness (`details.amplification`) | $R=1,2,3,4$; degree $1,2,4,8$; gap $-1$, $1/4$, $3/256$, $315/16777216$ |
| Threshold learner (assertions in `check_C13`) | equal mean $1/2$; values $2/3$ vs $0$ |
| Logistic (35) remainder/bound, max over 15 instances (`details.logistic`) | $\eta=0.01$: 0.274; $0.1$: 0.274; $0.5$: 0.270; $1$: 0.265 |
| Largest remainder/bound overall (`best_ratio`) | 0.2743 |

**What this shows.** The degree bound (34) is exact on random chains and is attained by the squaring witness. The order a source must match therefore grows with the composition of updates, as the theorem states. The threshold learner shows that no fixed moment order suffices once the update is discontinuous. The expansion (35) holds with room to spare at every tested step size.

## V7 — accumulation of per-step errors

**Why.** A one-step $O(\eta^2)$ error does not by itself give an $O(R\eta^2)$ training error. The proof states a geometric bound that is sharp, and the claim that linear accumulation needs stability. Both must be checked exactly.

**Setup.** Exact rational recursions $e_{t+1}=s\cdot((1+\eta L)e_t+C\eta^2)$ with slack $s\in(0,1]$, over random $\eta\in(0,1]$, $L\in[0,1]$, $C\in(0,1]$ and $R\le40$. Fixed-seed random restarts and hill-climbing over $(\eta,L,C,R,s)$ supplement the random cases.

**Procedure.** Assert $e_R\le C\eta^2\sum_{s<R}(1+\eta L)^s$ and equality at $s=1$. Search for the largest ratio of the bound to the linear simplification $RC\eta^2$. Evaluate the witness $\eta=\tfrac12$, $L=C=1$, $R=4$.

**Result.** Source: `theory/v015_checks/results_v015_independent.json → groups[id=V7]`.

| Quantity | Value |
|---|---|
| Status, instances, assertions, worst residual | PASS, 717, 1435, 0.0 |
| Geometric bound attained (`data.geometric_adversarial.best.score`) | 1.0 (e.g. $\eta=4/5$, $L=1/5$, $C=7/10$, $R=13$) |
| Witness (`data.linear_failure`) | $e_4=65/32$, bound $65/32$, linear $RC\eta^2=1$ |
| Largest bound/linear ratio found (`best_ratio`) | $3.78\times10^{30}$ ($\eta=2$, $L=2$, $R=47$) |

**What this shows.** The geometric bound is exact. It holds in every instance and is attained with equality. The linear-in-$R$ reading fails as soon as $\eta LR$ is not small. Accumulating local expansions along training therefore requires the stated stability or short-horizon condition.

## C23 (SGD settings) — where the local affine description applies in a real network

**Why.** Node N10 predicts that, for SGD, the response is governed by the mean consumed gradient up to $O(\eta^2)$ per step, with errors that accumulate geometrically. In a real nonlinear network this should make a fixed initial-tangent affine-SGD prediction accurate at small $\eta R$, and inaccurate at large $\eta$ or long horizons.

**Setup.** Controlled neural setting, not Gemma and not an external benchmark.
- Network: one-hot context, then 4 tanh units, then a sigmoid; binary cross-entropy; float64.
- Training: plain SGD per sampled block, with $\eta\in\lbrace 0.01,0.1,1,3\rbrace $ and $R\in\lbrace 1,4,40\rbrace $ updates.
- Tasks: Prop. 5 and Prop. 6 sources, moment laws $N=2,3,4$, and parity laws $N=2,3$.
- Estimation: complete finite-history enumeration where it is enumerable, otherwise 2400 independent training runs per source.
- Comparator: the actual policy's initial-tangent affine-SGD response.
- Agreement margin, fixed in code before execution: $10^{-4}+0.1\lvert\text{prediction}\rvert+4\,\mathrm{SE}$.

**Procedure.** For each setting, compute the actual final expected-return gap between the two sources and the tangent-affine prediction, and record agreement under the margin.

**Result.** Source: `theory/independent_checks/results_independent_v014.json → controlled_neural.details.configurations` (SGD rows). Cohort totals are in `details.observed_affine_comparison_cohorts`.

| $\eta$ | $R=1$ | $R=4$ | $R=40$ |
|---|---|---|---|
| 0.01 | 7/7 agree (max dev $1.1\times10^{-6}$) | 7/7 ($4.1\times10^{-6}$) | 7/7 ($2.4\times10^{-5}$) |
| 0.1 | 7/7 ($1.1\times10^{-4}$) | 7/7 ($2.5\times10^{-4}$) | 5/7 ($1.9\times10^{-3}$) |
| 1 | 3/7 (0.013) | 0/7 (0.053) | 3/7 (0.084) |
| 3 | 0/7 (0.52) | 0/7 (0.53) | 1/7 (0.51) |

Cohort "SGD, step ≤ 0.1, at most four updates": 28/28 agreements, largest absolute deviation $2.52\times10^{-4}$. The additive one-step null (identical examples and weights, regrouped) has parameter residual exactly $0.0$ (`details.additive_null_parameter_residual`).

**What this shows.** The affine first-order description of SGD is accurate exactly where the expansion (35) and the stability condition of Proposition R say it should be: small step size and few updates, or a very small step over many updates. It breaks down as $\eta R$ grows. The departure grows with $\eta$, consistent with the curvature and accumulation terms. This regime boundary is what nodes N11–N12 refine for Adam.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --only C13,C23 --output theory/independent_checks/subset.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V7 --output theory/v015_checks/quick/v7.json
```

## Artifacts

- `theory/independent_checks/results_independent_v014.json`: `claims[id=C13]` and `controlled_neural`.
- `theory/independent_checks/figs/C23_controlled_neural.png`: the original C23 panel.
- `theory/v015_checks/results_v015_independent.json`: `groups[id=V7]`.
- Check code: `theory/independent_checks/check_v014_independent.py` (`check_C13`, `check_C23`) and `theory/v015_checks/mathematics.py` (`check_V7`).
