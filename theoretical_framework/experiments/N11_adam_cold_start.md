# N11 · Cold-start Adam quotient and optimizer memory — Experiments

[← Framework map](../README.md) · [Proof](../proofs/N11_adam_cold_start.md) · [中文](N11_adam_cold_start_ZH.md)

**Verdict.** N11 holds. Every statement of Propositions A2–A4 is confirmed exactly, and the controlled network shows the cold-start quotient at work.

- **A2 (cold quotient).** On 300 random finite gradient alphabets, V2 finds the following:
  - the cold Adam step equals $c-\eta h_\epsilon(G)$ to $4.4\times10^{-16}$;
  - exactly $s-1$ statistics are necessary and sufficient at $\epsilon=0$, and $m-1$ at $\epsilon>0$;
  - the inverse (A8) has derivative $\epsilon/(1-\lvert u\rvert)^2$, which reaches $10^{22}$.
- **A3 (margin bound).** V3 confirms the bound on 1,660 exact instances. Adversarial search reaches a gap/bound ratio of 0.994, so the bound is nearly tight.
- **A4 (optimizer memory).** V4 and E3 reproduce the exact counterexample: $\widehat m_2=\mp\tfrac16$, $\widehat v_2=\tfrac{17}{24},\tfrac{41}{24}$, and opposite second steps. The second steps stay opposite for every $\epsilon$ tested from $0$ to $10^{-1}$. Witnesses exist in every $(\beta_1,\beta_2)$ grid cell with $\beta_1>0$ and in none with $\beta_1=0$, as Corollary A4′ predicts.
- **Mechanism in a real network.** In the controlled network (E0–E2), two label laws that agree on all moments through order 2 have identical gradient moments through order 2. Yet their expected first Adam update differs by $\eta$ per coordinate, and Adam's return gap is 0.0769 (R=1) and 0.1196 (R=4) at $\eta=0.1$, while SGD's is about 0.
  - The gap survives a frozen Jacobian (0.0757 and 0.1225), so it is an optimizer effect, not feature learning.
  - It vanishes when the support is shifted so that all first gradients share one sign (gap 0).
  - It falls to 0.0034 and 0.0098 after five warm moment steps.

![Cold-start Adam: optimizer-matched comparison, ablations, and the two-step memory witness](../figures/N11_cold_adam.png)
[PDF](../figures/N11_cold_adam.pdf)

## V2 — cold map, exact quotient dimensions, inverse conditioning (Prop. A2)

**Why.** A2 claims three things: an exact first-step formula; that the immediate-parameter information is the joint sign law (dimension $s-1$) at $\epsilon=0$ and the full law (dimension $m-1$) at $\epsilon>0$; and that coordinate sign marginals are insufficient. Each is a finite statement that can be checked exactly.

**Setup.**
- 300 random finite gradient alphabets with $p=1,\dots,4$ coordinates, including zero coordinates.
- An independent recurrence for (A6), checked in float64 and in exact rationals where the square root is exact.
- Exact ranks and null spaces; HiGHS LPs to construct separating laws when fewer than $s-1$ features are used.
- Inverse round-trips on a $u\to1$ grid with $\epsilon=1/100$.

**Procedure.**
- Compare the first step with $c-\eta h_\epsilon(G)$.
- Count attainable sign patterns $s$, and verify that $s-1$ pattern indicators suffice.
- For any smaller feature set, exhibit a zero-mass perturbation and the two laws $P,Q$ that it separates.
- Repeat with $\epsilon>0$, which gives dimension $m-1$.
- Evaluate the joint-versus-marginal example.

**Result.** Source: `theory/v015_checks/results_v015_independent.json → groups[id=V2]`.

| Quantity | Value |
|---|---|
| Status, instances, assertions, worst residual | PASS, 300, 3288, $4.44\times10^{-16}$ |
| Quotient dimension $= s-1$ at $\epsilon=0$ (`data.quotients[*].necessary_and_sufficient`) | 300/300 alphabets (1 to 14 atoms, $s$ from 1 to 14) |
| Dimension $= m-1$ at $\epsilon>0$ (`finite_epsilon_dimension`) | 300/300 |
| Example separation (`quotients[0]`) | 4 atoms, 4 patterns, 2 features: $P=(\tfrac14,\tfrac18,\tfrac38,\tfrac14)$ vs $Q=(\tfrac14,\tfrac38,\tfrac18,\tfrac14)$ match both features, but the pattern indicator differs by $-\tfrac14$ |
| Inverse derivative, $\epsilon=1/100$ (`data.conditioning`) | $u=0.9$: 1; $u=0.999$: $10^4$; $u=1-10^{-12}$: $10^{22}$ |
| Joint vs marginal (`data.joint_vs_marginal`) | marginals identical; same-sign utility 1 vs 0 |

Package check (independent implementation; `docs/direction/v015_revision/results_v015.json → checks`):
- T03: 480 cases, update error $\le2.22\times10^{-16}$.
- T04: 25 atoms, 9 sign patterns, dimensions 8 vs 24.
- T05: mean updates equal, same-sign utility gap 1.0.
- T06: label laws matching orders 0–2 have SGD mean parameter gap $0$, ideal-sign Adam gap $1/10$ and finite-$\epsilon$ gap $250/2703$.

**What this shows.** The exact information content of a cold Adam step is the joint sign quotient at $\epsilon=0$ and the full gradient law at $\epsilon>0$. The counts are exactly $s-1$ and $m-1$. T06 shows the same thing at the level of means: laws that SGD cannot distinguish at first order are distinguished by Adam.

## V3 — margin-sensitive approximation (Prop. A3)

**Why.** (A9) quantifies when the sign quotient is a good *approximate* description for regular utilities. It is the bridge between the exact statement (any $\epsilon>0$ identifies magnitudes) and practice.

**Setup.** Random finite gradient laws $P,Q$, and coordinatewise-Lipschitz $F\in[0,1]$ built as clipped sums of piecewise-linear coordinate functions with known $L_i$. Random margins $a$ and $\epsilon$. Exact rational arithmetic throughout. Adversarial restarts with hill-climbing over $(G,P,Q,L,a,\epsilon,\eta)$.

**Procedure.** Assert (A9) and $\Delta_{\rm sign}\le\mathrm{TV}(P_{\rm sign},Q_{\rm sign})$ on every instance, and maximize gap/bound.

**Result.** Source: `groups[id=V3]`.

| Quantity | Value |
|---|---|
| Status, instances, assertions, worst residual | PASS, 1660, 3320, 0.0 |
| Largest gap/bound, adversarial (`data.adversarial.best.score`) | 0.9941 |
| Largest gap/bound, random (`data.worst_random.score`) | 0.837 |
| Largest $\Delta_{\rm sign}/\mathrm{TV}$ found (`data.sign_TV_adversarial.best.score`) | 0.274 |

Package check T07: 1,500 instances, maximum gap/bound 0.739, no violation.

**What this shows.** (A9) is valid and nearly tight (0.994). For regular utilities, the sign quotient is accurate up to the gradient mass near zero and the $\epsilon/(a+\epsilon)$ terms.

## V4 and E3 — magnitude-bearing optimizer state (Prop. A4)

**Why.** A4 claims that matching first-step parameter information does not suffice for a learning continuation, because $m_1$ retains magnitudes. The counterexample must be exact, realizable by real losses, robust to $\epsilon$, and present in the actual optimizer code.

**Setup.**
- V4: exact Fraction and symbolic arithmetic for $(1,-\tfrac34)$ versus $(2,-\tfrac34)$ with $\beta_1=\beta_2=\tfrac12$; the squared-loss realization; $\epsilon\in\lbrace 0,10^{-1},\dots,10^{-12}\rbrace $; and a $5\times5$ grid $\beta_1,\beta_2\in\lbrace 0,\tfrac1{10},\tfrac12,\tfrac9{10},\dots\rbrace $ with 300 search trials per cell.
- E3: the same pair through the scalar optimizer used by E1/E2, at $\epsilon=0$ and $10^{-8}$, plus a soft-label search in the controlled network for same-sign histories with opposite second updates.

**Procedure.** Compute every moment and both second steps, and check that their signs are opposite. In the network, search label pairs $(y_1,y_2)$ against $(y_1',y_2)$ whose first steps coincide but whose second updates on some coordinate are opposite.

**Result.** Sources: `groups[id=V4]` and `groups[id=E3]`.

| Quantity | Value |
|---|---|
| V4 status, instances | PASS, 39 |
| Exact moments (`data.exact`) | $m_2=-\tfrac18,+\tfrac18$; $\widehat m_2=-\tfrac16,+\tfrac16$; $\widehat v_2=\tfrac{17}{24},\tfrac{41}{24}$ |
| Second steps at $\eta=1$ | $+\sqrt{102}/51=+0.198$ and $-\sqrt{246}/123=-0.128$ |
| Squared-loss realization | first targets $-1,-2$; common second target $-\tfrac14$ |
| Opposite directions for positive $\epsilon$ (`data.positive_epsilon`) | all 14 values from $0$ to $10^{-1}$ |
| $\beta$-grid cells with a witness (`data.grid_fraction`) | 0.8 = 20/25: every cell with $\beta_1>0$, no cell with $\beta_1=0$ |
| E3 scalar trace, $\epsilon=0$ and $10^{-8}$ (`data.scalar`) | second updates $+0.19803$ and $-0.12752$ |
| E3 network witnesses (`data.neural_witnesses`) | found at $\epsilon=0$ and $10^{-8}$: labels $(0,0.7)$ vs $(0.1,0.7)$, identical first step, opposite second step on the output bias |

Package check T08: first outputs equal, sign histories equal, second increments $+0.019803$ and $-0.012752$ at $\eta=0.1$, threshold-task gap 1.

**What this shows.** The counterexample is exact, realizable, robust to $\epsilon$, and occurs in a real network. The $\beta$-grid result isolates the mechanism. The first moment $m$ carries the magnitude memory, so the effect exists precisely when $\beta_1>0$, which is Corollary A4′.

## E0 and E1 — the C23 Adam gap is the cold-start quotient

**Why.** In the controlled network, Adam separates label laws that agree on all moments through order $N-1$, where SGD does not. A2 predicts the mechanism: cold Adam responds to the sign law of $G=p-Z$ around the current prediction, not to moments. E0 fixes the exact reproduction; E1 traces the transitions from input moments to gradient moments, normalized update and return, and ablates one factor at a time.

**Setup.** Controlled neural, not Gemma.
- Network: one-hot context, then 4 tanh units, then a sigmoid; mean BCE; float64.
- Adam: $\beta=(0.9,0.999)$, $\epsilon=10^{-8}$ outside the square root, bias correction on.
- Laws: `law_scalar(N)` matches label moments of orders $1,\dots,N-1$ only.
- Gap: $\mathbb E_P[p_R^N]-\mathbb E_Q[p_R^N]$, by complete finite-history enumeration (SE 0).
- E1 ablations (moment $N=3$, $\eta=0.1$): $\epsilon$ inside the square root; $\epsilon\in\lbrace 10^{-6},10^{-4},10^{-2},10^{-1}\rbrace $; no bias correction; five warm moment steps on a third law; global clip 0.1; sum reduction; and support shifted so that all first gradients share one sign.

**Procedure.** Enumerate per-support gradients, gradient moments, finite-$\epsilon$ updates and their expectations. Compare them with the executed `neural_step` at relative tolerance $10^{-12}$. Then rerun each ablation from the same snapshot.

**Result.** Sources: `groups[id=E0]`, `groups[id=E1]` and `theory/v015_checks/c23_bundle.json` (`laws`, `raw_runs`).

| E0 reproduction ($N=3$) | $\eta=0.01$ | $\eta=0.1$ |
|---|---|---|
| Adam gap, $R=1$ | 0.00757 | 0.07691 |
| Adam gap, $R=4$ | 0.01892 | 0.11961 |
| affine-SGD comparator, $R=1$ / $R=4$ | $-6.3\times10^{-9}$ / $-2.5\times10^{-8}$ | $-6.3\times10^{-6}$ / $-2.1\times10^{-5}$ |

E1 transitions for moment $N=3$ at $\eta=0.1$ (`data.transitions`):
- Input moment differences of orders 1, 2, 3: $0,0,-0.0556$.
- Joint gradient moments: orders 1–2 equal to $3\times10^{-17}$; order 3 differs by 0.056.
- Expected first update: $+0.05$ against $-0.05$ per coordinate, a difference of $\eta$.
- Return difference: 0.0769.
- Under $P$, labels are $0$ with probability $\tfrac14$ and $\tfrac23$ with probability $\tfrac34$, so the output-bias gradient $p-Z$ is positive with probability $\tfrac14$. Under $Q$, labels are $\tfrac13$ with probability $\tfrac34$ and $1$ with probability $\tfrac14$, so the gradient is positive with probability $\tfrac34$.

Measured map error is $2.1\times10^{-15}$ relative (`maximum_measured_relative_error`).

| E1 ablation (moment $N=3$, $\eta=0.1$) | gap $R=1$ | gap $R=4$ |
|---|---|---|
| baseline | 0.0769 | 0.1196 |
| $\epsilon$ inside sqrt; $\epsilon=10^{-6}$; sum reduction | 0.0769 | 0.1196 |
| $\epsilon=10^{-4}$ | 0.0767 | 0.1197 |
| $\epsilon=10^{-2}$ | 0.0610 | 0.1148 |
| $\epsilon=10^{-1}$ | 0.0211 | 0.0446 |
| no bias correction | 0.2477 | $-0.0081$ |
| five warm moment steps | **0.0034** | **0.0098** |
| global clip 0.1 | 0.0769 | 0.1840 |
| shifted support (one first-gradient sign) | **0.0000** | **0.0000** |

The shifted support preserves the matched label moments (`preserved_moments`: orders 1 and 2 still equal).

**What this shows.**
- Two laws with identical gradient moments through order 2 produce first Adam updates that differ by $\eta$ per coordinate. The cause is exactly the sign quotient of A2: the probability that the label lies above the current prediction is $1/4$ under one law and $3/4$ under the other.
- Removing sign disagreement removes the gap entirely. Warming the moments removes most of it.
- The gap is insensitive to the $\epsilon$ convention until $\epsilon$ becomes comparable to gradient magnitudes ($10^{-2}$), as A3 predicts.

## E2 — optimizer-matched frozen-Jacobian comparison

**Why.** A gap against an affine-SGD comparator could come from the optimizer, from nonlinear features, or from both. Freezing the network Jacobian while keeping the sigmoid, the cross-entropy and the same Adam recurrence isolates the optimizer.

**Setup.** Five branches from one snapshot, with identical within-source histories:
- actual network with Adam;
- frozen Jacobian, $\mathrm{logit}(\theta)=\mathrm{logit}(\theta_0)+J_0(\theta-\theta_0)$, with Adam and its own moments;
- actual network with SGD;
- frozen Jacobian with SGD;
- tangent affine SGD.

Grid: 5 tasks × $\eta\in\lbrace 0.01,0.1,1\rbrace $ × $R\in\lbrace 1,4,40\rbrace $ = 45 cells. Derivative checks of $J_0$ have residuals $\le2.6\times10^{-11}$.

**Procedure.** Report all five gaps per cell. A frozen-Adam gap *persists* if its magnitude exceeds $\max(10^{-10},3\,\mathrm{SE})$.

**Result.** Source: `groups[id=E2].data.comparisons`. The representative cells:

| Cell | actual Adam | frozen-Jacobian Adam | actual SGD | tangent SGD |
|---|---|---|---|---|
| moment $N=3$, $\eta=0.1$, $R=1$ | 0.0769 | 0.0757 | 0.0000 | 0.0000 |
| moment $N=3$, $\eta=0.1$, $R=4$ | 0.1196 | 0.1225 | 0.0000 | 0.0000 |
| moment $N=3$, $\eta=0.01$, $R=4$ | 0.0189 | 0.0188 | 0.0000 | 0.0000 |
| parity $N=2$, $\eta=0.1$, $R=4$ | 0.0762 | 0.0774 | 0.0034 | 0.0034 |
| moment $N=4$, $\eta=0.1$, $R=4$ | $-0.0218$ | $-0.0223$ | $-0.0000$ | 0.0000 |

A frozen-Jacobian Adam gap persists in 39 of 45 cells (`persistent_frozen_gaps`). Across all cells with $\eta\le0.1$ and $R\le4$, the three SGD branches agree with one another to within $2.8\times10^{-4}$, and none exceeds 0.0034.

**What this shows.** The Adam gap is reproduced almost exactly by an Adam that cannot learn features. It is therefore an optimizer-state effect, as N11 states. Moment matching (node N07) controls SGD here and does not control cold Adam.

## C23 (Adam settings) — the regime where the affine description fails

**Why.** This is the original observation that N11 explains: under the same laws, the tangent affine-SGD comparator fails for Adam.

**Setup.** As in node [N10](N10_update_composition.md)'s C23 section, with Adam ($\beta=(0.9,0.999)$, $\epsilon=10^{-8}$, bias correction), $\eta\in\lbrace 0.01,0.1\rbrace $ and $R\in\lbrace 1,4,40\rbrace $, over 7 task laws.

**Result.** Source: `theory/independent_checks/results_independent_v014.json → controlled_neural.details.observed_affine_comparison_cohorts`.

| Cohort | Configurations | Agreements | Departures | Largest deviation |
|---|---|---|---|---|
| Adam, all declared settings | 42 | 10 | 32 | 0.1196 (moment $N=3$, $\eta=0.1$, $R=4$) |

**What this shows.** For Adam, the affine-SGD prediction misses the gap in most settings. E0–E2 show that the missed part is the cold-start quotient of A2 and the momentum memory of A4.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V2,V3,V4,E0,E1,E2,E3 --output theory/v015_checks/quick/n11.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python docs/direction/v015_revision/verify_v015_extensions.py   # package T01–T13
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: groups V2, V3, V4, E0, E1, E2, E3.
- `theory/v015_checks/c23_bundle.json`: exact laws, network, optimizer code and per-history raw steps.
- `theory/v015_checks/figs/E2.png`: the original frozen-Jacobian panel.
- `docs/direction/v015_revision/results_v015.json`: checks T03–T08.
- `theory/independent_checks/results_independent_v014.json`: `controlled_neural`.
