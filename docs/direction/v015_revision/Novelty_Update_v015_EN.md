# Targeted Novelty Update v0.15: Optimizer-Relative Feedback and Few-Shot Coverage

**Date:** 10 October 2026, UTC.  
**Status:** Independent model-agent literature update, not external peer review. This note takes the reported independent v0.14 validation and C23 tanh-network result as supplied; it does not claim to have reproduced them. It supplements the previous audit without replacing or modifying it.

## 1. Assessment

The reported Adam discrepancy supports the framework's central requirement to derive information sufficiency from the **actual learner update and continuation**. It rejects an unqualified extension of affine-SGD predictions to Adam; it does not reject the general continuation-response characterization.

The immediate mechanism is already known. Adam's cold first parameter step approaches sign descent, and a 2025 paper explicitly derives this behavior. Distributional sign bias, persistent optimizer memory, passive demonstration coverage, and gains from active imitation all have close precedents. These should be supporting facts and boundary checks.

The stronger research opportunity remains a unified characterization connecting **accessible environmental distinctions, optimizer-state-dependent response classes, finite-information approximation, and verifiable task consequences**. Adam is a useful stress test within that theory. Neither a first-step identity nor a weighted missing-mass formula alone establishes a new central contribution.

## 2. Exact mathematical boundaries for the optimizer extension

The following are direct deductions from the stated optimizer, not priority claims.

### 2.1 Cold Adam: parameter output versus complete next state

For standard bias-corrected Adam, with \(m_0=v_0=0\), step counter zero, nonzero learning rate \(\eta\), and no additional update transformation,
\[
\theta_1=\theta_0-\eta h_\epsilon(G),\qquad
[h_\epsilon(g)]_i=\frac{g_i}{|g_i|+\epsilon}.
\]
This follows immediately from Algorithm 1 of Kingma and Ba [A1].

For \(\epsilon=0\), restrict to nonzero gradient coordinates or explicitly define the zero-gradient convention. The parameter output then depends on the **joint sign pattern**. Equal coordinatewise sign probabilities determine the mean parameter update, but need not determine a nonlinear deployment utility. Equality of the joint sign law is sufficient for every immediate deployment test depending only on \(\theta_1\), and necessary if the test family separates all reachable sign patterns. A restricted task family can require fewer statistics.

For \(\epsilon>0\), each coordinate map is injective:
\[
h_\epsilon^{-1}(u)=\frac{\epsilon u}{1-|u|},\qquad |u|<1.
\]
Consequently preserving the full transformed-gradient law is equivalent to preserving the full gradient law. This does **not** imply that a fixed task/continuation family requires that whole law. Nor does invertibility imply stable recovery from noisy measurements: the inverse is poorly conditioned near \(|u|=1\).

The approximation \(h_\epsilon(g)\approx\operatorname{sign}(g)\) must be checked against gradient scale. For \(|g_i|\ge\gamma>0\),
\[
|h_\epsilon(g_i)-\operatorname{sign}(g_i)|
\le\frac{\epsilon}{\gamma+\epsilon}.
\]
Small gradients can violate a sign approximation even when \(\epsilon\) is numerically small. Weight decay, clipping, accumulation, loss normalization, and epsilon placement must be included in the declared recipe.

Most importantly, the **complete next state** also contains
\[
m_1=(1-\beta_1)G,\qquad v_1=(1-\beta_2)G^{\odot2}.
\]
Thus at \(\epsilon=0\), matching joint signs preserves the immediate parameter distribution but generally does not preserve the optimizer-state distribution. A nonempty reference-learning tail can use the stored magnitudes. The first-step sign quotient is therefore not a multistep Adam sufficiency theorem.

### 2.2 Persistent state: an elementary separating example

Consider one-dimensional cold Adam with \(\epsilon=0\). First gradients \(a>0\) and \(b>0\) produce the same \(\theta_1=\theta_0-\eta\), but different \(m_1,v_1\). Apply the same positive second gradient \(c\). The second normalized update is
\[
\frac{(\beta_1a+c)/(1+\beta_1)}
{\sqrt{(\beta_2a^2+c^2)/(1+\beta_2)}},
\]
with \(a\) replaced by \(b\) in the other branch. These are generally different despite identical gradient-sign histories. Linear per-message losses realize such gradients, and a bounded deployment test can separate the resulting parameters.

This is a direct optimizer-state counterexample, not a new phenomenon in memory-based optimization. It establishes the correct scope of a sign-only interface. Special recipes can have additional invariances; no universal claim that every magnitude difference matters is warranted.

### 2.3 The unified response remains the right object

For optimizer \(o\), complete current state \(z\), and declared suffix,
\[
\psi_{o,t,z}(y)
=\mathbb E\!\left[V_{t+1}^{0,o}\bigl(U^o_t(z,y,\xi)\bigr)\right].
\]
The required tests are the family of these functions, not a universal bank of raw-gradient moments, signs, or parameter deltas. Equality of conditional full next-state kernels is a strong sufficient condition; equality against the actual continuation-response class can be strictly weaker.

Matching features averaged over a checkpoint bank does not automatically match them at subsequently visited states. Nor does matching the marginal distribution of optimizer moments establish the correct joint law with parameters and feedback.

## 3. Closest optimization literature

| Primary work | Exact relevance | Boundary for the proposed contribution |
|---|---|---|
| **Adam** [A1], ICLR 2015 | Defines first/second moment state, bias correction, and normalized parameter update | The cold-step formula and magnitude-bearing memory are immediate consequences of the original algorithm |
| **Dissecting Adam** [A2], ICML 2018 | Separates signed direction from a relative-variance magnitude factor; analyzes stochastic sign probabilities and expected one-step improvement | “Adam uses distributional information beyond the mean” and “sign and magnitude play different roles” are established |
| **signSGD** [A3], ICML 2018 | Sign-based updates and distributed majority vote; theory explicitly discusses noise shape and pathological skewed distributions | A mean gradient does not generally determine sign behavior; joint task-sensitive sign statistics remain an application of response-function matching |
| **Revisiting the Initial Steps** [A4], CPAL 2025 | Section 2.1 derives cold Adam's first step as sign descent after ignoring epsilon; studies nonzero second-moment initialization | This is direct prior art for the proposed explanation of C23 |
| **Error Feedback Fixes SignSGD** [A5], ICML 2019 | Gives failures of naive sign compression and restores useful information through a persistent residual state | Memory-dependent information sufficiency is not new; sign-only transmission differs from sign-only complete learner state |
| **Distributed Training with Heterogeneous Data** [A6], NeurIPS 2020 | Studies median/mean distinctions underlying sign-majority and median-based updates | Noise distribution and aggregation matter beyond a shared population mean |

These papers study optimizer behavior, convergence, or communication. The narrower proposed contribution concerns which **environment feedback** must be retained for a declared downstream learner/task family, and how that requirement can be acquired and certified. This distinction is legitimate but does not alone establish novelty: the operational theorem must yield a substantive new conclusion.

For empirical controls, include SGD, a sign-update baseline with an explicit zero convention, cold Adam, and Adam with retained state. A momentum or error-feedback condition is useful if memory is part of the claimed mechanism. Do not equate all of these algorithms merely because their first parameter step sometimes agrees.

## 4. Demonstration coverage: the exact formula and its limits

Suppose independent fair hidden bits \(B_j\) determine the correct decision in contexts \(j\), with nonnegative deployment weights \(w_j\). Each independent demonstration reveals bit \(j\) with probability \(q_j\), noiselessly, and the reveal mechanism is independent of the hidden labels. Observations about other bits provide no information about an unrevealed bit. Then after \(K\) demonstrations the Bayes-optimal expected weighted error is
\[
R_K^\star=\frac12\sum_j w_j(1-q_j)^K.
\]

The proof is a weighted missing-mass calculation: a bit remains unseen with probability \((1-q_j)^K\); conditional on being unseen it still requires a fair guess. When \(\sum_jw_j=1\), this is an expected classification error. Demonstrations may reveal several bits; independence across demonstrations for each reveal event is sufficient.

This formula is **not** an arbitrary neural student's achieved training risk. It is an information-limited Bayes floor for the stated model. The actual learner can incur additional approximation or optimization error. Correlated hidden facts, informative non-revelation, noisy labels, dependent demonstrations, and pretrained information about the bits change the formula.

The missing-mass mechanism is already central to imitation-learning bounds. Rajaraman et al. [I1] use unseen-state mass in both behavior-cloning upper bounds and lower bounds; their lower-bound construction includes the factor \(1-1/|\mathcal A|\), which becomes \(1/2\) for two actions. The proposed expression is an instructive exact specialization, not a new coverage law.

With a noiseless direct query that reveals one freely chosen unseen bit, the immediate reduction in Bayes error is \(w_j/2\). Querying the largest-weight unseen bits is optimal in this independent-bit model. “Adaptive” here can mean adapting to which bits demonstrations happened to reveal; label-dependent adaptivity adds nothing when labels are independent fair bits and costs are fixed. Richer adaptive-query claims require richer information structure or reachability constraints.

The relevant active-imitation precedents are:

- **Judah et al. [I2], JMLR 2014:** reductions from active imitation to IID active learning and conditions for label savings over passive learning.
- **Zhang et al. [I3], 2026 preprint:** offline coverage barriers under student misspecification, including horizon one with a perfect expert and an optimal policy in the student class; online guarantees under additional structure.
- **Watch, Try, Learn [I4], ICLR 2020:** few demonstrations combined with trial-and-error feedback to resolve task ambiguity and unobserved dynamics.
- **Active Coverage for PAC RL [I5], COLT 2023:** lower bounds and algorithms when obtaining coverage itself requires controlled interaction.

Therefore, neither “few demonstrations miss important information” nor “active queries can outperform more passive demonstrations” is available as a broad novelty claim.

## 5. Keep three coverage requirements separate

1. **Environment-information coverage:** whether demonstrations/queries reveal distinctions required to choose a useful policy. The missing-mass example addresses this.
2. **Learner-state coverage:** whether the conditional response and feedback distributions at actual training states are controlled by audited cells. This is the missing premise reported for v0.14 Theorem 19.
3. **Task/recipe coverage:** whether an audited response class approximates the new task, optimizer, or suffix under evaluation.

A bound on the first does not supply either of the others. A finite checkpoint bank alone supplies none of the needed extrapolation guarantees.

The Theorem 19 repair should explicitly provide a map from visited states to audited cells and a valid transfer bound for their conditional features/responses, or restrict the conclusion to actually audited states. Any feature-mean transport errors must enter the final discrepancy bound in addition to sampling error and approximation residual. A stopped-coverage bound requires a justified exit probability. Actual-state audits can avoid this extrapolation but retain the cost and restoration assumptions of complete update-plus-tail replicas. This is a scope repair, not a new coverage theorem.

## 6. What could make the joint contribution substantive?

A promising theorem target is an **optimizer-relative acquisition and compilation characterization** for an independently motivated family of valid environments and learners:

- Derive a state-conditional response representation for several update rules, including a memory-bearing normalized update, and prove both sufficiency and realizable necessity.
- Quantify how the attainable approximation changes with epsilon, gradient scale, persistent state, and the continuation horizon. A useful result must go beyond the one-line invertibility or sign identity.
- Link the representation to permitted demonstration/query channels, so the same model yields an unavoidable acquisition error and an achievable acquisition-plus-training guarantee.
- Include noisy measurement and valid visited-state coverage in the achievable result. Information dimension, coefficient stability, query reachability, and training-replica cost must remain distinct.

This is a target, not a claimed completed theorem. A direct sum of the Adam identity, missing-mass formula, and an existing concentration bound would still be insufficient. The contribution would be a new operational dependency or tight tradeoff that emerges only after acquisition and the declared learner dynamics are considered together.

A discriminating experiment can test the joint chain while keeping the theory general:

| Intervention | Held fixed | Predicted distinction |
|---|---|---|
| Feedback laws with matched raw low-order moments but different sign/normalized-update laws | Full initial state, task, message budget | Affine-SGD statistics can fail to predict a cold normalized update |
| Same joint sign law, different magnitudes | Zero-epsilon cold first step | Same immediate parameter law; potentially different retained optimizer state |
| Retain versus reset optimizer state before a common tail | First-step parameter values and future feedback protocol | Magnitude information can become relevant only through the stateful tail |
| Sweep epsilon relative to gradient scale | Task and valid feedback laws | Exact sign equivalence gives way to scale-dependent approximation |
| Vary \(K\), reveal probabilities, and task weights | Hidden-fact family and available query interface | The passive information floor follows weighted coverage; training can add excess error |
| Allocate queries by occupancy/uncertainty versus actual learner consequence | Total interaction and training budget | Optimizer-relative usefulness may differ from information or state coverage alone |

The most informative method result would show that a **state-conditional response criterion** selects better feedback across heldout tasks and optimizer states than raw moment fidelity, sign-only metrics, ordinary coverage/uncertainty, and direct rollout-cost criteria. Evaluate genuine trained agents and include the cost of learning/calibrating the criterion. The effect may arise from provider choice, query placement, packet content, or coupling; no one branch defines the general framework.

## 7. Novelty ledger and review limits

| Candidate statement | Judgment | Confidence |
|---|---|---|
| Cold Adam is sign-like; epsilon restores magnitude dependence | Established algorithmic fact; explicit first-step predecessor found | High |
| Joint signs can matter beyond coordinate marginals | Standard nonlinear-response fact specialized to a sign update | High |
| Persistent optimizer state invalidates general sign-only multistep sufficiency | Direct consequence of optimizer memory; supporting separation | High |
| Weighted few-shot missing-mass Bayes floor | Classical coverage specialization with close IL lower-bound precedent | High |
| A unified optimizer-relative environment-information characterization | Potential composition-level contribution; exact novelty not established | Medium-low |
| A cheap response-conditioned query/compilation method transfers across tasks and optimizer states | Unvalidated method hypothesis; requires direct comparisons | Undetermined |

Primary fulltext sections were inspected for Adam, Dissecting Adam, signSGD, Revisiting the Initial Steps, Error Feedback, Fundamental Limits of IL, Active Imitation Learning, and the 2026 online/offline IL paper. The median/mean paper, Watch/Try/Learn, and Active Coverage were assessed from primary records/abstracts. This is a targeted update, not a re-review of the previous 31-source register, and it supplies no global first-ever claim.

## References

- **[A1]** Diederik P. Kingma and Jimmy Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. [Primary paper](https://arxiv.org/pdf/1412.6980), Algorithm 1.
- **[A2]** Lukas Balles and Philipp Hennig. *Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients.* ICML 2018, PMLR 80:404–413. [Official record](https://proceedings.mlr.press/v80/balles18a.html), [paper](https://proceedings.mlr.press/v80/balles18a/balles18a.pdf).
- **[A3]** Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. *signSGD: Compressed Optimisation for Non-Convex Problems.* ICML 2018, PMLR 80:560–569. [Official record](https://proceedings.mlr.press/v80/bernstein18a.html), [paper](https://proceedings.mlr.press/v80/bernstein18a/bernstein18a.pdf).
- **[A4]** Abulikemu Abuduweili and Changliu Liu. *Revisiting the Initial Steps in Adaptive Gradient Descent Optimization.* CPAL 2025, as stated in the primary record; arXiv:2412.02153v2. [Record](https://arxiv.org/abs/2412.02153), [full text](https://arxiv.org/html/2412.02153v2), Section 2.1.
- **[A5]** Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian U. Stich, and Martin Jaggi. *Error Feedback Fixes SignSGD and other Gradient Compression Schemes.* ICML 2019, PMLR 97:3252–3261. [Official record](https://proceedings.mlr.press/v97/karimireddy19a.html), [paper](https://proceedings.mlr.press/v97/karimireddy19a/karimireddy19a.pdf).
- **[A6]** Xiangyi Chen, Tiancong Chen, Haoran Sun, Steven Z. Wu, and Mingyi Hong. *Distributed Training with Heterogeneous Data: Bridging Median- and Mean-Based Algorithms.* NeurIPS 2020. [Official record](https://papers.nips.cc/paper_files/paper/2020/hash/f629ed9325990b10543ab5946c1362fb-Abstract.html).
- **[I1]** Nived Rajaraman, Lin F. Yang, Jiantao Jiao, and Kannan Ramachandran. *Toward the Fundamental Limits of Imitation Learning.* arXiv:2009.05990, 2020. [Record](https://arxiv.org/abs/2009.05990), [paper](https://arxiv.org/pdf/2009.05990), Appendix A.1 and the lower-bound argument around Eq. (58).
- **[I2]** Kshitij Judah, Alan P. Fern, Thomas G. Dietterich, and Prasad Tadepalli. *Active Imitation Learning: Formal and Practical Reductions to I.I.D. Learning.* JMLR 15(120):4105–4143, 2014. [Official record](https://www.jmlr.org/papers/v15/judah14a.html), [paper](https://www.jmlr.org/papers/volume15/judah14a/judah14a.pdf).
- **[I3]** Huaqing Zhang, Jingchu Gai, Juno Kim, Bingbin Liu, and Andrej Risteski. *When Does Online Imitation Learning Help in LLM Post-Training? The Role of (Non-)Realizability Beyond Horizon.* arXiv:2606.30445, June 2026, preprint. [Record](https://arxiv.org/abs/2606.30445), [full text](https://arxiv.org/html/2606.30445v1).
- **[I4]** Allan Zhou et al. *Watch, Try, Learn: Meta-Learning from Demonstrations and Rewards.* ICLR 2020. [Official conference page](https://iclr.github.io/build/virtual/poster_SJg5J6NtDr.html), [primary record](https://arxiv.org/abs/1906.03352).
- **[I5]** Aymen Al-Marjani, Andrea Tirinzoni, and Emilie Kaufmann. *Active Coverage for PAC Reinforcement Learning.* COLT 2023, PMLR 195:5044–5109. [Official record](https://proceedings.mlr.press/v195/al-marjani23a.html).

