# Independent Novelty Audit: Environment Information and Few-Shot Agent Distillation

**Audit date:** 10 October 2026, UTC.  
**Object reviewed:** Environment_Information_Theory_v013_ZH.md, followed by a scope-consistency pass on v014/Environment_Information_Theory_EN.md; the continuation-response characterization and Consequence-Calibrated Experience Distillation (CED), including its fixed-marginal coupling branch.  
**Review status:** Independent model-agent literature audit. This is not external peer review, a priority certificate, or an assertion that all relevant work has been found. Mathematical correctness is addressed separately in the independent proof audit. No neural-training results were produced by this review.

## 1. Assessment

The framework has a useful organizing question: **which environment-derived information matters for the actual learning procedure and its eventual task performance?** The principal candidate contribution is a unified operational characterization through the learner's continuation-response test space, connecting necessity, sufficiency, approximation, identification, verification, and methods. Its building blocks include classical decision theory, learner-state control, weak approximation, polynomial moments, and statistical selection. Component-level precedent does not rule out an original unifying contribution; nor does putting these components in one document establish that contribution. The integrated claim needs its own comparison and substantive theorem, assessed in Section 2.1.

In particular, the finite moment hierarchy, polynomial degree propagation, and first-unmatched-cumulant rate should be presented as **useful corollaries for a restricted model**, not as an award-level theoretical core. The general learning-kernel identity is a simulation/telescoping identity on an augmented state space. A Taylor expansion of its continuation function closely follows established weak-approximation arguments.

The broad CED proposal also occupies a crowded area. Previous work already optimizes teaching data through downstream student updates, directly optimizes environmental return after behavior-cloning training, predicts teacher quality from student gradients, and gates teacher interventions by future student rollouts. An actual-update experiment is an excellent measurement design; it is not by itself a new learning principle.

The **fixed-marginal, consequence-optimized coupling** proposal is one possible method branch, not a replacement for the general theory. Its empirical question is whether joint arrangements of complete, valid experiences chosen by calibrated post-update task effect improve over antithetic sampling, gradient balancing, meta-reweighting, and rollout-based teaching. **Fixed-marginal coupling itself is already established.** No exact agent-distillation implementation of the full proposed design was verified in this bounded search, but that is low-confidence negative evidence, not novelty clearance. The theory should retain arbitrary environment-information providers; a world model is only one implementation.

There is currently no defensible basis for declaring “ICML best-paper-level novelty achieved.” A substantial integrated theorem can use established tools. What is still needed is a nontrivial operational consequence of the unified characterization, with an attainable method or compelling empirical implication that survives the closest framework comparisons below.

## 2. Claim-by-claim originality ledger

“Confidence” below concerns the overlap judgment, not the probability of paper acceptance.

| Framework claim | Closest prior principle or work | Appropriate originality label | Confidence |
|---|---|---|---|
| Representation, information, and algorithm gaps add exactly | Bayes decision-risk decompositions | Definition-level decomposition; useful bookkeeping | High |
| No information regret iff posterior-supported environments share an optimal decision; information cannot increase under garbling | Classical statistical decision theory and data processing | Classical foundation; state the measurable-selection/attainment assumptions | High |
| Passive traces from a privileged optimal teacher may not identify the best restricted student | Identifiability and asymmetric imitation; Robust Asymmetric Learning [R1] | Explicit separation example; not a new general impossibility principle | High on principle; medium on exact example priority |
| Active queries resolve the two-environment ambiguity with a KL-controlled sample rate | Sequential testing and change-of-measure lower bounds | Specialized application of standard testing tools | High |
| Equal static policy values need not imply equal downstream learning outcomes | Algorithm-dependent teaching; policy-aware learning [R2–R4] | Useful separation of two equivalence relations; do not claim a refutation of Bellman value equivalence | High |
| Augment state with parameters, optimizer, buffers, and history; compare learning kernels | Teaching MDPs and sequential optimal control [R2–R3] | Standard Markovization applied carefully to distillation | High |
| Exact continuation-value telescoping and drift/covariance expansion | Stochastic modified equations, especially one-step moment conditions and terminal-test weak error [R5] | Classical identity and weak-approximation corollary | High |
| Degree-\(N\) deployment return depends on update-block moments through degree \(N\) under affine averaging | Expectations of polynomials; moment matching [R6–R7] | Exact algebraic characterization in a restricted family | High |
| First unmatched order-\(r\) cumulant gives \(n^{-(r-1)}\) leading discrepancy | Cumulants of sample means and polynomial delta expansions | Useful finite-sample/asymptotic corollary, with a potentially vanishing leading contraction | High |
| Polynomial updates amplify the necessary moment degree | Composition of polynomial degrees | Algebraic corollary; an explicit sharp example may improve exposition | High |
| Minimal expected linear statistics equal a test-function span dimension; unrestricted tests require distributional fidelity | Linear algebra, integral probability metrics; value-equivalence span results [R4] | Classical characterization specialized to the learning response class | High |
| A frozen finite candidate panel admits a uniform empirical selection guarantee | Hoeffding concentration and a union bound | Standard calibration certificate; no automatic unseen-task guarantee | High |
| Equal sample marginals can yield different update outcomes because joint coupling changes | Antithetic SGD and learned antithetic sampling [R19–R20] | Known mechanism; agent-specific magnitude and usefulness remain experimental questions | High |
| A cheap calibrated signed post-update task predictor can choose useful packets across tasks/recipes | MetaOPD, GRACE, behavior distillation, and sequential teaching [R2, R8–R13] | Unvalidated method hypothesis; novelty depends on the exact implementation and incremental evidence | Medium on overlap; low on absence of an exact match |

Two distinctions must remain explicit. Raw joint moments of a **complete update block** are not identical to the reward/Q-function moments matched in imitation-learning papers. That difference prevents an incorrect claim of identical theorems, but does not make the general moment-matching principle new. Likewise, preserving values for a restricted policy class is weaker than the relevant Bellman or learning-response equivalence: a counterexample to the former cannot be advertised as a counterexample to the latter.

### 2.1 Composition-level originality: the continuation-response chain

The strongest proposed statement is: **for a declared learner/task family, the environment information that must be preserved is exactly the family of expectations tested by all admissible learning continuations; a corresponding approximation width characterizes lossy preservation, while acquisition and verification determine whether that information can actually be used.** This is a better candidate for the paper's central contribution than a list of unrelated local propositions. The judgment below does not require every lemma to be new.

The closest complete-framework neighbors are:

| Framework neighbor | Existing chain | Remaining distinction to establish |
|---|---|---|
| Value equivalence [R4] | Task-relative equivalence, policy/value test classes, span closure, and model-learning implications | Tests induced by actual learner updates and admissible training continuations rather than a declared planning/Bellman class; prove what is added beyond applying value equivalence to a learner-state MDP |
| Value-equivalent sampling and rate-distortion learning [R28–R29] | Environment-information compression, approximation/capacity tradeoff, learning targets, an algorithm, and regret guarantees | A fixed learner/optimizer and feedback-access-specific operational characterization, with matching identification/verification costs rather than only a newly named distortion |
| Statistical experiment deficiency and its bottleneck realization [R30] | Information ordering, approximation, operational decision-risk loss, and learned compression | The constrained family of sequential learning responses, feasible experience laws, and trainability constraints must change the answer in a demonstrable way |
| Algorithmic dataset-distillation guarantees [R16] | Finite test losses, sufficient control, a lower bound, and supervised/offline-RL distillation algorithms | A broader continuation-dependent response family with a nontrivial computable width or sample complexity, not just an unrestricted function-space restatement |
| Sequential teaching plus weak approximation [R2–R3, R5] | Learner evolution, teaching decisions, terminal utility, and propagation of local update errors | Tight information requirements and an observable certificate that these control/dynamics analyses do not already supply |
| Calibrated value-aware learning [R31] | A family of surrogate objectives, inconsistency results, corrected losses, and experiments | Distinguish statistical confidence calibration from surrogate-loss calibration; derive a learner-consequence-specific failure/correction beyond relabeling existing ones |

These neighbors prevent a broad claim that “previous work studied isolated pieces, whereas ours first connects useful information to learning and algorithms.” Some already connect several stages. Nevertheless, none of the inspected sources was verified to contain the entire proposed continuation-response characterization under the same declared learner, access, and feasible-feedback assumptions. **Confidence: high that the constituent connections have substantial precedent; medium-low that the precise integrated formulation is distinct; insufficient evidence to call its eventual central theorem novel.**

The following would make the unification substantive:

1. **A genuinely operational exact result.** Specify an admissible environment/feedback family and learner family independently of the desired answer. Prove both preservation and a realizable distinguishing task/learner whenever the required information is absent. Defining the test class to contain every possible final utility makes equivalence immediate; the contribution must identify a smaller constructive class or an unexpected structural characterization.
2. **A tight approximate result in the same model.** Connect an information width to matching achievable error and unavoidable loss using the same feasible experience laws and access constraints. A uniform approximation/annihilator duality is a useful tool, but is generally classical functional analysis. Its witnesses must be realizable environment-induced distributions. Arbitrary signed-measure witnesses need not be valid episodes.
3. **A nontrivial composition result.** Show how the operational information requirement changes with training horizon, deployment horizon, memory, optimizer state, or family size, including a matching lower construction within the admissible class. Polynomial degree multiplication alone is not yet such a general result.
4. **An acquisition/verification bridge.** Characterize when the needed response tests can be identified from allowed feedback and certified with finite data. The same low-dimensional response representation should drive the method and its sample bound. Structural dimension, feature conditioning, query accessibility, and sampling variance are different quantities; a low approximation width alone does not imply a cheap certificate.
5. **A consequence that was unavailable without the chain.** For example, a matched upper/lower information budget for an agent family, a provably unavoidable passive/active gap under the actual teacher interface, or a selection method whose attainable budget is governed by the new characterization and whose distinctive prediction is observed in neural training.

These are standards for evaluating an integrated contribution, not a demand that every ingredient or proof technique be new. The framework need not collapse to any one concrete implementation, including coupling, synthetic data, or world models. The practical branches should be consequences of the general object, and their failure should not be hidden by changing the theorem's scope.

### 2.2 Assessment of the completed v0.14 finite chain

The v0.14 manuscript strengthens the earlier formulation. Its central object now remains the same through one-boundary operational equivalence, exact feature sufficiency, approximate-width duality, noisy-feature stability, training composition, and auditing. Finite monomial and parity tasks provide realizable necessity witnesses; the grid count and broad-task barrier distinguish exact dimension from useful approximate complexity. The noisy-feature identity makes the approximation–coefficient-stability tradeoff explicit. Actual-state auditing addresses a specific checkpoint-coverage gap while retaining the cost of independent full-tail replicas.

These additions justify a stronger description than “a collection of separate observations”: **a proved finite operational characterization, with explicit specializations and verification limits, is now a reasonable description of the manuscript's mathematical organization.** This scope pass is not a replacement for the independent proof audit. It also does not establish the historical novelty of the whole characterization. The LP duality, dimension counting, parity independence, telescoping, and concentration remain credited tools; no claim to invent them is needed.

The remaining novelty question is whether the same chain yields a substantial conclusion about an independently motivated learner/task/access family beyond what follows immediately from value equivalence on the learner-state process, statistical-experiment comparison, and established distillation guarantees. A useful structured response theorem, an attainable identification/verification bound, or a distinct empirical prediction can answer that question. Success of fixed-marginal coupling is not a prerequisite.

## 3. Closest work that changes the proposed contribution

The following summaries are deliberately short. The important part is the implication for the proposed claim.

**Teaching as learner-state control — [R2–R3].** Active sequential machine teaching already treats the learner's evolving information/history as the state of a teaching problem. Sequential optimal-control teaching explicitly optimizes the sequence presented to a known learner. Therefore, “evaluate information by how the learner changes” and “include the full learner state” require attribution. A new contribution would need an identifiable, efficiently measurable response class or a new access-limited agent result, rather than the control formulation alone.

**Weak approximation of learning dynamics — [R5].** Li, Tai, and E use conditional update-moment conditions to control expectations of terminal test functions over many stochastic-gradient steps. Their argument uses continuation functions, telescoping, and Taylor expansion. This is a particularly close mathematical predecessor to the general learning-kernel and drift/covariance analysis. The framework's emphasis on environmental task utility may be useful, but substituting that terminal function does not create a new proof technique.

**Behavior distillation and offline behavior distillation — [R8–R10].** HaDES searches small synthetic state-action datasets using environmental return after supervised policy training; it also studies transfer across architectures and hyperparameters. Offline Behavior Distillation compares the policies produced by distilled data and includes action-value weighting. Single-batch RL distillation is another direct few-example precedent. Thus neither “few examples train an agent” nor “optimize post-training return instead of token fidelity” is available as a broad novelty claim.

**Meta-gradient weighting and MetaOPD — [R11–R12].** Meta-reweighting predates current agent distillation. MetaOPD, submitted on 8 October 2026, learns token weights through a virtual student update and reference-solution validation loss. Its detailed implementation includes the current AdamW state, scheduled learning rate, and clipping, and its checks distinguish weight learning from independent post-update evaluation. CED should distinguish its closed-loop environmental target and packet action space, not claim that update-effect weighting or optimizer-state fidelity was absent.

**FutureBridge-OPD — [R13].** This method branches from a common environment history, inserts teacher guidance or retains the student's response, and lets a frozen student continue. A future teacher-preference score gates the intervention. Its immediate object is a rollout intervention, rather than a parameter update. Nevertheless, it is a direct comparator for branching feedback packets, student continuations, and rejection of locally attractive teacher guidance.

**GRACE and LESS — [R14–R15].** GRACE scores teacher suitability using distributional properties of student gradients, without teacher internals/logits or test data. LESS uses optimizer-aware gradient similarity for targeted data selection. CED's learned response predictor must show incremental prediction and selection value over these signals under the same available information. A more expensive outcome predictor cannot claim practical superiority solely from higher correlation.

**Guarantees for dataset distillation — [R16].** Gupta, Saket, and Raghuveer connect losses over test regressors to dataset distillation, with a near-quadratic-in-dimension characterization for bounded linear regression and Bellman-loss variants for offline RL. This is important prior work for “a finite collection of tests certifies useful distilled data.” The finite candidate union bound in the present framework is a different, substantially more elementary certificate.

**Cost-sensitive/student-continuation imitation — [R17–R18].** LOLS and AggreVaTe use rollout costs or costs-to-go to select useful training decisions; LOLS explicitly incorporates learner continuation. These methods establish that teacher imitation error alone need not be the correct target. CED must test what is gained by measuring a training update's effect beyond measuring an action's downstream cost.

**Improvement fidelity in adaptive worlds — [R26].** A recent preprint studies whether proposed improvements remain improvements under a more faithful, policy-responsive environment, with paired decision-oriented selection. Its environment-response setting differs from a fixed environment and learner update. It still makes broad claims such as “first to rank by signed improvement rather than fidelity” unsafe.

Taken together, these papers answer the request for close precedents: **actual post-training environmental evaluation already exists; meta-update validation and out-of-domain distillation tests already exist; future agent-rollout gating already exists.** The exact combination of real post-parameter-update agent utility, learned packet-coupling selection, heldout calibration, and heldout task-family transfer was not verified as a single existing implementation.

## 4. Fixed-marginal coupling stress test

### 4.1 The precise candidate

Let \(P_1,\ldots,P_b\) be prescribed distributions over complete valid experience units, and let
\[
\mathcal C(P_1,\ldots,P_b)
=
\{\Gamma:\Gamma_j=P_j\text{ for each slot }j\}.
\]
At a frozen complete learner state \(z\), \(\Gamma\) controls only the joint packet \(B=(Y_1,\ldots,Y_b)\). Each \(Y_j\) must itself remain a valid whole episode or declared valid micro-update input. The method may change pairing or ordering but must not splice mutually inconsistent observations and actions.

Define the actual continuation utility
\[
c_z(B)
=
\mathbb E\!\left[J_E\!\left(\operatorname{Deploy}
\left(\operatorname{Tail}(U(z,B))\right)\right)\right],
\]
where the optimizer, loss normalization, stochasticity, and reference training tail are fully specified. A candidate coupling is selected by signed improvement over a declared \(\Gamma_0\), using proposal data to learn a cheap predictor and disjoint actual-update trials to calibrate a frozen finite set of candidates.

This yields a useful **experimental treatment**: all slot marginals can be identical while the joint experience law changes. It removes simple marginal data-quality explanations. It does not establish that ordinary sample weighting is globally powerless: reweighting can alter other aspects of training, and a sufficiently general method can represent pairs as its examples.

### 4.2 Direct overlap: coupling is not the new ingredient

Liu and Xu [R19] explicitly analyze two-example minibatches with identical marginals, changed dependence, preserved mean gradient, and changed covariance. Ren, Zhao, and Ermon [R20] learn antithetic samplers while retaining unbiased estimation. Thus the proposal's exact “hold marginals fixed, optimize dependence” lever has a direct predecessor.

Diversified mini-batches [R21] and repulsive point processes [R22] provide further joint-selection baselines. GraB [R23] optimizes sample ordering through gradient balancing. PartitionSel [R24] optimizes a joint validation-gradient utility under per-domain budgets; domain budgets are not exact per-example slot marginals, but it is a relevant structured-batch competitor. Exposure-matched curriculum comparisons [R25] already separate ordering from aggregate exposure.

An additional high-priority unresolved source is **Same Sample or Fresh? The Coupling Geometry of Stochastic Extragradient**, posted on OpenReview on 4 October 2026 [R27]. Indexed primary-source text discusses coupling-polytope choices and cross-covariance geometry. The full PDF was blocked by a browser-verification challenge. Its title/setting does not establish an exact agent-distillation overlap, but the generic mathematical coupling claim must remain unclaimed until its full theorem is checked.

### 4.3 What the elementary derivation does and does not establish

For a single common-snapshot averaged SGD update, let
\[
G_\Gamma=\frac{1}{b}\sum_{j=1}^{b}g(z,Y_j),
\qquad
\theta^+=\theta-\eta G_\Gamma.
\]
Fixed slot marginals imply a common mean \(\mu\). For a \(C^3\) continuation utility \(V\), Taylor's theorem gives
\[
\mathbb E_\Gamma V(\theta-\eta G_\Gamma)
-\mathbb E_{\Gamma_0}V(\theta-\eta G_{\Gamma_0})
=
\frac{\eta^2}{2}
\left\langle
\nabla^2 V(\theta),
\operatorname{Cov}(G_\Gamma)
-
\operatorname{Cov}(G_{\Gamma_0})
\right\rangle
+
O(\eta^3).
\]
A uniform remainder requires explicit derivative and third-moment bounds. This is a Taylor corollary, not a new theorem family.

It explains why minimizing the scalar gradient variance is not universally equivalent to maximizing downstream utility. The relevant contraction uses the curvature of **deployment continuation utility**, whose directions and signs may differ from those of training loss. This observation is mathematically straightforward. Its research value would come from predicting consequential ranking reversals in actual few-shot agent training and selecting useful couplings with small measurement cost.

For **two ordered micro-updates**, equal input marginals generally do not preserve the mean total update. The second gradient is evaluated at the state changed by the first update; optimizer state also changes. Such experiments test update interactions and ordering, not a clean covariance-only intervention. The write-up must not quietly transfer the averaged-SGD identity to AdamW, clipping, sequential updates, or changing normalization.

Finally, a finite transportation objective
\[
\max_{\Gamma\in\mathcal C(P_1,P_2)}
\sum_{i,j}\Gamma_{ij}\,\widehat c_z(i,j)
\]
is an ordinary optimal-transport/linear-programming formulation once costs are supplied. The hard question is how to estimate useful **training-consequence costs** cheaply and transferably. The familiar \(2\epsilon\) selection-regret result from uniformly \(\epsilon\)-accurate costs is a corollary, not a central contribution.

### 4.4 The narrow claim worth attempting

A defensible proposed claim is:

> In a defined few-shot agent-distillation regime, preserving all slot marginals while optimizing complete-experience coupling by calibrated post-update task consequences improves deployment performance at an equal total training-and-evaluation budget; the gains and coupling rankings are not explained by generic gradient variance reduction, marginal teacher quality, or action-level continuation scores.

This is deliberately a falsifiable empirical/method claim. It is not yet established. Its mathematical supporting statements can be correct and useful even if the empirical claim fails.

A second possible contribution is a **controlled learner-consequence benchmark**: a public panel of paired packet interventions with exact full-state restoration, fixed marginals, known toy mechanisms, neural outcomes, and heldout task/recipe splits. Such an artifact would help distinguish source fidelity, action-level usefulness, and learning usefulness. Its contribution would be measurement quality and reusable evidence, not a new universal information measure.

## 5. Required comparisons and ablations

The first three rows below concern the optional coupling/ordering branch. The remaining rows apply when the corresponding CED predictor, calibration, or transfer claim is made. They are method-specific empirical obligations, not conditions for the finite theory's mathematical validity.

| Question being tested | Minimum comparison | What would count against CED |
|---|---|---|
| Is the gain only better marginal data? | Identical packet marginals; independent pairing; random matching; repeated matching with equal exposure | Gains disappear after exact marginal and token-budget control |
| Is the gain ordinary variance reduction? | Antithetic gradient pairing; gradient-diverse batches; adaptive antithetic baseline where feasible | Variance minimization explains both rankings and gains |
| Is the gain ordinary ordering? | Random reshuffling; GraB; exposure-matched forward/reverse curricula | A cheap order rule matches the full method |
| Does post-update utility add information beyond gradients? | LESS, GRACE, gradient norm/diversity, validation-gradient matching | No incremental out-of-sample prediction or selection benefit |
| Does update consequence add information beyond rollout consequence? | LOLS/AggreVaTe-style cost-to-go and a FutureBridge-compatible intervention gate | Action-continuation score performs equally at lower cost |
| Is the learned predictor useful rather than a surrogate oracle? | Direct small-budget actual-update selection; random candidate search at equal evaluation cost | Predictor training costs exceed its saved evaluations |
| Is calibration doing useful work? | Ungated predictor; mean-only selection; disjoint calibrated lower-bound gate | Gate rejects nearly everything or loses all budget-normalized gains |
| Does the effect transfer? | Frozen predictor and protocol on heldout task families, model scales, and optimizer recipes | Claimed cross-task benefit is only within-task interpolation |
| Is the reference tail essential? | Zero-step, short, and longer declared reference tails | Ranking is unstable or reverses under the actual deployed training recipe |
| Is the method a genuinely few-shot procedure? | Strong SFT/BC, on-policy correction, and distilled-data baselines at matched total information and compute | Few demonstrations hide large uncapped teacher/environment search |

Access parity is essential. The faithful MetaOPD and FutureBridge-OPD implementations require teacher token probabilities/scores. If the main setting supplies only teacher text/actions, use an explicitly labeled accessible meta-reweighting or rollout-cost adaptation in the main track, and run faithful methods in a separate teacher-scoring track. Do not report an inaccessible method as faithfully reproduced, and do not compare methods with unreported privileged access.

All costs count: teacher generations, exploratory branches, candidate actual updates, reference-tail training, calibration rollouts, and final evaluation rollouts. “Few-shot” should separately report demonstrations used to update the final student, privileged training information gathered, and total search cost.

A finite-panel calibration theorem certifies only its declared candidate family, state/task cells, and sampling assumptions. It cannot certify arbitrary new neural architectures, new tasks, or changed training recipes. Those need an explicit distributional assumption and new generalization result, or must remain empirical heldout tests.

## 6. Evidence gates before a strong contribution claim

### 6.1 Integrated-theory gates

1. **Correctness and scope.** Complete the independent proof audit and maintain the same declared message laws, task/learner family, access assumptions, and response object throughout the chain. Separate finite exact statements, smooth approximations, and neural hypotheses. V0.14 already makes substantial progress here.
2. **Operational necessity and sufficiency.** Retain realizable finite counterexamples and constructive sufficient interfaces. State precisely when a worst-pair witness ranges over all message laws, when it is valid for a restricted family, and when it does not establish a fixed-reference lower bound.
3. **Complexity and composition.** Exhibit informative families where the theory computes a sharp or useful information requirement and predicts its change with task or training composition. Distinguish exact algebraic dimension, approximate width, coefficient stability, acquisition cost, and verification cost.
4. **Measurable verification.** Test the noisy-feature tradeoff and the actual-state audit in regimes where their assumptions can be checked. Report the independent-replica cost and whether the bound is nonvacuous. A successful finite audit does not establish a uniform neural-state certificate.
5. **A distinguishing consequence.** Demonstrate at least one important conclusion that the unified characterization makes accessible: a structured information requirement, an identification/verification tradeoff, or a falsifiable learner-consequence prediction. This consequence can arise from any valid information-provider branch.
6. **Whole-chain novelty.** Complete the closest-framework comparisons from Section 2.1 and identify the precise additional operational conclusion. Reusing classical tools is compatible with originality; completeness of presentation by itself does not settle originality.

### 6.2 Method-branch gates

1. **Coupling mechanism, if that branch is claimed.** Show a reproducible coupling-only effect with valid units and exact marginal equality. Distinguish common-snapshot covariance effects from sequential-update interactions and compare the baselines in the first three rows of Section 5.
2. **Prediction, if a response predictor is claimed.** On disjoint trials, predict the signed ranking of actual deployment gains beyond established gradient and rollout predictors. Include calibration curves, sign errors, and selection regret; correlation alone is insufficient.
3. **Efficiency, if a practical improvement is claimed.** Beat direct evaluation and the simplest relevant method at equal total cost, including metatraining and calibration. Structured-batching comparisons apply to the coupling branch.
4. **Transfer, if that claim is made.** Freeze the selection machinery before heldout task-family and recipe evaluation. Report failures and negative transfer rather than only pooled averages.

If no coupling effect remains beyond variance reduction, reject the coupling branch's distinctive empirical claim. If a response predictor does not repay its measurement cost, reject that implementation's efficiency claim. Neither finding refutes the proved finite characterization or requires abandoning the ambition of a general theory. Continue with other consequences or provider branches whose assumptions and novelty are independently supported; preserve failed-branch results as boundary evidence. Do not retrospectively change the original branch hypothesis to make it appear successful.

## 7. Search coverage and limitations

The audit used primary papers and official proceedings/arXiv/OpenReview records. Search families included decision-relevant information and value equivalence; learner-aware simulator learning; active and sequential machine teaching; gradient/trajectory dataset distillation; moment matching in imitation and language-model distillation; weak approximations and conditional update moments; teacher-quality prediction; meta-gradient weighting; action and student-continuation costs; antithetic SGD; fixed-marginal couplings; batch diversification and matching; gradient-balanced ordering; and exposure-controlled curricula. Recent 2026 searches included the October preprints above. The integrated-framework follow-up covered value-equivalence rate-distortion learning, operational channel deficiency, and calibrated value-aware losses.

The review inspected relevant full-text sections of [R2], [R4–R8], [R10], [R12–R14], [R16], [R19–R20], [R25–R26], [R31], and policy-aware model/simulator papers. Other comparisons, including [R28–R30], used primary abstracts/official metadata plus clearly identified standard objectives; they are not represented as independently re-proved.

Material access limitations:

- **Two Mathematical Models of Knowledge Distillation** by Audrey Xie, Ludwig Schmidt, and John Duchi (AISTATS 2026; PMLR 300:4726–4734) was verified through its [official abstract](https://proceedings.mlr.press/v300/xie26b.html). Fulltext retrieval was blocked. Detailed gradient-covariance overlap remains to be checked.
- **Behavior Cloning is Not All You Need: The Optimality of On-Policy Distillation for Noisy Expert Feedback** by Ved Sriraman, Peihan Liu, Daniel Hsu, and Adam Block (arXiv:2606.30923, 2026) was [abstract-reviewed](https://arxiv.org/abs/2606.30923); HTML/PDF retrieval failed. Its noisy-expert setting should not be conflated with the framework's perfect privileged teacher.
- FutureBridge-OPD was examined in **v1**; the attempted v2 fulltext was unavailable.
- [R27] fulltext was blocked; its indexed primary-source text is a warning, not a completed theorem comparison.
- GraB, PartitionSel, Offline Behavior Distillation, and several established baselines were assessed at the primary abstract/objective level rather than through exhaustive implementation reproduction.

This review does not cover every obscure workshop paper, unpublished manuscript, language, patent, or source released after the cutoff. No “first ever” statement follows from these searches.

## 8. Primary reference register

URLs below identify the exact sources used. Venue labels are used only where verified; otherwise the entry is explicitly a preprint. The retrieval reference IDs are supplied separately to the integrating author.

- **[R1]** Andrew Warrington et al. *Robust Asymmetric Learning in POMDPs.* ICML 2021. [Official proceedings](https://proceedings.mlr.press/v139/warrington21a.html).
- **[R2]** Tomi Peltola, Mustafa Mert Çelikok, Pedram Daee, Samuel Kaski. *Machine Teaching of Active Sequential Learners.* arXiv:1809.02869, version 3, 2019. [Full text](https://arxiv.org/html/1809.02869v3).
- **[R3]** Laurent Lessard, Xuezhou Zhang, Xiaojin Zhu. *An Optimal Control Approach to Sequential Machine Teaching.* AISTATS 2019, PMLR 89:2495–2503. [Official proceedings](https://proceedings.mlr.press/v89/lessard19a.html).
- **[R4]** Christopher Grimm et al. *The Value Equivalence Principle for Model-Based Reinforcement Learning.* 2020. [Primary record](https://arxiv.org/abs/2011.03506), [PDF](https://arxiv.org/pdf/2011.03506).
- **[R5]** Qianxiao Li, Cheng Tai, Weinan E. *Stochastic Modified Equations and Dynamics of Stochastic Gradient Algorithms I: Mathematical Foundations.* JMLR 20(40):1–47, 2019. [Official article](https://jmlr.org/papers/v20/17-526.html), [PDF](https://jmlr.org/papers/volume20/17-526/17-526.pdf).
- **[R6]** Gokul Swamy, Sanjiban Choudhury, J. Andrew Bagnell, Steven Wu. *Of Moments and Matching: A Game-Theoretic Framework for Closing the Imitation Gap.* ICML 2021, PMLR 139:10022–10032. [Official proceedings](https://proceedings.mlr.press/v139/swamy21a.html).
- **[R7]** Chen Jia. *Adversarial Moment-Matching Distillation of Large Language Models.* NeurIPS 2024. [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cbae8efcc23a0cb6d15a20f245514020-Abstract-Conference.html).
- **[R8]** Andrei Lupu, Chris Lu, Jarek Liesen, Robert Tjarko Lange, Jakob Foerster. *Behaviour Distillation.* ICLR 2024. [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/36d13b71487b29965a2d3f5fcc109afc-Abstract-Conference.html), [full text](https://arxiv.org/html/2406.15042v1). The June 2024 arXiv upload date does not change the venue: both official proceedings and arXiv comments state ICLR 2024.
- **[R9]** Shiye Lei, Sen Zhang, Dacheng Tao. *Offline Behavior Distillation.* NeurIPS 2024. [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d0724f5d6108517c3eab35f77f156967-Abstract-Conference.html).
- **[R10]** Connor Wilhelm, Dan Ventura. *Distilling Reinforcement Learning into Single-Batch Datasets.* arXiv:2508.09283, 2025. [Primary record](https://arxiv.org/abs/2508.09283), [full text](https://arxiv.org/html/2508.09283v1).
- **[R11]** Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun. *Learning to Reweight Examples for Robust Deep Learning.* ICML 2018, PMLR 80:4334–4343. [Official proceedings](https://proceedings.mlr.press/v80/ren18a.html).
- **[R12]** Zipeng Wang, Xinpeng Dong, Yuefan Wang, Pingchen Lu, Xian Wei, Kun Kuang, Fei Wu, Zhongxiang Dai, Min Zhang. *MetaOPD: Meta-Learned Token Weighting for On-Policy Distillation.* arXiv:2610.11989, 8 October 2026, preprint. [Primary record](https://arxiv.org/abs/2610.11989), [full text](https://arxiv.org/html/2610.11989v1).
- **[R13]** Chishui Chen et al. *Look Ahead Before You Distill: Future Trajectory Validation of Teacher Guidance for Agentic On-Policy Distillation.* arXiv:2608.01953, 2026, preprint. [Primary record](https://arxiv.org/abs/2608.01953), [v1 full text reviewed](https://arxiv.org/html/2608.01953v1).
- **[R14]** Abhishek Panigrahi, Bingbin Liu, Sadhika Malladi, Sham Kakade, Surbhi Goel. *In Good GRACEs: Principled Teacher Selection for Knowledge Distillation.* ICLR 2026; arXiv:2511.02833. [Official paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/59fa5d049b777d880fdcddba2b24738f-Paper-Conference.pdf), [full text](https://arxiv.org/html/2511.02833v3).
- **[R15]** Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. *LESS: Selecting Influential Data for Targeted Instruction Tuning.* ICML 2024, PMLR 235:54104–54132. [Official proceedings](https://proceedings.mlr.press/v235/xia24c.html).
- **[R16]** Aaryan Gupta, Rishi Saket, Aravindan Raghuveer. *Algorithmic Guarantees for Distilling Supervised and Offline RL Datasets.* ICLR 2026. [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6bfd8e0962d08cbfee31f2a53937a8d3-Abstract-Conference.html), [full text](https://arxiv.org/html/2512.00536v1).
- **[R17]** Kai-Wei Chang, Akshay Krishnamurthy, Alekh Agarwal, Hal Daumé III, John Langford. *Learning to Search Better than Your Teacher.* ICML 2015, PMLR 37:2058–2066. [Official proceedings](https://proceedings.mlr.press/v37/changb15.html).
- **[R18]** Stéphane Ross, J. Andrew Bagnell. *Reinforcement and Imitation Learning via Interactive No-Regret Learning.* arXiv:1406.5979, 2014. [Primary record](https://arxiv.org/abs/1406.5979).
- **[R19]** Jingchang Liu, Linli Xu. *Accelerating Stochastic Gradient Descent Using Antithetic Sampling.* arXiv:1810.03124, 2018. [Primary record](https://arxiv.org/abs/1810.03124), [full text](https://arxiv.org/html/1810.03124v1).
- **[R20]** Hongyu Ren, Shengjia Zhao, Stefano Ermon. *Adaptive Antithetic Sampling for Variance Reduction.* ICML 2019, PMLR 97:5420–5428. [Official proceedings](https://proceedings.mlr.press/v97/ren19b.html), [PDF](https://proceedings.mlr.press/v97/ren19b/ren19b.pdf).
- **[R21]** Cheng Zhang, Hedvig Kjellstrom, Stephan Mandt. *Determinantal Point Processes for Mini-Batch Diversification.* arXiv:1705.00607, 2017. [Primary record](https://arxiv.org/abs/1705.00607), [PDF](https://arxiv.org/pdf/1705.00607).
- **[R22]** Cheng Zhang, Cengiz Öztireli, Stephan Mandt, Giampiero Salvi. *Active Mini-Batch Sampling using Repulsive Point Processes.* arXiv:1804.02772, 2018. [Primary record](https://arxiv.org/abs/1804.02772).
- **[R23]** Yucheng Lu, Wentao Guo, Christopher De Sa. *GraB: Finding Provably Better Data Permutations than Random Reshuffling.* arXiv:2205.10733, 2022. [Primary record](https://arxiv.org/abs/2205.10733).
- **[R24]** Prayas Agrawal, Prateek Chanda, Ishita Khatri, Ganesh Ramakrishnan, Bamdev Mishra, Pratik Jawanpuria. *Minibatch Selection via Partition Matroid Constrained Gradient Matching.* arXiv:2606.07954, 2026, preprint. [Primary record](https://arxiv.org/abs/2606.07954).
- **[R25]** Changho Shin, David Alvarez-Melis. *Curriculum Learning as Transport: Understanding Curricula with Wasserstein Geodesics.* arXiv:2609.09099, 2026. [Full text](https://arxiv.org/html/2609.09099v1). Related instruction-tuning comparator: Bingxiang He et al., *The Right Time Matters: Data Arrangement Affects Zero-Shot Generalization in Instruction Tuning*, Findings of ACL 2025, pp. 222–243, [official article](https://aclanthology.org/2025.findings-acl.13/).
- **[R26]** Ke Wang, Zijie Zhao, Zhiyi Yuan, Changlun Li. *When Better Gets Worse: Improvement Fidelity for Self-Improving Agents in Adaptive Worlds.* arXiv:2609.32677, 2026, preprint. [Full text](https://arxiv.org/html/2609.32677v1).
- **[R27]** Xiaoyu Li, Dai Shi, Jiaojiao Jiang, Junbin Gao, Andi Han. *Same Sample or Fresh? The Coupling Geometry of Stochastic Extragradient.* OpenReview posting, 4 October 2026; venue/acceptance not asserted. [Record](https://openreview.net/forum?id=Tq62MyX4US), [indexed PDF, full access blocked](https://openreview.net/pdf/8512d44f756b40782c616125f688fb844968cff4.pdf).
- **[R28]** Dilip Arumugam, Benjamin Van Roy. *Deciding What to Model: Value-Equivalent Sampling for Reinforcement Learning.* NeurIPS 2022. [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3b18d368150474ac6fc9bb665d3eb3da-Abstract-Conference.html).
- **[R29]** Dilip Arumugam, Benjamin Van Roy. *Deciding What to Learn: A Rate-Distortion Approach.* ICML 2021, PMLR 139:373–382. [Official proceedings](https://proceedings.mlr.press/v139/arumugam21a.html).
- **[R30]** Pradeep Kr. Banerjee, Guido Montúfar. *The Variational Deficiency Bottleneck.* IJCNN 2020, as identified in the primary record; arXiv:1810.11677, version 3. [Primary record](https://arxiv.org/abs/1810.11677).
- **[R31]** Claas Voelcker, Anastasiia Pedan, Arash Ahmadian, Romina Abachi, Igor Gilitschenski, Amir-massoud Farahmand. *Calibrated Value-Aware Model Learning with Probabilistic Environment Models.* arXiv:2505.22772, version 2, 2025; venue not asserted here. [Primary record](https://arxiv.org/abs/2505.22772), [v2 full text](https://arxiv.org/html/2505.22772v2).

Additional foundational comparisons: Tongzhou Wang et al., *Dataset Distillation*, [arXiv:1811.10959](https://arxiv.org/abs/1811.10959); Bo Zhao, Konda Reddy Mopuri, Hakan Bilen, *Dataset Condensation with Gradient Matching*, [arXiv:2006.05929](https://arxiv.org/abs/2006.05929); George Cazenavette et al., *Dataset Distillation by Matching Training Trajectories*, [arXiv:2203.11932](https://arxiv.org/abs/2203.11932); Romina Abachi, Mohammad Ghavamzadeh, Amir-massoud Farahmand, *Policy-Aware Model Learning for Policy Gradient Methods*, [arXiv:2003.00030](https://arxiv.org/abs/2003.00030); Christoph Dann, Yishay Mansour, Mehryar Mohri, *Theoretical Foundations and Effective Algorithms for Policy-Aware Simulator Learning*, [arXiv:2605.29032](https://arxiv.org/abs/2605.29032).

