# Learner-Relative Experience Sufficiency for Agent Distillation — Theoretical Framework

[中文版](README_ZH.md)

**Question.** Relative to the declared deployment tasks, the permitted experiments and the actual learning procedure, which differences between feedback sources change the final task performance of a distilled student?

**Central results.**

- **Theorem A (§6, learner-relative information bound).** For any distiller (replay, rewriting and synthesis included), the smallest achievable first-order error is governed by the Fisher information that demonstrations and queries carry about one learner-dependent functional. Its gradient $`c=\sum_tJ_t^{*}g_t`$ combines the learner's Jacobian with the task's continuation gradients. Under identifiability and growth conditions, the bound is attained up to a constant factor by a regularized Elfving c-optimal query design ([Elfving, 1952](#ref-elfving1952optimum)). Hence the same demonstrations can be sufficient for one learner and insufficient for another.
- **Theorem B (§7, Adam's second-moment channel).** Under warm Adam ([Kingma & Ba, 2015](#ref-kingma2015adam)), the first-order value of a message splits exactly into a parameter channel, a momentum channel and a second-moment channel. In a stationary regime the response to a message of relative magnitude $`\lambda`$ has an exact mean-field form. For small increments it is a concave quadratic that peaks at an explicit $`\lambda^*`$ and turns negative near $`2\lambda^*`$. Scores that look only at the parameter step, or only at directions, are therefore inconsistent rankers.
- **Theorem C (§8, auditable certificate).** A common-random-numbers (CRN) paired replacement of one training round (CRN is a classical variance-reduction device in simulation; [Glasserman, 2004](#ref-glasserman2004monte)) measures the finite effect of a feedback law directly. The resulting certificate needs no continuation-value gradient and no curvature constant. Its radius has only a statistical term and a surrogate-calibration term.

These rest on foundations (§2–§5): an exact regret decomposition and the cost of acquiring missing information, the continuation response as the object of sufficiency, the complexity of sufficiency, and the statistic that each optimizer makes feedback carry. §9 derives the method, CED, from all of them.

**How to read this page.** The page goes from the whole to the parts, breadth first.

| Level | Content | After reading it you can answer |
|---|---|---|
| [Level 0](#level-0--one-paragraph) | One paragraph on the whole framework | What question the framework answers, and in which steps |
| [Level 1](#level-1--eight-sections) | The eight sections and how they connect | What each section asks, what it concludes, what it hands on |
| [Level 2](#level-2--all-results) | Every numbered result, in order, with links to its statement, proof and experiments pages | What each result says, the intuition, its role in the chain, and where its exact statement, proof and checks are |

Notation is fixed in [§1](#1-setting-and-notation). Every numbered result has its own **statement** page (setting, exact statement, derivation outline) and, where applicable, a **proof** page with the complete derivation and an **experiments** page with the checks that test it: why each check is run, its setup, its result and why it supports the statement. Each of these pages can be read on its own; it restates the setting and the statement it needs. All measured results are on the experiments pages; this page contains statements and illustrative examples only.

**References.** Every page ends with its own list of the works it cites. The complete list, with authors, title, venue and DOI or arXiv ID checked against the primary source and the pages that use each work, is [references.md](references.md).

**Scope.** Every statement on this page is proved under the assumptions it states. A few are proved to leading order in an explicitly stated asymptotic regime, and the page says so where this happens. Proofs are finite-dimensional or carry explicit regularity assumptions. The checks on the experiments pages are exact finite computations or controlled small networks. Real-model tests of Theorems A–C are designed ([Remark 9.4](statements/09-04_scope_of_evidence.md#s9-4)); the first runs are under way and their results are not yet on these pages.

---

## Level 0 · One paragraph

To distill an agent we have a pretrained student, $`K`$ trusted teacher demonstrations per task family, a budget of further permitted experiments (teacher calls, environment branches, resets), and a fixed training recipe (optimizer, learning rate, number of steps). Many feedback sources can be fed to the student: the demonstrations themselves, replays, synthetic rewrites, new queries, different packagings of the same units. The framework asks which differences between these sources actually change the final task performance of the distilled student. It answers in eight steps.

1. **Can the data determine what to do? (§2)**
   - The student's regret splits exactly into a representation term, an information term and an algorithm term.
   - An information gap can be closed only by new experiments, never by replay or rewriting, and even an everywhere-optimal teacher can leave one.
   - Closing it costs on the order of $`\log(1/\delta)/\Delta^2`$ queries, and demonstrations and queries draw on one information budget.
2. **What does feedback do to the learner? (§3)**
   - Teacher quality, policy value and the mean gradient do not determine what the student learns.
   - The invariant is the *continuation response*: the expected final return after receiving one message, applying the actual update and finishing training.
3. **How much must two sources agree on? (§4)**
   - Exactly the span of the response class. For averaging learners this is a moment order set by the deployment length.
   - Approximate complexity can be much smaller than the exact order, and nonlinear updates multiply the order.
4. **What does the actual optimizer make feedback carry? (§5)**
   - To first order, what matters is the mean increment of the *complete learner state* (parameters and optimizer state). This first-order representation is a tool, not a claim of novelty.
   - The statistic is explicit for SGD, momentum, cold and warm Adam, clipping, Lion ([Chen et al., 2023](#ref-chen2023symbolic)) and LoRA ([Hu et al., 2022](#ref-hu2022lora)). For warm Adam the exact statistic is the mean Adam increment, and the familiar (mean, second moment) description holds only under a frozen-denominator approximation with an explicit error.
   - An Adam coordinate's warmth falls like $`1/(\beta_2s)`$ only early in training; it levels off at $`(1-\beta_2)/\beta_2`$.
5. **How much information is needed, and in which direction? (Theorem A, §6)**
   - The learner decides the direction $`c`$ in which the data must be informative. The minimax error of any distiller is the learner-relative information norm, up to a constant.
   - Consequences: replay has zero value against aliasing, demonstrations and queries trade at a learner-relative exchange rate, queries are allocated by water-filling, and SGD and Adam can need very different data for the same task.
6. **How does a message act through Adam's second moment? (Theorem B, §7)**
   - A message also changes all later step sizes through $`v`$.
   - This gives an amplitude law with an optimal magnitude and, for small increments, a sign change near twice that magnitude. Parameter-only and cosine scores misrank in an explicit region.
7. **How can finite measurements certify a choice? (Theorem C and §8)**
   - Finite audits need a coverage (transport) term, and full-tail audits at the actual states are safe but expensive.
   - First-order scores need the continuation gradient, and they certify a pair only when its first-order term exceeds the second-order remainder; pairs whose mean increments agree along the gradient are never certified.
   - CRN paired replacement measures the finite effect directly. Two problems remain open: a provable range much smaller than the a-priori one, and a testable condition for transferring the surrogate calibration.
8. **The method (§9).**
   - A mechanism test with a strict null (coupling width).
   - Consequence-Calibrated Experience Distillation (CED): acquire by learner-relative design, select by certified measured consequence, match the learner's statistic, account for coverage, freeze and then test transfer.

---

## Level 1 · Eight sections

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 28, "rankSpacing": 36, "padding": 10, "wrappingWidth": 260}}}%%
flowchart TB
  subgraph P1["Part I · What data and feedback determine"]
    direction LR
    S2["<b>§2 Feasibility<br/>and acquisition</b><br/>Can the data<br/>determine what to do?"]:::b1
    S3["<b>§3 Learning<br/>consequence</b><br/>What does feedback<br/>make the student learn?"]:::b2
    S4["<b>§4 Complexity<br/>of sufficiency</b><br/>How many statistics<br/>must match?"]:::b3
    S2 --> S3 --> S4
  end
  subgraph P2["Part II · What the actual learner needs"]
    direction LR
    S5["<b>§5 Optimizer state</b><br/>What must feedback<br/>carry for a real optimizer?"]:::b4
    S6["<b>§6 Theorem A</b><br/>How much information,<br/>in which direction?"]:::b4
    S7["<b>§7 Theorem B</b><br/>How does a message<br/>act through Adam's v?"]:::b4
    S5 --> S6
    S5 --> S7
  end
  subgraph P3["Part III · How to verify and use it"]
    direction LR
    S8["<b>§8 Verification</b><br/>Theorem C: certify with<br/>finite measurements"]:::b5
    S9["<b>§9 Mechanism<br/>and method</b><br/>CED"]:::b6
    S8 --> S9
  end
  P1 --> P2 --> P3
  style P1 fill:#fafafa,stroke:#999
  style P2 fill:#fafafa,stroke:#999
  style P3 fill:#fafafa,stroke:#999
  classDef b1 fill:#e8f1fb,stroke:#2f6db5,color:#111
  classDef b2 fill:#eaf6ec,stroke:#2e8540,color:#111
  classDef b3 fill:#fdf3e1,stroke:#b8741a,color:#111
  classDef b4 fill:#f6e9f4,stroke:#9c3d8f,color:#111
  classDef b5 fill:#e9f5f6,stroke:#1f7f8c,color:#111
  classDef b6 fill:#fbeaea,stroke:#b53b3b,color:#111
```

Arrows show the main flow of the argument. Further dependencies:

- §2 fixes the acquisition budget used by Theorem A (§6).
- §3 supplies the continuation decomposition used throughout §8.
- §4 supplies the approximation width and the coefficient stability used by §8.
- §5 supplies the statistic that §8 measures.

### §2 · Feasibility and acquisition

**Question.** Within the policies the student can execute, can the available data determine what to do, and what does it cost to acquire what is missing?

**Conclusion.**

- The student's regret against the best reference policy splits exactly into representation error, information error and algorithm error. All three are nonnegative, and none can substitute for another.
- The information error is zero if and only if all environments consistent with the data share an optimal policy. No environment-independent post-processing of the data (replay, rewriting, synthesis) can reduce it.
- Even a teacher that is optimal in every environment can leave an information gap. Active experiments such as reset queries close it, at a cost of order $`\log(1/\delta)/\Delta^2`$; this order is both sufficient and necessary. Compiling the acquired fact into the student is a separate, much smaller cost.
- In a hidden-fact model, the information left after $`K`$ demonstrations is explicit, the optimal hard-budget query rule is greedy by weight, and demonstrations and adaptive queries satisfy one linear information-budget inequality.

**Hand-off.** Missing facts must be acquired. What a learner makes of the information is the question of §3, and how much information a *specific* learner needs is answered by Theorem A in §6.

### §3 · Learning consequence

**Question.** Given a feedback source, what will the student actually learn, and what should be compared when two sources are compared?

**Conclusion.**

- Static measures are not enough. Two sources can have equally good teachers and give every student the same policy value, yet distill different students.
- Two label packets can have equal marginals and equal mean gradients, yet different learning values, with the sign of the difference reversing on another task.
- The right object is the continuation response: the expected final task return after receiving message $`y`$, applying the actual update and finishing training under the reference recipe.
- The gap over a whole run is exactly a sum over rounds of one-boundary replacement effects. Two message laws are interchangeable if and only if they have equal expectation under every response.
- When guarantees hold only on part of the state space, the probability of leaving that part is paid once, not once per round.

**Hand-off.** The object of sufficiency is the response class $`\mathcal G`$. §4 measures its complexity, §5–§7 compute it for real optimizers, and §8 checks it with finite data.

### §4 · Complexity of sufficiency

**Question.** How many statistics must two feedback sources agree on to be equivalent for learning?

**Conclusion.**

- When equivalence is judged by the means of a set of features, exact sufficiency holds if and only if every response lies in their span. The number of statistics needed is the dimension of the span of the response class.
- Approximately, the worst-case error is twice the best uniform approximation error of the responses by the features. An unrestricted learner needs the whole distribution.
- For learners that average independent experience blocks and for deployments with at most $`N`$ decisions, exactly the moments up to order $`N`$ are needed, and every order is necessary.
- The gap decays with the number of *independent* blocks at a rate set by the first unmatched cumulant, independently of the number of training rounds. Replay and deterministic synthesis do not make it decay.
- Approximate complexity can be far below the exact order (about $`\sqrt N`$ for smooth values), with parity-type values as the exception.
- Composing nonlinear updates multiplies the required order.

**Hand-off.** What must match is decided by the learner itself, so the actual optimizer must be modelled (§5). In approximations the stability of the coefficients matters (§8).

### §5 · Optimizer state and the first-order representation

**Question.** What must feedback carry for the learners actually in use: SGD, momentum, Adam/AdamW, clipping, Lion, LoRA?

**Conclusion.**

- Write any stateful first-order optimizer as a map from a consumed gradient to an increment of the complete learner state. To first order, the effect of a feedback law on the training outcome is the inner product between the continuation-value gradient and the mean complete-state increment.
- Higher orders correspond to higher moments of the increment. This expansion is a standard Taylor-plus-telescoping computation, in the same family as stochastic modified equations ([Li, Tai & E, 2019](#ref-li2019stochastic)), TracIn ([Pruthi et al., 2020](#ref-pruthi2020estimating)) and metagradients ([Engstrom et al., 2025](#ref-engstrom2025optimizing)), and it is used here as a tool.
- What is specific to each optimizer is the increment:
  - Cold-start Adam looks only at gradient signs in its first step, while its state keeps the magnitudes.
  - For warm Adam, the exact first-order statistic is the mean Adam increment. Under a frozen-denominator approximation it reduces to the mean and second moment of the gradient, with an error controlled coordinatewise by the coordinate's warmth.
  - Warmth falls like $`1/(\beta_2s)`$ only while $`s(1-\beta_2)\ll1`$, and then levels off at $`(1-\beta_2)/\beta_2`$, so the frozen-denominator error has a per-step floor.
  - Clipping makes the statistic the clipped mean. When clipping is active with probability $`p_{\rm act}`$, only the mean direction matters, up to an error at most $`c(1-p_{\rm act})`$.

**Hand-off.** The increment map tells us *what* a feedback law must carry. Theorem A (§6) asks how much information the data must hold about it, Theorem B (§7) follows a message through Adam's second moment, and §8 measures it.

### §6 · Theorem A: the learner-relative information bound

**Question.** Given $`K`$ demonstrations and $`n`$ permitted queries, how well can *any* distiller reproduce the first-order effect of the oracle feedback, and how should the queries be spent?

**Conclusion.**

- Let $`c=\sum_tJ_t^*g_t`$, which combines the learner's Jacobian $`J_t`$ (how the mean increment moves with the unknown environment parameter) with the task's continuation gradients $`g_t`$.
- The minimax error of any distiller is, up to a factor between $`1/4`$ and $`1`$, the learner-relative information norm: the largest ratio of $`c^\top h`$ to the Fisher information that demonstrations and queries carry in direction $`h`$.
- Asymptotically optimal queries follow an Elfving c-optimal design ([Elfving, 1952](#ref-elfving1952optimum); [Pukelsheim, 1993](#ref-pukelsheim1993optimal)) with target $`c`$, regularized so that every direction is eventually identified.
- Consequences:
  - directions the demonstrations cannot see make every amount of replay worthless;
  - one demonstration is worth a learner-relative number of queries;
  - in a state-coverage model, queries go by water-filling to states with high learner-relative weight and low teacher coverage;
  - SGD weights a state by the square of its visitation, while warm Adam in the frequent-visit regime weights it by the visitation itself, so Adam amplifies the student's long tail;
  - clipping makes the learner blind to gradient scale.

**Hand-off.** Theorem A says in which direction information is needed. For Adam that direction has a component through the second moment, which §7 analyses.

### §7 · Theorem B: Adam's second-moment channel

**Question.** How does one message act on the final outcome under warm Adam, and when does a larger gradient hurt?

**Conclusion.**

- A message's first-order value splits exactly into parameter, momentum and second-moment channels, with costates given by an explicit backward recursion.
- In a stationary mean-field regime the response to a message of relative magnitude $`\lambda`$ has an exact closed form:
  - for small increments it is a concave quadratic up to a proved relative error of order $`(1-\beta_2)\max\lbrace1,\lambda^2\rbrace`$, with optimum $`\lambda^*=(1-\beta_1^{T+1})/(\rho\,\omega_T)`$ and a sign change near $`2\lambda^*`$;
  - for large $`\lambda`$ the direct effect saturates while the suppression of later steps keeps accumulating; the limit is explicit, and when it is negative the response has a positive zero.
- Parameter-only scores and cosine scores therefore misrank message pairs in an explicit region. Under SGD the value would simply be linear in $`\lambda`$.

**Hand-off.** Where the second-moment channel dominates, increments are large relative to $`v`$, and first-order rankings fail; there one must measure finite replacements (§8).

### §8 · Verification and Theorem C

**Question.** With finitely many measurements, how can we prove that one feedback source is better than another, or select a good one?

**Conclusion.**

- An error in measuring a feature is amplified by that feature's coefficient; a fit with small residual and huge coefficients is not a proof.
- Selecting by discrepancies measured on a fixed set of audit cells requires an explicit coverage (transport) term. If training reaches uncovered states, the audit can read zero while the learning gap is 1.
- Paired full-tail experiments at the actual states, with selection by a lower confidence bound, never lose more than $`\delta`$ against the reference, but they need many paired replicas per candidate.
- The mean increment itself is cheap to estimate. A first-order score, however, needs the continuation gradient and the remainder constants. It certifies a pair only when the pair's first-order term exceeds the second-order remainder, so pairs whose mean increments agree along the gradient are never certified, however large their true gap.
- Theorem C replaces the gradient by a CRN paired replacement of one round. This is exact in expectation, has a radius with only a statistical term and a calibration term, and costs $`(M+1)r`$ surrogate tails per audited state.
- Open: a provable range much smaller than the a-priori one, and a testable calibration-transfer condition. Without them the strict certificate is valid but rarely decisive.

**Hand-off.** §9 assembles acquisition, selection and certification into a method.

### §9 · Mechanism and method

**Question.** What method follows, and which mechanism deserves a test of its own?

**Conclusion.**

- *Coupling width:* fixing each unit's marginal law and regrouping units into joint training packets changes nothing if and only if a packet's return is additive across units. One SGD step is the strict null.
- *CED (Consequence-Calibrated Experience Distillation):* record the complete learner state; acquire missing information by learner-relative design before compiling; select candidate feedback laws by CRN paired-replacement scores, calibrated against full-tail audits and with coverage accounted for; match the learner's statistic and control message magnitude relative to the running second moment; freeze the selector and then test transfer.
- Each design rule is a corollary of a cited statement.

---

## Level 2 · All results

Each section's graph shows how its results depend on each other (an arrow A → B means that B's statement or proof uses A). Below it, every numbered result of the section is listed in order with three lines (what it says, intuition, role) and links to its own statement, proof and experiments pages.

### §2 · Feasibility and acquisition — results

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T2["<b>Theorem 2.2</b><br/>Regret decomposition"]:::b1
  P3["<b>Prop. 2.3</b><br/>Robust ambiguity<br/>duality"]:::b1
  P4["<b>Prop. 2.4</b><br/>Post-processing<br/>cannot help"]:::b1
  T6["<b>Theorem 2.6</b><br/>Passive impossibility,<br/>active attainability"]:::b1
  P7["<b>Prop. 2.7</b><br/>Compilation cost"]:::b1
  L9["<b>Lemma 2.9</b><br/>No leakage about<br/>unrevealed bits"]:::b1
  T10["<b>Theorem 2.10</b><br/>Few-shot<br/>information floor"]:::b1
  P11["<b>Prop. 2.11</b><br/>One information<br/>budget"]:::b1
  T2 --> P3
  T2 --> P4
  T2 --> T6
  T6 --> P7
  T6 --> T10
  L9 --> T10
  T10 --> P11
  classDef b1 fill:#e8f1fb,stroke:#2f6db5,color:#111
```

**[Definition 2.1](statements/02-01_bayes_value_three_regret_terms.md#s2-1) · Bayes value and the three regret terms** — [statement](statements/02-01_bayes_value_three_regret_terms.md)
- *What it is.* $`B(D)`$ is the value of the best decision a Bayesian could make with the transcript, using a policy the student can execute. $`\mathcal R_{\rm rep}`$ measures the student class against the reference class, $`\mathcal R_{\rm info}`$ measures the transcript against perfect knowledge of the environment, and $`\mathcal R_{\rm alg}`$ measures the learner against the Bayes value.

**[Theorem 2.2](statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2) · Regret decomposition and decision ambiguity** — [statement](statements/02-02_regret_decomposition_decision_ambiguity.md) · [proof](proofs/02-02_regret_decomposition_decision_ambiguity.md) · [experiments](experiments/02-02_regret_decomposition_decision_ambiguity.md)
- *What it says.* The expected regret against the best reference policy is the sum of three nonnegative terms: representation error (the student class cannot reach the reference optimum), information error (the data cannot tell the environments apart) and algorithm error (the learner does not use the data well). The information error is zero if and only if, for every transcript of positive probability, all environments with positive posterior probability share an optimal policy.
- *Intuition.* The information error is what is lost even by a perfect learner, because the data leave the right decision ambiguous. More data can only lower it; a better learner can only lower the algorithm error.
- *Role.* It separates "not enough information" from "learned badly". Every information result of the framework bounds one of these terms; [Theorem 2.6](statements/02-06_passive_impossibility_active_attainability.md#s2-6) and [Theorem 2.10](statements/02-10_few_shot_information_floor.md#s2-10) compute the information term in explicit families.

**[Proposition 2.3](statements/02-03_robust_ambiguity_duality.md#s2-3) · Robust ambiguity duality** — [statement](statements/02-03_robust_ambiguity_duality.md) · [proof](proofs/02-03_robust_ambiguity_duality.md) · [experiments](experiments/02-03_robust_ambiguity_duality.md)
- *What it says.* The minimax regret of hedging over an ambiguity class equals the largest Bayes information regret over priors on the class. If the experiments cannot distinguish the members of the class, no learner does better in the worst case, and the minimax mixture attains the value. The value is zero exactly when one policy is optimal in all members.
- *Intuition.* Indistinguishable worlds force the learner to hedge. The price of hedging is the value of a zero-sum game against the least favourable prior.
- *Role.* It gives the information error a computable minimax form and a certificate of its size (a least favourable prior). The two-world family of [Theorem 2.6](statements/02-06_passive_impossibility_active_attainability.md#s2-6) is an instance with value $`\Delta`$.

**[Proposition 2.4](statements/02-04_post_processing_cannot_reduce_information.md#s2-4) · Post-processing cannot reduce information regret** — [statement](statements/02-04_post_processing_cannot_reduce_information.md) · [proof](proofs/02-04_post_processing_cannot_reduce_information.md) · [experiments](experiments/02-04_post_processing_cannot_reduce_information.md)
- *What it says.* Any randomized transformation of the transcript that does not depend on the environment can only raise the information term or leave it unchanged. Adding data that is independent of the environment given the transcript leaves the Bayes value and the information term unchanged.
- *Intuition.* Processing data can make existing information easier to learn, which lowers the algorithm error, but it cannot create a missing fact.
- *Role.* It rules out generated data as a substitute for new experiments. [Corollary 6.7](statements/06-07_aliasing_zero_value_replay.md#s6-7) states the same for the learner-relative first-order effect, for every distiller.

**[Definition 2.5](statements/02-05_two_world_aliased_family.md#s2-5) · Two-world aliased family** — [statement](statements/02-05_two_world_aliased_family.md)
- *What it is.* Two worlds differ only in how well each action works in the state where the teacher does not use it. The teacher never reveals that difference, because it always takes the action that succeeds for sure.

**[Theorem 2.6](statements/02-06_passive_impossibility_active_attainability.md#s2-6) · Passive impossibility, active attainability** — [statement](statements/02-06_passive_impossibility_active_attainability.md) · [proof](proofs/02-06_passive_impossibility_active_attainability.md) · [experiments](experiments/02-06_passive_impossibility_active_attainability.md)
- *What it says.* Passive learners, however they process the demonstrations, lose at least $`\Delta`$ in one of the worlds. With reset queries, about $`\log(1/\delta)/\Delta^2`$ queries are necessary and sufficient, up to a constant factor, to output a near-optimal student with probability $`1-\delta`$ in both worlds.
- *Intuition.* A perfect demonstration shows what the teacher did, not what a student with different abilities should do; the student has to try.
- *Role.* It shows why the method needs an active acquisition step and why demonstration quality is not information sufficiency. Its change-of-measure argument is reused in [Proposition 2.11](statements/02-11_shared_information_budget.md#s2-11), and its passive part is generalized by [Corollary 6.7](statements/06-07_aliasing_zero_value_replay.md#s6-7).

**[Proposition 2.7](statements/02-07_compilation_cost.md#s2-7) · Compilation cost** — [statement](statements/02-07_compilation_cost.md) · [proof](proofs/02-07_compilation_cost.md) · [experiments](experiments/02-07_compilation_cost.md)
- *What it says.* After the sign is estimated, $`O(\log(1/\delta_L))`$ synthetic labels drawn toward the estimated optimum train a frequency learner to within $`\tfrac18`$ of the target, with probability at least $`1-\delta_L`$, independently of $`\Delta`$. Together with a correct sign, this gives a student with regret at most $`\Delta/2`$ with probability at least $`1-\delta-\delta_L`$.
- *Intuition.* Acquiring the fact costs real environment queries and grows like $`\Delta^{-2}`$; compiling it costs cheap synthetic labels and does not grow with $`1/\Delta`$.
- *Role.* It separates the acquisition budget (authentic queries) from the compilation budget (synthetic labels) in the accounting of the method ([Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2)).

**[Definition 2.8](statements/02-08_hidden_bit_model.md#s2-8) · Hidden-bit model** — [statement](statements/02-08_hidden_bit_model.md)
- *What it is.* Facts are independent fair bits, one per context; demonstrations reveal random subsets of them; queries reveal chosen bits one at a time, up to a hard budget $`n`$.

**[Lemma 2.9](statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9) · Demonstrations leak nothing about unrevealed bits** — [statement](statements/02-09_no_leakage_about_unrevealed_bits.md) · [proof](proofs/02-09_no_leakage_about_unrevealed_bits.md) · [experiments](experiments/02-09_no_leakage_about_unrevealed_bits.md)
- *What it says.* Given the complete transcript, the bits no demonstration revealed are independent, fair and independent of the transcript and of the learner's randomness.
- *Intuition.* Demonstrations that reveal some facts by a mechanism independent of the facts carry no evidence about the rest.
- *Role.* It makes every unrevealed bit a coin flip for any learner, which gives the information floor of [Theorem 2.10(a)](statements/02-10_few_shot_information_floor.md#s2-10), and it makes value-adaptive querying useless in [Theorem 2.10(b)](statements/02-10_few_shot_information_floor.md#s2-10).

**[Theorem 2.10](statements/02-10_few_shot_information_floor.md#s2-10) · Few-shot information floor and optimal hard-budget querying** — [statement](statements/02-10_few_shot_information_floor.md) · [proof](proofs/02-10_few_shot_information_floor.md) · [experiments](experiments/02-10_few_shot_information_floor.md)
- *What it says.* After $`K`$ demonstrations the information floor is $`\tfrac12\sum_jw_j(1-q_j)^K`$: half the expected weight of the facts no demonstration revealed. With a hard budget of $`n`$ queries, querying the heaviest unseen facts is optimal, and adapting the queries to the revealed values does not help. The expectation must stay outside the positive part.
- *Intuition.* Each demonstration reveals a fact with some probability; what it leaves unrevealed must be queried, and the most valuable gaps first.
- *Role.* It puts $`K`$ into the theory and is the model case for the query allocation of [Proposition 6.10](statements/06-10_water_filling_allocation.md#s6-10) and the acquisition step of [Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2).

**[Proposition 2.11](statements/02-11_shared_information_budget.md#s2-11) · Demonstrations and queries share one information budget** — [statement](statements/02-11_shared_information_budget.md) · [proof](proofs/02-11_shared_information_budget.md) · [experiments](experiments/02-11_shared_information_budget.md)
- *What it says.* To tell two worlds apart with error at most $`\delta`$, the KL information of the $`K`$ demonstrations plus the KL information per query times the expected number of queries must reach $`\mathrm{kl}(1-\delta,\delta)`$. If queries carry no information and the demonstrations do not suffice, no query budget helps.
- *Intuition.* Demonstrations and queries pay into the same account; informative demonstrations reduce the number of queries needed, and replays pay nothing.
- *Role.* It is the exchange rate between $`K`$ and the query budget. [Corollary 6.8](statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8) makes the rate depend on the learner.

**[Remark 2.12](statements/02-12_learner_relative_versions.md#s2-12) · Learner-relative versions** — [statement](statements/02-12_learner_relative_versions.md)
- *What it says.* The passive impossibility of [Theorem 2.6(a)](statements/02-06_passive_impossibility_active_attainability.md#s2-6) and the data-processing result of [Proposition 2.4](statements/02-04_post_processing_cannot_reduce_information.md#s2-4) become [Corollary 6.7](statements/06-07_aliasing_zero_value_replay.md#s6-7): if demonstrations cannot distinguish two environments that differ on $`\Psi`$, no distiller, including replay, rewriting and synthesis, beats half the difference, for any $`K`$. The budget of [Proposition 2.11](statements/02-11_shared_information_budget.md#s2-11) becomes the per-direction budget of [Corollary 6.8](statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8), where the directions are weighted by the learner.
- *Role.* It links the decision-theoretic results of §2 to the learner-relative information bound (Theorem A) of §6.

### §3 · Learning consequence — results

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  P1["<b>Prop. 3.1</b><br/>Static value ≠<br/>learning value"]:::b2
  T2["<b>Theorem 3.2</b><br/>Continuation-response<br/>characterization"]:::b2
  P1 --> T2
  classDef b2 fill:#eaf6ec,stroke:#2e8540,color:#111
```

**[Proposition 3.1](statements/03-01_static_value_not_learning_value.md#s3-1) · Static value is not learning value** — [statement](statements/03-01_static_value_not_learning_value.md) · [proof](proofs/03-01_static_value_not_learning_value.md) · [experiments](experiments/03-01_static_value_not_learning_value.md)
- *What it says.* Two data sources can give every student in the class the same policy value and give the teacher the same value, yet distill students with returns $`\tfrac{43}{80}`$ and $`\tfrac{41}{80}`$. Two label packets can have the same marginals and the same mean gradient, yet learning values that differ by $`\eta^2/8`$, with the sign reversed on another task.
- *Intuition.* What is learned depends on how the data act on the student *through the update*, and on what the deployment rewards; how good the data look by themselves does not decide it.
- *Role.* It rules out teacher quality, policy value and the mean gradient as sufficiency criteria and motivates the continuation response of [Theorem 3.2](statements/03-02_continuation_response_characterization.md#s3-2).

**[Theorem 3.2](statements/03-02_continuation_response_characterization.md#s3-2) · Continuation-response characterization** — [statement](statements/03-02_continuation_response_characterization.md) · [proof](proofs/03-02_continuation_response_characterization.md) · [experiments](experiments/03-02_continuation_response_characterization.md)
- *What it says.* The gap between the reference and an alternative training process is exactly the sum over rounds of the effect of replacing that round's reference step by the alternative step, measured by the reference continuation value at the states the alternative actually visits. Two message laws are interchangeable for every one-boundary replacement if and only if they have the same expectation under every response. When the per-round guarantee holds only on certified sets, the probability of leaving them is paid once, not once per round.
- *Intuition.* The response $`g(y)`$ answers "if this round feeds the student $`y`$ and training then continues as declared, what is the final score"; it builds in the update rule, the rest of training and the deployment task.
- *Role.* It fixes the object of sufficiency, the response class $`\mathcal G`$. [Theorem 4.2](statements/04-02_sufficiency_duality_transfer.md#s4-2) measures its size, [Proposition 5.1](statements/05-01_optimizer_transformed_expansion.md#s5-1) expands it to first order, and [Theorem 8.2](statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) and [Theorem 8.3](statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) check it with finite data.

**[Remark 3.3](statements/03-03_response_class_derived_not_chosen.md#s3-3) · The response class is derived, not chosen** — [statement](statements/03-03_response_class_derived_not_chosen.md) · [proof](proofs/03-03_response_class_derived_not_chosen.md)
- *What it says.* $`\mathcal G`$ is generated by starting from the task score $`F_E`$, applying the reference kernels backward, and pulling back through the actual update. Enlarging the declared task, learner, suffix or snapshot family can only enlarge $`\mathcal G`$, and therefore only refine the equivalence of [Theorem 3.2(b)](statements/03-02_continuation_response_characterization.md#s3-2).
- *Role.* It fixes the response class as a property of the learner and the task, not a modelling convenience; every later sufficiency statement is relative to the declared family.

### §4 · Complexity of sufficiency — results

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T2["<b>Theorem 4.2</b><br/>Sufficiency duality<br/>and transfer"]:::b3
  T4["<b>Theorem 4.4</b><br/>Affine learners:<br/>moment order"]:::b3
  T5["<b>Theorem 4.5</b><br/>Round-independent bound,<br/>averaging rates"]:::b3
  P6["<b>Prop. 4.6</b><br/>Exact order ≠<br/>approximate complexity"]:::b3
  T7["<b>Theorem 4.7</b><br/>Update composition"]:::b3
  T2 --> T4
  T4 --> T5
  T2 --> P6
  T4 --> T7
  classDef b3 fill:#fdf3e1,stroke:#b8741a,color:#111
```

**[Definition 4.1](statements/04-01_feature_sufficiency_unresolved_discrepancy.md#s4-1) · Feature sufficiency and unresolved discrepancy** — [statement](statements/04-01_feature_sufficiency_unresolved_discrepancy.md)
- *What it is.* A feature interface summarizes a feedback law by the means of fixed features $`\phi_1,\dots,\phi_d`$ of the message. Two laws with equal feature means form a feature-matched pair; the unresolved discrepancy $`\mathfrak D_\phi(\mathcal G)`$ is the largest response difference such a pair can still produce, and the exact and approximate dimensions $`d_0`$, $`d_\varepsilon`$ count the features needed to make it $`0`$ or at most $`\varepsilon`$. These count linear expectation statistics, not bits, samples or queries.

**[Theorem 4.2](statements/04-02_sufficiency_duality_transfer.md#s4-2) · Sufficiency duality and transfer** — [statement](statements/04-02_sufficiency_duality_transfer.md) · [proof](proofs/04-02_sufficiency_duality_transfer.md) · [experiments](experiments/04-02_sufficiency_duality_transfer.md)
- *What it says.* Judge two feedback laws by the means of a fixed set of features. Equal feature means imply equal response means exactly when every response lies in the span of the features, so the number of statistics needed is $`\dim\mathrm{span}(\lbrace\mathbf 1\rbrace\cup\mathcal G)-1`$. Approximately, the worst unresolved gap is exactly twice the best uniform approximation error of the responses by the features; a response written as features plus a small residual transfers feature-mean errors with its coefficient size. An unrestricted learner needs the whole distribution.
- *Intuition.* Equal feature means rule out differences only in the directions the features span; whatever part of a response lies outside that span is risk the interface cannot see.
- *Role.* It turns "how much information must match" into computable quantities, a span dimension or an approximation width, used by every later count in §4, §5 and §8.

**[Definition 4.3](statements/04-03_binary_context_student_affine_learner.md#s4-3) · Binary-context student and convex-affine learner** — [statement](statements/04-03_binary_context_student_affine_learner.md)
- *What it is.* The computable model of §4: a student with $`k`$ binary-action contexts and action probabilities $`p\in[0,1]^k`$; deployments that are finite trees with at most $`N`$ decisions per path and expected reward $`J_E(p)`$; experience in IID blocks $`Z_i\in[0,1]^k`$; and a learner $`p_R=w_0p_0+\sum_iw_iZ_i`$ with convex weights fixed in advance. The learning value of a source is $`\mathbb EJ_E(p_R)`$.

**[Theorem 4.4](statements/04-04_affine_learners_moment_order.md#s4-4) · Affine learners: moment order is necessary and sufficient** — [statement](statements/04-04_affine_learners_moment_order.md) · [proof](proofs/04-04_affine_learners_moment_order.md) · [experiments](experiments/04-04_affine_learners_moment_order.md)
- *What it says.* For a student with binary-action contexts, deployments with at most $`N`$ decisions and learners that take convex averages of independent experience blocks, two sources are equivalent for every such task and learner exactly when their joint moments agree up to order $`N`$. Every order is necessary, the exact count on a grid is $`\binom{k+N}N-1`$, and on binary blocks fewer than $`2^k-1`$ statistics can leave a gap of 1.
- *Intuition.* The deployment value is a polynomial of degree $`N`$ in the student's action probabilities, and averaging turns its expectation into a combination of block moments of order at most $`N`$.
- *Role.* A fully computable instance of Theorem 4.2, showing that the information order is set by the deployment length.

**[Theorem 4.5](statements/04-05_round_independent_bound_averaging_rates.md#s4-5) · A round-independent bound and averaging rates** — [statement](statements/04-05_round_independent_bound_averaging_rates.md) · [proof](proofs/04-05_round_independent_bound_averaging_rates.md) · [experiments](experiments/04-05_round_independent_bound_averaging_rates.md)
- *What it says.* For convex-affine learners the learning gap is at most a coefficient mass times the largest moment mismatch, whatever the number of training rounds. When $`n`$ independent blocks are averaged, the gap decays as $`n^{-(r-1)}`$, where $`r`$ is the order of the first unmatched cumulant. Replaying one block, or training on deterministic rewrites of it, gives a gap that does not decay at all.
- *Intuition.* Only independent evidence shrinks the gap. Repetition and deterministic synthesis add exposure, not information.
- *Role.* It separates the number of independent blocks from the number of epochs. This counting discipline is used in §2 ([Theorem 2.10](statements/02-10_few_shot_information_floor.md#s2-10)) and in the method of §9 ([Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2)).

**[Proposition 4.6](statements/04-06_exact_order_not_approximate_complexity.md#s4-6) · Exact order is not approximate complexity** — [statement](statements/04-06_exact_order_not_approximate_complexity.md) · [proof](proofs/04-06_exact_order_not_approximate_complexity.md) · [experiments](experiments/04-06_exact_order_not_approximate_complexity.md)
- *What it says.* For the value $`p^N`$, matching moments up to order about $`\sqrt{N\log(1/\varepsilon)}`$ brings the gap below $`\varepsilon`$, although exact equality needs order $`N`$. For parity, no polynomial of degree below $`N`$ approximates better than $`1/2`$.
- *Intuition.* Smooth responses have good low-order approximations; combinatorial responses do not.
- *Role.* It shows that the operative complexity is the approximation width, not the degree, and leads to the coefficient-stability question answered in §8 ([Theorem 8.1](statements/08-01_stability_approximation_tradeoff.md#s8-1)).

**[Theorem 4.7](statements/04-07_update_composition_amplifies_information_requirement.md#s4-7) · Update composition amplifies the information requirement** — [statement](statements/04-07_update_composition_amplifies_information_requirement.md) · [proof](proofs/04-07_update_composition_amplifies_information_requirement.md) · [experiments](experiments/04-07_update_composition_amplifies_information_requirement.md)
- *What it says.* Composed polynomial updates can multiply the degree of the response in an early block by the degrees of all later updates, and the bound is attained. A threshold learner breaks first-moment sufficiency with a single decision. For one smooth SGD step, the difference between two feedback laws is a mean-gradient term plus a gradient-second-moment term plus a third-order remainder, and per-step errors add up linearly only under a stability condition.
- *Intuition.* The same data fed to different learners needs different amounts of information; what must match is a property of the learner composed with the task.
- *Role.* It shows that the actual optimizer must be modelled, which is the subject of §5 ([Proposition 5.1](statements/05-01_optimizer_transformed_expansion.md#s5-1)).

### §5 · Optimizer state and the first-order representation — results

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart TB
  P3["<b>Prop. 5.3</b><br/>Cold-start Adam"]:::b4
  P7["<b>Prop. 5.7</b><br/>Cold steps separate<br/>equal means"]:::b4
  P1["<b>Prop. 5.1</b><br/>First-order<br/>representation (tool)"]:::b4
  T10["<b>Theorem 5.10</b><br/>First-order statistic<br/>by learner"]:::b4
  P9["<b>Prop. 5.9</b><br/>Partially active<br/>clipping"]:::b4
  P5["<b>Prop. 5.5</b><br/>Exact Adam statistic,<br/>frozen-denominator error"]:::b4
  P6["<b>Prop. 5.6</b><br/>Warm Adam is<br/>first-order affine"]:::b4
  P8["<b>Prop. 5.8</b><br/>Evolution of warmth"]:::b4
  P3 --> P7
  P1 --> T10
  P3 --> T10
  P9 --> T10
  P5 --> T10
  P5 --> P6
  P6 --> P8
  classDef b4 fill:#f6e9f4,stroke:#9c3d8f,color:#111
```

**[Proposition 5.1](statements/05-01_optimizer_transformed_expansion.md#s5-1) · Optimizer-transformed expansion** — [statement](statements/05-01_optimizer_transformed_expansion.md) · [proof](proofs/05-01_optimizer_transformed_expansion.md) · [experiments](experiments/05-01_optimizer_transformed_expansion.md)
- *What it says.* Write any stateful first-order optimizer, clipping included, as a map from a consumed gradient to an increment of the complete state (parameters, momentum, second moments, counter). The effect of two feedback laws on the training outcome is, to first order, the sum over rounds of the inner product between the continuation-value gradient and the difference of mean state increments; the remainder is controlled by second moments of the increment. Matching the mean increment in the directions of the continuation-value gradients is sufficient to first order, necessary where the first-order term exceeds the remainder, and the number of statistics is the dimension of the linearized response class.
- *Intuition.* Feedback affects the student only through the change of state it causes. To first order only the mean of that change matters, and only in the directions the continuation value cares about.
- *Role.* A tool, of the same kind as stochastic modified equations, TracIn and metagradients (Remark 5.2). Every later statement about what a learner makes the feedback carry, and Theorems A–C, are computed from it.

**[Remark 5.2](statements/05-02_relation_to_prior_work.md#s5-2) · Relation to stochastic modified equations, TracIn, influence functions and metagradients** — [statement](statements/05-02_relation_to_prior_work.md)
- *What it says.* Proposition 5.1 is a Taylor expansion inserted into a telescoping sum, the same kind of calculation as stochastic modified equations ([Li, Tai & E, 2019](#ref-li2019stochastic)), TracIn ([Pruthi et al., 2020](#ref-pruthi2020estimating)), influence functions ([Koh & Liang, 2017](#ref-koh2017understanding)), metagradients ([Engstrom et al., 2025](#ref-engstrom2025optimizing)) and In-Run Data Shapley ([Wang et al., 2025](#ref-wang2025data)). What its bookkeeping adds is the complete state: Adam's second-moment block makes the remainder first order in the learning rate.
- *Role.* Fixes the novelty claim (none for Proposition 5.1) and explains when first-order detection needs a margin condition.

**[Proposition 5.3](statements/05-03_cold_start_adam_sign_quotient.md#s5-3) · Cold-start Adam: sign quotient and optimizer memory** — [statement](statements/05-03_cold_start_adam_sign_quotient.md) · [proof](proofs/05-03_cold_start_adam_sign_quotient.md) · [experiments](experiments/05-03_cold_start_adam_sign_quotient.md)
- *What it says.* From a zero optimizer state, Adam's first step depends only on $`g_i/(\lvert g_i\rvert+\epsilon)`$. With $`\epsilon=0`$ the joint sign law must match (sign marginals are not enough); with $`\epsilon>0`$ the whole gradient law must. Momentum keeps the first gradient's magnitude, so two histories with the same signs can take second steps in opposite directions.
- *Intuition.* Adam's first step is almost sign descent: the magnitude is divided out of the step but kept in the optimizer state.
- *Role.* Shows that moment sufficiency does not transfer to Adam, and that what must match depends on the optimizer state.

**[Definition 5.4](statements/05-04_warmth_frozen_denominator_surrogate.md#s5-4) · Warmth and the frozen-denominator surrogate** — [statement](statements/05-04_warmth_frozen_denominator_surrogate.md)
- *What it is.* The warmth $`\kappa_i=(1-\beta_2)G_i^2/(\beta_2v_i)`$ of each coordinate and the frozen-denominator surrogate step; together they measure how nonlinear a warm Adam step is in the gradient.

**[Proposition 5.5](statements/05-05_exact_first_order_statistic_adam.md#s5-5) · The exact first-order statistic of Adam and the frozen-denominator error** — [statement](statements/05-05_exact_first_order_statistic_adam.md) · [proof](proofs/05-05_exact_first_order_statistic_adam.md) · [experiments](experiments/05-05_exact_first_order_statistic_adam.md)
- *What it says.* For Adam the first-order statistic is the exact mean Adam increment, which is not determined by the mean and second moment of the gradient. Freezing the denominator at its retained value turns the step into preconditioned momentum SGD, with a per-coordinate error at most $`\eta u_i\min\lbrace\kappa_i/2,1\rbrace`$, where $`\kappa_i`$ is the warmth.
- *Intuition.* Adam divides by a running root-mean-square that the new gradient itself perturbs. When that perturbation is small (small warmth) the step is nearly linear in the gradient.
- *Role.* Separates the two approximations hidden in "warm Adam only needs $`(\mathbb E G,\mathbb E G^{\odot2})`$": the first-order expansion and the frozen denominator.

**[Proposition 5.6](statements/05-06_warm_adam_first_order_affine.md#s5-6) · Warm Adam is first-order affine in the consumed gradient** — [statement](statements/05-06_warm_adam_first_order_affine.md) · [proof](proofs/05-06_warm_adam_first_order_affine.md) · [experiments](experiments/05-06_warm_adam_first_order_affine.md)
- *What it says.* At a warm state, the one-boundary response is an affine function of the consumed gradient plus an explicit residual: a frozen-denominator term, a second-moment sensitivity term, and a curvature term. Laws with equal mean consumed gradient differ by at most the expected residuals.
- *Intuition.* Once the second moment has accumulated, one Adam step behaves like one preconditioned momentum-SGD step.
- *Role.* Says from which point of training the affine and first-order theory applies to Adam, with a computable error.

**[Proposition 5.7](statements/05-07_cold_steps_separate_equal_mean.md#s5-7) · Cold steps separate equal-mean feedback at first order** — [statement](statements/05-07_cold_steps_separate_equal_mean.md) · [proof](proofs/05-07_cold_steps_separate_equal_mean.md) · [experiments](experiments/05-07_cold_steps_separate_equal_mean.md)
- *What it says.* At a cold state two gradient laws with the same mean, or even with all moments below any fixed order equal, give responses that differ at first order in the learning rate.
- *Intuition.* A cold step sees signs relative to zero, and the sign probabilities are not functions of low-order moments.
- *Role.* The witness that the cold window is a genuine first-order effect, not a higher-order correction.

**[Proposition 5.8](statements/05-08_evolution_warmth.md#s5-8) · Evolution of warmth** — [statement](statements/05-08_evolution_warmth.md) · [proof](proofs/05-08_evolution_warmth.md) · [experiments](experiments/05-08_evolution_warmth.md)
- *What it says.* With stationary gradients the mean-field warmth is exactly $`(1-\beta_2)/(\beta_2(1-\beta_2^s))`$ after $`s`$ absorbed evaluations. It falls like $`1/(\beta_2s)`$ only while $`s(1-\beta_2)\ll1`$, then levels off at $`(1-\beta_2)/\beta_2`$. Accumulated over a run, the warmth-controlled error is at most $`c\log(1+R/W)+c(1-\beta_2)R`$; bias correction does not change warmth; learning-rate-zero warm-up advances the second moment exactly like real steps.
- *Intuition.* A fixed $`\beta_2`$ means a fixed averaging window, so every new gradient keeps perturbing the denominator by a fixed relative amount.
- *Role.* Fixes how far into training the warm description is accurate, and shows it never becomes exact under a fixed $`\beta_2`$.

**[Proposition 5.9](statements/05-09_partially_active_clipping.md#s5-9) · Partially active clipping** — [statement](statements/05-09_partially_active_clipping.md) · [proof](proofs/05-09_partially_active_clipping.md) · [experiments](experiments/05-09_partially_active_clipping.md)
- *What it says.* The mean clipped gradient splits exactly into an unclipped part and a clipped-direction part; replacing it by $`c`$ times the mean direction errs by at most $`c(1-p_{\rm act})`$, where $`p_{\rm act}`$ is the probability that clipping is active.
- *Intuition.* Only gradients below the threshold keep their magnitude; the rest contribute only their direction.
- *Role.* Makes the clipping statistic exact for any activation rate and quantifies the "only the direction matters" simplification.

**[Theorem 5.10](statements/05-10_first_order_statistic_by_learner.md#s5-10) · The first-order statistic by learner** — [statement](statements/05-10_first_order_statistic_by_learner.md) · [proof](proofs/05-10_first_order_statistic_by_learner.md) · [experiments](experiments/05-10_first_order_statistic_by_learner.md)
- *What it says.* The statistic of Proposition 5.1 for common learners: SGD and momentum — the mean gradient; warm Adam — the exact mean Adam increment, equal to $`(\mathbb EG,\mathbb EG^{\odot2})`$ only under the frozen-denominator approximation, with the second moment entering through $`v`$; a cold Adam step — the sign law in the parameters; clipping — the clipped mean; Lion — flip probabilities; LoRA at $`B=0`$ — a rank-$`r`$ sketch, with $`A`$ fixed at step one.
- *Intuition.* Two feedback sources with the same mean gradient can still differ at first order under cold Adam, clipping, or warm Adam through the second-moment channel.
- *Role.* Tells the method what to match and measure under each training recipe, and feeds the learner Jacobian of Theorem A and the state channels of Theorem B.

**[Remark 5.11](statements/05-11_checking_warm_state_conditions.md#s5-11) · Checking the warm-state conditions on a real model** — [statement](statements/05-11_checking_warm_state_conditions.md) · [experiments](experiments/05-11_checking_warm_state_conditions.md)
- *What it says.* To check the warm-state condition on a real model, report the weighted sum $`\sum_i\hat L_iu_i\min\lbrace\kappa_i/2,1\rbrace`$, its largest terms and $`\max_i\kappa_i`$, not quantiles of $`\kappa`$. Sensitivities $`\hat L_i`$ are proxies; non-additivity of the update map is not non-additivity of the response; late in training the reference warmth is the plateau $`(1-\beta_2)/\beta_2`$.
- *Role.* Turns the warm-state results into a measurement protocol for real models.

### §6 · Theorem A: the learner-relative information bound — results

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T3["<b>Theorem 6.3</b><br/>Learner-relative<br/>lower bound"]:::b4
  L4["<b>Lemma 6.4</b><br/>Design minimax"]:::b4
  T5["<b>Theorem 6.5</b><br/>Achievability by<br/>c-optimal design"]:::b4
  C7["<b>Cor. 6.7</b><br/>Aliasing; replay<br/>has zero value"]:::b4
  C8["<b>Cor. 6.8</b><br/>Demonstration–query<br/>exchange rate"]:::b4
  P10["<b>Prop. 6.10</b><br/>Water-filling<br/>allocation"]:::b4
  P11["<b>Prop. 6.11</b><br/>Learner weights:<br/>SGD vs warm Adam"]:::b4
  C12["<b>Cor. 6.12</b><br/>Learner separations"]:::b4
  T3 --> T5
  L4 --> T5
  T3 --> C7
  T3 --> C8
  T5 --> P10
  P10 --> P11
  P11 --> C12
  C7 --> C12
  classDef b4 fill:#f6e9f4,stroke:#9c3d8f,color:#111
```

**[Definition 6.1](statements/06-01_environment_family_and_distiller.md#s6-1) · Environment family, data experiment, distiller, audited states** — [statement](statements/06-01_environment_family_and_distiller.md)
- *What it is.* The environments indexed by an unknown $`\theta`$, the data (demonstrations and queries), the distillers (any map of the transcript, replay and rewriting included) and the audited learner states.
- *Role.* It fixes what Theorem A quantifies over.

**[Definition 6.2](statements/06-02_learner_relative_functional_information_norm.md#s6-2) · Learner-relative functional and information norm** — [statement](statements/06-02_learner_relative_functional_information_norm.md)
- *What it is.* The learner-relative functional $`\Psi`$ (first-order effect of the oracle feedback on the task), its gradient $`c=\sum_tJ_t^*g_t`$, the fidelity error, and the learner-relative information norm $`\mathfrak n_{K,n}`$.
- *Role.* Every bound of §6 is stated through $`\mathfrak n_{K,n}`$.

**[Theorem 6.3](statements/06-03_learner_relative_lower_bound.md#s6-3) · Learner-relative lower bound** — [statement](statements/06-03_learner_relative_lower_bound.md) · [proof](proofs/06-03_learner_relative_lower_bound.md) · [experiments](experiments/06-03_learner_relative_lower_bound.md)
- *What it says.* No distiller, replay, rewriting and synthesis included, can match the oracle feedback's first-order effect on the declared task better than a quarter of the learner-relative information norm $`\mathfrak n_{K,n}`$. The norm measures how much the $`K`$ demonstrations and $`n`$ queries reveal about the environment in the one direction $`c`$ that the learner and the task care about.
- *Intuition.* Two environments that the data cannot tell apart, but that this learner would treat differently, force an error on one of them; what counts is the learner's sensitivity $`c`$, not the size of the environment difference.
- *Role.* It is the lower half of Theorem A and the quantitative core of "acquire before compiling"; the aliasing and exchange-rate corollaries follow from it.

**[Lemma 6.4](statements/06-04_design_minimax_equality.md#s6-4) · Design minimax equality** — [statement](statements/06-04_design_minimax_equality.md) · [proof](proofs/06-04_design_minimax_equality.md) · [experiments](experiments/06-04_design_minimax_equality.md)
- *What it says.* The best variance any query design can achieve for estimating the learner-relative functional equals the square of the information norm, and an optimal design exists. It is the c-optimal (Elfving) design whose target vector is the learner Jacobian applied to the continuation-value gradients.
- *Intuition.* The best way to spend queries is the one that is most informative about the single direction the learner will use; this is a classical design problem with a learner-specific target.
- *Role.* It turns the lower bound into an acquisition rule and makes the upper bound computable.

**[Theorem 6.5](statements/06-05_achievability.md#s6-5) · Achievability; tight to constants in the scalar case** — [statement](statements/06-05_achievability.md) · [proof](proofs/06-05_achievability.md) · [experiments](experiments/06-05_achievability.md)
- *What it says.* Estimating the environment with a (slightly regularized) c-optimal query design and emitting the oracle law at the estimate reaches error $`\mathfrak n_{K,n}(1+o(1))`$. Together with the lower bound, the best achievable first-order error is pinned between $`\mathfrak n_{K,n}/4`$ and $`\mathfrak n_{K,n}`$.
- *Intuition.* Once the information is collected in the right direction, a plug-in distiller loses at most a constant factor.
- *Role.* It is the upper half of Theorem A: the information norm is the right measure of how sufficient a batch of experience is for a given learner.

**[Remark 6.6](statements/06-06_class_uniform_fidelity.md#s6-6) · Class-uniform fidelity** — [statement](statements/06-06_class_uniform_fidelity.md) · [proof](proofs/06-06_class_uniform_fidelity.md)
- *What it says.* For fidelity over a whole declared class of tasks, the lower and upper bounds of Theorem A extend and differ by at most $`4d^{3/2}`$, with $`d`$ the dimension of the class.
- *Role.* It covers declared classes of continuation values; sharper constants are not claimed.

**[Corollary 6.7](statements/06-07_aliasing_zero_value_replay.md#s6-7) · Aliasing and the zero value of replay** — [statement](statements/06-07_aliasing_zero_value_replay.md) · [proof](proofs/06-07_aliasing_zero_value_replay.md) · [experiments](experiments/06-07_aliasing_zero_value_replay.md)
- *What it says.* If two environments give identical demonstration laws and the learner-relative functional differs between them, every distiller errs by at least half the difference, for every $`K`$; replay and rewriting cannot help.
- *Intuition.* More copies of the same evidence never separate worlds that the evidence cannot separate.
- *Role.* It is the learner-relative form of passive impossibility: whether demonstrations suffice depends on the learner.

**[Corollary 6.8](statements/06-08_exchange_rate_between_demonstrations_queries.md#s6-8) · Exchange rate between demonstrations and queries** — [statement](statements/06-08_exchange_rate_between_demonstrations_queries.md) · [proof](proofs/06-08_exchange_rate_between_demonstrations_queries.md)
- *What it says.* Demonstrations and queries enter one linear information budget direction by direction; at the binding direction one demonstration is worth a definite number of queries, and that number depends on the learner.
- *Intuition.* A demonstration and a query are both evidence; their relative value is how much each reveals in the direction this learner cares about.
- *Role.* It gives the trade-off between collecting more demonstrations and spending queries.

**[Assumption 6.9](statements/06-09_state_coverage_model.md#s6-9) · State-coverage model** — [statement](statements/06-09_state_coverage_model.md)
- *What it is.* Finitely many agent states with independent oracle message laws; demonstrations supply samples at the teacher's visitation rate, queries where the distiller chooses.
- *Role.* It makes Theorem A explicit for agent distillation.

**[Proposition 6.10](statements/06-10_water_filling_allocation.md#s6-10) · Explicit information bound and optimal query allocation by water-filling** — [statement](statements/06-10_water_filling_allocation.md) · [proof](proofs/06-10_water_filling_allocation.md) · [experiments](experiments/06-10_water_filling_allocation.md)
- *What it says.* When demonstrations cover teacher states and training happens at student states, the error is governed by $`\sum_sW_s/N_s`$, with $`W_s`$ the learner-relative weight of state $`s`$ and $`N_s`$ its sample count. The optimal query allocation is water-filling, $`n_s^\star=(\sqrt{W_s/\lambda}-k_s)_+`$; a state with positive weight and no coverage leaves an irreducible error.
- *Intuition.* Queries should go where the learner is sensitive and the teacher has not been.
- *Role.* It makes Theorem A operational for agent distillation and gives the acquisition step of the method its allocation rule.

**[Proposition 6.11](statements/06-11_learner_weights_sgd_warm_adam.md#s6-11) · Learner weights of SGD and warm Adam** — [statement](statements/06-11_learner_weights_sgd_warm_adam.md) · [proof](proofs/06-11_learner_weights_sgd_warm_adam.md) · [experiments](experiments/06-11_learner_weights_sgd_warm_adam.md)
- *What it says.* For SGD the weight of a state grows like the square of its visitation; for warm Adam in the frequent regime it grows only linearly and includes a second-moment term; for very rare states Adam returns to the squared scaling but sees only the sign of the feedback.
- *Intuition.* Adam normalizes by the running second moment, so rarely visited states take steps as large as common ones; their feedback therefore matters more than under SGD.
- *Role.* It shows that the optimizer changes where information must be collected, and it supplies the constant $`\kappa_{\rm iso}`$ used again in §7.

**[Corollary 6.12](statements/06-12_learner_separations.md#s6-12) · Learner separations** — [statement](statements/06-12_learner_separations.md) · [proof](proofs/06-12_learner_separations.md) · [experiments](experiments/06-12_learner_separations.md)
- *What it says.* Demonstrations that reveal a feedback direction but not its scale are sufficient for an always-clipped learner and insufficient, for every $`K`$, for SGD. Quantitatively, a long tail of rarely covered states costs warm Adam far more information than SGD.
- *Intuition.* What a learner ignores, it does not need to be told; what it amplifies, it needs to be told precisely.
- *Role.* It is the formal content of "the same demonstrations can be sufficient for one learner and not for another".

**[Example 6.13](statements/06-13_long_tail_example.md#s6-13) · One common state and N long-tail states** — [statement](statements/06-13_long_tail_example.md)
- *What it says.* One common state and $`N`$ tail states: relative to the common state, the tail's share of the information requirement is larger under warm Adam than under SGD by $`N(1-q)/q`$ (for example $`80`$ when $`N=20`$, $`q=0.2`$).
- *Role.* It shows the size of the learner-relative amplification and why water-filling sends more queries to the tail under Adam.

### §7 · Theorem B: Adam's second-moment channel — results

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T1["<b>Theorem 7.1</b><br/>State-channel<br/>decomposition"]:::b4
  C2["<b>Cor. 7.2</b><br/>Stationary<br/>amplitude law"]:::b4
  P3["<b>Prop. 7.3</b><br/>Two proxy classes<br/>are inconsistent"]:::b4
  T1 --> C2
  C2 --> P3
  classDef b4 fill:#f6e9f4,stroke:#9c3d8f,color:#111
```

**[Theorem 7.1](statements/07-01_state_channel_decomposition.md#s7-1) · State-channel decomposition** — [statement](statements/07-01_state_channel_decomposition.md) · [proof](proofs/07-01_state_channel_decomposition.md) · [experiments](experiments/07-01_state_channel_decomposition.md)
- *What it says.* For Adam/AdamW (bias correction, decoupled weight decay, $`\epsilon`$ and global-norm clipping included), the gradient of the continuation value with respect to $`(\theta,m,v)`$ is given by an explicit reverse costate recursion. The first-order value of a message then splits exactly into a parameter channel, a momentum channel and a second-moment channel. The last depends on the message only through $`g\odot g`$.
- *Intuition.* A message changes not only the current step but also the running second moment, and through it every later step size. The second-moment channel is the price or benefit of that change.
- *Role.* It makes the optimizer-state part of [Proposition 5.1](statements/05-01_optimizer_transformed_expansion.md#s5-1) computable for Adam. It is the object behind the amplitude law ([Corollary 7.2](statements/07-02_stationary_amplitude_law.md#s7-2)) and the failure of parameter-only influence scores ([Proposition 7.3](statements/07-03_inconsistency_two_proxy_classes.md#s7-3)); with Corollary 7.2 it forms Theorem B.

**[Corollary 7.2](statements/07-02_stationary_amplitude_law.md#s7-2) · Stationary amplitude law** — [statement](statements/07-02_stationary_amplitude_law.md) · [proof](proofs/07-02_stationary_amplitude_law.md) · [experiments](experiments/07-02_stationary_amplitude_law.md)
- *What it says.* At a warm stationary state, the mean-field response to a same-direction message of amplitude $`\lambda`$ (in units of the running RMS) is an explicit finite sum, valid for every $`\lambda`$.
  - For small increments it is a concave quadratic with optimum $`\lambda^\star=(1-\beta_1^{T+1})/(\rho\,\omega_T)`$ and sign change near $`2\lambda^\star`$, with an explicit error bound.
  - For very large amplitudes the direct effect saturates. A large message ends up worse than no message exactly when the signal-to-noise ratio $`\rho`$ exceeds $`\kappa_{\rm iso}(T)/\Lambda(T)`$.
- *Intuition.* A big gradient inflates $`v`$ and so shrinks the following steps; beyond some size the shrinkage costs more than the big step gains. Under SGD there is no such effect: bigger is always better.
- *Role.* The first quantitative statement that, under Adam, the value of a message is non-monotone in its magnitude. It tells the method to control message magnitude relative to the running second moment ([Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2), step 4), and it predicts where linearized scores mislead.

**[Proposition 7.3](statements/07-03_inconsistency_two_proxy_classes.md#s7-3) · Inconsistency of two proxy classes** — [statement](statements/07-03_inconsistency_two_proxy_classes.md) · [proof](proofs/07-03_inconsistency_two_proxy_classes.md) · [experiments](experiments/07-03_inconsistency_two_proxy_classes.md)
- *What it says.* A statistic that ignores the deployment task cannot rank feedback laws correctly for every task. In the regime of Corollary 7.2, parameter-only first-order scores rank two same-direction messages in the wrong order. Examples are TracIn-style scores with Adam steps and fixed-state Adam data Shapley. Cosine scores (LESS-type) cannot separate the two messages at all.
- *Intuition.* The parameter-only score sees the bigger immediate step and misses the damage done through $`v`$; the cosine sees only direction, and the two messages point the same way.
- *Role.* A proof, not just an empirical observation, that two widely used families of influence scores are not consistent rankers under Adam. It is one reason the method selects by measured consequence ([Remark 9.3](statements/09-03_derivation_design_rules.md#s9-3), Rule 2).

**[Remark 7.4](statements/07-04_relation_prior_work_role_clipping.md#s7-4) · Relation to prior work; the role of clipping** — [statement](statements/07-04_relation_prior_work_role_clipping.md)
- *What it says.* The closest analysis of AdamW's state dynamics (Liu & Li, 2026) gives no amplitude law, optimum or threshold; In-Run Data Shapley for Adam and TracIn-style scores with Adam steps are parameter-only scores, and LESS is a cosine score, so all three fall under Proposition 7.3(b). Global-norm clipping caps the amplitude $`\lambda`$ of every message and with it the second-moment harm; this is stated as a testable mechanism, not a result.
- *Role.* It fixes what §7 claims as new, and it motivates recording the clipping rate in the method and the clipping arm of the real-model test R3 ([Remark 9.4](statements/09-04_scope_of_evidence.md#s9-4)).

**[Remark 7.5](statements/07-05_where_first_order_rankers_fail.md#s7-5) · Where first-order rankers fail** — [statement](statements/07-05_where_first_order_rankers_fail.md) · [experiments](experiments/07-05_where_first_order_rankers_fail.md)
- *What it says.* Where the second-moment channel carries a large share of the first-order difference between two candidates, the increments are large relative to $`v`$ and the response saturates, so the linearization overstates the $`v`$-channel harm. There every first-order score, including the full-state score with the exact continuation gradient, is unreliable, and candidates should be ranked by a finite paired-replacement measurement ([Theorem 8.8](statements/08-08_crn_paired_replacement_certificate.md#s8-8)) or the nonlinear amplitude formula.
- *Role.* It limits first-order screening in the method to low-share pairs with a dominant margin ([Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2), steps 4–5), and explains why first-order attribution that works under SGD can fail under Adam.

### §8 · Verification and Theorem C — results

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  T1["<b>Theorem 8.1</b><br/>Stability–approximation<br/>tradeoff"]:::b5
  T2["<b>Theorem 8.2</b><br/>Frozen selection<br/>with transport"]:::b5
  T3["<b>Theorem 8.3</b><br/>Adaptive audit at<br/>actual states"]:::b5
  P4["<b>Prop. 8.4</b><br/>Measuring the<br/>mean increment"]:::b5
  T8["<b>Theorem 8.8</b><br/>CRN paired-replacement<br/>certificate"]:::b5
  C9["<b>Cor. 8.9</b><br/>First-order version<br/>and its cap"]:::b5
  P10["<b>Prop. 8.10</b><br/>Cost and CRN"]:::b5
  T1 --> T2
  T2 --> T3
  P4 --> C9
  T8 --> C9
  T8 --> P10
  T3 --> T8
  classDef b5 fill:#e9f5f6,stroke:#1f7f8c,color:#111
```

**[Theorem 8.1](statements/08-01_stability_approximation_tradeoff.md#s8-1) · Stability–approximation tradeoff** — [statement](statements/08-01_stability_approximation_tradeoff.md) · [proof](proofs/08-01_stability_approximation_tradeoff.md) · [experiments](experiments/08-01_stability_approximation_tradeoff.md)
- *What it says.* When each feature mean is known only up to a tolerance, the largest response difference that cannot be ruled out equals the best tradeoff between the approximation error of the response by the features and the size of the approximation's coefficients, each coefficient weighted by its feature's tolerance.
- *Intuition.* The cost of measuring a feature imprecisely is its coefficient in the response model. A fit with a small residual and huge coefficients proves nothing.
- *Role.* It decides which features an audit should measure and which response models are acceptable as certificates. The frozen audit of Theorem 8.2 uses its linear term, and the fitting rule of the method minimizes its right side.

**[Theorem 8.2](statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) · Frozen finite-interface selection with transport** — [statement](statements/08-02_frozen_finite_interface_selection_transport.md) · [proof](proofs/08-02_frozen_finite_interface_selection_transport.md) · [experiments](experiments/08-02_frozen_finite_interface_selection_transport.md)
- *What it says.* Fix audit cells and features in advance, measure each candidate's feature discrepancy from the reference, and select the smallest. The selected candidate's learning gap is bounded by the measured discrepancy plus statistical error plus an explicit transport term for the states training actually visits, plus the probability of leaving the certified states. The transport term cannot be dropped.
- *Intuition.* An audit sees only the states it covers. If training goes elsewhere, the audit can report zero discrepancy while the learning gap is maximal.
- *Role.* It gives the correct form of any finite, frozen audit and identifies its blind spot, which the actual-state audits of Theorems 8.3 and 8.8 avoid.

**[Theorem 8.3](statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) · Actual-state adaptive audit; fidelity versus improvement** — [statement](statements/08-03_actual_state_adaptive_audit_fidelity.md) · [proof](proofs/08-03_actual_state_adaptive_audit_fidelity.md) · [experiments](experiments/08-03_actual_state_adaptive_audit_fidelity.md)
- *What it says.* At the states training actually reaches, run paired full-tail experiments for each candidate against the reference and select by a lower confidence bound. The resulting training procedure is never worse than the reference by more than the confidence level, and it keeps the certified improvement. The required number of paired replicas is explicit and large.
- *Intuition.* It measures directly whether replacing this one round helps, with no extrapolation to unseen states; the price is many full training tails.
- *Role.* It is the reliable but expensive reference audit, used to calibrate cheaper measurements such as the surrogate of Theorem 8.8.

**[Proposition 8.4](statements/08-04_measuring_mean_increment_actual_states.md#s8-4) · Measuring the mean increment at the actual states** — [statement](statements/08-04_measuring_mean_increment_actual_states.md) · [proof](proofs/08-04_measuring_mean_increment_actual_states.md) · [experiments](experiments/08-04_measuring_mean_increment_actual_states.md)
- *What it says.* The mean optimizer increment of a feedback law can be estimated from gradient evaluations alone, with an error that falls like one over the square root of the sample size in the Hilbert norm. Combined with the continuation's gradient norm and curvature, this bounds the one-round effect and, along sampled runs, the whole-run gap.
- *Intuition.* The data side of a first-order certificate is cheap; the continuation side (its gradient and curvature) is not measured by these samples at all.
- *Role.* It is the cheap half of first-order auditing and explains why first-order audits still need calibration constants.

**[Remark 8.5](statements/08-05_norms_constants.md#s8-5) · Norms and constants** — [statement](statements/08-05_norms_constants.md) · [experiments](experiments/08-05_norms_constants.md)
- *What it says.* The bound of Proposition 8.4 holds in the Hilbert norm of the state space and not in arbitrary norms with the same constant; the increment bound can grow with dimension (cold-start Adam moves every coordinate); a ranking needs only a scalar projection; and Theorem 8.8 needs no increment bound at all.
- *Role.* It keeps the first-order measurement honest about its constants and points to the finite measurement of Theorem 8.8.

**[Definition 8.6](statements/08-06_crn_paired_replacement_score.md#s8-6) · CRN paired-replacement score** — [statement](statements/08-06_crn_paired_replacement_score.md)
- *What it is.* The CRN paired-replacement score of an ordered pair of candidates at an audited state: one round with each candidate, followed by a shared surrogate tail, scored by a surrogate objective, averaged over independent replicas.
- *Role.* It is the measurement of Theorem 8.8 and its first-order variant in Corollary 8.9, and the selection statistic of the method.

**[Assumption 8.7](statements/08-07_certificate_assumptions.md#s8-7) · Independent replicas, bounded scores, surrogate alignment** — [statement](statements/08-07_certificate_assumptions.md)
- *What it is.* Independence of replicas given the history, a known a-priori range of each paired score, and alignment of surrogate differences with task differences up to a factor and an error.
- *Role.* These are the only assumptions of Theorem 8.8; the range and the alignment are exactly the two open problems of Remark 8.12.

**[Theorem 8.8](statements/08-08_crn_paired_replacement_certificate.md#s8-8) · CRN paired-replacement certificate; Theorem C** — [statement](statements/08-08_crn_paired_replacement_certificate.md) · [proof](proofs/08-08_crn_paired_replacement_certificate.md) · [experiments](experiments/08-08_crn_paired_replacement_certificate.md)
- *What it says.* At an audited state, give each candidate one round and then run the same surrogate tail with the same random numbers for every candidate. The mean paired score difference estimates the finite effect of the replacement without bias, and an empirical Bernstein radius plus a surrogate-calibration term turns it into a certificate that holds for all candidate pairs at once.
- *Intuition.* Instead of linearizing and then bounding the error of the linearization, measure the finite effect directly; sharing the random numbers cancels the tail noise that two independent runs would add.
- *Role.* It replaces the oracle continuation gradient by a measurement, and it is the selection step of the method. Its rigor rests on a known score range and a calibrated surrogate, which are the open problems of Remark 8.12.

**[Corollary 8.9](statements/08-09_first_order_version_margin_criterion.md#s8-9) · The first-order version, its margin criterion, and its cap** — [statement](statements/08-09_first_order_version_margin_criterion.md) · [proof](proofs/08-09_first_order_version_margin_criterion.md) · [experiments](experiments/08-09_first_order_version_margin_criterion.md)
- *What it says.* A first-order score ranks two candidates correctly whenever their first-order difference exceeds the second-order remainder. Measured by finite differences, the first-order certificate pays a finite-difference term and the remainder on top of the statistical radius, and it can never certify a pair whose first-order term is below the remainder, whatever the true gap.
- *Intuition.* Candidates that differ mainly through second moments of their increments, as under Adam, are invisible to any first-order certificate.
- *Role.* It marks where gradient-based scores are trustworthy and why Theorem 8.8 measures finite replacement instead; it also separates oracle diagnostics from measurable certificates.

**[Proposition 8.10](statements/08-10_cost_and_crn.md#s8-10) · Cost, and what common random numbers buy** — [statement](statements/08-10_cost_and_crn.md) · [proof](proofs/08-10_cost_and_crn.md) · [experiments](experiments/08-10_cost_and_crn.md)
- *What it says.* The certificate costs one surrogate tail per candidate and replica. Its range term needs on the order of a hundred replicas to be small, and common random numbers reduce the variance of each paired score to the size of the difference between the two candidates' increments.
- *Intuition.* Paired tails with shared randomness differ only because the candidates differ; independent tails also differ because training is noisy.
- *Role.* It sets the replica budget of the method and points to the pathwise Lipschitz bound as the route to a provable range.

**[Remark 8.11](statements/08-11_surrogate_calibration_and_transfer.md#s8-11) · Calibrating the surrogate; transfer along training** — [statement](statements/08-11_surrogate_calibration_and_transfer.md) · [experiments](experiments/08-11_surrogate_calibration_and_transfer.md)
- *What it says.* The alignment constants are fitted by regressing paired full-tail task contrasts on surrogate scores; transferring them to new pairs is an exchangeability assumption, not a theorem; inside the adaptive audit the certified bound of Theorem 8.8 can replace the full-tail bound with Theorem 8.3(c) unchanged; using a selection later needs transport or a re-audit.
- *Role.* It connects Theorem C to the safe adaptive audit and states the calibration-transfer condition that Remark 8.12 leaves open.

**[Remark 8.12](statements/08-12_open_problems.md#s8-12) · Open problems** — [statement](statements/08-12_open_problems.md) · [experiments](experiments/08-12_open_problems.md)
- *What it says.* Two open problems: a provable score range much smaller than the a-priori range, for example from a certified pathwise Lipschitz constant; and a testable calibration-transfer condition. Until both are settled, Theorem 8.8 is in practice an unbiased, low-variance ranking score, and a rigorous certificate only where the range and the calibration are justified independently.
- *Role.* It states precisely what remains to make the measurable certificate both rigorous and useful.

### §9 · Mechanism and method — results

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"nodeSpacing": 24, "rankSpacing": 34, "padding": 8, "wrappingWidth": 260}}}%%
flowchart LR
  P1["<b>Prop. 9.1</b><br/>Coupling width"]:::b6
  A2["<b>Algorithm 9.2</b><br/>CED"]:::b6
  P1 --> A2
  classDef b6 fill:#fbeaea,stroke:#b53b3b,color:#111
```

**[Proposition 9.1](statements/09-01_coupling_width.md#s9-1) · Coupling width** — [statement](statements/09-01_coupling_width.md) · [proof](proofs/09-01_coupling_width.md) · [experiments](experiments/09-01_coupling_width.md)
- *What it says.* Fix the marginal law of each feedback unit and change only how units are grouped into joint training packets. Every such regrouping leaves the learning outcome unchanged exactly when the joint-packet response is a sum of per-unit contributions; otherwise the change is at most twice the distance of the response from additive tables. One SGD step at a common parameter is exactly invariant to regrouping; sequential steps are not.
- *Intuition.* If the learner "sees" each unit on its own, grouping is irrelevant. Grouping matters only through interactions between units, and those appear once one update feeds into the next.
- *Role.* A mechanism test with a strict null, and the reason a candidate feedback law is specified by the packets it emits, not by a bag of units ([Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2), step 3).

**[Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2) · Consequence-Calibrated Experience Distillation, CED** — [statement](statements/09-02_ced_algorithm.md) · [proof](proofs/09-02_ced_algorithm.md) · [experiments](experiments/09-02_ced_algorithm.md)
- *What it says.* Record the complete optimizer state; acquire missing information first, spending queries where the learner-relative weight is high and teacher coverage is low; compile candidate feedback laws as packet laws; record each candidate's learner statistic and flag messages that are too large relative to the running second moment; measure each candidate's consequence at the actual state by CRN paired replacement, calibrated against a few full-tail audits; select by lower confidence bound, re-audit, freeze, then test method and policy transfer separately. Each output carries a stated guarantee (G1–G3) or is labelled empirical (G4).
- *Intuition.* Choose feedback by what happens after this learner, in its current state, consumes it, not by how good the feedback looks on its own.
- *Role.* Where the framework lands: every step is required by a statement of §2–§8, and the planned real-model tests R0–R3 ([Remark 9.4](statements/09-04_scope_of_evidence.md#s9-4)) test those steps.

**[Remark 9.3](statements/09-03_derivation_design_rules.md#s9-3) · Derivation of the design rules** — [statement](statements/09-03_derivation_design_rules.md) · [proof](proofs/09-03_derivation_design_rules.md) · [experiments](experiments/09-03_derivation_design_rules.md)
- *What it says.* Six rules behind CED, each a necessary condition for the corresponding guarantee and derived from cited statements: acquire before compiling; select by measured consequence; match the learner's statistic, not the raw gradient; fit response models with stable coefficients; account for coverage, certify by paired replacement and use first-order audits only where the margin dominates; freeze, then test transfer. In the controlled learner, the static proxies tested did not reliably select, while measured responses did.
- *Role.* It explains why Algorithm 9.2 has the steps it has; each rule closes one way a method can claim more than it knows, and an ablation of the method should remove one rule at a time.

**[Remark 9.4](statements/09-04_scope_of_evidence.md#s9-4) · What the controlled evidence covers, and the planned real-model tests** — [statement](statements/09-04_scope_of_evidence.md)
- *What it says.* All evidence in the framework is exact finite computation or a controlled small network; no statement is supported by a real-model or benchmark result. The pre-registered real-model tests on ALFWorld ([Shridhar et al., 2021](#ref-shridhar2021alfworld)) are R0 (warm-state conditions and response non-additivity), R1 (Theorem C: same-state ranking of four feedback laws by the CRN score), R2 (Theorem A: learner-relative query allocation against uniform and SGD-weighted allocation) and R3 (Theorem B: one batch at several magnitudes, clipping off and on).
- *Role.* It marks the boundary of the claims and ties each planned test to the statement it tests, so real-model results can be read against predictions written in advance.

---

## 1. Setting and notation

The objects below are used by every section. Section-specific definitions are numbered within their section.

<a id="s1-1"></a>

**Definition 1.1 (Environment, student class, reference class).**

- An environment $`E`$ comprises a task distribution, transitions, feedback semantics, scoring and a fixed execution harness. All returns lie in $`[0,1]`$, and $`J_E(\pi)`$ is the expected return of policy $`\pi`$.
- The student's deployable policies form a class $`\Pi\subseteq\Pi_{\rm ref}`$, where $`\Pi_{\rm ref}`$ is a reference class (for example, the policies the teacher could execute). Write

```math
J_E^\Pi=\max_{\pi\in\Pi}J_E(\pi),\qquad J_E^{\rm ref}=\max_{\pi\in\Pi_{\rm ref}}J_E(\pi).
```

<a id="s1-2"></a>

**Definition 1.2 (Information and distillers).**

- $`I_0`$ is the common initial information: the pretrained student, the known rules, and the $`K`$ trusted demonstrations per task family. The teacher model's pretrained knowledge is part of $`I_0`$; it is the same in every environment of a family.
- $`D`$ is the transcript of all further information obtained from permitted experiments: teacher calls on new states, environment branches, resets. A new teacher call that observes a new state is an experiment, i.e. part of $`D`$.
- A *distiller* is any measurable map from $`(I_0,D)`$ and private randomness (independent of the environment) to a feedback protocol (Definition 1.3). Replay, rewriting and synthesis are distillers. They obtain nothing beyond $`(I_0,D)`$ that depends on the environment.

<a id="s1-3"></a>

**Definition 1.3 (Complete learner state, feedback law, update).**

- The *complete learner state* is $`z=(\theta,o)\in\mathcal Z`$. Here $`\theta`$ is the parameter vector and $`o`$ is the optimizer state (for Adam, the moments $`m,v`$ and the step counter), together with buffers, sampler state and the history that the update uses. $`\mathcal Z`$ is a real Hilbert space with inner product $`\langle\cdot,\cdot\rangle`$, Euclidean in finite dimension.
- At round $`t`$ a *feedback law* $`P_t(\cdot\mid z)`$ emits a message $`Y_t`$. The message yields a *consumed gradient* $`G_t(z,Y_t)`$, defined after accumulation, normalization and any clipping that the stated optimizer includes. The actual update is $`Z_{t+1}=U_t(z,Y_t,\xi_t)`$, with update randomness $`\xi_t`$.
- For a first-order optimizer, $`U_t(z,y,\xi)=z+\Delta_t\big(z,G_t(z,y,\xi)\big)`$. The *increment map* $`\Delta_t`$ contains the learning-rate schedule, momentum, second moments, weight decay and clipping. The *mean increment* of a law $`P`$ is

```math
\mu^P_t(z)=\mathbb E_{Y\sim P_t(\cdot\mid z)}\,\Delta_t\big(z,G_t(z,Y)\big).
```

- A *protocol* $`S`$ is a sequence of feedback laws $`(P^S_t)_{t<R}`$. Integrating $`\xi_t`$ gives its learning kernel $`K^S_t`$. The reference protocol is written $`K^0_t`$. All protocols start from the same initial state law.
- The final student is scored by $`F_E(z)=J_E(\pi_z)\in[0,1]`$, and $`L_E(S)=\mathbb E\,F_E(Z^S_R)`$. For a protocol that uses a single law $`P`$ in every round we write $`L(P)`$.

<a id="s1-4"></a>

**Definition 1.4 (Continuation value, response, declared class).**

- The *continuation value* under the reference tail is

```math
V^0_R=F_E,\qquad V^0_t=K^0_tV^0_{t+1}\quad(t<R).
```

- $`V^0_{t+1}(z)`$ is the expected final score when training continues from state $`z`$ after round $`t`$ under the reference protocol. When the tail starts from a different protocol $`P`$ we write $`V^P_{t+1}`$.
- The *one-boundary response* of message $`y`$ at round $`t`$ and state $`z`$ is

```math
g_{t,z}(y)=\mathbb E_{\xi_t}\,V^0_{t+1}\big(U_t(z,y,\xi_t)\big).
```

- It is the expected final real-task return after receiving $`y`$, applying the actual update, and completing the declared reference tail.
- The *declared class* $`\mathcal V`$ collects the continuation values of the declared deployment tasks and reference tails. The *response class* $`\mathcal G`$ collects the corresponding responses. Both are fixed before any feedback is selected.

<a id="s1-5"></a>

**Definition 1.5 (Counts).** The following counts are never conflated.

| Symbol | Meaning |
|---|---|
| $`K`$ | trusted demonstrations per task family |
| $`n_{\rm query}`$, or $`n`$ in §6 | further permitted experiments (queries) |
| $`n`$ in §4 | independent experience blocks |
| $`R`$ | training updates |
| $`N`$ | decisions in one deployment episode |
| $`d`$ | linear statistics (features) |
| $`r`$, $`M`$ in §8 | CRN replicas per audited state, number of candidate feedback laws |

Replaying one block $`n`$ times is not $`n`$ independent blocks.

**Conventions.**

- Statements are numbered by section with one shared counter (Definition 5.4 is followed by Proposition 5.5).
- *Theorem A* denotes Theorems 6.3 and 6.5, *Theorem B* Theorem 7.1 and Corollary 7.2, and *Theorem C* Theorem 8.8.
- Where a letter is reused locally, the section says so on first use; for example, $`s`$ counts absorbed gradient evaluations in §5 and denotes an agent state in §6.

---

## Status of the results

| Section | Statements | Status |
|---|---|---|
| §2 | [Theorem 2.2](statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2), [Propositions 2.3–2.4](statements/02-03_robust_ambiguity_duality.md#s2-3), [Theorem 2.6](statements/02-06_passive_impossibility_active_attainability.md#s2-6), [Proposition 2.7](statements/02-07_compilation_cost.md#s2-7), [Lemma 2.9](statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9), [Theorem 2.10](statements/02-10_few_shot_information_floor.md#s2-10), [Proposition 2.11](statements/02-11_shared_information_budget.md#s2-11) | proved |
| §3 | [Proposition 3.1](statements/03-01_static_value_not_learning_value.md#s3-1), [Theorem 3.2](statements/03-02_continuation_response_characterization.md#s3-2) | proved |
| §4 | [Theorems 4.2](statements/04-02_sufficiency_duality_transfer.md#s4-2), [4.4](statements/04-04_affine_learners_moment_order.md#s4-4), [4.5](statements/04-05_round_independent_bound_averaging_rates.md#s4-5), [Proposition 4.6](statements/04-06_exact_order_not_approximate_complexity.md#s4-6), [Theorem 4.7](statements/04-07_update_composition_amplifies_information_requirement.md#s4-7) | proved |
| §5 | [Propositions 5.1](statements/05-01_optimizer_transformed_expansion.md#s5-1), [5.3](statements/05-03_cold_start_adam_sign_quotient.md#s5-3), [5.5–5.9](statements/05-05_exact_first_order_statistic_adam.md#s5-5), [Theorem 5.10](statements/05-10_first_order_statistic_by_learner.md#s5-10) | proved (finite dimension, explicit assumptions); Proposition 5.1 is a standard tool |
| §6 (Theorem A) | [Theorem 6.3](statements/06-03_learner_relative_lower_bound.md#s6-3), [Lemma 6.4](statements/06-04_design_minimax_equality.md#s6-4), [Theorem 6.5](statements/06-05_achievability.md#s6-5), [Corollaries 6.7–6.8](statements/06-07_aliasing_zero_value_replay.md#s6-7), [Proposition 6.10](statements/06-10_water_filling_allocation.md#s6-10), [Corollary 6.12](statements/06-12_learner_separations.md#s6-12) | proved (local asymptotic minimax for 6.3(b), 6.5) |
| §6 | [Proposition 6.11](statements/06-11_learner_weights_sgd_warm_adam.md#s6-11) | proved to leading order in the stated visitation regimes |
| §7 (Theorem B) | [Theorem 7.1](statements/07-01_state_channel_decomposition.md#s7-1), [Proposition 7.3](statements/07-03_inconsistency_two_proxy_classes.md#s7-3) | proved |
| §7 | [Corollary 7.2](statements/07-02_stationary_amplitude_law.md#s7-2) | proved in the stationary mean-field model: (a) exact formula, (b) quadratic law with an explicit error bound, (c) explicit limit |
| §8 (Theorem C) | [Theorems 8.1–8.3](statements/08-01_stability_approximation_tradeoff.md#s8-1), [Proposition 8.4](statements/08-04_measuring_mean_increment_actual_states.md#s8-4), [Theorem 8.8](statements/08-08_crn_paired_replacement_certificate.md#s8-8), [Corollary 8.9](statements/08-09_first_order_version_margin_criterion.md#s8-9), [Proposition 8.10](statements/08-10_cost_and_crn.md#s8-10) | proved; the surrogate calibration of [Remark 8.11](statements/08-11_surrogate_calibration_and_transfer.md#s8-11) is an assumption |
| §9 | [Proposition 9.1](statements/09-01_coupling_width.md#s9-1); [Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2) and its design rules | proved; design rules derived |

**Open problems.**

- **A tight provable range for Theorem C.** The certificate is valid with the a-priori range of the surrogate. To be decisive it needs a provable range much smaller than that, for example from a pathwise Lipschitz bound on the reference tail ([Remark 8.12](statements/08-12_open_problems.md#s8-12)).
- **Transfer of the surrogate calibration.** Theorem C uses a calibration $`(\alpha,\varepsilon_{\rm sur})`$ fitted on some candidate pairs. Extending it to new pairs needs a testable exchangeability condition, which is not yet available ([Remarks 8.11–8.12](statements/08-11_surrogate_calibration_and_transfer.md#s8-11)).
- **Real-model evidence.** The experiments pages contain exact finite computations and controlled small networks. Tests of Theorems A–C on a real agent-distillation recipe are designed in [Remark 9.4](statements/09-04_scope_of_evidence.md#s9-4); the first runs are under way and their results are not yet on these pages.

**Checks.** Every experiments page opens with a verdict table. One check fails its pre-set tolerance: a boundary setting of Corollary 7.2(a), recorded on the [experiments page of Corollary 7.2](experiments/07-02_stationary_amplitude_law.md).

## Reproducing the checks

From the repository root, CPU only:

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/independent_checks/check_v014_independent.py --output theory/independent_checks/results_independent_v014.json
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --output theory/v015_checks/results_v015_independent.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 theory/.venv/bin/python theory/v016_checks/check_v016.py
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/figures/make_figures.py
```

Check code IDs on the experiments pages name the suite and the check:

| Prefix | Suite | Location |
|---|---|---|
| `v014:` | independent checks `C1`–`C23` | `theory/independent_checks` |
| `v015:` | exact finite checks `V1`–`V9` and controlled-network experiments `E0`–`E12` | `theory/v015_checks` |
| `pkg:` | checks shipped with the revision package (`T01`–`T13` and named package checks) | `docs/direction/v014_handoff`, `docs/direction/v015_revision` |
| `v016:` | checks of §5–§8 additions (`C1`–`C6b`) | `theory/v016_checks` |

The page-rendering checker is described in [tools/README.md](tools/README.md); figure provenance is in [figures/README.md](figures/README.md).

## Appendix · Sources

Each statement was assembled from the source documents below; links go to the exact sections.

| This framework | Source |
|---|---|
| [Theorem 2.2](statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2), [Propositions 2.3–2.4](statements/02-03_robust_ambiguity_duality.md#s2-3) | [v0.14.1 §2](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#2-first-principles-feasibility-identification-and-use) |
| [Theorem 2.6](statements/02-06_passive_impossibility_active_attainability.md#s2-6), [Proposition 2.7](statements/02-07_compilation_cost.md#s2-7) | [v0.14.1 §3](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#3-a-complete-acquisition-example-perfect-demonstrations-can-be-insufficient) |
| [Lemma 2.9](statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9), [Theorem 2.10](statements/02-10_few_shot_information_floor.md#s2-10), [Proposition 2.11](statements/02-11_shared_information_budget.md#s2-11) | [v0.15 §4](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#4-make-the-number-of-demonstrations-explicit) |
| [Proposition 3.1](statements/03-01_static_value_not_learning_value.md#s3-1) | [v0.14.1 §4](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#4-why-static-evaluation-is-not-learning-sufficiency) |
| [Theorem 3.2](statements/03-02_continuation_response_characterization.md#s3-2), [Remark 3.3](statements/03-03_response_class_derived_not_chosen.md#s3-3) | [v0.14.1 §5](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#51-complete-learning-state-and-continuation-response); [v0.15 §1](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#1-what-the-validation-changes) |
| [Theorem 4.2](statements/04-02_sufficiency_duality_transfer.md#s4-2) | [v0.14.1 §6](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#6-exact-and-approximate-information-complexity) |
| [Theorem 4.4](statements/04-04_affine_learners_moment_order.md#s4-4) | [v0.14.1 §7](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#7-an-explicit-closed-theory-deployment-dependence-composed-with-learning) |
| [Theorem 4.5](statements/04-05_round_independent_bound_averaging_rates.md#s4-5) | [v0.14.1](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-14-a-discrepancy-bound-independent-of-training-rounds) |
| [Proposition 4.6](statements/04-06_exact_order_not_approximate_complexity.md#s4-6) | [v0.14.1 §8](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#8-exact-order-is-not-approximate-information-complexity) |
| [Theorem 4.7](statements/04-07_update_composition_amplifies_information_requirement.md#s4-7) | [v0.14.1 §9](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#9-learning-dynamics-can-amplify-the-information-requirement); [v0.15 §7](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#7-experiments-that-discriminate-the-theory) |
| [Proposition 5.1](statements/05-01_optimizer_transformed_expansion.md#s5-1), [Remark 5.2](statements/05-02_relation_to_prior_work.md#s5-2) | [OTLR §2–§3, §6](../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#2-the-expansion); [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Proposition 5.3](statements/05-03_cold_start_adam_sign_quotient.md#s5-3) | [v0.15 §3](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#3-what-adam-requires-three-distinct-levels-of-sufficiency) |
| [Definition 5.4](statements/05-04_warmth_frozen_denominator_surrogate.md#s5-4), [Propositions 5.5–5.7](statements/05-05_exact_first_order_statistic_adam.md#s5-5) | [warm-Adam §2–§3](../docs/direction/v015_extension/Warm_Adam_Response_EN.md#2-warm-one-step-approximation); [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Proposition 5.8](statements/05-08_evolution_warmth.md#s5-8) | [warm-Adam §4](../docs/direction/v015_extension/Warm_Adam_Response_EN.md#4-how-warmth-evolves-and-a-free-way-to-warm-the-state); [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Proposition 5.9](statements/05-09_partially_active_clipping.md#s5-9), [Remark 5.11](statements/05-11_checking_warm_state_conditions.md#s5-11) | [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Theorem 5.10](statements/05-10_first_order_statistic_by_learner.md#s5-10) | [OTLR §4](../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#4-what-each-optimizer-makes-the-feedback-carry); [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [§6](statements/06-01_environment_family_and_distiller.md), Definitions 6.1 – Example 6.13 | [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [§7](statements/07-01_state_channel_decomposition.md), Theorem 7.1 – Remark 7.5 | [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Theorem 8.1](statements/08-01_stability_approximation_tradeoff.md#s8-1) | [v0.14.1](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-18-an-exact-stabilityapproximation-tradeoff) |
| [Theorem 8.2](statements/08-02_frozen_finite_interface_selection_transport.md#s8-2) | [v0.14.1](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#theorem-19-frozen-finite-interface-selection); [v0.15 §2](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#2-the-corrected-finite-audit-bridge) |
| [Theorem 8.3](statements/08-03_actual_state_adaptive_audit_fidelity.md#s8-3) | [v0.14.1 §11](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#11-fidelity-improvement-and-adaptive-verification); [v0.15 §5](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#5-how-the-new-pieces-compose) |
| [Proposition 8.4](statements/08-04_measuring_mean_increment_actual_states.md#s8-4), [Remark 8.5](statements/08-05_norms_constants.md#s8-5) | [OTLR §5](../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#5-using-the-statistic-at-the-actual-states); [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Corollary 8.9](statements/08-09_first_order_version_margin_criterion.md#s8-9) | [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Definition 8.6 – Remark 8.12](statements/08-06_crn_paired_replacement_score.md#s8-6) | [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |
| [Proposition 9.1](statements/09-01_coupling_width.md#s9-1) | [v0.14.1 §12](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#12-one-derived-mechanism-coupling-valid-experience-units) |
| [Algorithm 9.2](statements/09-02_ced_algorithm.md#s9-2), [Remarks 9.3–9.4](statements/09-03_derivation_design_rules.md#s9-3) | [v0.14.1 §13](../docs/direction/v015_revision/Environment_Information_Theory_EN.md#13-the-derived-few-shot-method-consequence-calibrated-experience-distillation); [v0.15 §6](../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#6-method-consequence-an-optimizer-aware-instance-of-ced); [v0.16 revision draft](../docs/direction/2026-10-11_v016_revision_ZH.md) |

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](references.md).

<a id="ref-chen2023symbolic"></a>

**Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le (2023).** Symbolic discovery of optimization algorithms. *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 49205–49233. arXiv [2302.06675](https://arxiv.org/abs/2302.06675).

<a id="ref-elfving1952optimum"></a>

**Gustav Elfving (1952).** Optimum allocation in linear regression theory. *The Annals of Mathematical Statistics* 23, no. 2, 255–262. DOI [10.1214/aoms/1177729442](https://doi.org/10.1214/aoms/1177729442).

<a id="ref-engstrom2025optimizing"></a>

**Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry (2025).** Optimizing ML training with metagradient descent. arXiv preprint. arXiv [2503.13751](https://arxiv.org/abs/2503.13751).

<a id="ref-glasserman2004monte"></a>

**Paul Glasserman (2004).** Monte Carlo Methods in Financial Engineering. Springer, New York (Stochastic Modelling and Applied Probability 53). DOI [10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1).

<a id="ref-hu2022lora"></a>

**Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022).** LoRA: Low-rank adaptation of large language models. *The Tenth International Conference on Learning Representations (ICLR)*. [openreview.net/forum?id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9); arXiv [2106.09685](https://arxiv.org/abs/2106.09685).

<a id="ref-kingma2015adam"></a>

**Diederik P. Kingma, Jimmy Ba (2015).** Adam: A method for stochastic optimization. *3rd International Conference on Learning Representations (ICLR)*. arXiv [1412.6980](https://arxiv.org/abs/1412.6980).

<a id="ref-koh2017understanding"></a>

**Pang Wei Koh, Percy Liang (2017).** Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:1885–1894. [proceedings.mlr.press/v70/koh17a](https://proceedings.mlr.press/v70/koh17a.html).

<a id="ref-li2019stochastic"></a>

**Qianxiao Li, Cheng Tai, Weinan E (2019).** Stochastic modified equations and dynamics of stochastic gradient algorithms I: Mathematical foundations. *Journal of Machine Learning Research* 20, no. 40, 1–47. [jmlr.org/papers/v20/17-526](https://jmlr.org/papers/v20/17-526.html).

<a id="ref-pruthi2020estimating"></a>

**Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan (2020).** Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems 33 (NeurIPS)*, 19920–19930. arXiv [2002.08484](https://arxiv.org/abs/2002.08484).

<a id="ref-pukelsheim1993optimal"></a>

**Friedrich Pukelsheim (1993).** Optimal Design of Experiments. Wiley, New York (Wiley Series in Probability and Mathematical Statistics); reprinted as SIAM Classics in Applied Mathematics 50 (2006). DOI of the reprint [10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109).

<a id="ref-shridhar2021alfworld"></a>

**Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, Matthew Hausknecht (2021).** ALFWorld: Aligning text and embodied environments for interactive learning. *The Ninth International Conference on Learning Representations (ICLR)*. arXiv [2010.03768](https://arxiv.org/abs/2010.03768).

<a id="ref-wang2025data"></a>

**Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia (2025).** Data Shapley in one training run. *13th International Conference on Learning Representations (ICLR)*. arXiv [2406.11011](https://arxiv.org/abs/2406.11011).
