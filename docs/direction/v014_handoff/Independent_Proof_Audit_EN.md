# Independent Mathematical Audit of the Environment–Learner Information Framework

**Version reviewed:** Environment_Information_Theory_v013_ZH.md, 813 lines, SHA-256 ec2648ad01b80a28e6c5476b0bb515672d183add67af12bf1b6ac95d436ab4e3.  
**Audit date:** 2026-10-10.  
**Review type:** independent model-agent mathematical review. The reviewer rederived the statements from the definitions, attempted counterexamples, and performed a separate exact-arithmetic check. This is neither external human peer review nor formal verification. The source was not edited.

## Overall assessment

The finite mathematical core is internally consistent. I did not identify a false central proposition under its explicitly stated finite, bounded, independent-block, and fixed-update assumptions. Several claims become false if those assumptions are dropped. The most important repairs concern the semantics of calibrated candidate selection, the scope of moment order, and the interpretation of fidelity as improvement.

The source already distinguishes many known tools from proposed contributions. That distinction should be retained and strengthened. The Bayesian decomposition, testing lower bound, learning-process telescoping identity, feature-span criterion, and finite-candidate concentration inequality are standard arguments. The moment hierarchy, composition degree bound, and exposure law are correct consequences of polynomial composition and cumulant identities. They can support a useful application-specific contribution, but mathematical correctness is not evidence that they are independently novel or sufficiently substantial for a major conference contribution.

The framework has a complete exact answer for a restricted affine learner class. It does not yet have a computationally useful, unconditional certificate for neural agent training. In the neural setting, coverage and continuation-function approximation are substantive unknowns. Held-out predictive success can test the relevance of proposed metrics; it cannot remove those mathematical assumptions.

**Recommended disposition:** accept the finite results after the clarifications below; present the neural method as a separately specified algorithm with falsifiable predictions and conditional guarantees. Do not label the entire package a completed general theory of agent distillation.

## Review conventions

- **Valid:** the statement follows from the declared assumptions.
- **Valid with explicit conditions:** the derivation is sound, but an easy-to-miss condition must appear in the theorem, not only elsewhere.
- **Interpretation repair:** the mathematics is correct, but a stronger natural-language reading is false.
- **Open:** a bridge required for the proposed neural application has not been proved.
- Equation numbers and section numbers below refer to v0.13.
- Let \(L_E(S)\) always mean expected true-environment deployment utility after the declared learner receives a fresh training stream from the fixed interface \(S\). Any different convention must be defined separately.

## 1. Decision information and the perfect-teacher construction

### A1. Basic objects and attainable maxima — Section 1.1

**Verdict: valid with explicit conditions.**

The policy class must fix execution information, available memory, and permissible randomization. The environment and harness may influence observations through execution but must not silently expose the unknown environment identity to the learner. Conditioning on a common initial information object \(I_0\) is correct; the environment prior is then the conditional prior after that information.

The source appropriately requires attainable maxima. For a finite environment family and a finite policy class there is no measurable-selection difficulty. A compact policy cube with continuous utility also suffices for the later examples. In an unrestricted neural-policy statement, replacing these maxima by suprema does not automatically preserve the zero-regret equivalence without an attainment argument.

**Required wording in an English theorem:** “We work with a finite environment family and attained policy optima. The learner’s residual randomization is conditionally independent of the environment given its complete observed transcript.”

### A2. Proposition 1: representation–information–algorithm decomposition — Equation (2)

**Verdict: valid; standard decision-theoretic decomposition.**

Let \(M(E)=J_E^\Pi\), and let \(b(D)=\max_{\pi}\mathbb E[J_E(\pi)\mid D]\). Adding and subtracting \(\mathbb E M(E)\) and \(\mathbb E b(D)\) proves the identity. For every transcript,

\[
\mathbb E[M(E)\mid D]\ge b(D)
\ge \mathbb E[J_E(\hat\pi)\mid D].
\]

The second inequality uses the conditional-independence assumption: conditional on \(D\), the learner produces a mixture of feasible policies without gaining another observation of \(E\). Thus all terms are nonnegative.

**Limit:** this is a Bayes decomposition. Replacing each expectation by a separate worst-case supremum does not yield an additive minimax identity. The first term also does not separately identify information-access and parameter-capacity restrictions.

### A3. Zero information regret — Equation (3)

**Verdict: valid in the attained finite setting.**

For a fixed transcript and a maximizing posterior policy \(\pi_D\),

\[
\mathbb E[J_E^\Pi-J_E(\pi_D)\mid D]
=\sum_e \mu(e\mid D)\bigl[J_e^\Pi-J_e(\pi_D)\bigr].
\]

The sum is zero exactly when every positive-weight summand is zero. The common-optimum intersection criterion therefore follows.

**Limits and repairs:**

1. The criterion concerns the posterior support after the transcript, not whether finite noisy data identify an observational-equivalence class.
2. The no-prior discussion is valid only with the relevant joint equivalence class. Pairwise intersections of optimal-policy sets do not imply a common intersection for three or more environments.
3. Approximate guarantees must specify whether randomized policy selection is allowed. Feasibility is then a collection of linear expected-utility inequalities over that randomization; an intersection of deterministic near-optimal policy sets can be too restrictive.

The source acknowledges the third point. Preserve it.

### A4. Data processing and synthetic feedback — Equation (4)

**Verdict: valid.**

For a Markov chain \(E\to D\to D'\), the posterior-optimal value is convex in the posterior. Conditional Jensen and the tower property give the inequality. If synthetic data \(Z\) satisfy \(Z\perp E\mid(D,I_0)\), observing \(Z\) in addition to \(D\) leaves the posterior unchanged.

**Interpretation limit:** this rules out additional information about the unknown world conditional on the existing information. It does not rule out computational advantages, better optimization, improved conditioning, or information already present in a pretrained generator. An implementation must decide exactly what is contained in \(I_0\).

### A5. Perfect teacher and aliased student — Equation (5)

**Verdict: valid.**

At state 0, action \(A\) has reward 1, and at state 1, action \(B\) has reward 1. All alternative success probabilities lie in \([1/4,3/4]\), so the teacher is strictly optimal. Its recorded state, action, and reward distribution is independent of the sign \(v\).

For an aliased student choosing \(A\) with probability \(p\),

\[
\begin{aligned}
J_v(p)
&=\tfrac12[p+(1-p)(\tfrac12-2v\Delta)]
+\tfrac12[p(\tfrac12+2v\Delta)+(1-p)]\\
&=\tfrac34+v\Delta(2p-1).
\end{aligned}
\]

The optimal restricted student obtains \(3/4+\Delta\), leaving representation loss \(1/4-\Delta\) relative to the teacher.

**Limit:** the student must not retain the hidden state through another observation channel or within-episode memory. The example has one decision, so this restriction is explicit and easy to implement.

### A6. Proposition 2: passive impossibility — Equation (6)

**Verdict: valid; sharp for this family.**

The output distribution is the same under \(v=+1\) and \(v=-1\). If \(m=\mathbb E\hat p\), their expected regrets are \(2\Delta(1-m)\) and \(2\Delta m\). Their maximum is at least \(\Delta\), attained by \(p=1/2\).

The statement covers every learner using only the specified demonstrations and their conditionally independent postprocessing, including randomized learners. It is not merely a behavioral-cloning failure.

**Novelty classification:** the reasoning is a two-environment indistinguishability argument. The combination with a perfect privileged teacher is a useful witness, but neither indistinguishability nor teacher–student information asymmetry is a new proof principle.

### A7. Adaptive active-query lower bound — Equation (7)

**Verdict: valid with the declared query interface.**

The informative query distribution is Bernoulli\((1/2+2v\Delta)\). A learner achieving regret at most \(\Delta/2\) must output \(p\ge3/4\) in the positive world and \(p\le1/4\) in the negative world. Consequently the event \(\{\hat p>1/2\}\) defines a test with errors at most \(\delta\).

For the stopped transcript, the KL chain rule gives

\[
\mathrm{KL}(\mathbb P_+^D,\mathbb P_-^D)
=\mathbb E_+N\,d_\Delta,
\qquad
d_\Delta=\operatorname{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta),
\]

if every counted query is one informative branch. If uninformative operations are also counted, the equality becomes an upper bound by \(\mathbb E_+N\,d_\Delta\), which still proves the lower bound. Finite expected stopping time and bounded per-query log-likelihood ratios justify passing from truncated transcripts to the stopped transcript. The same argument applies in the opposite direction because \(d_\Delta\) is symmetric for these two Bernoulli laws.

Data processing to the decision event gives at least \(\operatorname{kl}(1-\delta,\delta)\). For \(u=2\Delta\le1/4\),

\[
\operatorname{kl}(\tfrac12+u,\tfrac12-u)
=2u\log\frac{1+2u}{1-2u}\le16u^2.
\]

The stated constant 64 is conservative and valid.

**Required assumptions:** no additional query reveals \(v\); resets and initial state access mean what the theorem says; the same algorithm and noninformative initial knowledge are used in both environments. This is not a general active-learning complexity theorem.

### A8. Active upper bound and compilation — Equations (8)–(9)

**Verdict: valid.**

The mean lies \(2\Delta\) from the threshold. Hoeffding gives a one-sided error at most \(\exp(-8n\Delta^2)\). The upper bound therefore matches the two-world lower bound in \(\Delta^{-2}\log(1/\delta)\) up to constants.

For the compiled frequency student, two-sided Hoeffding gives

\[
\Pr(|\hat p-q|>1/8)\le2e^{-m/32}.
\]

On correct direction identification and successful compilation, the total probability of choosing the inferior action is at most \(1/4\), yielding regret at most \(\Delta/2\).

**Limits:**

- \(\Delta\) is known as a family parameter in the proposed stopping rule.
- The labels drawn from \(q\) are synthetic target actions, not newly observed teacher executions.
- The joint success level is \(1-\delta-\delta_L\), not \(1-\delta\).
- The compilation result concerns a frequency learner with an exactly representable target. It does not establish convergence of neural cross-entropy training.

## 2. Evaluation–learning separation and complete learner dynamics

### A9. Static policy-value matching versus learning — Equations (10)–(11)

**Verdict: valid.**

Independently recomputing the table gives

\[
\mathbb E_E r(A)=\mathbb E_S r(A)=11/20,\quad
\mathbb E_E r(B)=\mathbb E_S r(B)=1/2,
\]

and teacher utility \(3/5\) under both state distributions. Teacher \(A\)-frequencies are \(3/4\) and \(1/4\). A frequency learner has those expected probabilities for every positive sample size, so its true \(E\)-utilities are \(43/80\) and \(41/80\).

**Important limits:**

1. The reference teacher is optimal in this example, but its realized reward is not always 1. Do not call this the “perfect-success teacher” example.
2. Equality covers the specified aliased student class, not every history-dependent policy.
3. The result does not refute full Bellman value equivalence.
4. The source changes the sampled state distribution; it does not merely relabel an otherwise identical joint transcript.
5. The gap is an algorithm-specific learning-value gap. It is not a lower bound for every possible learning algorithm, which could ignore both datasets and choose \(p=1\).

### A10. Mean update is insufficient — Equation (12)

**Verdict: valid; direction of benefit is task dependent.**

Both batch means have expectation \(1/2\). Their variances are \(1/8\) and 0. Thus the updated parameter variances differ by \(\eta^2/8\), and the expected square differs by the same amount.

**Interpretation repair:** matching additional moments controls fidelity to the reference learner. It does not necessarily improve task success. If the valid terminal utility is instead \(J(p)=1-p^2\), the sign of the performance difference reverses. More reference-like noise can help, hurt, or make no difference depending on the task curvature.

### A11. Complete learner state — Section 4.1

**Verdict: necessary and correctly specified.**

Parameters alone need not be Markov. Optimizer state, replay contents, environment-interaction history, and the relevant randomization mechanism can change future training even at identical weights. A full-history state always defines a conditional kernel in the finite setting.

**Repair for implementation:** a saved “checkpoint” is a complete learner snapshot only if these additional components and their restoration semantics are included. Matching only weights cannot validate a theorem whose state includes optimizer and buffer state.

### A12. Proposition 3: learning-process telescoping — Equation (13)

**Verdict: valid; a standard simulation identity.**

Along the \(S\) process,

\[
\mathbb E[V_r^E(Z_r)-V_{r+1}^E(Z_{r+1})]
=\mathbb E[(K_r^E-K_r^S)V_{r+1}^E(Z_r)].
\]

Summing leaves \(\mathbb E V_0^E(Z_0)-\mathbb E F_E(Z_R^S)\), which is exactly \(L_E(E)-L_E(S)\) for the common initial law.

**Nonnegotiable details:**

- The occupancy law in this orientation is \(d_r^S\), not \(d_r^E\).
- Both terminal evaluations use the same true \(F_E\).
- \(V_r^E\) follows the reference learning protocol after the current point.
- The difference can have either sign.
- If the update rule differs across interfaces, that difference must be included in the compared kernels; it cannot be hidden while interpreting the result as a message-only comparison.

### A13. Learning-consequence tests and one-boundary necessity — Equations (14)–(15)

**Verdict: valid with the stated intervention quantifier.**

At a fixed snapshot, an interface affects the one-boundary replacement experiment only through the expectation of \(g(y)=V_{r+1}^E(U_r(z,y))\). Equality of those expectations is therefore exactly equivalent to equality of that experiment’s terminal mean. Quantifying over the declared snapshots and suffixes gives the test class.

**Limit:** this necessity is partly definitional. It does not show that every local test must match to preserve one fixed aggregate training outcome. Signed errors can cancel. It also does not supply a cheap way to calculate the tests.

### A14. Stopping at the first uncertified state — Equation (16)

**Verdict: valid, with an explicit stopping convention.**

Define \(\tau\) as the first uncertified pre-update state time, capped at \(R\); let “exit” mean \(\tau<R\). Stopped telescoping gives

\[
L_E(E)-L_E(S)
=
\mathbb E_S\sum_{r<\tau}(K_r^E-K_r^S)V_{r+1}^E(Z_r)
+
\mathbb E_S[V_\tau^E(Z_\tau)-F_E(Z_R^S)].
\]

On \(\tau=R\), the last integrand is zero. On exit, its conditional expectation lies in \([-1,1]\). Hence its absolute expectation is at most \(\Pr_S(\tau<R)\).

**Required wording:** define whether the initial state can already be outside the certified set and include that event in the exit probability. Finite stored checkpoints do not by themselves imply small exit probability for continuous neural trajectories.

### A15. SGD Taylor expansion — Equation (17)

**Verdict: valid under smoothness and integrability.**

Apply Taylor’s theorem to \(V(\theta-\eta G)\) for each realization, integrate under each source, and subtract. The quadratic term involves raw second moments \(\mathbb E GG^\top\), not covariance alone unless mean terms are handled separately. The two absolute third-order remainder bounds add.

**Required assumptions:** a common \(C^3\) continuation function on every relevant update segment, a uniform operator norm bound on its third derivative there, and finite third gradient moments. If the horizon includes discontinuous parsing, threshold selection, or discrete update decisions, smoothness cannot simply be assumed.

**Open bridge:** the supervised loss is not the continuation value. Replacing one by the other requires a separate approximation theorem or empirical hypothesis.

## 3. Joint moments, update composition, and exposure

### A16. Polynomial deployment utility — Equation (19)

**Verdict: valid under the declared action interface.**

Every action factor is \(p_j\) or \(1-p_j\); environment and harness factors are independent of the numerical parameter vector. Summing finite trajectory probabilities gives degree at most \(N\). The backward coefficient-norm recurrence \(B_{n+1}\le3B_n\), \(B_0\le1\), yields \(3^N\).

**Conditions worth restating:**

- The conditional probability of the next action, given full past history, is the declared coordinate \(p_j\). Arbitrarily correlated private action randomization not satisfying this condition is outside the class.
- Context selection may depend on past observations and actions, but cannot inspect the numerical parameter \(p\) directly.
- Terminal utility is fixed independently of \(p\).
- The actual degree can be lower than the horizon.

**Interpretation repair:** polynomial degree is relative to the policy coordinates and the learning update. Reparameterizing \(p=\sigma(\theta)\) generally destroys finite polynomial degree in \(\theta\). Therefore “information order” is not an invariant bit-complexity of an arbitrary neural agent.

### A17. Theorem 4: moment equivalence — Section 5.3

**Verdict: valid; the universal necessity quantifier is essential.**

Sufficiency follows from expanding \(J(w_0p_0+\sum_iw_iZ_i)\). Independence across blocks factors expectations; a block contributes only moments of total degree at most \(N\).

For necessity, use one update with weight 1 and the task requiring exactly the action successes in a multi-index \(\alpha\). Its value is \(p^\alpha\), so universal value matching forces \(\mathbb E_P Z^\alpha=\mathbb E_Q Z^\alpha\).

**Limits:**

1. This is not necessary for a fixed task or fixed update recipe.
2. Matching each coordinate’s entire marginal distribution does not determine mixed moments.
3. It preserves terminal expected utility, not the trained-parameter law or individual-run safety probability.
4. The block distribution is fixed and independent across updates. On-policy query distributions that change with the student are not covered.
5. A block must actually contain all coordinates claimed to be observed.

### A18. Theorem 5: strict hierarchy — Equations (20)–(21)

**Verdict: valid.**

The even and odd binomial masses both normalize. Their \(\ell\)-th moment difference is

\[
2^{1-N}N^{-\ell}\sum_{i=0}^N(-1)^i\binom Ni i^\ell,
\]

which vanishes for \(\ell<N\) and equals the claimed signed value at \(\ell=N\). Requiring \(N\) successful actions realizes \(J(p)=p^N\).

I independently verified all lower moments and the exact highest-order gap using rational arithmetic for \(N=1,\ldots,12\).

**Important interpretation repair:** this proves exact necessity, not high-order necessity at a fixed practical tolerance. The separation size is

\[
g_N=\frac{N!}{2^{N-1}N^N}
\sim 2\sqrt{2\pi N}\,(2e)^{-N}.
\]

It becomes exponentially small. A fixed-\(\varepsilon\) lower bound would require a different argument or a tolerance that scales with \(N\). Do not describe this construction as showing that practical agents must estimate arbitrarily high moments accurately.

### A19. Theorem 6: a training-round-independent fidelity bound — Equation (22)

**Verdict: valid.**

For a monomial \(p^\alpha\), the convex affine substitution has nonnegative expansion coefficients of total at most 1. Each term contains at most \(|\alpha|\) nonconstant block moments in \([0,1]\). A product telescoping bound controls its expectation difference by \(|\alpha|\delta_N\). Summing coefficients and then task monomials yields \(C_J\delta_N\).

The result does not hide a factor \(R\). The nonnegative convex weights are the reason the coefficient mass stays bounded as the number of blocks grows.

**Limits:**

- Signed or unconstrained affine weights invalidate this proof and can introduce amplification.
- Data-dependent weights require conditioning and may alter the relevant moments.
- The bound can be vacuous through \(C_J\), and the number of moments can be combinatorially large.
- The result concerns bias between population learning procedures; it does not give a high-probability bound for an individual trained student.
- The basis coefficient norm is coordinate dependent.

I independently enumerated a two-coordinate, three-block example with nonuniform weights and all monomials through degree 3. Every bound held exactly. This is a sanity check, not a replacement for the proof.

### A20. Theorem 7: first unmatched cumulant and exposure rate — Equation (23)

**Verdict: valid for fixed bounded distributions and a fixed polynomial task.**

Let \(W_n=\bar Z_n-\mu\). For \(s\ge2\),

\[
\kappa_s(W_n)=n^{1-s}\kappa_s(Z).
\]

The polynomial Taylor expansion is finite. The centered moment–cumulant formula contains only partitions with blocks of size at least 2. In the difference between \(P\) and \(Q\), at least one block must have size at least \(r\). The smallest possible inverse power is \(n^{-(r-1)}\), obtained only by the single \(r\)-block. Every remaining differing partition has inverse power at least \(n^{-r}\). This proves the coefficient and the remainder order.

**Required wording:**

- \(P,Q,k,N,J\) are fixed while \(n\) grows.
- The mean is common, and cumulants below order \(r\) match.
- The conclusion is \(C n^{1-r}+O(n^{-r})\), not automatically \(\Theta(n^{1-r})\).
- A zero derivative contraction can remove the leading term.
- A finite training-bank replay count is not the number of independent population blocks.

The statement is a finite polynomial version of a higher-order expectation expansion. A novelty claim must explain what is new in its use for distillation; cumulant scaling itself is established mathematics.

### A21. Exact hierarchy exposure law — Equation (24)

**Verdict: valid.**

In the parity construction, all moments and hence cumulants below \(N\) agree. The degree-\(N\) task has no higher derivative terms. Thus the first differing \(N\)-th cumulant contributes exactly the original \(N\)-th moment gap divided by \(n^{N-1}\).

I independently checked this by exact discrete convolution for \(N=2,3,4\) and \(n=1,2,3,7\), without invoking the cumulant formula.

**Limit:** the exact law is for this constructed pair and its prescribed mean learner. Fitting a power law to neural training results cannot by itself identify a first unmatched cumulant.

### A22. Nonlinear one-step counterexample — Section 5.7

**Verdict: valid.**

Both sources have mean \(1/2\), but thresholding \(p=1\{Z>1/2\}\) gives expected \(p\) equal to \(2/3\) and 0. This correctly prevents extending the affine theorem based on deployment horizon alone.

**Interpretation:** even a smooth nonlinearity would generally require more than the mean; the discontinuity is convenient, not the sole reason for failure.

### A23. Polynomial-update composition — Equation (25a)

**Verdict: valid as a sufficient blockwise degree bound.**

Working backward from a degree-\(N\) terminal polynomial, substitution of \(U_t\) multiplies the degree in the previous parameter by at most \(a_t\). The degree in block \(Z_i\) is therefore at most

\[
D_i=N b_i\prod_{t>i}a_t.
\]

Independence across blocks then proves moment sufficiency through each \(D_i\).

The squaring example attains the exponential degree for the first block. It shows that deployment horizon alone does not bound the exact required order.

**Limits and repairs:**

1. The degree bound need not be tight for a particular update, task, or block.
2. Degrees in \(p\) and \(Z\) must be separate total-degree bounds, as the source states.
3. The update must keep the policy in its declared domain.
4. State-dependent input distributions violate the unconditional independent-block factorization.
5. Exact polynomial degree can be large while the finite-difference performance separation is tiny.
6. Polynomial approximation of a neural update requires its own approximation error bound after composition. Degree counting alone does not provide it.

## 4. Minimal statistics and finite calibration

### A24. Proposition 8: minimum linear expectation statistics — Equations (25)–(26)

**Verdict: valid; standard linear algebra.**

If \(g\) lies in the span of the constant and transmitted features, its expectation is determined by those feature means. Conversely, if a \(g\) lies outside, choose \(h\) in the orthogonal complement of the transmitted span with \(g^\top h\ne0\). Since \(h^\top1=0\), sufficiently small perturbations of the uniform law by \(\pm ah\) are probability laws with the same transmitted means and different \(g\)-means.

The minimal dimension is the span dimension modulo constants.

**Limits:**

- This counts fixed linear expectation statistics, not communication bits, model size, or observations.
- It assumes all probability laws on the finite alphabet are admissible.
- For a constrained law family, the kernel criterion must use differences of admissible laws.
- A single infinitely precise nonlinear real encoding is outside this linear-statistic model; it cannot be used to interpret the dimension as an unrestricted communication lower bound.

### A25. Full-distribution boundary — Equation (27)

**Verdict: valid for the constructed rich learner family.**

An event-triggered choice between policies with utility gap \(\gamma\) produces \(g_A=a+\gamma1_A\). The supremum of the expectation difference is \(\gamma\operatorname{TV}(P,Q)\). Singletons span the finite function space, yielding \(m-1\) independent mean statistics.

**Limit:** this does not establish that practical SGD or LLM fine-tuning realizes every event-triggered learner. It shows why a universal compression claim needs restrictions; it does not quantify practical neural compressibility.

### A26. Reference gain plus fidelity — Equation (28)

**Verdict: valid; elementary but essential.**

Subtracting the two assumptions yields the claimed gain. The result uses a one-sided fidelity loss; a source that improves on the reference is permitted.

**Required interpretation:** a fidelity method can only inherit the quality that the reference training procedure actually has. A strong teacher does not prove a positive \(\gamma\). Nor does closer matching guarantee superiority to other distillation algorithms.

### A27. Finite-candidate calibration — Equations (29)–(31)

**Verdict: valid with fixed candidates and independent within-cell audit samples.**

Each reference feature mean has two-sided Hoeffding failure probability at most \(\delta/(2Lr)\), and each candidate feature mean has failure probability at most \(\delta/(2MLr)\). The two union bounds yield a joint event of probability at least \(1-\delta\). Dependence among different features or cells does not invalidate the union bound; the sample sequence used for each individual empirical mean must meet its concentration assumption.

On that event, \(|D_j-\widehat D_j|\le e_E+e_S\) uniformly in \(j\). Applying the bound to the selected and true-best candidates gives (31). Selection using these audit data is allowed because the event is simultaneous over the fixed candidate set.

**Required repairs:**

1. Condition explicitly on all pilot randomness used to create candidates and features.
2. Candidates, features, cells, and the declared recipe must be fixed before the audit, or an alternative uniform-complexity bound must replace the finite union bound.
3. A candidate may not be further fine-tuned on the audit and then inherit its old certificate.
4. The final training-value quantity is the population value of the selected, frozen interface under the declared fresh training experiment.
5. Reusing the exact audit realizations as downstream training data changes the experiment and requires a separate analysis or an explicitly modeled reuse kernel.
6. If candidates are frozen replay distributions, independent resampling is conditional on the bank. A claim about the original environment must still account for bank error.

**Why point 5 matters:** two interfaces can each have population mean \(1/2\), while choosing between them based on realized audit outputs and then reusing those outputs produces an effective training mean different from \(1/2\). For example, take one constant-\(1/2\) source and one Bernoulli-\(1/2\) source. When both population discrepancies tie at zero, select the Bernoulli source if its auxiliary audit draw is 1 and otherwise select the constant source. Fresh selected-source training still has mean \(1/2\); reusing the selected audit output has mean \(3/4\). Thus “selected fixed source” and “selected reused audit data” are distinct laws, even when all population moments being compared agree.

This example diagnoses the missing experiment specification. It does not claim that every numerical Hoeffding bound in (31) is violated by reuse.

### A28. General continuation-feature certificate — Equations (32)–(33)

**Verdict: valid conditionally; the expensive assumptions are real.**

For \(g=c+b^\top\phi+r\),

\[
|(\mathbb E_E-\mathbb E_S)g|
\le\|b\|_1D+2\|r\|_\infty.
\]

Use the stopped learning-process bound and then the simultaneous calibration event. The stated certificate follows.

**Required conditions:**

- The representation and residual bound hold for every claimed recipe, certified state, and message in the declared domain.
- The \(B_t,\tau_t\) bounds are uniform over that domain.
- The exit probability applies to the selected interface, either through a simultaneous candidate bound or through a separate valid audit of the selected interface.
- Any statistical estimation of \(\tau\), \(B\), or \(\rho\) contributes its own failure event to the final confidence accounting.
- The statement controls expected terminal utility, not every seed’s realized success rate.

**Open issue:** finite neural probes do not imply a uniform supremum-norm residual bound on an open message/state space. A fitted predictor may be useful without being a certificate.

### A29. Closed affine calibration — Equation (34)

**Verdict: valid under the affine theorem and calibration conditions.**

The moment features are directly observable from complete blocks. Uniform feature-mean error supplies \(\delta_N\), and Theorem 6 gives the value certificate. No continuation-value oracle is needed in this subclass.

The probability event is uniform over the fixed moment family. For each choice of \(R,p_0,w\), the certified quantity is the population expected training value; it is not a simultaneous high-probability statement about all actual sampled students.

**Computational limit:** \(\binom{k+N}{N}-1\) features and a potentially exponential coefficient constant can make the bound unusable. “Finite” must not be rewritten as “efficient.”

### A30. Estimating continuation tests — Section 7.5

**Verdict: a valid expensive construction in a finite, restorable system; open in the intended neural regime.**

If a complete state can be restored, the exact next update can be executed, the reference continuation can be run, and true deployment can be sampled, then each resulting bounded return is an unbiased sample of the appropriate continuation experiment. Enumerating a finite family supports a uniform concentration bound.

If every \(g(y)\) has a uniform estimation error at most \(e\), then for any fitted predictor \(\hat g(y)\),

\[
\sup_y|g(y)-\hat g(y)|
\le e+\sup_y|\hat g_{\rm empirical}(y)-\hat g(y)|.
\]

This remains valid even if the fit uses the uniformly controlled estimates.

**Conditions:** the entire declared finite domain must be covered; residual fitting on only observed messages does not certify unseen ones. Complete restoration and all continuation costs must be logged. If “forcing message \(y\)” violates the update or access model, the experiment must instead use a realizable sampled-message protocol with its own estimand.

## 5. Mandatory repairs for the next manuscript

1. **Define the post-selection training experiment.** State that candidate interfaces are frozen conditional on pilot information and that downstream training uses fresh draws from the selected interface. If data are reused, include reuse in the kernel and rederive the guarantee.
2. **Separate fidelity from improvement.** Keep the reference-gain premise and report both signed learning-value change and absolute fidelity. Do not predict monotonic task improvement from matching more moments.
3. **Qualify information order.** It is the order of a declared coordinate-dependent polynomial consequence class. It is not an intrinsic entropy, bit-complexity, or universal property of neural agents.
4. **Separate exact and approximate necessity.** The parity construction proves exact moment necessity; its effect size shrinks exponentially. Add an explicit practical-tolerance limitation.
5. **Make all probability levels refer to named randomness.** Audit-sample confidence, training-seed variability, deployment sampling, and continuation-fit uncertainty are separate.
6. **State complete snapshot and coverage requirements.** An optimizer checkpoint or a small state bank is not automatically the state space of the theorem.
7. **Retain every nonlinearity and dependence boundary.** Adaptive data selection, replay dependence, thresholding, neural parameterization, and candidate refinement are not covered by the affine theorem merely because the deployment horizon is finite.
8. **Do not promote empirical metrics to certificates.** Label them predictive diagnostics until approximation and coverage assumptions have been independently validated or bounded.

## 6. What this audit establishes and does not establish

### Established in this review

- Every stated finite proposition was rederived from its hypotheses.
- No central algebraic contradiction was found.
- The passive lower bound and active upper bound concern the same two-environment access model.
- The evaluation–learning separation has the stated exact numerical gap.
- The joint-moment sufficiency, universal necessity, strict hierarchy, approximate fidelity bound, cumulant exposure law, and polynomial-update extension are valid within their scopes.
- The minimal-statistic and calibration statements have correct finite-dimensional and concentration arguments.
- Independent rational-arithmetic checks reproduced key finite constructions.

### Not established

- Novelty relative to all existing literature.
- A fixed-tolerance high-order information lower bound.
- Efficient neural continuation-function estimation.
- Uniform state/message coverage for a real agent.
- A general relation between moment matching and better task performance.
- A proof that the proposed practical method beats behavioral cloning, on-policy distillation, dataset distillation, or other adaptive data methods.
- External human peer review or machine-checked formal verification.
- Any best-paper-level assessment.

## 7. Contribution assessment for the authors

The strongest justified statement is:

> For an explicitly declared learner and deployment interface, the information that must be preserved is determined by the composition of the learning update and deployment utility. In a finite polynomial subclass, this produces an exact blockwise moment characterization, strict exact hierarchy, and a testable exposure law.

This statement should not be advertised as a new general statistical principle. Its potential contribution lies in the particular operational problem, the joint characterization, and whether it produces a useful algorithm and predictions unavailable from simpler baselines.

The next research step should therefore test a specific neural approximation of learning consequences, with a preregistered failure criterion. For example, if a consequence metric cannot predict which training block changes true deployment utility after controlling for teacher loss, trajectory length, and reward, then the neural bridge has failed even if the finite theorems remain correct. If complete-block dependence has no additional predictive or intervention value beyond simpler weighting, the higher-order motivation should be narrowed.

A defensible paper can use classical tools and simple proofs. What it cannot do is infer originality, practical importance, or broad validity from the existence of several mathematically correct lemmas.


## 8. Independent check of proposed v0.14 additions

The author supplied the following additional statements after the v0.13 review. They are reviewed separately here because they are not part of the hashed source above.

### B1. Exact approximation duality for information preservation

**Statement reviewed.** On a finite message alphabet, let \(H=\operatorname{span}\{1,\phi_1,\ldots,\phi_d\}\). Then

\[
\sup_{\substack{P,Q\ {\rm probability\ laws}\\
\mathbb E_P\phi=\mathbb E_Q\phi}}
|(\mathbb E_P-\mathbb E_Q)f|
=
2\inf_{h\in H}\|f-h\|_\infty.
\]

**Verdict: valid; standard finite-dimensional duality.**

For any feasible pair and \(h\in H\),

\[
|(P-Q)^\top f|
=|(P-Q)^\top(f-h)|
\le\|P-Q\|_1\|f-h\|_\infty
\le2\|f-h\|_\infty.
\]

This proves the upper bound. Finite-dimensional linear-program duality for best uniform approximation gives a vector \(\sigma\) satisfying

\[
\sigma^\top h=0\quad(h\in H),\qquad
\|\sigma\|_1\le1,\qquad
\sigma^\top f=\inf_{h\in H}\|f-h\|_\infty.
\]

Since \(1\in H\), \(\sigma\) has zero total mass. Its positive and negative parts each have mass at most \(1/2\). Therefore \(2\sigma\) is the difference of two probability laws: add the same nonnegative leftover probability mass to \(2\sigma_+\) and \(2\sigma_-\). These laws share all feature means and attain the lower bound. The absolute value is harmless because the feasible set is symmetric.

**Critical quantifier:** the supremum ranges over all pairs sharing some common feature means. This need not be equality if one reference law \(P\) is fixed in advance, or if both laws must lie in one prescribed feature-mean fiber. Restricted source families likewise replace the full dual ball by their realizable law differences.

This result improves the formulation of approximate information sufficiency. It does not change the count from “linear expectation features” into bits or query complexity.

### B2. Scalar high degree can require much lower approximate information order

**Statement reviewed.** For \(x\in[0,1]\), truncate the Chebyshev expansion of \(x^N\) at order \(q\). The resulting degree-\(q\) polynomial has uniform approximation error at most

\[
\Pr(|B-N|>q)\le2e^{-q^2/N},
\qquad B\sim{\rm Binomial}(2N,1/2).
\]

Consequently, if two scalar laws match moments through \(q\), their \(N\)-th moments differ by at most \(4e^{-q^2/N}\). The same consequence holds for the scalar affine-mean learner because its final parameter laws also match moments through \(q\).

**Verdict: valid for this scalar task and learner.**

Put \(2x-1=\cos\theta\). Then \(x^N=\cos^{2N}(\theta/2)\), and the exact cosine expansion is

\[
x^N=a_0+\sum_{k=1}^N a_kT_k(2x-1),
\]

where

\[
a_0=4^{-N}\binom{2N}{N},\qquad
a_k=2\,4^{-N}\binom{2N}{N-k}.
\]

All coefficients are nonnegative. Since \(|T_k|\le1\), truncation error is at most the omitted coefficient sum, which is exactly \(\Pr(|B-N|>q)\). At \(x=1\), every \(T_k=1\), so this particular truncation’s uniform error actually equals that tail. Hoeffding gives the displayed exponential bound. When \(q\ge N\), the truncation error is exactly zero.

If moments through \(q\) match, the two laws agree on the truncated polynomial’s expectation. The difference on \(x^N\) is at most twice its uniform approximation error. Under independent affine averaging, degree-\(q\) polynomial tests of the final mean depend only on block moments through \(q\), so the same argument applies.

**Limits:**

- This is not a bound for every degree-\(N\) or every \(N\)-step task.
- It is an exact-low-moment-matching argument. With noisy raw moments, the approximation polynomial’s power-basis coefficient norm controls error amplification and can be large.
- Bounded Chebyshev features provide a stable basis for a single scalar message. Their pullback through a multi-block learner requires its own coefficient or approximation analysis before claiming a low finite-sample cost.
- The existence of a low-degree approximation does not imply that the required source distribution can be learned or implemented within the few-shot budget.

### B3. A constant-gap multivariate hierarchy

**New reviewer construction.**

The exponentially shrinking gap in the scalar parity-of-binomial-coefficients example is not universal. For \(k=N\), let \(P\) be the uniform law on even-parity vectors in \(\{0,1\}^N\), and let \(Q\) be uniform on odd-parity vectors. Every subset of fewer than \(N\) coordinates is uniform under both laws. Therefore all joint raw moments of total degree below \(N\) match.

Use a one-block update \(p=Z\), visit context \(j\) once for each \(j=1,\ldots,N\), and reward the student exactly when the action vector has even parity. The policy value is

\[
J(p)=\frac{1+\prod_{j=1}^N(1-2p_j)}{2}.
\]

For \(p\in\{0,1\}^N\), the actions are deterministic. Thus

\[
\mathbb E_PJ(Z)=1,\qquad
\mathbb E_QJ(Z)=0.
\]

This is a valid \(N\)-step finite task with a constant learning-value gap despite matching all moments through degree \(N-1\).

Moreover, the best uniform approximation of \(J\) by total-degree-\(q\) polynomials with \(q<N\) has error exactly \(1/2\). The constant polynomial \(1/2\) achieves that error; the above moment-matching pair forces an error at least \(1/2\) by the two-law expectation argument. The same lower bound holds on the whole cube because the binary vertices are a subset.

**Interpretation:** both scalar \(x^N\) and multivariate parity have exact degree \(N\), but their approximate information requirements are very different. Approximation of the actual consequence class is more informative than exact degree alone. The constant-gap example lets dimension grow with \(N\); it does not repair a fixed-one-coordinate lower bound.

This is also a standard parity construction, not a newly discovered lower-bound technique. Its usefulness here is to prevent an incorrect universal compressibility claim.

### B4. Certified improvement by selecting a message provider

**Statement reviewed.** Let a fixed reference training kernel be \(K_t^0\), with continuation value \(V_{t+1}^0\). Define the provider advantage at a complete learner state as

\[
A_t(z,j)=(K_t^j-K_t^0)V_{t+1}^0(z).
\]

A possibly state-dependent provider policy \(\nu\) generates actual learner occupancies \(d_t^\nu\). Then

\[
L_E(\nu)-L_E(0)
=\sum_t\mathbb E_{d_t^\nu}A_t(Z_t,\nu_t).
\]

On a simultaneous audit event, suppose

\[
A_t(z,j)\ge\widehat A_t(z,j)-b_t(z,j)-e_t(z,j).
\]

Assign the reference provider lower confidence bound 0 exactly. Select a provider maximizing the lower confidence bound. If the conditions hold at every state where a nonreference provider is used, the selected process cannot degrade the reference’s expected terminal utility.

**Verdict: valid; standard policy improvement applied to the learner-state process.**

The identity is the same telescoping calculation as A12 with the sign reversed. Let

\[
\ell_t(z,j)=\widehat A_t(z,j)-b_t(z,j)-e_t(z,j),\qquad
\ell_t(z,0)=0.
\]

On the simultaneous event, the chosen provider has

\[
A_t(z,\nu_t)\ge\max\{0,\max_j\ell_t(z,j)\}\ge0.
\]

Integrating along the actual selected process proves the no-degradation claim and the stronger lower bound

\[
L_E(\nu)-L_E(0)
\ge
\mathbb E_\nu\sum_t
\max\{0,\max_j\ell_t(Z_t,j)\}.
\]

This bound concerns population expected terminal value conditional on the completed audit, with the declared fresh downstream training law.

**Paired audit concentration.** If each audit sample is an independently replicated contrast \(X_i\in[-1,1]\), then

\[
\Pr\!\left(
|\bar X-\mathbb EX|>e
\right)\le2e^{-ne^2/2}.
\]

For \(M\) nonreference providers and \(L\) fixed audit cells, a simultaneous radius is

\[
e=\sqrt{\frac{2\log(2ML/\delta)}{n}}.
\]

The two arms within a pair may share randomness, provided their marginals are the specified candidate and reference experiments; distinct pairs must satisfy the concentration assumptions. Shared randomness can reduce variance but does not change the range-based proof.

**Proxy bias.** The term \(b_t\) must bound the difference between the expected audited proxy contrast and the true continuation advantage. If a bounded proxy \(\widetilde V\) obeys a uniform value error \(\|\widetilde V-V^0\|_\infty\le\beta_t\) on both next-state supports, then \(b_t=2\beta_t\) is valid. If full reference continuations and real deployments are actually run in both audit arms, the Monte Carlo contrast is unbiased for \(A_t\), and this particular proxy-bias term is zero. A learned critic does not receive \(b_t=0\) merely because it has low training error.

**Coverage and fallback.**

- If the provider is exactly the reference outside the certified domain, its advantage there is exactly zero. No exit penalty is needed.
- If arbitrary provider behavior continues after the first uncertified state, the stopped identity instead yields the certified pre-exit improvement minus \(\rho\), where \(\rho\) bounds the first-exit probability. The penalty is charged once.
- Mapping a continuous neural state to a nearby stored checkpoint requires a uniform within-cell approximation bound. Without it, “audited cell” is not a valid certificate for the mapped state.
- A finite exact checkpoint bank can force fallback nearly everywhere in a continuous neural trajectory. A practical generalized provider gate is empirical until this coverage/bias problem is solved.

**Selection and reuse conditions.** Freeze proxies, candidate providers, and audit definitions conditional on pilot data. Do not train them further on the same audit and retain the old confidence claim. A state-dependent provider policy is permitted because the confidence event is simultaneous over every cell it may use. Unseen cells need a valid additional confidence allocation, a uniform function-class argument, or reference fallback.

**Novelty classification:** this theorem is policy improvement on a lifted state space. The research contribution would need to come from a useful, independently validated way to construct or estimate the learning-consequence advantage under the few-shot, black-box-teacher, real-agent constraints.


## 9. Couplings, on-demand audits, and the unified formulation

These checks were requested after the initial v0.14 additions. They preserve the general environment–learner formulation; the affine examples are special cases rather than the definition of the problem.

### C1. Fixed-marginal packet couplings

**Statement reviewed.** Let \(H\in\mathbb R^{m\times n}\) be a finite packet-response table. Fix strictly positive probability marginals \(\mu,\nu\), and let \(\mathcal C(\mu,\nu)\) be their coupling polytope. Then

\[
\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H\rangle
-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H\rangle=0
\]

if and only if \(H_{ij}=a_i+b_j\). For any two couplings,

\[
|\langle\Gamma-\Gamma',H\rangle|
\le
2\inf_{a,b}\|H-a-b\|_\infty.
\]

**Verdict: valid, with no general equality in the second display.**

If \(H=a+b\), its expectation is \(\sum_i\mu_i a_i+\sum_j\nu_jb_j\), independent of the coupling.

Conversely, \(\Gamma_0=\mu\nu^\top\) is strictly positive. For any rectangle with distinct rows \(i,i'\) and columns \(j,j'\), both sufficiently small positive and negative perturbations in the direction

\[
e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}
\]

remain feasible couplings. Zero width therefore forces

\[
H_{ij}+H_{i'j'}-H_{ij'}-H_{i'j}=0
\]

for every rectangle. Fix a base row and column, and set
\(a_i=H_{i1}\), \(b_j=H_{1j}-H_{11}\). The rectangle identities give \(H_{ij}=a_i+b_j\).

For the inequality, coupling differences annihilate additive functions, and their \(\ell_1\) norm is at most 2. Subtract an arbitrary \(a+b\), apply the norm inequality, and minimize.

**Strict-inequality example.** Take \(\mu=\nu=(0.9,0.1)\) and

\[
H=\begin{pmatrix}1&0\\0&0\end{pmatrix}.
\]

The feasible value of \(\Gamma_{11}\) ranges from \(0.8\) to \(0.9\), so the response width is \(0.1\). The best additive uniform approximation error is \(1/4\): the unit rectangle difference forces error at least \(1/4\), and a checkerboard residual of entries \(\pm1/4\) attains it. Thus \(2\operatorname{dist}_\infty(H,\mathrm{additive})=0.5\), strictly greater than the width.

**Limits and interpretation:**

- With zero marginal entries, the characterization applies on the positive marginal support, not on unused rows or columns.
- \(H(i,j)\) must be the response to the entire joint packet under the declared learning experiment. A table of token-level losses is not automatically a table of final learning consequences.
- The result identifies when packet dependence matters even though marginal message laws are fixed.
- This is a coupling/linear-algebra consequence. It does not establish that an estimated neural response table generalizes to unobserved packet pairs.

### C2. Finite-grid moment dimension and binary universal-task lower bound

**Statement reviewed.** On

\[
Y=\{0,1/N,\ldots,1\}^k,
\]

the universally sufficient linear expectation statistics for the affine learner and every at-most-\(N\)-step binary-context task require exactly

\[
\binom{k+N}{N}-1
\]

independent nonconstant coordinates. On \(Y=\{0,1\}^k\) with \(N\ge k\), unrestricted terminal action payoffs force \(2^k-1\) such coordinates, even if the permitted worst-pair utility gap is any number strictly below 1.

**Verdict: valid under the universal quantifiers.**

The total-degree-at-most-\(N\) polynomial space has dimension \(\binom{k+N}{N}\). Its evaluation map on the stated tensor grid is injective. Indeed, a polynomial of degree at most \(N\) in each variable that vanishes on all \(N+1\) values of the first coordinate has all its coefficients in that coordinate vanish on the remaining tensor grid; repeat the argument in each coordinate. The total-degree subspace is therefore also injectively represented.

One-block, unit-weight updates and monomial deployment tasks realize every monomial of total degree at most \(N\). These responses span the whole polynomial space on the grid. Moment sufficiency supplies the upper bound; the linear-statistic annihilator argument supplies the matching lower bound modulo the constant.

For binary \(Y\), take a one-block update \(p=Z\) and visit each of the \(k\) contexts once. The resulting action vector is exactly \(Z\). Any specified payoff \(r:\{0,1\}^k\to[0,1]\) can therefore be realized as a terminal reward. This produces all bounded response functions on \(Y\).

If \(d<2^k-1\) features are transmitted, the span of the constant and those features has dimension below \(2^k\). Choose a nonzero signed annihilator \(h\). Its total mass is zero, so its positive and negative parts have the same positive mass \(s\). Define \(P=h_+/s\), \(Q=h_-/s\). They have disjoint supports and identical transmitted feature means. Set \(r=1\) on the positive support and \(r=0\) elsewhere. Then

\[
\mathbb E_Pr-\mathbb E_Qr=1.
\]

Thus fewer features cannot guarantee worst-pair error below 1 uniformly over this task class and all source laws.

**Critical scope:**

1. The feature interface is fixed across all of these tasks. For one known one-update response \(r\), transmitting its expectation alone would be sufficient.
2. The task family permits arbitrary terminal action-vector payoffs; practical task restrictions can lower the dimension.
3. The statement counts linear expectation statistics. It is neither a communication-bit bound nor a query-sample lower bound.
4. Finite feature precision, estimability, and the realizability of the two witness laws are separate questions.
5. The affine learner includes one-block, full-replacement updates. Excluding that learner changes the necessity claim.

### C3. Adaptive on-demand auditing at actual learner states

**Problem.** Auditing only a fixed checkpoint bank does not cover a continuous neural trajectory. Auditing at the actual current state avoids that particular coverage problem, but the statistical event is now interleaved with the trajectory. One must not condition terminal expectations on a global “all audits succeeded” event without accounting for selection.

**Assumptions for the reviewed result.**

- There is a fixed finite training horizon \(R\).
- \(Z_t\) is the complete actual learner/environment state. A fixed reference recipe has bounded continuation values \(V_t^0\in[0,1]\), with \(V_R^0=F_E\).
- Auxiliary audit records may guide the provider controller, but the reference continuation ignores them. Running and discarding an audit does not change the actual learner/environment state or the reference value. Required environment and learner restoration is exact.
- Conditional on all history before the current audit, the finite candidate set, its kernels, \(M_t\), sample count \(n_t\), and confidence allocation \(\delta_t\) are frozen.
- Each candidate is audited with independent paired replicas. Each replica executes the candidate update or the reference update, then the full reference-learning suffix and real deployment. Its two marginals match those experiments, giving a contrast in \([-1,1]\).
- The actual selected update uses a fresh draw from the frozen selected kernel, independent of the audit replicas conditional on the current state, history, and selected provider.
- Unaudited rounds use the reference update. Predictable confidence allocations satisfy \(\sum_t\delta_t\le\delta\) almost surely.

Under these assumptions, no neural interpolation, finite-bank coverage premise, or unbounded critic proxy is needed. The cost may be prohibitive; it must include every branch, suffix, restoration, and deployment.

#### Filtrations and conditional audit event

Let \(\mathcal H_t\) denote history just before audit \(t\), and \(\mathcal F_t\) history after that audit and provider selection but before the fresh actual update. For candidate \(j\), define

\[
A_t(j)=
(K_t^j-K_t^0)V_{t+1}^0(Z_t).
\]

Conditional on \(\mathcal H_t\), the audit contrast has mean \(A_t(j)\). With

\[
e_t=\sqrt{\frac{2\log(2M_t/\delta_t)}{n_t}},
\]

a union bound over the current candidates gives an event \(G_t\) satisfying

\[
\Pr(G_t^c\mid\mathcal H_t)\le\delta_t,\qquad
|\widehat A_t(j)-A_t(j)|\le e_t\quad\text{on }G_t.
\]

The radius assumes \(n_t\) is chosen before observing the current audit outcomes. Adaptive stopping within that audit requires a confidence-sequence or other valid sequential argument.

Set the reference lower bound to 0 exactly and all other lower bounds to \(\widehat A_t(j)-e_t\). Choose their maximizer and let the selected lower bound be \(B_t\). Then \(0\le B_t\le1\). On \(G_t\), the true selected advantage \(A_t=A_t(J_t)\) obeys \(A_t\ge B_t\ge0\). At fallback rounds, define \(A_t=B_t=0\).

Conditional probability and the union bound yield

\[
\Pr\!\left(\bigcap_tG_t\right)\ge1-\delta.
\]

This remains valid for actual, randomly reached states because each audit statement is conditional on the pre-audit history. It does not assert uniform accuracy at states that were never audited.

#### Exact predictable-drift identity

Because the real update is fresh,

\[
A_t
=
\mathbb E[V_{t+1}^0(Z_{t+1})\mid\mathcal F_t]
-V_t^0(Z_t).
\]

Define

\[
D_{t+1}=
V_{t+1}^0(Z_{t+1})
-\mathbb E[V_{t+1}^0(Z_{t+1})\mid\mathcal F_t].
\]

These are martingale differences in the interleaved audit/update filtration. Exact telescoping gives

\[
F_E(Z_R)-V_0^0(Z_0)
=
\sum_{t=0}^{R-1}A_t+\sum_{t=0}^{R-1}D_{t+1}.
\]

Thus, with probability at least \(1-\delta\),

\[
\sum_t A_t\ge\sum_tB_t.
\]

This is a high-probability statement about the predictable cumulative drift. It is not by itself a high-probability terminal-value statement, and it is not a statement about a conditional terminal expectation given the global audit event.

#### Valid unconditional expected-value guarantees

Always \(A_t\in[-1,1]\). Consequently,

\[
A_t\ge-1_{G_t^c},
\qquad
A_t\ge B_t-2\,1_{G_t^c}.
\]

Taking unconditional expectations and using the martingale identity gives

\[
\boxed{
L_E(\mathrm{adaptive})-L_E(\mathrm{reference})
\ge-\sum_t\mathbb E\delta_t\ge-\delta,
}
\]

and

\[
\boxed{
L_E(\mathrm{adaptive})-L_E(\mathrm{reference})
\ge
\mathbb E\sum_tB_t-2\sum_t\mathbb E\delta_t.
}
\]

These expectations include audit randomness, selected updates, and the remaining declared training randomness. They do not condition on successful audits.

An exact unconditional no-degradation claim with no statistical penalty is generally false. A slightly inferior candidate can pass a finite-sample optimistic gate with small but positive probability. Reference fallback on the other outcomes then produces a small but strictly negative expected change.

A stopped refinement is also valid. Let \(\tau\) be the first failed audit, capped at \(R\). Stopping the martingale before that update and bounding the remaining terminal-value difference by 1 gives

\[
L_E(\mathrm{adaptive})-L_E(\mathrm{reference})
\ge
\mathbb E\sum_{t<\tau}B_t-\Pr(\tau<R)
\ge
\mathbb E\sum_{t<\tau}B_t-\delta.
\]

The stopped sum is a theoretical quantity because the true audit-failure event is generally unobservable.

#### Why conditioning on global audit success is unsafe

Future audit outcomes depend on visited learner states. Therefore the event \(\bigcap_tG_t\) may select trajectories with unusually low or high continuation values, even when every applied provider has exactly the reference kernel. The martingale sum has unconditional mean zero, but need not have zero conditional mean given that event.

A simple conceptual construction uses two intermediate states with reference continuation values \(1/4\) and \(3/4\). Give them equal probability. Use a statistically identical candidate. In the lower-value state, couple the two audit arms identically so their contrasts are always zero. In the higher-value state, use independent paired rollouts, so a nonzero probability of a confidence failure remains. The all-audits-success event disproportionately retains lower-value trajectories. Conditioning on it lowers the expected final value although the actual training process is exactly the reference.

This is why the predictable-drift statement and the unconditional expected-value statement must be kept separate.

One can also combine the global audit event with an independent martingale tail inequality. Since each conditional increment has range width at most 1, for deterministic \(R\),

\[
\Pr\!\left(
\sum_tD_{t+1}<-\sqrt{\frac R2\log(1/\eta)}
\right)\le\eta.
\]

Therefore, with probability at least \(1-\delta-\eta\),

\[
F_E(Z_R)-V_0^0(Z_0)
\ge
\sum_tB_t-\sqrt{\frac R2\log(1/\eta)}.
\]

This generic terminal bound is often vacuous for long training. It should not be advertised as a useful single-run neural performance certificate without a sharper variance analysis.

#### Practical boundaries

- Exact current-state auditing removes the need to extrapolate from a checkpoint bank. It does not remove restoration, independent replication, full-suffix cost, or stationarity requirements.
- Reusing a winning audit-trained checkpoint as the actual update changes the selected kernel and invalidates the fresh-update argument.
- If audit outcomes themselves alter the reference learner’s future behavior, audit transitions have their own effect and must be included in the state-process decomposition.
- Short-horizon or critic-proxy audits are different estimands. They require an explicit proxy-bias term; they do not inherit the full-suffix result automatically.
- This is a general sequential decision argument on learner state, not an original policy-improvement principle.

### C4. Assessment of the unified chain

The proposed chain is mathematically coherent when its quantifiers remain explicit:

1. **Sufficiency quotient.** Two admissible information laws are equivalent when every declared learning-consequence test has the same expectation. At a single boundary, this corresponds exactly to substitutability for the declared updates and suffixes.
2. **Approximation width.** For a specified message interface, the maximum consequence disagreement among laws with the same transmitted information measures unresolved ambiguity. Finite unrestricted law classes yield the uniform-approximation duality; constrained classes, including fixed-marginal couplings, have their own feasible widths.
3. **Composition.** Learning updates, subsequent learning, and deployment determine the consequence class. Affine moments, nonlinear degree growth, parity tasks, and packet-response tables are different specializations of this same operation.
4. **Audit and action.** Real experiments estimate selected consequence contrasts. They can test fidelity or select a beneficial provider. Frozen global audits and adaptive actual-state audits require different probability statements.

This formulation is broader than one proposed neural metric, a world-model implementation, or a particular moment order. It also explains why no fixed, universally small interface can cover arbitrary tasks and learning algorithms.

Three distinctions must remain visible:

- A one-boundary substitutability quotient is stronger than equality of one final aggregate mean; endpoint cancellations can hide local errors.
- Fidelity to a reference, improvement over that reference, and information-theoretic learnability are different targets. They can be connected, but none implies the others without additional conditions.
- A mathematical quotient or width is a characterization. Efficiently learning its features and implementing a good provider remain algorithmic questions.

The chain is a valid organizing theory. Its ingredients are established mathematical ideas; the audit does not establish that their integration is novel. A substantial contribution would require a new nontrivial characterization or algorithmic consequence within this chain and evidence that it predicts or changes actual agent-learning behavior.


## 10. Tolerant feature measurement and robust decision ambiguity

### D1. Exact duality with nonzero feature tolerances

**Statement reviewed.** On a finite message alphabet, let \(g\) and fixed features \(\phi_1,\ldots,\phi_d\) be finite real-valued vectors, and let \(\varepsilon_j\ge0\). Define

\[
W_\varepsilon(g)=
\sup_{\substack{P,Q\ \text{probability laws}\\
|(\mathbb E_P-\mathbb E_Q)\phi_j|\le\varepsilon_j}}
|(\mathbb E_P-\mathbb E_Q)g|.
\]

Then

\[
\boxed{
W_\varepsilon(g)
=
\min_{b,c}
\left[
2\left\|g-c-\sum_jb_j\phi_j\right\|_\infty
+\sum_j\varepsilon_j|b_j|
\right]
=
\min_b
\left[
\operatorname{osc}\!\left(g-\sum_jb_j\phi_j\right)
+\sum_j\varepsilon_j|b_j|
\right].
}
\]

Here \(\operatorname{osc}(h)=\max_yh(y)-\min_yh(y)\).

**Verdict: valid, including zero tolerances and linearly dependent features.**

Every probability-law difference \(v=P-Q\) satisfies \(1^\top v=0\) and \(\|v\|_1\le2\). Conversely, any such vector is a probability-law difference: its positive and negative parts have equal mass at most 1, and the same leftover mass can be added to both. The feasible set is symmetric, so the absolute value does not change the supremum.

Write \(\Phi b=\sum_jb_j\phi_j\). The primal is the finite linear program

\[
\max_v g^\top v
\quad\text{subject to}\quad
1^\top v=0,\quad
\|v\|_1\le2,\quad
-\varepsilon\le\Phi^\top v\le\varepsilon.
\]

Dualizing the last pair of inequalities introduces nonnegative multipliers \(u,w\), with \(b=u-w\). Their penalty is \(\sum_j\varepsilon_j(u_j+w_j)\). For any fixed \(b\), its minimum over these representations is \(\sum_j\varepsilon_j|b_j|\). The remaining support function is

\[
\sup_{1^\top v=0,\ \|v\|_1\le2}(g-\Phi b)^\top v
=\operatorname{osc}(g-\Phi b).
\]

Finite LP strong duality gives equality, rather than only weak-duality's upper bound. The primal is nonempty and compact, and its optimum is finite. The dual has a feasible point, for example \(b=0\) with a finite residual bound, and its optimum is attained. Zero tolerances or rank-deficient features do not require a strict-interior condition.

Finally,

\[
\operatorname{osc}(h)=2\min_c\|h-c1\|_\infty,
\]

with midpoint centering \(c=(\max h+\min h)/2\). This proves both displayed forms.

**Scope.** This is exact for unrestricted pairs of probability laws satisfying the tolerances. For a fixed reference \(P\), restricted realizable laws, or fixed marginal couplings, it remains an upper bound but need not be exact. Taking \(b=0\) gives \(W_\varepsilon(g)\le\operatorname{osc}(g)\), which is at most 1 when \(g\in[0,1]\).

**Independent numerical check.** Take \(Y=\{0,1/2,1\}\), \(\phi(y)=y\), and \(g(y)=1\{y=1\}\). Direct primal and dual LPs agree with

\[
W_\varepsilon(g)=\min\{1,\tfrac12+\varepsilon\}.
\]

For \(0\le\varepsilon\le1/2\), use \(Q=\delta_{1/2}\) and let \(P\) put mass \(1/2+\varepsilon\) at 1 and \(1/2-\varepsilon\) at 0. The feature discrepancy is \(\varepsilon\) and the response discrepancy is \(1/2+\varepsilon\). The dual choice \(b=1\) has residual oscillation \(1/2\), giving the matching upper bound. For larger tolerances, \(P=\delta_1,Q=\delta_{1/2}\) attain width 1 and \(b=0\) certifies it. Separate SciPy primal/dual solves reproduced these values at tolerances \(0,0.05,0.25,0.5,0.8,1.5\), including the zero-tolerance endpoint.

**How this closes the finite measurement bridge.** On a simultaneous feature-audit event, set

\[
\varepsilon_j=
|\widehat\mu_{P,j}-\widehat\mu_{Q,j}|+e_{P,j}+e_{Q,j}.
\]

The actual pair is feasible for \(W_\varepsilon\). Therefore its consequence error is bounded by the optimized residual-plus-measurement expression above. This explicitly trades representation residual against sensitivity to feature measurement error.

The coefficients \(b\) may be optimized after observing these valid simultaneous feature bounds: that event controls every linear combination of the fixed features, so choosing \(b\) does not itself require another finite-candidate union bound. If the feature family changes after seeing the audit, a new uniform argument is needed.

The formula still requires the actual response function or a certified approximation. If a finite response estimate obeys \(\|\widehat g-g\|_\infty\le\eta\) on a separate valid uniform event, then

\[
W_\varepsilon(g)\le W_\varepsilon(\widehat g)+2\eta.
\]

All event failure probabilities must be combined. A fitted neural response predictor with no uniform residual control does not receive this guarantee.

This is a direct application of LP duality, not a new duality principle. It does provide an exact connection between the framework's approximation question and its finite-feature measurement error.

### D2. Robust ambiguity value and its probability interpretation

I also checked the finite-matrix extension in Proposition 2 of the current English framework. For a known finite ambiguity class \(C\) and finite policy class \(\Pi\), define

\[
r(C)=
\min_{\rho\in\Delta(\Pi)}
\max_{E\in C}
\left[J_E^\Pi-\mathbb E_{\pi\sim\rho}J_E(\pi)\right].
\]

Then

\[
r(C)=
\max_{\mu\in\Delta(C)}
\left[
\sum_E\mu_EJ_E^\Pi
-\max_{\pi\in\Pi}\sum_E\mu_EJ_E(\pi)
\right].
\]

**Verdict: valid; finite zero-sum minimax.**

Apply minimax to the nonnegative matrix
\(M_{E,\pi}=J_E^\Pi-J_E(\pi)\). The inner minimization over policy mixtures for a fixed environment mixture is attained by a pure policy, giving the displayed formula.

If all permitted observations have identical laws across \(C\), any learner induces a common output mixture, so its worst-case expected regret is at least \(r(C)\). The known-class minimax mixture attains that value. Furthermore, \(r(C)=0\) if and only if there is a common optimal policy: a zero expected nonnegative gap in every environment forces every positive-mass policy of an optimal finite mixture to be optimal in every environment.

**Probability clarification.** If a random confidence set \(C_D\) contains the true environment with probability at least \(1-\delta\), and the selected mixture has worst-case gap at most \(\varepsilon\) on that set, then with probability at least \(1-\delta\) over the data its **mixture-averaged** regret is at most \(\varepsilon\). This does not automatically imply that one policy sampled from the mixture has regret at most \(\varepsilon\) with probability \(1-\delta\). Such a statement needs a further tail argument, a stronger per-policy support guarantee, or a definition treating the mixture itself as the deployed randomized policy.

The finite-policy convention is sufficient here. A compact infinite policy class can be handled separately with continuity and the appropriate minimax/attainment assumptions; one should not silently invoke the finite-matrix theorem in that setting.


## 11. Final integration signoff and audit freeze

**Standalone manuscript reviewed:** Environment_Information_Theory_EN.md, version 0.14, 880 lines, 67,221 bytes.  
**Reviewed SHA-256:** 9271ae051b6a31fc15907a0469354e46a8d043493f9cf353daca2809c58ac1e6.  
**Integration review date:** 2026-10-10.

I read the completed standalone manuscript and checked that its actual statements, assumptions, and probability interpretations match the reviewed claims above. The integration review identified five wording/assumption omissions, all of which were corrected and then checked in the hashed version:

1. The active testing lower bound now requires \(0<\delta<1/2\) and success in each of the two worlds.
2. Learning-process comparisons explicitly start from the same complete initial-state law.
3. Response estimation distinguishes the provider-integrated mean from a pointwise response conditional on a fixed realizable message.
4. An empirical provider constructed from audit observations explicitly requires a new independent audit or a valid correction; freezing and resampling it does not inherit the original confidence event.
5. The coupling characterization explicitly assumes the full coupling polytope and validity of every Cartesian-product pair, with a caveat for forbidden pairings.

The particularly consequential final statements are correctly qualified:

- Proposition 2's confidence guarantee concerns mixture-averaged regret on the high-probability data event.
- Theorem 18's noisy-feature duality is sharp for unrestricted pairs of finite message laws and only an upper bound under additional unmodeled restrictions.
- Theorem 19 separates fixed-provider population guarantees from individual training-seed outcomes and requires a fresh audited training law.
- Theorem 20 states high-probability predictable-drift control and unconditional expected-value bounds with statistical penalties. It does not incorrectly condition terminal expectation on global audit success.
- The main text distinguishes finite mathematical claims from empirical neural metrics, proxy-bias assumptions, access/restoration requirements, and unverified cross-task transfer.

**Integration disposition:** no further specific mathematical integration corrections were identified in this hashed version. This is a qualified independent model-agent signoff on consistency with the reviewed assumptions and proofs. It is not formal verification, external human peer review, a novelty certification, or evidence of neural-agent efficacy.

Earlier repair requests in this audit are retained as the review history; the five items above are resolved in the final standalone manuscript. The audit is frozen after this integration record.





