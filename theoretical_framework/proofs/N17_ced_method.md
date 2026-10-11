# N17 · Consequence-Calibrated Experience Distillation (CED) — Derivation

[← Framework map](../README.md) · [Experiments](../experiments/N17_ced_method.md) · [中文](N17_ced_method_ZH.md)

This page derives the six CED design rules from the framework. Each rule is a proposition stating what any method that claims to preserve or improve the learning value $`L_E`$ must do, followed by the theorems that force it. The rules are necessary conditions for the stated guarantees. They do not by themselves establish that a CED instance outperforms a given baseline.

## Setting and assumptions

**Few-shot agent distillation.** A student starts from pretrained information $`I_{\rm pre}`$ with $`K`$ trusted teacher demonstrations per task family.

**Acquisition budget.** Further permitted experiments (teacher calls, environment branches, resets) form a separately accounted budget.

**Providers.** A provider $`j`$ is an executable rule that, at a complete learner state $`z`$, emits feedback messages $`Y\sim P^j_{t,z}`$ (training units, packets, targets).

**Recipe.** Training uses a declared recipe: update $`U_t`$, optimizer (for example AdamW), horizon $`R`$, and reference suffix. The final student is deployed alone, with value $`F_E(z)=J_E(\pi_z)`$.

**Response and provider value.** As in N05,

```math
g_{t,z}(y)=\mathbb E_\xi V^0_{t+1}(U_t(z,y,\xi)),\qquad G(z,j)=\mathbb E_{Y\sim P^j_{t,z}}g_{t,z}(Y),
```

and the signed local advantage is $`A_t(z,j)=G(z,j)-G(z,0)`$.

## Statement

**(a)** **Rule 1 (acquire before compiling).**

- Let $`D`$ be the transcript and $`D'=C(D)`$ any randomized post-processing: replay, rewriting, paraphrase or synthetic descendants. Then $`\mathcal R_{\rm info}(D')\ge\mathcal R_{\rm info}(D)`$.
- Consequently, residual decision ambiguity can be reduced only by enlarging $`D`$ with new permitted experiments. The size of this acquisition is governed by:
  - $`\mathbb E n_{\rm query}\ge\mathrm{kl}(1-\delta,\delta)/\mathrm{kl}(\tfrac12+2\Delta,\tfrac12-2\Delta)`$ in the two-world family;
  - $`R_{\rm info}(K,n)=\tfrac12\mathbb E\sum_{r>n}w_{(r)}`$ in the hidden-bit model;
  - $`K\kappa_D+\kappa_X\mathbb E N\ge\mathrm{kl}(1-\delta,\delta)`$ in general two-world models.

**(b)** **Rule 2 (select by measured consequence).**

- A selection rule is consistent with the learning-value ordering on a declared family only if it is a function of $`\lbrace G(z,j)\rbrace _j`$, or of a statistic proved sufficient for $`G`$ on that family.
- No rule that depends only on teacher value and every student's static policy value, or only on the label marginals and the mean consumed gradient, can select the better provider on every instance.

**(c)** **Rule 3 (record the optimizer state).**

- The statistic of the feedback law that determines $`G(z,\cdot)`$ depends on the complete state $`z=(\theta,m,v,t,\dots)`$.
- At a cold Adam state it is the joint sign law of the consumed gradient, together with the magnitudes retained in $`m`$. At a warm state with warmth $`\kappa`$ it is the mean preconditioned consumed gradient, up to the residual $`r(y)`$ of [N12(W1)](N12_optimizer_warmth.md).
- Response measurements and their transport must therefore be indexed by complete states and their warmth, not by parameters alone.

**(d)** **Rule 4 (fit stable response models).**

- Consider response models $`c+b^\top\phi`$ on features measured to simultaneous tolerances $`\epsilon_j`$, with a response estimate $`\widehat g`$ satisfying $`\lVert\widehat g-g\rVert_\infty\le\eta`$.
- The certified unresolved discrepancy is minimized by minimizing $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert`$. Its optimal value plus $`2\eta`$ is the exact bound.
- A model with smaller residual but larger $`\sum_j\epsilon_j\lvert b_j\rvert`$ yields a strictly worse certificate whenever the sum of the two terms is larger.

**(e)** **Rule 5 (account for coverage).** A guarantee on $`L_E`$ for the selected provider must take one of three forms:

- (i) the frozen-audit certificate [N14(b)](N14_frozen_audit_transport.md), including measured transport $`a_{t,j}(z)`$ and exit probability $`\rho`$;
- (ii) the actual-state lower-confidence-bound guarantees [N15(c)](N15_adaptive_audit.md);
- (iii) a population-average comparison of the frozen method by whole-run randomized trials, labelled empirical.

Selection by uncorrected point estimates, or a transport-free extrapolation, carries no such guarantee.

**(f)** **Rule 6 (freeze, then test transfer).**

- Selectors, features, cells and providers must be frozen before the data that evaluates them.
- Transfer to a target task family is guaranteed only through the decomposition of target responses [N06(c)](N06_sufficiency_duality.md). It must otherwise be measured on held-out families, recipes and states.
- Method transfer and policy transfer are distinct quantities and are evaluated separately.

## Proof

**(a)** **Rule 1.**

- [N01(d)](N01_decomposition_and_ambiguity.md): $`\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D']\le\mathbb E\max_\pi\mathbb E[J_E(\pi)\mid D]`$ for randomized post-processing $`D'`$ of $`D`$. This is the tower property plus convexity of the maximum. Hence $`\mathcal R_{\rm info}(D')=\mathbb EJ^\Pi_E-\mathbb E B(D')\ge\mathcal R_{\rm info}(D)`$. Compilation can lower only $`\mathcal R_{\rm alg}`$.
- [N02(a)–(d)](N02_passive_vs_active_acquisition.md) (N02): passive data alone leave worst-case regret at least $`\Delta`$, and the query lower and upper bounds [N02(b)](N02_passive_vs_active_acquisition.md)–[N02(c)](N02_passive_vs_active_acquisition.md) quantify the acquisition needed.
- [N03(a)](N03_few_shot_information_floor.md), [N03(b)](N03_few_shot_information_floor.md) (N03): unrevealed bits remain fair given the transcript, so replaying revealed content changes nothing. $`R_{\rm info}(K,n)`$ is achieved by querying the heaviest unseen bits.
- [N03(c)](N03_few_shot_information_floor.md) (N03): demonstration and query information add in one KL budget.

These four facts give Rule 1.

**(b)** **Rule 2.**

- *Necessity of the response.* By [N05(b)](N05_continuation_response.md), two message laws are interchangeable for every one-boundary replacement with the declared suffix if and only if $`\mathbb E_Pg=\mathbb E_Qg`$ for all $`g`$ in the declared response family. The value of replacing the reference by provider $`j`$ at state $`z`$ is exactly $`A_t(z,j)`$, and by the signed identity [N15(b)](N15_adaptive_audit.md) (N15) the total effect of a provider policy is $`\sum_t\mathbb E_{d^S_t}A_t`$. Any rule ordering providers consistently with $`L_E`$ on the family must therefore order them as $`\lbrace G(z,j)\rbrace`$ does.
- *Static statistics.* [N04(a)](N04_static_vs_learning_sufficiency.md) (N04) gives two sources $`S_1,S_2`$ that agree on the teacher value $`3/5`$ and on $`J(\pi_p)=\tfrac12+p/20`$ for every aliased student. Frequency distillation from them yields $`43/80`$ and $`41/80`$. A rule depending only on the shared statistics returns the same choice, by position or by a common randomization, for the instance $`(S_1,S_2)`$ and its relabelling $`(S_2,S_1)`$. So it selects the worse provider in at least one of the two, or each with probability $`1/2`$.
- *Mean gradients.* [N04(b)](N04_static_vs_learning_sufficiency.md) (N04) gives two packets with equal label marginals and equal mean gradient whose learning values differ by $`\eta^2/8`$ under $`J(p)=p^2`$; under $`J(p)=1-p^2`$ the preference reverses. These label statistics do not involve $`J`$. A rule depending only on them therefore makes the same choice for both tasks, and is wrong for one of them.
- *Packet organization is part of the provider.* By [N16(a)–(b)](N16_coupling_width.md) (N16), regrouping units with fixed marginals changes the learning value exactly through the non-additive part of the joint-packet response $`H_z`$. A measured response must therefore be the response to the packets actually emitted, not a sum of per-unit scores.
- *No universal small statistic.* For unrestricted learners, [N06(d)](N06_sufficiency_duality.md) (N06) shows that no fixed family of fewer than $`m-1`$ statistics determines all response means. Sufficient statistics exist only for restricted families, for example joint moments to order $`N`$ for affine learners (N07).

**(c)** **Rule 3.**

- [N11(a)–(c)](N11_adam_cold_start.md) (N11): at a cold state and $`\epsilon\to0`$, $`\theta_1=c-\eta\mathrm{sign}(G)`$, and bounded immediate utilities are determined exactly by the joint sign law.
- [N11(e)](N11_adam_cold_start.md) (N11): the complete next state contains $`m_1=(1-\beta_1)G`$. Two histories with identical sign sequences yield opposite second steps, so the sign law does not determine the continuation.
- [N12(W1)](N12_optimizer_warmth.md) (N12): at a warm state, $`\lvert\psi(y)-c-b^\top G(y)\rvert\le r(y)`$, with $`r`$ controlled by $`\kappa`$ and the curvature $`H_G`$. Equal mean consumed gradients thus imply response gaps of at most $`\mathbb E_{P_0}r+\mathbb E_{P_1}r`$.
- [N12(W2)](N12_optimizer_warmth.md) (N12): at a cold state, equal-mean laws separate at $`\Theta(\eta)`$.
- The identifying statistic therefore changes with $`(m,v,t)`$.
- The witness in N14 (identical $`\theta`$, different $`(m,v)`$, next first moments $`\mp1/8`$) shows that two states that agree in parameters can differ in their next state. The transport condition [N14(b)](N14_frozen_audit_transport.md) must therefore be stated on complete states.

**(d)** **Rule 4.** [N13](N13_stability_tradeoff.md) (N13) gives $`W_{\mathbf{\epsilon}}(g)=\min_b[\mathrm{osc}(g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert]`$. By [N13(d)](N13_stability_tradeoff.md) of N13, $`\lvert(P-Q)g\rvert\le W_{\mathbf{\epsilon}}(\widehat g)+2\eta`$ for the actual pair on the simultaneous event. For every fixed $`b`$ the bound is $`\mathrm{osc}(\widehat g-\Phi b)+\sum_j\epsilon_j\lvert b_j\rvert+2\eta`$, so the minimizing $`b`$ gives the best certified bound. [N13(c)](N13_stability_tradeoff.md) of N13 ($`\phi=\alpha g`$, residual $`0`$, $`W=\min\lbrace 1,\epsilon/\alpha\rbrace`$) shows that a zero residual can coexist with a vacuous certificate.

**(e)** **Rule 5.**

- By N14, a frozen audit certifies $`L_E`$ through [N14(b)](N14_frozen_audit_transport.md). Its Part 6 shows that dropping the transport term can produce a bound of $`0.0296`$ for a gap of $`1`$, so form (i) must include $`a`$ and $`\rho`$.
- By N15, auditing at the actual states with the lower confidence bound gives [N15(c)](N15_adaptive_audit.md), which is form (ii). By [N15, Remark 1](N15_adaptive_audit.md#remarks), point-estimate selection can have negative expected gain.
- Neither form guarantees anything about a single training seed, and both are costly: on the order of $`10^3`$ paired full tails per candidate and state, by N15 Part 4.
- A claim about the mean outcome of a frozen method over a declared task/seed population is a different estimand. It is supported by whole-run randomized comparisons (form iii) and must be labelled as such. It does not become a local certificate.

**(f)** **Rule 6.**

- *Freezing.* By [N14(a)–(d)](N14_frozen_audit_transport.md) (N14), the audit event is simultaneous over a candidate set fixed *before* the audit. Reusing audit samples to construct or train the provider changes the provider law, and the certificate then needs a new independent audit ([N14, Remark 3](N14_frozen_audit_transport.md#remarks)). The same holds for any selector tuned on its evaluation data.
- *Transfer.* By [N06(c)](N06_sufficiency_duality.md), if every target-family response satisfies $`g=c+b^\top\phi+r`$ with $`\lVert b\rVert_1\le B`$ and $`\lVert r\rVert_\infty\le\tau`$, then $`\sup_g\lvert(P-Q)g\rvert\le B\lVert\mathbb E_P\phi-\mathbb E_Q\phi\rVert_\infty+2\tau`$. Conversely, by [N06(a)](N06_sufficiency_duality.md), if some target response lies outside $`H_\phi`$, there are feature-matched laws $`P,Q`$ that it separates. No audit on the training family alone can then certify the target. Absent a verified $`(B,\tau)`$ for the target family, transfer is an empirical quantity measured on held-out families.
- *Two kinds of transfer.* Method transfer evaluates $`L_{E'}(S)`$, where $`S`$ is the frozen selection protocol applied to a new family $`E'`$ with its own $`K`$ demonstrations; its response class is $`\mathcal G_{E'}`$, defined through updates. Policy transfer evaluates $`J_{E'}(\pi_z)`$ for the already distilled student $`z`$, with no further update; its test functions are $`F_{E'}`$. These are different functionals with different test classes, so a bound on one does not bound the other, and they are reported separately. $`\square`$

## Remarks

**The derived procedure.** The rules compose into one loop:

1. Expose residual ambiguity and acquire new evidence within the declared budget (Rule 1). Keep a ledger that separates new evidence from replay and from synthetic descendants.
2. Compile the evidence into valid units, and build a finite provider menu that includes the raw teacher reference.
3. At restorable complete states, with warmth recorded, measure each provider's actual update-plus-tail response (Rules 2–3).
4. Fit bounded-coefficient response models under the [N13](N13_stability_tradeoff.md) objective (Rule 4).
5. Select with form (i) or (ii) of Rule 5, or select empirically and validate by form (iii).
6. Freeze, then evaluate method transfer and policy transfer separately on held-out families, recipes and states (Rule 6).

**What the controlled experiments add** (see the experiments page):

- Static update statistics cannot replace measured responses, which strengthens Rule 2.
- Short measured tails are good proxies except where the deployed task has a threshold, so the tail length is part of the estimand.
- Moment warm-up compresses differences between providers without making static statistics predictive. Under Rule 3, warmth changes *which* statistic matters; it does not make a static statistic sufficient.

## Dependencies and source

- Uses [N01](N01_decomposition_and_ambiguity.md)–[N06](N06_sufficiency_duality.md) and [N11](N11_adam_cold_start.md)–[N16](N16_coupling_width.md), as cited in each rule.

**Source:** [v0.14.1 §13](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md#13-the-derived-few-shot-method-consequence-calibrated-experience-distillation); [v0.15 §6](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md#6-method-consequence-an-optimizer-aware-instance-of-ced); [warm-Adam §4](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md#4-how-warmth-evolves-and-a-free-way-to-warm-the-state); [OTLR §5](../../docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md#5-using-the-statistic-at-the-actual-states) — v0.14.1 §13; addendum §6; warm-Adam §4; OTLR §5. [Environment_Information_Theory_EN.md](../../docs/direction/v015_revision/Environment_Information_Theory_EN.md) §13, [Optimizer_State_and_Few_Shot_Theory_v015_EN.md](../../docs/direction/v015_revision/Optimizer_State_and_Few_Shot_Theory_v015_EN.md) §5–§6, [CED_Method_EN.md](../../docs/direction/v014_handoff/CED_Method_EN.md) §§3–6, and [Warm_Adam_Response_EN.md](../../docs/direction/v015_extension/Warm_Adam_Response_EN.md) §§2–4.
