# Proposition 9.1 · Coupling width

*Statement*

[← Framework](../README.md) · [Statement](09-01_coupling_width.md) · [Proof](../proofs/09-01_coupling_width.md) · [Experiments](../experiments/09-01_coupling_width.md) · [中文](09-01_coupling_width_ZH.md)

**In one paragraph.** Training data is consumed in packets: a training update sees several units at once. Suppose the content of the units is fixed, in the sense that the marginal law of what fills each slot of a packet is fixed, and only the pairing of units into packets is changed. Can that change what the student learns? This proposition answers exactly: regrouping never matters if and only if the response to a packet is a sum of per-unit contributions, and otherwise the possible change is bounded by how far the response is from additive. It also gives a strict null: one SGD step at a common parameter is exactly invariant to regrouping, while sequential steps are not. This turns "does packet organization matter for this learner?" into a measurable quantity with an exact null.

**Setting used on this page.**

- *Units and slots.* $`\mathcal Y_1=\lbrace1,\dots,m\rbrace`$ and $`\mathcal Y_2=\lbrace1,\dots,n\rbrace`$ are finite sets of valid feedback units, for example the units that fill two slots of a training packet. Every pair $`(i,j)`$ is a valid packet unless stated otherwise.
- *Marginals.* $`\mu\in\Delta(\mathcal Y_1)`$ and $`\nu\in\Delta(\mathcal Y_2)`$ are the slot marginals, assumed strictly positive. With zero entries, everything below holds on the positive-support product.
- *Couplings.* The coupling polytope is $`\mathcal C(\mu,\nu)=\lbrace\Gamma\in\mathbb R^{m\times n}_{\ge0}:\Gamma\mathbf 1=\mu,\ \Gamma^\top\mathbf 1=\nu\rbrace`$: all ways of pairing the units while keeping both slot marginals, and hence the per-unit content, fixed.
- *Packet response.* At a complete learner state $`z`$ ([Definition 1.3](../README.md#s1-3)), $`H_z(i,j)`$ is the expected final task return after the actual update on the joint packet $`(i,j)`$, followed by the reference tail. It is a one-boundary response ([Definition 1.4](../README.md#s1-4), [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2)) with the packet as the message. The expected learning value of a pairing $`\Gamma`$ is $`\langle\Gamma,H_z\rangle=\sum_{i,j}\Gamma_{ij}H_z(i,j)`$.
- *Coupling width.* $`w(H_z)=\max_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle-\min_{\Gamma\in\mathcal C(\mu,\nu)}\langle\Gamma,H_z\rangle`$: the largest change in expected learning value obtainable by reorganizing which units are packed together.
- *Additive table.* $`H=a\mathbf 1^\top+\mathbf 1b^\top`$, i.e. $`H(i,j)=a_i+b_j`$.

<a id="s9-1"></a>

**Proposition 9.1 (Coupling width).** In the setting above:

- **(a) Zero width.** $`w(H_z)=0`$ if and only if $`H_z(i,j)=a_i+b_j`$ for some $`a\in\mathbb R^m`$ and $`b\in\mathbb R^n`$.
- **(b) Bound by the additive residual.** For all $`\Gamma,\Gamma'\in\mathcal C(\mu,\nu)`$,

```math
\lvert\langle\Gamma-\Gamma',H_z\rangle\rvert\le2\inf_{a,b}\big\lVert H_z-a\mathbf 1^\top-\mathbf 1b^\top\big\rVert_\infty .
```

- The inequality in (b) can be strict.
- **(c) Restricted pairings.** If some pairs are forbidden, so that the feasible couplings form a proper face of $`\mathcal C(\mu,\nu)`$, zero feasible width does not imply additivity.
- **(d) Strict additive null.** One SGD step at a common parameter, on the union of all packets with identical per-example weights, is invariant to how the examples are grouped into packets. Two or more sequential steps are not invariant in general; for quadratic losses the difference is of order $`\eta^2`$: when packet $`P_1`$ is applied before $`P_2`$, the grouping-dependent part of the result is $`\eta^2H_{P_2}g_{P_1}(\theta_0)`$.

**What it says.**

- Regrouping units with fixed marginals can change the learning outcome only through the non-additive part of the packet response.
- The size of the possible change is at most twice the sup-norm distance of the response from additive tables.
- A single step at a common parameter is an exact null; real effects of grouping appear once one update feeds into the next.

**Intuition.** If the learner "sees" each unit on its own, grouping is irrelevant. Grouping matters only through interactions between units, and those appear when the second packet's gradient is evaluated after the first packet has already moved the parameters.

**Role in the framework.**

- It is a mechanism test with a strict null: any measured single-step regrouping effect signals an implementation or randomness difference.
- It is the reason a candidate feedback law in the method is specified by the packets it emits, not by a bag of units ([Algorithm 9.2](09-02_ced_algorithm.md#s9-2), step 3; [Remark 9.3](09-03_derivation_design_rules.md#s9-3), Rule 2).
- The result is coupling linear algebra; no new duality is claimed.

**Derivation outline.**

- (a) An additive table has the same inner product $`\mu^\top a+\nu^\top b`$ with every coupling. Conversely, the product coupling $`\mu\nu^\top`$ is interior, so every rectangle move $`e_{ij}+e_{i'j'}-e_{ij'}-e_{i'j}`$ is feasible in both directions; zero width forces every rectangle sum of $`H_z`$ to vanish, which is additivity.
- (b) The difference of two couplings has zero row and column sums, so it annihilates additive tables, and its $`\ell_1`$ norm is at most 2; Hölder's inequality gives the bound.
- (c) Forbidding one pair makes the indicator of that pair invisible to every feasible coupling, although it is not additive.
- (d) One step sums the per-example gradients at the same parameter, which does not depend on the partition. In two steps the second packet's gradient is evaluated after the first update, which produces the cross term $`\eta^2H_{P_2}g_{P_1}`$.

**Illustration.** With $`\mu=\nu=(0.9,0.1)`$ and $`H=e_{11}`$, every coupling has $`\Gamma_{11}=x\in[0.8,0.9]`$, so the width is $`0.1`$; the best additive approximation leaves a checkerboard residual of size $`1/4`$, so the bound (b) is $`0.5`$. The bound is valid but loose.

**Uses.**

- [Definition 1.4](../README.md#s1-4) and [Theorem 3.2](03-02_continuation_response_characterization.md#s3-2): the one-boundary response, here with a packet as the message.
- [Theorem 4.2](04-02_sufficiency_duality_transfer.md#s4-2): the contrast with unrestricted laws, where the analogous bound is an equality.

**Used by.**

- [Algorithm 9.2](09-02_ced_algorithm.md#s9-2), step 3: candidates are packet laws.
- [Remark 9.3](09-03_derivation_design_rules.md#s9-3), Rule 2: packet organization is part of the candidate.

## Related work

- The coupling polytope $`\mathcal C(\mu,\nu)`$ is the feasible set of the transportation problem of linear programming ([Bertsimas & Tsitsiklis, 1997](#ref-bertsimas1997introduction)); (a)–(c) are linear algebra on that polytope.
- The cross term $`\eta^2H_{P_2}g_{P_1}`$ in (d) is the second-order effect of evaluating a gradient after an earlier update, the same term that produces the drift correction in stochastic modified equations for SGD ([Li et al., 2019](#ref-li2019stochastic)).
