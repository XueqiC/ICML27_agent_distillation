# Proposition 2.7 · Compilation cost

*Statement*

[← Framework](../README.md) · [Statement](02-07_compilation_cost.md) · [Proof](../proofs/02-07_compilation_cost.md) · [Experiments](../experiments/02-07_compilation_cost.md) · [中文](02-07_compilation_cost_ZH.md)

**In one paragraph.** Finding out what to do and teaching the student to do it are different jobs. Once the sign of the two-world family is estimated, the student still has to be trained toward the right action; this proposition shows that this compilation step needs a number of synthetic labels that depends only on the target confidence, not on the gap $`\Delta`$.

**Setting used on this page.**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects one of two worlds; the gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform. The teacher sees $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$; both succeed with probability 1, so the teacher is optimal in both worlds.
- The student is aliased (it cannot see $`s`$) and plays $`A`$ with probability $`p`$. The other two state–action pairs succeed with probability $`\tfrac12\mp2v\Delta`$, so the student's value is $`J_v(p)=\tfrac34+v\Delta(2p-1)`$ and its best value is $`J_v^\Pi=\tfrac34+\Delta`$, at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$.
- *Passive interface:* the learner sees teacher demonstrations (possibly with $`s`$) and post-processing of them. *Query interface:* it may also reset to $`s=1`$, play $`A`$ and observe the Bernoulli $`(\tfrac12+2v\Delta)`$ outcome, adaptively, with a stopping time $`N`$.
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ is the binary KL divergence.
- Full definition: [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5).
- $`\widehat v`$ is the learner's estimate of the sign, for example from the query test of [Theorem 2.6(c)](../statements/02-06_passive_impossibility_active_attainability.md#s2-6). A *frequency learner* sets the student's $`\widehat p`$ to the empirical frequency of $`A`$ among $`m`$ training labels; $`\delta_L`$ is the allowed compilation failure probability.

<a id="s2-7"></a>

**Proposition 2.7 (Compilation cost).** In the family of [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5), after the sign is estimated as $`\widehat v`$:

- choose the target $`q=7/8`$ if $`\widehat v=+1`$ and $`q=1/8`$ otherwise;
- generate $`m`$ Bernoulli $`(q)`$ target labels, and let the student be their empirical frequency $`\widehat p`$.

If $`m\ge\lceil32\log(2/\delta_L)\rceil`$, then $`\lvert\widehat p-q\rvert\le1/8`$ with probability at least $`1-\delta_L`$. If $`\widehat v`$ is correct with probability at least $`1-\delta`$ ([Theorem 2.6(c)](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)), the compiled student has regret at most $`\Delta/2`$ with probability at least $`1-\delta-\delta_L`$.

**What it says.**

- After the sign is estimated, $`O(\log(1/\delta_L))`$ synthetic labels drawn toward the estimated optimum train a frequency learner to within $`\tfrac18`$ of the target, with probability at least $`1-\delta_L`$, independently of $`\Delta`$.
- Together with a correct sign, this gives a student with regret at most $`\Delta/2`$ with probability at least $`1-\delta-\delta_L`$.

**Intuition.**

- Acquiring the fact costs real environment queries and grows like $`\Delta^{-2}`$; compiling it costs cheap synthetic labels and does not grow with $`1/\Delta`$.

**Role in the framework.**

- It separates the acquisition budget (authentic queries) from the compilation budget (synthetic labels) in the accounting of the method ([Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2)).

**Derivation outline.**

- The empirical frequency of the compiled labels is controlled by the two-sided Hoeffding inequality, and a union bound combines the identification and compilation failures. The number of labels does not depend on $`\Delta`$.

**Illustration.** Identification costs authentic environment queries, order $`\Delta^{-2}\log(1/\delta)`$; compilation costs synthetic labels, order $`\log(1/\delta_L)`$. At small $`\Delta`$ almost the whole budget is acquisition, and the two must be accounted for separately.

**Comments.**

- **Acquisition and compilation are separate costs.**
  - Identification costs $`\Theta(\Delta^{-2}\log(1/\delta))`$ authentic environment queries; the lower and upper bounds of [Theorem 2.6(b),(c)](../statements/02-06_passive_impossibility_active_attainability.md#s2-6) match up to constants.
  - Compilation then costs $`O(\log(1/\delta_L))`$ synthetic target labels, independent of $`\Delta`$. These labels are compiled, not new teacher executions, and are accounted for separately.

**Uses.**

- [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5): the two-world family.
- [Theorem 2.6](../statements/02-06_passive_impossibility_active_attainability.md#s2-6): identification of the sign by queries.

**Used by.**

- [Algorithm 9.2](../statements/09-02_ced_algorithm.md#s9-2): separate acquisition and compilation budgets.

## Related work

- Both the identification step and the compilation step are controlled by Hoeffding's inequality for bounded independent variables ([Hoeffding, 1963](#ref-hoeffding1963probability)). Compilation needs only the two-sided form at the fixed accuracy $`\tfrac18`$, which is why the number of labels does not depend on $`\Delta`$.

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-hoeffding1963probability"></a>

**Wassily Hoeffding (1963).** Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58, no. 301, 13–30. DOI [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830).
