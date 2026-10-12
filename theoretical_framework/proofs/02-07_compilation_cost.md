# Proposition 2.7 · Compilation cost

*Proof*

[← Framework](../README.md) · [Statement](../statements/02-07_compilation_cost.md) · [Proof](02-07_compilation_cost.md) · [Experiments](../experiments/02-07_compilation_cost.md) · [中文](02-07_compilation_cost_ZH.md)

**What is proved here.** Hoeffding's inequality controls the empirical frequency of the compiled labels, and a union bound combines the identification and compilation failures.

**Setting used on this page.**

- An unknown sign $`v\in\lbrace -1,+1\rbrace`$ selects one of two worlds; the gap $`0<\Delta\le1/8`$ is known.
- A hidden state $`s\in\lbrace 0,1\rbrace`$ is uniform. The teacher sees $`s`$ and plays $`A`$ at $`s=0`$ and $`B`$ at $`s=1`$; both succeed with probability 1, so the teacher is optimal in both worlds.
- The student is aliased (it cannot see $`s`$) and plays $`A`$ with probability $`p`$. The other two state–action pairs succeed with probability $`\tfrac12\mp2v\Delta`$, so the student's value is $`J_v(p)=\tfrac34+v\Delta(2p-1)`$ and its best value is $`J_v^\Pi=\tfrac34+\Delta`$, at $`p=1`$ if $`v=+1`$ and at $`p=0`$ if $`v=-1`$.
- *Passive interface:* the learner sees teacher demonstrations (possibly with $`s`$) and post-processing of them. *Query interface:* it may also reset to $`s=1`$, play $`A`$ and observe the Bernoulli $`(\tfrac12+2v\Delta)`$ outcome, adaptively, with a stopping time $`N`$.
- $`\mathrm{kl}(a,b)=a\log\frac ab+(1-a)\log\frac{1-a}{1-b}`$ is the binary KL divergence.
- Full definition: [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5).
- $`\widehat v`$ is the estimated sign; the frequency learner sets $`\widehat p`$ to the empirical frequency of $`m`$ Bernoulli $`(q)`$ labels.

**Statement ([Proposition 2.7](../statements/02-07_compilation_cost.md#s2-7)).** In the family of [Definition 2.5](../statements/02-05_two_world_aliased_family.md#s2-5), after the sign is estimated as $`\widehat v`$:

- choose the target $`q=7/8`$ if $`\widehat v=+1`$ and $`q=1/8`$ otherwise;
- generate $`m`$ Bernoulli $`(q)`$ target labels, and let the student be their empirical frequency $`\widehat p`$.

If $`m\ge\lceil32\log(2/\delta_L)\rceil`$, then $`\lvert\widehat p-q\rvert\le1/8`$ with probability at least $`1-\delta_L`$. If $`\widehat v`$ is correct with probability at least $`1-\delta`$ ([Theorem 2.6(c)](../statements/02-06_passive_impossibility_active_attainability.md#s2-6)), the compiled student has regret at most $`\Delta/2`$ with probability at least $`1-\delta-\delta_L`$.

**Proof.**

The two-sided Hoeffding inequality gives

```math
\Pr(\lvert\widehat p-q\rvert>\tfrac18)\le2\exp(-2m/64)=2e^{-m/32}\le\delta_L
```

when $`m\ge32\log(2/\delta_L)`$. Suppose $`\widehat v=v=+1`$ and $`\lvert\widehat p-q\rvert\le\tfrac18`$. Then $`\widehat p\ge\tfrac34`$ and the regret is $`2\Delta(1-\widehat p)\le\Delta/2`$. The case $`v=-1`$ is symmetric, with $`\widehat p\le\tfrac14`$. A union bound over the identification failure ($`\le\delta`$) and the compilation failure ($`\le\delta_L`$) gives the stated probability. $`\square`$

## Related work

- The sufficiency bounds use Hoeffding's inequality for bounded independent variables ([Hoeffding, 1963](../references.md#hoeffding1963probability)).

## References

- [Hoeffding, 1963](../references.md#hoeffding1963probability)
