# Lemma 2.9 · Demonstrations leak nothing about unrevealed bits

*Experiments*

[← Framework](../README.md) · [Statement](../statements/02-09_no_leakage_about_unrevealed_bits.md) · [Proof](../proofs/02-09_no_leakage_about_unrevealed_bits.md) · [Experiments](02-09_no_leakage_about_unrevealed_bits.md) · [中文](02-09_no_leakage_about_unrevealed_bits_ZH.md)

**What is tested here.** The lemma has no separate numerical statement; [Check 2.4](../experiments/02-10_few_shot_information_floor.md#c2-4) tests its consequences by brute force. In all 100 exhaustively enumerated cases, including 80 with correlated multi-bit reveal sets, the Bayes-optimal risk equals the closed form that the lemma implies, and the best value-adaptive query tree is no better than greedy querying (PASS).

**Setting used on this page.**

- Unknown bits $`\Theta_1,\dots,\Theta_d`$, independent and fair. Context $`j`$ occurs at deployment with weight $`w_j`$ ($`\sum_jw_j=1`$) and its correct decision is $`\Theta_j`$; the student can learn any context-to-action table, so there is no representation error.
- Each of the $`K`$ demonstrations reveals a random set $`S_k`$ of bits with their values; the sets are IID and independent of the bits, so the unrevealed labels are missing completely at random ([Rubin, 1976](#ref-rubin1976inference)), and $`q_j=\Pr(j\in S)`$ is the chance that one demonstration reveals bit $`j`$.
- After the demonstrations a hard budget allows $`n`$ queries, each revealing one chosen bit; choices may depend on everything seen so far.
- $`U_K`$ is the set of bits revealed by no demonstration and $`M_K=\lvert U_K\rvert`$. $`R_{\rm info}`$ denotes the minimum achievable expected regret (the information term of [Theorem 2.2](../statements/02-02_regret_decomposition_decision_ambiguity.md#s2-2)).
- Full definition: [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8).

**Statement under test ([Lemma 2.9](../statements/02-09_no_leakage_about_unrevealed_bits.md#s2-9)).** In the model of [Definition 2.8](../statements/02-08_hidden_bit_model.md#s2-8), conditional on the full transcript $`T`$, the unseen bits $`(\Theta_j)_{j\in U_K}`$ are IID fair and independent of $`T`$ and of the learner's private randomness.

### [Check 2.4](../experiments/02-10_few_shot_information_floor.md#c2-4) · Hidden-bit coverage, hard query budget and adaptivity

- *What it tests for this lemma.* If the unrevealed bits stay independent fair coins given the transcript, then (i) every learner's risk on an unseen bit is exactly one half, so the Bayes risk is half the expected unseen weight, and (ii) adapting queries to revealed values cannot help. A brute-force posterior that put any information on an unrevealed bit would break both.
- *Setup.* 100 cases with $`d\le5`$ bits, $`K\le4`$ demonstrations, $`n\le3`$ queries, random rational weights and random reveal-set laws over bitmasks (within-trajectory dependence allowed); all $`2^d`$ worlds and all ordered reveal histories enumerated exactly, label-dependent query trees minimized exactly.
- *Result.* Bayes risk equals the closed form in 100/100 cases; the adaptive optimum equals greedy in 100/100 cases; 80 cases have multi-coordinate reveal sets.
- *Verdict.* PASS.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theory/v015_checks/check_v015.py --only V5 --output theory/v015_checks/quick/sec2.json
```

## Artifacts

- `theory/v015_checks/results_v015_independent.json`: the group of `v015:V5`; code `theory/v015_checks/mathematics.py` (`check_V5`).

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-rubin1976inference"></a>

**Donald B. Rubin (1976).** Inference and missing data. *Biometrika* 63, no. 3, 581–592. DOI [10.1093/biomet/63.3.581](https://doi.org/10.1093/biomet/63.3.581).
